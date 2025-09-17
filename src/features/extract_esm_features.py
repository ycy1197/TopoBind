#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract ESM-2 (3B) embeddings for antibody-antigen sequences
and save them for downstream training.

- Input : alldata.csv (must contain antibody_seq_a, antibody_seq_b, antigen_seq)
- Output: esm_features/esm_embeddings.pkl (dict: pdb_id -> embedding)

Notes:
- Uses mean-pooled token embeddings from the last layer (layer 36).
- Truncates sequences longer than 2046 tokens.
"""

import os
import torch
import pickle
import pandas as pd
import numpy as np
from esm import pretrained

# Avoid CUDA memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Device setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No GPU found, using CPU")

# ------------------------------
# 1) Load ESM-2 model
# ------------------------------
print("Loading ESM2_t36_3B_UR50D model (~3B params)...")
model, alphabet = pretrained.esm2_t36_3B_UR50D()
batch_converter = alphabet.get_batch_converter()

model = model.to(device)
model.eval()

# ------------------------------
# 2) Read dataset
# ------------------------------
csv_file = "alldata.csv"
data_df = pd.read_csv(csv_file)

# Concatenate antibody heavy/light chains + antigen
data_df["full_sequence"] = (
    data_df["antibody_seq_a"] + data_df["antibody_seq_b"] + data_df["antigen_seq"]
)
sequences = data_df["full_sequence"].tolist()
pdb_ids = data_df["pdb"].str.split('_').str[0].tolist()

print(f"Total samples: {len(sequences)}")

# ------------------------------
# 3) Embedding extraction
# ------------------------------
def extract_embeddings(sequences, model, batch_converter, batch_size=4):
    """Extract mean-pooled embeddings for a list of sequences."""
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]

            # Truncate long sequences
            batch_data = []
            for j, seq in enumerate(batch_sequences):
                if len(seq) > 2046:
                    print(f"Warning: sequence {i+j} length {len(seq)} > 2046, truncating.")
                    seq = seq[:2046]
                batch_data.append((f"seq_{i+j}", seq))

            try:
                _, _, batch_tokens = batch_converter(batch_data)
                batch_tokens = batch_tokens.to(device)

                # Forward pass
                results = model(batch_tokens, repr_layers=[36], return_contacts=False)
                token_representations = results["representations"][36]

                # Mean pool over tokens
                sequence_embeddings = token_representations.mean(1)

                # Move to CPU
                all_embeddings.append(sequence_embeddings.cpu().numpy())

                # Free memory
                del batch_tokens, results, token_representations, sequence_embeddings
                torch.cuda.empty_cache()

                if (i // batch_size) % 5 == 0 or i + batch_size >= len(sequences):
                    print(f"Processed {min(i+batch_size, len(sequences))}/{len(sequences)} sequences")

            except RuntimeError as e:
                print(f"Error at batch {i}: {e}")
                torch.cuda.empty_cache()

    return np.vstack(all_embeddings)

print("Extracting embeddings...")
embeddings = extract_embeddings(sequences, model, batch_converter, batch_size=4)

# ------------------------------
# 4) Save embeddings
# ------------------------------
os.makedirs("esm_features", exist_ok=True)
esm_feature_dict = {pdb_ids[i]: embeddings[i] for i in range(len(pdb_ids))}

with open("esm_features/esm_embeddings.pkl", "wb") as f:
    pickle.dump(esm_feature_dict, f)

print("ESM embeddings saved to esm_features/esm_embeddings.pkl")

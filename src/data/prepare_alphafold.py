#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper to locate sequences for a given PDB ID from alldata.csv and
guide AlphaFold/ColabFold usage (no local prediction is run here).

It writes a FASTA file with antigen/antibody sequences for manual upload.

Usage:
    python alphafold_sequences.py  # runs example for '4FQR' at the bottom
"""

import os
import pandas as pd

CSV_FILE = "alldata.csv"

def find_uniprot_for_pdb(pdb_id: str, csv_file: str):
    """
    Look up antigen and antibody sequences for a PDB ID in the CSV.

    Expected CSV columns:
        - 'pdb'
        - 'antibody_seq_a'
        - 'antibody_seq_b'
        - 'antigen_seq'
    """
    df = pd.read_csv(csv_file)
    entry = df[df["pdb"].str.contains(pdb_id, case=False, na=False)]
    if not entry.empty:
        row = entry.iloc[0]
        return row["antibody_seq_a"], row["antibody_seq_b"], row["antigen_seq"]
    return None, None, None

def download_alphafold_prediction(pdb_id: str, output_dir: str = "PDB"):
    """
    Prepare FASTA file(s) and show instructions for AlphaFold/ColabFold usage.
    This function does NOT download predictions from AlphaFold automatically.
    """
    ab_a, ab_b, antigen = find_uniprot_for_pdb(pdb_id, CSV_FILE)
    if not antigen:
        print(f"Could not find sequences in CSV for {pdb_id}")
        return False

    # Save sequences to a FASTA file for convenience
    fasta_path = f"{pdb_id}_sequences.fasta"
    with open(fasta_path, "w") as f:
        f.write(f">{pdb_id}_antigen\n{antigen}\n")
        f.write(f">{pdb_id}_antibody_A\n{ab_a}\n")
        f.write(f">{pdb_id}_antibody_B\n{ab_b}\n")

    print("\nAlphaFold/ColabFold options:")
    print("Option 1: ColabFold (recommended)")
    print("  - Open: https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb")
    print("  - Upload the FASTA file:")
    print(f"    {os.path.abspath(fasta_path)}")
    print("Option 2: AlphaFold EBI portal")
    print("  - https://alphafold.ebi.ac.uk/  (submit sequences manually)")
    print(f"\nFASTA saved at: {fasta_path}")
    return True

# Example run
if __name__ == "__main__":
    download_alphafold_prediction("4FQR")

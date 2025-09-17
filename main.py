"""
main.py

End-to-end orchestration script:
1) Load ESM/topology features and CSV (alldata.csv)
2) Build dataset and split into train/val/test
3) Initialize and train the EnhancedCrossAttentionModel
4) Evaluate neural model on test set; save plots
5) Extract fused features for the entire dataset
6) Run LassoCV on a fresh train/test split (as in your original logic)
7) Save ROC and metrics

All hyperparameters, dimensions, thresholds, and logic are preserved exactly as you wrote.
Only the code is reorganized into modules with English comments/docstrings.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from model import EnhancedCrossAttentionModel
from train import set_seed, train_model
from evaluate import evaluate_model
from visualization import plot_results, evaluate_classification_and_plot_roc


# ---------------- Dataset & Collate ----------------

class ProteinBindingDataset(Dataset):
    """
    Dataset that:
    - reads alldata.csv (expects columns incl. 'pdb' and 'delta_g')
    - aligns PDBs present in both esm_embeddings.pkl and topo_features.pkl
    - standardizes ESM and topology vectors
    - resolves topology dimension mismatches via percentile-based target length with pad/truncate
    """
    def __init__(self, csv_file, esm_features, topo_features, pdb_ids=None, normalize=True,
                 topo_dim_percentile=90):
        self.df = pd.read_csv(csv_file)
        self.esm_features = esm_features
        self.topo_features = topo_features
        self.normalize = normalize

        # Valid PDBs must exist in both feature dicts
        if pdb_ids is None:
            self.valid_pdbs = sorted(set(esm_features.keys()) & set(topo_features.keys()))
        else:
            self.valid_pdbs = pdb_ids

        # Align 'pdb_base' to PDB IDs in feature dicts
        self.df['pdb_base'] = self.df['pdb'].str.split('_').str[0]
        self.filtered_df = self.df[self.df['pdb_base'].isin(self.valid_pdbs)].copy()

        print(f"Raw rows: {len(self.df)}")
        print(f"Filtered rows: {len(self.filtered_df)}")
        if len(self.filtered_df) == 0:
            raise ValueError("No data after filtering! Check PDB ID alignment.")

        # ESM dim inferred from the first entry
        self.esm_dim = next(iter(esm_features.values())).shape[0]

        # Infer target topo dimension using percentile of observed lengths
        topo_dims = [self.topo_features[pdb_id]['feature_values'].shape[0]
                     for pdb_id in self.valid_pdbs if pdb_id in self.topo_features]
        unique_dims = sorted(set(topo_dims))

        if len(unique_dims) > 1:
            print(f"Warning: inconsistent topo dims: {unique_dims}")
            self.topo_dim = int(np.percentile(topo_dims, topo_dim_percentile))
            print(f"Use {topo_dim_percentile}th percentile dim: {self.topo_dim}")
            adjusted_count = sum(1 for dim in topo_dims if dim != self.topo_dim)
            print(f"Samples to adjust: {adjusted_count}")
        else:
            self.topo_dim = unique_dims[0]

        print(f"ESM dim: {self.esm_dim}")
        print(f"Topo standard dim: {self.topo_dim}")

        # Compute mean/std for standardization
        if normalize:
            print("Compute feature statistics for standardization...")
            all_esm_features = []
            all_topo_features = []

            for pdb_id in self.valid_pdbs:
                if pdb_id in self.esm_features:
                    all_esm_features.append(self.esm_features[pdb_id])
                if pdb_id in self.topo_features:
                    topo_feature = self.topo_features[pdb_id]['feature_values']
                    # Adjust to target length
                    if len(topo_feature) != self.topo_dim:
                        if len(topo_feature) < self.topo_dim:
                            padded = np.zeros(self.topo_dim, dtype=topo_feature.dtype)
                            padded[:len(topo_feature)] = topo_feature
                            topo_feature = padded
                        else:
                            topo_feature = topo_feature[:self.topo_dim]
                    all_topo_features.append(topo_feature)

            all_esm_features = np.vstack(all_esm_features)
            self.esm_mean = np.mean(all_esm_features, axis=0)
            self.esm_std  = np.std(all_esm_features, axis=0) + 1e-8

            all_topo_features = np.vstack(all_topo_features)
            self.topo_mean = np.mean(all_topo_features, axis=0)
            self.topo_std  = np.std(all_topo_features, axis=0) + 1e-8

            print("Standardization prepared.")

    def __len__(self):
        return len(self.filtered_df)

    def __getitem__(self, idx):
        row = self.filtered_df.iloc[idx]
        pdb_id = row['pdb_base']
        delta_g = row['delta_g']

        esm_feature = self.esm_features[pdb_id]
        topo_feature = self.topo_features[pdb_id]['feature_values']

        # Adjust to target topo length (pad/truncate)
        if len(topo_feature) != self.topo_dim:
            if len(topo_feature) < self.topo_dim:
                padded_feature = np.zeros(self.topo_dim, dtype=topo_feature.dtype)
                padded_feature[:len(topo_feature)] = topo_feature
                topo_feature = padded_feature
            else:
                topo_feature = topo_feature[:self.topo_dim]

        # Standardize
        if self.normalize:
            esm_feature  = (esm_feature  - self.esm_mean)  / self.esm_std
            topo_feature = (topo_feature - self.topo_mean) / self.topo_std

        return (
            torch.tensor(esm_feature, dtype=torch.float32),
            torch.tensor(topo_feature, dtype=torch.float32),
            torch.tensor(delta_g, dtype=torch.float32)
        )


def custom_collate_fn(batch):
    """
    Collate that pads variable-length topo vectors as a safeguard.
    (Your dataset already aligns lengths; this is kept as an extra safety net.)
    """
    esm_features = torch.stack([item[0] for item in batch])

    topo_features_list = [item[1] for item in batch]
    topo_sizes = [t.shape[0] for t in topo_features_list]

    if len(set(topo_sizes)) > 1:
        max_size = max(topo_sizes)
        padded_topo_features = []
        for tensor in topo_features_list:
            if tensor.shape[0] < max_size:
                padded = torch.zeros(max_size, dtype=tensor.dtype, device=tensor.device)
                padded[:tensor.shape[0]] = tensor
                padded_topo_features.append(padded)
            else:
                padded_topo_features.append(tensor)
        topo_features = torch.stack(padded_topo_features)
    else:
        topo_features = torch.stack(topo_features_list)

    labels = torch.stack([item[2] for item in batch])
    return esm_features, topo_features, labels


# ---------------- Main Orchestration ----------------

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load features
    with open('esm_features/esm_embeddings.pkl', 'rb') as f:
        esm_features = pickle.load(f)
    with open('param_output/d8.0_k6/topo_features.pkl', 'rb') as f:
        topo_features = pickle.load(f)

    # 2) Dataset
    csv_file = 'alldata.csv'
    dataset = ProteinBindingDataset(csv_file, esm_features, topo_features, normalize=True, topo_dim_percentile=90)

    # 3) Split (70/15/15) with fixed seed
    g = torch.Generator().manual_seed(42)
    train_size = int(0.7 * len(dataset))
    val_size   = int(0.15 * len(dataset))
    test_size  = len(dataset) - train_size - val_size

    train_dataset, temp_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size + test_size], generator=g
    )
    val_dataset, test_dataset = torch.utils.data.random_split(
        temp_dataset, [val_size, test_size], generator=g
    )

    # 4) Loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  collate_fn=custom_collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

    # 5) Infer topo partitions from feature_names (matches your logic)
    for pdb_id in dataset.valid_pdbs:
        feat = topo_features[pdb_id]
        if 'feature_names' in feat and len(feat['feature_values']) == dataset.topo_dim:
            feature_names = feat['feature_names']
            break
    else:
        raise ValueError("No sample found whose feature_names match the padded topo_dim.")

    feature_names = feature_names[:dataset.topo_dim]
    contact_dim   = sum('contact'  in name.lower() for name in feature_names)
    interface_dim = sum('interface' in name.lower() for name in feature_names)
    distance_dim  = sum(('dist' in name.lower()) or ('mean' in name.lower()) or ('min' in name.lower())
                        for name in feature_names)
    known_dims    = contact_dim + interface_dim + distance_dim
    topology_dim  = dataset.topo_dim - known_dims

    print(f"Inferred dims -> contact={contact_dim}, interface={interface_dim}, distance={distance_dim}, topology={topology_dim}")

    # 6) Model
    model = EnhancedCrossAttentionModel(
        esm_dim=dataset.esm_dim,
        topo_dim=dataset.topo_dim,
        contact_dim=contact_dim,
        interface_dim=interface_dim,
        distance_dim=distance_dim,
        topology_dim=topology_dim,
        hidden_dim=256,
        num_heads=8,
        dropout=0.1
    ).to(device)

    # 7) Train
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)

    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device,
        num_epochs=100, patience=15
    )

    # 8) Evaluate NN on test and plot
    os.makedirs("results", exist_ok=True)
    nn_results = evaluate_model(model, test_loader, device)
    plot_results(nn_results, train_losses, val_losses, output_dir='results')

    # 9) Extract fused features for the entire dataset (your original logic)
    model.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        full_loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
        for esm, topo, labels in full_loader:
            esm, topo = esm.to(device), topo.to(device)
            features = model.extract_features(esm, topo)
            all_features.append(features)
            all_labels.append(labels.numpy())

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)

    # 10) LassoCV with fresh split (kept exactly as your script)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lasso = LassoCV(alphas=np.logspace(-4, 1, 30), cv=5, max_iter=10000, n_jobs=-1)
    lasso.fit(X_train, y_train)
    print(f"Best alpha: {lasso.alpha_:.5f}")
    y_pred = lasso.predict(X_test)

    # 11) Regression metrics and ROC
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    corr = np.corrcoef(y_test, y_pred)[0, 1]

    acc, roc_auc, fpr, tpr = evaluate_classification_and_plot_roc(
        preds=y_pred, labels=y_test, delta_g_threshold=-10,
        output_pdf='results/roc_curve_crossattn_lasso.pdf'
    )

    print("CrossAttention + Lasso results:")
    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}, Correlation: {corr:.4f}")
    print(f"Accuracy (ΔG < -10): {acc:.4f}")
    print(f"AUC: {roc_auc:.4f}")

    # 12) Save ROC arrays and metrics
    roc_dict = {
        "model": "TopoBind (CrossAttn + Lasso)",
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "auc": float(roc_auc)
    }
    with open("results/roc_data_lasso.pkl", "wb") as f:
        pickle.dump(roc_dict, f)
    print("[✓] ROC data saved to: results/roc_data_lasso.pkl")

    metrics_json = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "corr": float(corr),
        "acc@-10": float(acc),
        "auc": float(roc_auc),
        "best_alpha": float(lasso.alpha_)
    }
    with open("results/metrics_lasso.json", "w") as f:
        json.dump(metrics_json, f, indent=2)
    print("[✓] Metrics saved to: results/metrics_lasso.json")


if __name__ == "__main__":
    main()

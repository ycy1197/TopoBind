"""
evaluate.py

Test-set regression evaluation helper:
- Computes MSE, RMSE, MAE, Pearson correlation, and R^2.
- Preserves your original metric set and returns predictions/labels for plotting.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(model, test_loader, device):
    """
    Run forward pass on test_loader and compute regression metrics.

    Args:
        model: trained PyTorch model
        test_loader: DataLoader for test split
        device: torch.device

    Returns:
        dict with 'predictions', 'true_values', 'mse', 'rmse', 'mae', 'correlation', 'r2'
    """
    model.eval()
    all_preds = []
    all_labels = []

    import torch
    with torch.no_grad():
        for esm_features, topo_features, labels in test_loader:
            esm_features = esm_features.to(device)
            topo_features = topo_features.to(device)
            outputs = model(esm_features, topo_features)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_preds)
    correlation = np.corrcoef(all_preds, all_labels)[0, 1]
    r2 = r2_score(all_labels, all_preds)

    return {
        'predictions': all_preds,
        'true_values': all_labels,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'correlation': correlation,
        'r2': r2
    }

"""
visualization.py

Plot utilities used by the pipeline:
- plot_results(): loss curves, prediction-vs-true scatter, error histogram, summary file
- evaluate_classification_and_plot_roc(): convert ΔG to binary at a threshold, plot ROC, return AUC

All plotting behavior (filenames, labels, threshold semantics) matches your original code.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc


def plot_results(results, train_losses, val_losses, output_dir: str = 'results'):
    """
    Create:
      - loss_curve.png
      - prediction_scatter.png
      - error_distribution.png
      - results_summary.txt
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Training/validation loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=300)
    plt.close()

    # 2) Prediction vs True
    predictions = results['predictions']
    true_values = results['true_values']
    plt.figure(figsize=(8, 8))
    plt.scatter(true_values, predictions, alpha=0.5)

    min_val = min(min(true_values), min(predictions))
    max_val = max(max(true_values), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.xlabel('True ΔG Values')
    plt.ylabel('Predicted ΔG Values')
    plt.title(f'Prediction vs True (r = {results["correlation"]:.4f})')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'prediction_scatter.png'), dpi=300)
    plt.close()

    # 3) Error distribution
    errors = predictions - true_values
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=30, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title(f'Error Distribution (MAE = {results["mae"]:.4f})')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=300)
    plt.close()

    # 4) Save a brief text summary
    with open(os.path.join(output_dir, 'results_summary.txt'), 'w') as f:
        f.write(f"MSE: {results['mse']:.4f}\n")
        f.write(f"RMSE: {results['rmse']:.4f}\n")
        f.write(f"MAE: {results['mae']:.4f}\n")
        f.write(f"Correlation: {results['correlation']:.4f}\n")


def evaluate_classification_and_plot_roc(preds, labels, delta_g_threshold: float = -10,
                                         output_pdf: str = 'results/roc_curve-lasso.pdf'):
    """
    Convert regression outputs to binary via ΔG < threshold, compute accuracy and ROC curve,
    and save the ROC plot as a PDF. Returns (acc, auc, fpr, tpr).
    """
    # Binary labels from ΔG threshold
    y_true = (labels < delta_g_threshold).astype(int)
    y_pred = (preds < delta_g_threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)

    # Higher y_scores = more likely positive => use negative ΔG
    y_scores = -preds
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (ΔG threshold = {delta_g_threshold})')
    plt.legend(loc="lower right")
    plt.grid(True)

    os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"ROC saved to: {output_pdf}")

    return acc, roc_auc, fpr, tpr

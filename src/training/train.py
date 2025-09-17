"""
train.py

Training utilities:
- set_seed(): reproducibility helpers
- train_model(): training loop with mixed-precision (if CUDA), early stopping, and ReduceLROnPlateau

This file intentionally contains only the training loop (and seed setup).
Dataset/model/evaluation/plots are handled in other modules as per your request.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def set_seed(seed: int = 42):
    """Set random seeds and deterministic flags for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU consistency
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(model: torch.nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
                device: torch.device,
                num_epochs: int = 100,
                patience: int = 15):
    """
    Standard training loop:
    - mixed precision if CUDA is available
    - early stopping using best val loss
    - ReduceLROnPlateau scheduler on val loss

    Returns:
        best_model (with best val loss), train_losses, val_losses
    """
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    train_losses = []
    val_losses = []

    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for esm_features, topo_features, labels in train_loader:
            esm_features, topo_features, labels = esm_features.to(device), topo_features.to(device), labels.to(device)
            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(esm_features, topo_features)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(esm_features, topo_features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * esm_features.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # ---- validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for esm_features, topo_features, labels in val_loader:
                esm_features, topo_features, labels = esm_features.to(device), topo_features.to(device), labels.to(device)
                outputs = model(esm_features, topo_features)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * esm_features.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(f"[Epoch {epoch+1:03d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # ---- early stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # restore best weights
    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses

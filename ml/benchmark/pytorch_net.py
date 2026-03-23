"""
PyTorch fully-connected network for binary tabular classification.

Architecture: Linear → BN → ReLU → Dropout → Linear → ReLU → Dropout → Linear → Sigmoid
Training: Adam, BCELoss, early stopping on val_loss.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from ml.runtime.gpu import describe_torch_runtime
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class TabularNet(nn.Module):
    """Fully-connected network for binary classification of tabular data."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run_pytorch(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 100,
) -> dict:
    runtime = describe_torch_runtime()
    device = torch.device(runtime["device"])
    if runtime["cuda_available"]:
        print(f"  [PyTorch] Device: {runtime['device']} ({runtime['gpu_name']})")
    else:
        print(f"  [PyTorch] Device: {runtime['device']}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    X_tr = torch.tensor(X_train)
    y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    # Hold-out validation split (15 %)
    val_n = int(len(X_tr) * 0.15)
    X_val = X_tr[:val_n].to(device)
    y_val = y_tr[:val_n].to(device)
    X_tr = X_tr[val_n:].to(device)
    y_tr = y_tr[val_n:].to(device)

    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=64,
        shuffle=True,
        pin_memory=runtime["cuda_available"],
    )

    model = TabularNet(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    patience = 10
    no_improve = 0
    best_state: dict | None = None

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stopping at epoch {epoch}")
                break

        if epoch % 20 == 0:
            print(f"    Epoch {epoch}/{epochs}: val_loss={val_loss:.4f}")

    train_time = time.perf_counter() - t0

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    X_te = torch.tensor(X_test).to(device)
    t_inf = time.perf_counter()
    with torch.no_grad():
        preds = (model(X_te).cpu().numpy().ravel() >= 0.5).astype(int)
    inf_ms = (time.perf_counter() - t_inf) / len(X_test) * 1000

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="binary")
    print(f"    → Accuracy: {acc:.3f} | F1: {f1:.3f} | Time: {train_time:.2f}s")

    return {
        "model": "TabularNet",
        "framework": "pytorch",
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "train_time_sec": round(train_time, 3),
        "inference_time_ms": round(inf_ms, 3),
    }

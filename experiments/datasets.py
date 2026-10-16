"""
Dataset loaders for the activation benchmark.

Each loader returns (train_loader, val_loader, input_dim_or_shape, n_classes).
Tabular and vision datasets (MNIST/FashionMNIST) all come from scikit-learn's
`fetch_openml`, cached locally after the first download. torchvision is
deliberately not used here: this environment has a torch/torchvision ABI
mismatch (torch 2.13.0 pip install vs. a torchvision 0.18.0 build expecting
torch 2.3.0), which breaks `import torchvision` with a
"torchvision::nms does not exist" error. Fixing that is a global environment
change outside this project's scope, so we route around it instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = Path(__file__).resolve().parent / "data"

TABULAR_LOADERS = {
    "digits": load_digits,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}


def load_tabular(name: str, seed: int, batch_size: int = 64) -> Tuple[DataLoader, DataLoader, int, int]:
    if name not in TABULAR_LOADERS:
        raise ValueError(f"Unknown tabular dataset: {name!r}")
    data = TABULAR_LOADERS[name]()
    X, y = data.data, data.target
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    n_classes = int(y.max() + 1)
    return train_loader, val_loader, X.shape[1], n_classes


OPENML_VISION_NAMES = {
    "mnist": "mnist_784",
    "fashion_mnist": "Fashion-MNIST",
}

_openml_cache: dict = {}


def load_vision(name: str, seed: int, batch_size: int = 128, n_subsample: int = 12000) -> Tuple[DataLoader, DataLoader, Tuple[int, int, int], int]:
    from sklearn.datasets import fetch_openml

    if name not in OPENML_VISION_NAMES:
        raise ValueError(f"Unknown vision dataset: {name!r}")

    if name not in _openml_cache:
        bunch = fetch_openml(OPENML_VISION_NAMES[name], version=1, as_frame=False, parser="auto")
        X = bunch.data.astype(np.float32) / 255.0
        y = bunch.target.astype(np.int64) if np.issubdtype(bunch.target.dtype, np.number) else \
            np.asarray(bunch.target, dtype=np.int64) if str(bunch.target.dtype).startswith("int") else None
        if y is None:
            # OpenML labels often come back as strings (e.g. '0', '1', ...).
            y = bunch.target.astype(int)
        _openml_cache[name] = (X, y)
    X, y = _openml_cache[name]

    # Subsample for speed (full 70k is overkill for this benchmark); keep it
    # class-balanced and seeded so runs are reproducible per seed.
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=min(n_subsample, len(X)), replace=False)
    X, y = X[idx], y[idx]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_val = X_val.reshape(-1, 1, 28, 28)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    shape = (1, 28, 28)
    n_classes = int(np.unique(y).shape[0])
    return train_loader, val_loader, shape, n_classes


__all__ = ["load_tabular", "load_vision", "TABULAR_LOADERS"]

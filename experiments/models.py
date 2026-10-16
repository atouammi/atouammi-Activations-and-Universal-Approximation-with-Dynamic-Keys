"""
Models for the activation benchmark. Each takes an `act_factory` — a
zero-arg callable returning a fresh activation nn.Module — so the same
activation (and, for keyed ones, the same key) is reused layer-to-layer
within one model instance, but different model instances never
accidentally share a module.
"""

from __future__ import annotations

from typing import Callable, Tuple

import torch.nn as nn


class MLP(nn.Module):
    """Generic MLP for tabular data."""

    def __init__(self, in_dim: int, n_classes: int, act_factory: Callable[[], nn.Module], hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            act_factory(),
            nn.Linear(hidden, hidden),
            act_factory(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    """Small CNN for MNIST/FashionMNIST/CIFAR-10-sized images."""

    def __init__(self, in_shape: Tuple[int, int, int], n_classes: int, act_factory: Callable[[], nn.Module]):
        super().__init__()
        c, h, w = in_shape
        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            act_factory(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            act_factory(),
            nn.MaxPool2d(2),
        )
        flat_dim = 64 * (h // 4) * (w // 4)
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 128),
            act_factory(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


__all__ = ["MLP", "SmallCNN"]

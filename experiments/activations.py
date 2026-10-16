"""
Registry of activation functions used across the benchmark.

Classical activations (ReLU, Sigmoid, Tanh, GELU, SiLU) are plain
nn.Module instances. CSwish/KGELU additionally take a `key_seed` so the
benchmark can test sensitivity to the (alpha, theta) "key" independently
of the network's own init seed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cswish_activation import CSwish, KGELU  # noqa: E402

# Activations considered relevant per task family (see strategy discussion):
# sigmoid is excluded for vision (CNN) due to well-known vanishing-gradient
# behavior in hidden layers; everything else is included everywhere.
TASK_ACTIVATIONS = {
    "tabular": ["relu", "sigmoid", "tanh", "gelu", "silu", "cswish", "kgelu"],
    "vision": ["relu", "tanh", "gelu", "silu", "cswish", "kgelu"],
    "sequence": ["gelu", "silu", "tanh", "cswish", "kgelu"],
}

# Default keyed-activation hyperparameters, used unless overridden.
DEFAULT_K = 3
DEFAULT_BETA = 1.0


def make_activation(
    name: str,
    key_seed: Optional[int] = None,
    k: int = DEFAULT_K,
    beta: float = DEFAULT_BETA,
) -> nn.Module:
    """Build a fresh activation module by name.

    `key_seed` only affects CSwish/KGELU: it seeds the (alpha, theta) key.
    It is ignored for classical activations (passing one is a no-op).
    """
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "cswish":
        return CSwish(k=k, beta=beta, seed=key_seed)
    if name == "kgelu":
        return KGELU(k=k, beta=beta, seed=key_seed)
    raise ValueError(f"Unknown activation: {name!r}")


def activation_factory(name: str, key_seed: Optional[int] = None, **kwargs) -> Callable[[], nn.Module]:
    """Return a zero-arg factory, so a model can build one instance per layer
    without accidentally sharing the same module (and hence the same key)."""
    return lambda: make_activation(name, key_seed=key_seed, **kwargs)


def is_keyed(name: str) -> bool:
    return name.lower() in ("cswish", "kgelu")


__all__ = ["TASK_ACTIVATIONS", "make_activation", "activation_factory", "is_keyed"]

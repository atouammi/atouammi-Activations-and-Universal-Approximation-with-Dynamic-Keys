"""
CSwish (Cryptographic Swish) and KGELU (Keyed GELU) activation functions.

Reference
---------
P. Tiebekabe, "Cryptographically-Inspired Transformer Architectures:
Controlled-Sensitivity Activations and Universal Approximation with
Dynamic Keys", ICMAI 2026.

Definitions (paper notation)
-----------------------------
Security trigonometric polynomial (Def. 3.1):

    T_{k,alpha,theta}(x) = sum_{j=1}^{k} alpha_j * sin(j*x + theta_j)

    stability condition:  sum_{j=1}^{k} j * |alpha_j| <= beta / 4      (Eq. 6)

CSwish (Def. 3.3):

    sigma_{k,theta}(x) = x * sigmoid(beta*x + T_{k,alpha,theta}(x))    (Eq. 7)

KGELU (Def. 3.5):

    gamma_{k,theta}(x) = x * Phi(beta*x + T_{k,alpha,theta}(x))        (Eq. 13)

where Phi is the standard normal CDF. KGELU reduces to standard GELU
when k = 0 and beta = 1.

Lipschitz constant on [-M, M] (Eq. 8), used here only as a diagnostic:

    L_sigma(M) = 1 + 5*beta*M/16 + M * sum_{j=1}^{k} j*|alpha_j|
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# Shared trigonometric "key" term T_k
# ----------------------------------------------------------------------
def trigonometric_key_term(
    x: torch.Tensor, alpha: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    """Evaluate T_{k,alpha,theta}(x) = sum_j alpha_j * sin(j*x + theta_j).

    Parameters
    ----------
    x : tensor of arbitrary shape (...,)
    alpha, theta : 1-D tensors of shape (k,)

    Returns
    -------
    tensor of the same shape as x.
    """
    k = alpha.shape[0]
    # j = 1..k, broadcast against x by adding a trailing dimension
    j = torch.arange(1, k + 1, device=x.device, dtype=x.dtype)  # (k,)
    # x: (...,)  ->  (..., 1);  j, alpha, theta: (k,)  ->  broadcast to (..., k)
    phase = x.unsqueeze(-1) * j + theta  # (..., k)
    return (alpha * torch.sin(phase)).sum(dim=-1)


def _stability_bound(beta: float) -> float:
    """Right-hand side of the stability condition (Eq. 6): beta / 4."""
    return beta / 4.0


def project_to_stability(alpha: torch.Tensor, beta: float) -> torch.Tensor:
    """Rescale alpha (if needed) so that sum_j j*|alpha_j| <= beta/4 (Eq. 6).

    This keeps the Lipschitz guarantee of Proposition 3.4(iii) valid.
    """
    k = alpha.shape[0]
    j = torch.arange(1, k + 1, device=alpha.device, dtype=alpha.dtype)
    weighted_l1 = (j * alpha.abs()).sum()
    bound = _stability_bound(beta)
    if weighted_l1 > bound and weighted_l1 > 0:
        alpha = alpha * (bound / weighted_l1)
    return alpha


# ----------------------------------------------------------------------
# Functional forms
# ----------------------------------------------------------------------
def cswish(
    x: torch.Tensor, alpha: torch.Tensor, theta: torch.Tensor, beta: float = 1.0
) -> torch.Tensor:
    """Functional CSwish: x * sigmoid(beta*x + T_k(x))."""
    u = beta * x + trigonometric_key_term(x, alpha, theta)
    return x * torch.sigmoid(u)


def kgelu(
    x: torch.Tensor, alpha: torch.Tensor, theta: torch.Tensor, beta: float = 1.0
) -> torch.Tensor:
    """Functional KGELU: x * Phi(beta*x + T_k(x)), Phi = standard normal CDF."""
    u = beta * x + trigonometric_key_term(x, alpha, theta)
    return x * torch.special.ndtr(u)


# ----------------------------------------------------------------------
# nn.Module wrappers with fixed / configurable keys
# ----------------------------------------------------------------------
class _KeyedActivationBase(nn.Module):
    """Common machinery for CSwish / KGELU: holds the (alpha, theta, beta, k)
    'key' as fixed buffers (not trained by the optimizer), with helpers to
    set the key explicitly or derive it deterministically from a seed.
    """

    def __init__(
        self,
        k: int = 3,
        beta: float = 1.0,
        alpha: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        enforce_stability: bool = True,
    ):
        super().__init__()
        if k < 1:
            raise ValueError("k (security parameter) must be >= 1")
        self.k = k
        self.beta = beta
        self.enforce_stability = enforce_stability

        if alpha is None or theta is None:
            alpha, theta = self._generate_key(k, beta, seed)
        else:
            alpha = torch.as_tensor(alpha, dtype=torch.float32)
            theta = torch.as_tensor(theta, dtype=torch.float32)
            if alpha.shape != (k,) or theta.shape != (k,):
                raise ValueError(f"alpha and theta must have shape ({k},)")
            if enforce_stability:
                alpha = project_to_stability(alpha, beta)

        # Fixed (non-trainable) parameters: registered as buffers so they
        # still move with .to(device)/.cuda() and are saved in state_dict,
        # but are never touched by the optimizer.
        self.register_buffer("alpha", alpha)
        self.register_buffer("theta", theta)

    @staticmethod
    def _generate_key(k: int, beta: float, seed: Optional[int]):
        """Derive a pseudo-random (alpha, theta) pair, analogous to deriving
        round keys from a secret seed in a block cipher.
        """
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(int(seed))
        # Random phases in [0, 2*pi)
        theta = torch.rand(k, generator=gen) * 2 * math.pi
        # Random signed weights, later rescaled to satisfy the stability
        # condition sum_j j*|alpha_j| <= beta/4 with equality (full budget).
        raw = torch.rand(k, generator=gen) * 2 - 1  # in [-1, 1)
        j = torch.arange(1, k + 1, dtype=torch.float32)
        weighted_l1 = (j * raw.abs()).sum()
        alpha = raw * (_stability_bound(beta) / weighted_l1)
        return alpha, theta

    def set_key(self, alpha: torch.Tensor, theta: torch.Tensor) -> None:
        """Replace the activation's key (e.g. to simulate a key rotation)."""
        alpha = torch.as_tensor(alpha, dtype=self.alpha.dtype, device=self.alpha.device)
        theta = torch.as_tensor(theta, dtype=self.theta.dtype, device=self.theta.device)
        if alpha.shape != (self.k,) or theta.shape != (self.k,):
            raise ValueError(f"alpha and theta must have shape ({self.k},)")
        if self.enforce_stability:
            alpha = project_to_stability(alpha, self.beta)
        self.alpha.copy_(alpha)
        self.theta.copy_(theta)

    def lipschitz_constant(self, M: float) -> float:
        """Theoretical Lipschitz constant on [-M, M] (Eq. 8)."""
        j = torch.arange(1, self.k + 1, dtype=self.alpha.dtype)
        weighted_l1 = float((j * self.alpha.abs()).sum())
        return 1.0 + 5.0 * self.beta * M / 16.0 + M * weighted_l1

    def extra_repr(self) -> str:
        return f"k={self.k}, beta={self.beta}"


class CSwish(_KeyedActivationBase):
    """Cryptographic Swish activation (Def. 3.3), x * sigmoid(beta*x + T_k(x)).

    Example
    -------
    >>> act = CSwish(k=3, beta=1.0, seed=42)
    >>> x = torch.randn(8, 16)
    >>> y = act(x)
    >>> act.lipschitz_constant(M=5.0)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cswish(x, self.alpha, self.theta, self.beta)


class KGELU(_KeyedActivationBase):
    """Keyed GELU activation (Def. 3.5), x * Phi(beta*x + T_k(x)).

    With k small and beta = 1, this closely tracks standard GELU while
    retaining a key-dependent perturbation.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return kgelu(x, self.alpha, self.theta, self.beta)


__all__ = [
    "trigonometric_key_term",
    "project_to_stability",
    "cswish",
    "kgelu",
    "CSwish",
    "KGELU",
]

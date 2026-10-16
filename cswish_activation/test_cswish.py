"""
Sanity checks / demo for CSwish & KGELU (see cswish.py).

Run:
    python test_cswish.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cswish import CSwish, KGELU, project_to_stability, trigonometric_key_term


def check_shapes():
    act = CSwish(k=3, beta=1.0, seed=0)
    for shape in [(), (5,), (4, 8), (2, 3, 16)]:
        x = torch.randn(shape, requires_grad=True)
        y = act(x)
        assert y.shape == x.shape, f"shape mismatch for input shape {shape}"
        y.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
    print("[OK] shapes & gradients flow correctly through CSwish")


def check_stability_projection():
    beta = 1.0
    k = 4
    # Deliberately violate the stability condition (Eq. 6): sum j*|alpha_j| <= beta/4
    alpha = torch.tensor([10.0, -5.0, 3.0, 2.0])
    projected = project_to_stability(alpha, beta)
    j = torch.arange(1, k + 1, dtype=torch.float32)
    weighted_l1 = (j * projected.abs()).sum().item()
    bound = beta / 4.0
    assert weighted_l1 <= bound + 1e-5, "stability projection failed"
    print(f"[OK] stability projection: sum j|alpha_j| = {weighted_l1:.4f} <= beta/4 = {bound:.4f}")


def check_lipschitz_bound_empirically():
    """Empirically estimate the Lipschitz constant on [-M, M] via dense
    finite differences and compare it to the theoretical bound (Eq. 8).
    The theoretical bound must not be violated (up to discretization error).
    """
    M = 5.0
    act = CSwish(k=3, beta=1.0, seed=1)
    theoretical = act.lipschitz_constant(M)

    x = torch.linspace(-M, M, steps=20001)
    y = act(x)
    dy = (y[1:] - y[:-1]).abs()
    dx = (x[1:] - x[:-1]).abs()
    empirical = (dy / dx).max().item()

    print(f"[INFO] theoretical Lipschitz bound L_sigma({M}) = {theoretical:.4f}")
    print(f"[INFO] empirical max slope on grid           = {empirical:.4f}")
    assert empirical <= theoretical + 1e-2, "empirical slope exceeds theoretical Lipschitz bound!"
    print("[OK] empirical slope stays within the theoretical Lipschitz bound")


def check_non_polynomial_behavior():
    """Qualitative check: unlike GELU/ReLU, CSwish's derivative should show
    visible oscillation because of the trigonometric key term T_k.
    """
    act = CSwish(k=5, beta=1.0, seed=2)
    x = torch.linspace(-3, 3, steps=4000, requires_grad=True)
    y = act(x)
    (grad,) = torch.autograd.grad(y.sum(), x, create_graph=False)
    # crude oscillation proxy: count sign changes of the second difference
    second_diff = grad[2:] - 2 * grad[1:-1] + grad[:-2]
    sign_changes = (second_diff[1:] * second_diff[:-1] < 0).sum().item()
    print(f"[INFO] sign changes in curvature of sigma'(x): {sign_changes} "
          f"(0 would indicate a monotone/convex derivative like GELU)")


def check_key_rotation_changes_output():
    """Two different keys (same k, beta) should produce different outputs,
    illustrating the 'key' role of (alpha, theta) -- Section 5 of the paper."""
    x = torch.randn(32, requires_grad=False)
    act_a = CSwish(k=3, beta=1.0, seed=10)
    act_b = CSwish(k=3, beta=1.0, seed=11)
    y_a, y_b = act_a(x), act_b(x)
    diff = (y_a - y_b).abs().mean().item()
    assert diff > 1e-6, "different keys produced identical outputs (unexpected)"
    print(f"[OK] different keys -> different outputs (mean abs diff = {diff:.4f})")


def demo_drop_in_ffn():
    """Shows CSwish used as a drop-in replacement in a standard
    position-wise feed-forward block, as in Eq. (18) of the paper."""

    class FeedForward(nn.Module):
        def __init__(self, d_model=64, d_ff=256, activation=None):
            super().__init__()
            self.w1 = nn.Linear(d_model, d_ff)
            self.w2 = nn.Linear(d_ff, d_model)
            self.act = activation or nn.GELU()

        def forward(self, x):
            return self.w2(self.act(self.w1(x)))

    ffn_cswish = FeedForward(activation=CSwish(k=3, beta=1.0, seed=42))
    ffn_gelu = FeedForward(activation=nn.GELU())

    x = torch.randn(4, 10, 64)  # (batch, seq_len, d_model)
    out_cswish = ffn_cswish(x)
    out_gelu = ffn_gelu(x)
    assert out_cswish.shape == x.shape == out_gelu.shape
    print("[OK] CSwish works as a drop-in activation inside a Transformer FFN block")


if __name__ == "__main__":
    check_shapes()
    check_stability_projection()
    check_lipschitz_bound_empirically()
    check_non_polynomial_behavior()
    check_key_rotation_changes_output()
    demo_drop_in_ffn()
    print("\nAll checks passed.")

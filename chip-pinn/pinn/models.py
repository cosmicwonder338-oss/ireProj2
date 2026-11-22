"""Networks. All inputs are nondimensional coordinates x̂ ∈ [0, Lx/L] × [0, Ly/L]."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

ACTIVATIONS = {"tanh": nn.Tanh, "silu": nn.SiLU, "sin": None}


class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class MLP(nn.Module):
    """Fully connected net  (x, y) -> output.

    Inputs are affinely mapped from the box [lo, hi] to [-1, 1] inside the model, so tanh units start
    in their sensitive range regardless of the physical domain size. Xavier init, zero biases.
    `out_transform`: None | "softplus" (strictly positive outputs, e.g. conductivity or power).
    """

    def __init__(self, in_dim=2, out_dim=1, hidden=64, layers=4, activation="tanh",
                 lo=(0.0, 0.0), hi=(1.0, 1.0), out_transform=None, out_scale=1.0, out_bias=0.0,
                 fourier_n=0, fourier_sigma=1.0):
        super().__init__()
        act = Sine if activation == "sin" else ACTIVATIONS[activation]
        # Optional random Fourier features (Tancik et al. 2020): [z, sin(πBz), cos(πBz)] with
        # B ~ N(0, σ²). Counteracts spectral bias — the MLP sees high frequencies from the start.
        self.fourier_n = int(fourier_n)
        if self.fourier_n > 0:
            self.register_buffer("B", torch.randn(in_dim, self.fourier_n) * float(fourier_sigma))
        feat_dim = in_dim + 2 * self.fourier_n
        dims = [feat_dim] + [hidden] * layers + [out_dim]
        mods = []
        for a, b in zip(dims[:-1], dims[1:]):
            lin = nn.Linear(a, b)
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            mods += [lin, act()]
        self.net = nn.Sequential(*mods[:-1])        # no activation after the last layer
        self.register_buffer("lo", torch.tensor(lo, dtype=torch.float32))
        self.register_buffer("hi", torch.tensor(hi, dtype=torch.float32))
        self.out_transform = out_transform
        self.out_scale, self.out_bias = float(out_scale), float(out_bias)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        z = 2.0 * (xy - self.lo) / (self.hi - self.lo) - 1.0
        if self.fourier_n > 0:
            zb = torch.pi * (z @ self.B)
            z = torch.cat([z, torch.sin(zb), torch.cos(zb)], dim=1)
        out = self.net(z)
        if self.out_transform == "softplus":
            out = nn.functional.softplus(out)
        return self.out_scale * out + self.out_bias


def build_mlp(mcfg, lo, hi, **overrides) -> MLP:
    """Build an MLP from a config block (hidden, layers, activation, optional fourier{n, sigma})."""
    ff = mcfg.get("fourier") or {}
    kw = dict(hidden=int(mcfg.hidden), layers=int(mcfg.layers), activation=str(mcfg.activation),
              lo=lo, hi=hi, in_dim=len(lo), fourier_n=int(ff.get("n", 0)),
              fourier_sigma=float(ff.get("sigma", 1.0)))
    kw.update(overrides)
    return MLP(**kw)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class PositiveScalar(nn.Module):
    """A single trainable positive number, parametrised as exp(θ) (e.g. unknown constant k̂).

    Log-parametrisation keeps it positive and makes Adam steps relative (a step of 1e-3 in θ is a
    0.1 % change whatever the magnitude)."""

    def __init__(self, init: float):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(float(init)).log())

    def forward(self, *_) -> torch.Tensor:
        return self.theta.exp()


class GaussianSources(nn.Module):
    """Structured power-map prior: q̂(x, y) = b + Σ_k a_k · exp(−(x−cx_k)²/2sx_k² − (y−cy_k)²/2sy_k²).

    Chip power comes from a handful of functional blocks, so a few localised sources is a far
    stronger (and physically motivated) prior than a free-form network, which — given only point
    sensors — prefers the smoothest map that fits them (a ramp towards the hottest sensors).
    Everything is constrained: centres inside the die (sigmoid), widths in [0.03, 0.35]·L, amplitudes
    and background ≥ 0 (softplus). Gaussians (not boxes) keep gradients w.r.t. position non-zero
    everywhere, which makes the source positions trainable from any initial guess.
    """

    def __init__(self, n_sources: int, Lx: float, Ly: float, q_mean: float):
        super().__init__()
        g = int(np.ceil(np.sqrt(n_sources)))
        cs = [((i + 0.5) / g, (j + 0.5) / g) for j in range(g) for i in range(g)][:n_sources]
        frac = torch.tensor(cs, dtype=torch.float32).clamp(0.05, 0.95)
        self.register_buffer("L", torch.tensor([Lx, Ly], dtype=torch.float32))
        self.c_raw = nn.Parameter(torch.log(frac / (1 - frac)))           # sigmoid⁻¹
        self.s_raw = nn.Parameter(torch.zeros(n_sources, 2))              # mid-range widths
        self.a_raw = nn.Parameter(torch.zeros(n_sources))
        self.b_raw = nn.Parameter(torch.tensor(0.0))
        self.scale = float(q_mean)

    def params(self):
        c = torch.sigmoid(self.c_raw) * self.L
        s = (0.03 + 0.32 * torch.sigmoid(self.s_raw)) * self.L
        a = nn.functional.softplus(self.a_raw) * self.scale * 4.0
        b = nn.functional.softplus(self.b_raw) * self.scale * 0.2
        return c, s, a, b

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        c, s, a, b = self.params()
        d = (xy[:, None, :] - c[None]) / s[None]                         # (N, K, 2)
        return b + (a[None] * torch.exp(-0.5 * (d ** 2).sum(-1))).sum(1, keepdim=True)

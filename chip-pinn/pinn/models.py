"""Networks. All inputs are nondimensional coordinates x̂ ∈ [0, Lx/L] × [0, Ly/L]."""
from __future__ import annotations

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
                 lo=(0.0, 0.0), hi=(1.0, 1.0), out_transform=None, out_scale=1.0, out_bias=0.0):
        super().__init__()
        act = Sine if activation == "sin" else ACTIVATIONS[activation]
        dims = [in_dim] + [hidden] * layers + [out_dim]
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
        out = self.net(z)
        if self.out_transform == "softplus":
            out = nn.functional.softplus(out)
        return self.out_scale * out + self.out_bias


def build_mlp(mcfg, lo, hi, **overrides) -> MLP:
    """Build an MLP from a config block (hidden, layers, activation)."""
    kw = dict(hidden=int(mcfg.hidden), layers=int(mcfg.layers), activation=str(mcfg.activation),
              lo=lo, hi=hi)
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

"""Collocation / boundary / evaluation point sets (nondimensional coordinates)."""
from __future__ import annotations

import numpy as np
import torch

# outward unit normals
NORMALS = {"bottom": (0.0, -1.0), "top": (0.0, 1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0)}


def interior(n: int, Lx: float, Ly: float, device) -> torch.Tensor:
    """n uniform random points in (0, Lx) × (0, Ly), requires_grad for autograd derivatives."""
    xy = torch.rand(n, 2, device=device) * torch.tensor([Lx, Ly], device=device)
    return xy.requires_grad_(True)


def edge(name: str, n: int, Lx: float, Ly: float, device):
    """n uniform random points on one edge + their outward normals."""
    t = torch.rand(n, device=device)
    if name in ("bottom", "top"):
        xy = torch.stack([t * Lx, torch.full_like(t, 0.0 if name == "bottom" else Ly)], dim=1)
    else:
        xy = torch.stack([torch.full_like(t, 0.0 if name == "left" else Lx), t * Ly], dim=1)
    nrm = torch.tensor(NORMALS[name], device=device).expand(n, 2)
    return xy.requires_grad_(True), nrm


def grid_points(x_nd: np.ndarray, y_nd: np.ndarray, device) -> torch.Tensor:
    """Tensor (ny*nx, 2) of cell-centre points ordered so `.reshape(ny, nx)` matches FV arrays."""
    X, Y = np.meshgrid(x_nd, y_nd)
    return torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), dtype=torch.float32, device=device)

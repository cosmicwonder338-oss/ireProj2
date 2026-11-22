"""Synthetic sensor readings from a reference solution.

Sensors read T at arbitrary (x, y) by bilinear interpolation of the cell-centre field, plus Gaussian
noise with std = noise_rel × (max(T) − T_amb), i.e. noise is relative to the temperature *rise*
(1% of a 50 K rise = 0.5 K, realistic for on-die thermal diodes).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from common.seed import rng


def interpolator(sol):
    g = sol.grid
    return RegularGridInterpolator((g.y, g.x), sol.T, method="linear",
                                   bounds_error=False, fill_value=None)  # linear extrapolation in the
                                                                         # outer half-cell


def placement(n: int, Lx: float, Ly: float, kind: str = "random", seed: int = 0,
              margin: float = 0.5e-3) -> np.ndarray:
    """Sensor positions (n, 2) in metres. 'random' = uniform; 'grid' = near-square regular grid."""
    g = rng(seed)
    if kind == "random":
        return np.column_stack([g.uniform(margin, Lx - margin, n), g.uniform(margin, Ly - margin, n)])
    if kind == "grid":
        nc = int(np.ceil(np.sqrt(n * Lx / Ly)))
        nr = int(np.ceil(n / nc))
        xs = np.linspace(margin, Lx - margin, nc)
        ys = np.linspace(margin, Ly - margin, nr)
        pts = np.array([(x, y) for y in ys for x in xs])
        return pts[np.sort(g.choice(len(pts), n, replace=False))] if len(pts) > n else pts
    raise ValueError(f"placement '{kind}' not implemented (smart placement arrives in Stage 8)")


def read_sensors(sol, xy: np.ndarray, noise_rel: float, T_amb: float, seed: int = 0) -> dict:
    T_clean = interpolator(sol)(xy[:, ::-1])           # interpolator wants (y, x)
    sigma = noise_rel * (sol.T.max() - T_amb)
    T_noisy = T_clean + rng(seed + 1).normal(0.0, sigma, len(xy)) if sigma > 0 else T_clean.copy()
    return {"xy": xy, "T_clean": T_clean, "T_noisy": T_noisy, "sigma": float(sigma)}

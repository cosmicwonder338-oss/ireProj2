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


def pod_library(cfg, n_maps: int = 60, n_grid: int = 50, seed0: int = 10_000):
    """Temperature snapshots of *random* chips (never the test layout), for sensor placement."""
    from .fd_solver import Grid, Problem, solve
    from .layouts import LAYOUTS, MM, cooling_setup
    ge = cfg.geometry
    g = Grid(float(ge.Lx), float(ge.Ly), n_grid, n_grid)
    X, Y = g.mesh()
    bc, sink = cooling_setup(cfg, cfg.physics.cooling)
    k = np.full_like(X, float(cfg.physics.materials.silicon))
    snaps = []
    for i in range(n_maps):
        q = 1e9 * LAYOUTS["random_mixed"](X, Y, float(cfg.layouts.edge_width_mm) * MM,
                                          seed=seed0 + i)["q_rel"]
        T = solve(Problem(g, k, q, bc, sink=sink, T_sink=0.0)).T
        snaps.append(T / T.max())
    return g, np.stack(snaps)


def qr_placement(n: int, grid, snaps: np.ndarray, margin: float = 0.5e-3) -> np.ndarray:
    """Data-driven sparse sensor placement (Manohar et al., IEEE Control Systems 2018):
    POD modes Ψ_r of the snapshot library, then QR with column pivoting on Ψ_rᵀ — the pivots are
    the locations that best condition the reconstruction of the r leading modes. Returns (n, 2) m."""
    import scipy.linalg
    X, Y = grid.mesh()
    ok = ((X >= margin) & (X <= grid.Lx - margin) & (Y >= margin) & (Y <= grid.Ly - margin)).ravel()
    D = snaps.reshape(len(snaps), -1)[:, ok].T                     # (candidates, maps)
    U, _, _ = np.linalg.svd(D, full_matrices=False)
    r = min(n, U.shape[1])
    Psi = U[:, :r]
    M = Psi.T if n <= r else Psi @ Psi.T
    _, _, piv = scipy.linalg.qr(M, pivoting=True, mode="economic")
    idx = np.flatnonzero(ok)[piv[:n]]
    return np.column_stack([X.ravel()[idx], Y.ravel()[idx]])

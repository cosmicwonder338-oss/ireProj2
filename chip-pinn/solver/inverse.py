"""Classical inverse baselines built on the FV solver (no neural networks).

fit_constant_k: 1-parameter least squares — find the uniform k whose FV solution best matches the
sensor readings. Bounded scalar minimisation over log k; each evaluation is one sparse solve.

The baseline solves on a *coarser* grid (default 100²) than the one that generated the synthetic
data (200²). Inverting with the exact model that produced the data (the "inverse crime") makes
noise-free recovery trivially perfect and flatters every method.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize_scalar

from .fd_solver import Grid, Problem, solve
from .layouts import cooling_setup, q_function
from .sensors import interpolator


def fit_constant_k(cfg, layout: str, cooling: str, q_scale_W_m3: float, sensor_xy: np.ndarray,
                   T_obs: np.ndarray, n: int = 100, k_bounds=(10.0, 1000.0)) -> dict:
    ge = cfg.geometry
    g = Grid(float(ge.Lx), float(ge.Ly), n, n)
    X, Y = g.mesh()
    q = q_function(cfg, layout, q_scale_W_m3)(X, Y)
    bc, sink = cooling_setup(cfg, cooling)
    T_amb = float(cfg.physics.T_amb)
    n_eval = [0]

    def misfit(logk):
        n_eval[0] += 1
        sol = solve(Problem(g, np.full_like(X, np.exp(logk)), q, bc, sink=sink, T_sink=T_amb))
        return float(np.sum((interpolator(sol)(sensor_xy[:, ::-1]) - T_obs) ** 2))

    t0 = time.perf_counter()
    res = minimize_scalar(misfit, bounds=np.log(k_bounds), method="bounded",
                          options={"xatol": 1e-6})
    return {"k": float(np.exp(res.x)), "rms_misfit_K": float(np.sqrt(res.fun / len(T_obs))),
            "n_solves": n_eval[0], "time_s": time.perf_counter() - t0, "grid": f"{n}x{n}"}


def rbf_interpolate(sensor_xy: np.ndarray, T_obs: np.ndarray, grid, smoothing: float = 0.0):
    """Physics-free baseline: thin-plate-spline interpolation of the sensor readings onto the grid
    (what you would do with a spreadsheet of sensor values). Returns T (ny, nx) in °C."""
    from scipy.interpolate import RBFInterpolator
    X, Y = grid.mesh()
    rbf = RBFInterpolator(sensor_xy * 1e3, T_obs, kernel="thin_plate_spline", smoothing=smoothing)
    pts = np.column_stack([X.ravel() * 1e3, Y.ravel() * 1e3])
    return rbf(pts).reshape(X.shape)

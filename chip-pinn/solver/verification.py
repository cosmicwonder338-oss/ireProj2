"""Verification problems with known exact answers (used by tests/ and scripts/stage2_verify.py).

1. Manufactured solution (MMS) on the unit square, variable k, sink, and all three BC types:
     T = 1 + x²/2 + sin(2x) cos(πy) + 0.3 (y − 1)²
     k = 1 + ½ sin(πx) sin(πy)
     s = 2
   q is *derived* so that ∇·(k∇T) + q − sT = 0 holds exactly. BCs: left/right Dirichlet with the
   exact T, top insulated (∂T/∂y = 0 at y = 1 by construction), bottom convective with h = 3 and an
   edge-varying T∞ chosen to match. The same formulas serve numpy (FV solver) and torch (autograd
   residual), via the `lib` argument.

2. Two-layer slab (silicon over an air void) with Dirichlet top/bottom: exact piecewise-linear T.
   With the interface on a cell face, harmonic-mean FV should reproduce it to round-off.
"""
from __future__ import annotations

import numpy as np

from .fd_solver import Grid, Problem, solve

MMS_S = 2.0
MMS_H = 3.0
PI = np.pi


def mms_fields(x, y, lib=np) -> dict:
    """Exact T, its derivatives, k, and the consistent source q at points (x, y)."""
    sin, cos = lib.sin, lib.cos
    T = 1 + 0.5 * x ** 2 + sin(2 * x) * cos(PI * y) + 0.3 * (y - 1) ** 2
    Tx = x + 2 * cos(2 * x) * cos(PI * y)
    Ty = -PI * sin(2 * x) * sin(PI * y) + 0.6 * (y - 1)
    Txx = 1 - 4 * sin(2 * x) * cos(PI * y)
    Tyy = -PI ** 2 * sin(2 * x) * cos(PI * y) + 0.6
    k = 1 + 0.5 * sin(PI * x) * sin(PI * y)
    kx = 0.5 * PI * cos(PI * x) * sin(PI * y)
    ky = 0.5 * PI * sin(PI * x) * cos(PI * y)
    div = k * (Txx + Tyy) + kx * Tx + ky * Ty          # ∇·(k∇T)
    q = -div + MMS_S * T
    return {"T": T, "Tx": Tx, "Ty": Ty, "k": k, "q": q, "div": div}


def mms_problem(n: int) -> Problem:
    g = Grid(1.0, 1.0, n, n)
    X, Y = g.mesh()
    f = mms_fields(X, Y)
    T_ex = lambda x, y: mms_fields(np.asarray(x, float), np.asarray(y, float))["T"]  # noqa: E731
    # bottom: −k ∂T/∂n = k T_y = h (T − T∞)  ⇒  T∞ = T − k T_y / h   (k = 1 on y = 0)
    T_inf = lambda xs: T_ex(xs, 0.0) - mms_fields(xs, 0.0 * xs)["Ty"] / MMS_H  # noqa: E731
    bc = {"left": {"type": "dirichlet", "value": lambda ys: T_ex(0.0 * ys, ys)},
          "right": {"type": "dirichlet", "value": lambda ys: T_ex(1.0 + 0.0 * ys, ys)},
          "top": {"type": "insulated"},
          "bottom": {"type": "convective", "h": MMS_H, "T_inf": T_inf}}
    return Problem(g, f["k"], f["q"], bc, sink=MMS_S, T_sink=0.0)


def mms_convergence(ns=(16, 32, 64, 128, 256)) -> list[dict]:
    """Solve the MMS problem on refined grids; return errors and observed orders."""
    rows = []
    for n in ns:
        p = mms_problem(n)
        sol = solve(p)
        X, Y = p.grid.mesh()
        e = sol.T - mms_fields(X, Y)["T"]
        rows.append({"n": n, "h": 1.0 / n, "err_L2": float(np.sqrt(np.mean(e ** 2))),
                     "err_max": float(np.abs(e).max()), "solve_s": sol.info["t_total_s"],
                     "energy_balance_rel_err": sol.info["energy_balance_rel_err"]})
    for a, b in zip(rows[:-1], rows[1:]):
        b["order_L2"] = float(np.log(a["err_L2"] / b["err_L2"]) / np.log(a["h"] / b["h"]))
        b["order_max"] = float(np.log(a["err_max"] / b["err_max"]) / np.log(a["h"] / b["h"]))
    return rows


def two_layer(n: int = 40, k_bottom: float = 150.0, k_top: float = 0.03,
              T_bottom: float = 0.0, T_top: float = 1.0) -> dict:
    """Layered slab, interface at y = 0.5 (on a face for even n). Returns max error vs exact."""
    g = Grid(1.0, 1.0, n, n)
    X, Y = g.mesh()
    k = np.where(Y < 0.5, k_bottom, k_top)
    bc = {"bottom": {"type": "dirichlet", "value": T_bottom},
          "top": {"type": "dirichlet", "value": T_top}}
    sol = solve(Problem(g, k, np.zeros_like(X), bc))
    flux = (T_top - T_bottom) / (0.5 / k_bottom + 0.5 / k_top)       # series resistance
    T_int = T_bottom + flux * 0.5 / k_bottom
    T_ex = np.where(Y < 0.5, T_bottom + flux * Y / k_bottom, T_int + flux * (Y - 0.5) / k_top)
    return {"n": n, "err_max": float(np.abs(sol.T - T_ex).max()), "T_interface_exact": float(T_int)}

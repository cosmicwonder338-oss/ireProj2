"""Chip layouts (top view, 10 mm × 10 mm silicon die) and problem construction.

A layout gives a *relative* power map (dimensionless shape). `build_problem` scales it so the hottest
point sits `target_peak_rise` K above ambient — the problem is linear in q, so this is exact — and the
resulting absolute power densities (W/m³, W/cm², total W) are reported, so they can be sanity-checked.

Two cooling models (config `physics.cooling`):
  edge      — the brief as written: bottom edge (y = 0) convective −k ∂T/∂n = h (T − T_amb),
              other edges insulated.
  backside  — a thin die cooled through its whole back face into a heat sink. Integrating the 3D
              equation over the die thickness t gives a distributed sink  −(h/t)(T − T_amb)  in the
              2D equation; all four edges insulated. Lateral spreading length  ℓ = sqrt(k t / h).
"""
from __future__ import annotations

import numpy as np

from common.seed import rng
from .fd_solver import Grid, Problem, solve


def smooth_box(X, Y, x0, x1, y0, y1, w):
    """≈1 inside [x0,x1]×[y0,y1], ≈0 outside, tanh edges of width w (w=0 → sharp)."""
    if w <= 0:
        return ((X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)).astype(float)
    sx = 0.5 * (np.tanh((X - x0) / w) - np.tanh((X - x1) / w))
    sy = 0.5 * (np.tanh((Y - y0) / w) - np.tanh((Y - y1) / w))
    return sx * sy


def gaussian(X, Y, xc, yc, r):
    return np.exp(-((X - xc) ** 2 + (Y - yc) ** 2) / (2 * r ** 2))


# ----- layouts: functions of (X, Y in metres, edge width w, seed) -> dict(q_rel, blocks) ------------
MM = 1e-3


def quad_core(X, Y, w, seed=0):
    """2×2 cores (3.5 mm), one busy core (top-right) at full power, three idle at 25%."""
    q = 0.05 * np.ones_like(X)                                   # uncore / leakage background
    cores = {"core0": (1.0, 4.5, 1.0, 4.5, 0.25), "core1": (5.5, 9.0, 1.0, 4.5, 0.25),
             "core2": (1.0, 4.5, 5.5, 9.0, 0.25), "core3_busy": (5.5, 9.0, 5.5, 9.0, 1.0)}
    for (x0, x1, y0, y1, a) in cores.values():
        q += (a - 0.05) * smooth_box(X, Y, x0 * MM, x1 * MM, y0 * MM, y1 * MM, w)
    return {"q_rel": q, "blocks": cores}


def cpu_gpu(X, Y, w, seed=0):
    """Dense CPU cluster (top-left), large lower-density GPU (right), memory controller strip."""
    q = 0.03 * np.ones_like(X)
    blocks = {"cpu": (1.0, 4.5, 5.5, 9.0, 1.0), "gpu": (5.5, 9.5, 1.0, 8.0, 0.45),
              "mem_ctrl": (1.0, 4.5, 0.8, 2.3, 0.2)}
    for (x0, x1, y0, y1, a) in blocks.values():
        q += (a - 0.03) * smooth_box(X, Y, x0 * MM, x1 * MM, y0 * MM, y1 * MM, w)
    return {"q_rel": q, "blocks": blocks}


def random_hotspots(X, Y, w, seed=0, n=5):
    """n Gaussian hotspots, random centres (≥1.5 mm from the edges), radii 0.4–1.0 mm."""
    g = rng(seed)
    q = 0.05 * np.ones_like(X)
    spots = {}
    for i in range(n):
        xc, yc = g.uniform(1.5, 8.5, size=2) * MM
        r = g.uniform(0.4, 1.0) * MM
        a = 1.0 if i == 0 else g.uniform(0.3, 0.8)
        q += a * gaussian(X, Y, xc, yc, r)
        spots[f"spot{i}"] = (xc / MM, yc / MM, r / MM, a)
    return {"q_rel": q, "blocks": spots}


def random_mixed(X, Y, w, seed=0):
    """Random library layout (for data-driven sensor placement): 1–3 rectangular blocks and
    1–5 Gaussian hotspots with random sizes and powers. Never used as a test chip."""
    g = rng(seed)
    q = 0.05 * np.ones_like(X)
    for _ in range(g.integers(1, 4)):
        x0, y0 = g.uniform(0.5, 7.0, size=2)
        wx, wy = g.uniform(1.0, 4.0, size=2)
        q += g.uniform(0.1, 1.0) * smooth_box(X, Y, x0 * MM, min(x0 + wx, 9.5) * MM, y0 * MM,
                                              min(y0 + wy, 9.5) * MM, w)
    for _ in range(g.integers(1, 6)):
        xc, yc = g.uniform(1.0, 9.0, size=2) * MM
        q += g.uniform(0.2, 1.0) * gaussian(X, Y, xc, yc, g.uniform(0.4, 1.2) * MM)
    return {"q_rel": q, "blocks": {}}


LAYOUTS = {"quad_core": quad_core, "cpu_gpu": cpu_gpu, "random_hotspots": random_hotspots,
           "random_mixed": random_mixed}


# ----- problem construction ---------------------------------------------------------------------
def cooling_setup(cfg, cooling: str):
    """Return (bc dict, sink coefficient W/m³K) for the chosen cooling model."""
    ph = cfg.physics
    T_amb, h = float(ph.T_amb), float(ph.h)
    if cooling == "edge":
        bc = {e: ({"type": "convective", "h": h, "T_inf": T_amb} if t == "convective"
                  else {"type": t}) for e, t in ph.boundary.items()}
        return bc, 0.0
    if cooling == "backside":
        bc = {e: {"type": "insulated"} for e in ("bottom", "top", "left", "right")}
        return bc, h / float(ph.thickness)
    raise ValueError(f"unknown cooling model '{cooling}'")


def build_problem(cfg, layout: str, cooling: str, nx: int | None = None, ny: int | None = None):
    """Build and solve the layout, with q scaled so that max(T) − T_amb = target_peak_rise."""
    ge, ph = cfg.geometry, cfg.physics
    grid = Grid(float(ge.Lx), float(ge.Ly), int(nx or cfg.solver.nx), int(ny or cfg.solver.ny))
    X, Y = grid.mesh()
    lay = LAYOUTS[layout](X, Y, float(cfg.layouts.edge_width_mm) * MM, seed=int(cfg.seed))
    k = np.full_like(X, float(ph.materials.silicon))
    bc, sink = cooling_setup(cfg, cooling)
    T_amb = float(ph.T_amb)

    # unit solve, then rescale (linear problem) and re-solve so reported fields are genuine solves
    q_unit = 1.0e8 * lay["q_rel"]
    unit = solve(Problem(grid, k, q_unit, bc, sink=sink, T_sink=T_amb))
    scale = float(cfg.layouts.target_peak_rise) / (unit.T.max() - T_amb)
    prob = Problem(grid, k, q_unit * scale, bc, sink=sink, T_sink=T_amb,
                   meta={"layout": layout, "cooling": cooling, "blocks": lay["blocks"],
                         "q_scale_W_m3": 1.0e8 * scale})
    return prob, solve(prob)


def q_function(cfg, layout: str, q_scale_W_m3: float):
    """Power density as a function of arbitrary points: q(x_m, y_m) -> W/m³ (numpy arrays).
    Lets the PINN evaluate the *exact* source at any collocation point, not a gridded copy."""
    w = float(cfg.layouts.edge_width_mm) * MM
    fn = LAYOUTS[layout]

    def q(x, y):
        return q_scale_W_m3 * fn(np.asarray(x, float), np.asarray(y, float), w,
                                 seed=int(cfg.seed))["q_rel"]
    return q


def summarize(cfg, prob: Problem, sol) -> dict:
    """Physical numbers for the report (°C, mm, W)."""
    g, T, q = prob.grid, sol.T, prob.q
    t = float(cfg.physics.thickness)
    j, i = np.unravel_index(np.argmax(T), T.shape)
    jq, iq = np.unravel_index(np.argmax(q), q.shape)
    P_total = float(np.sum(q) * g.dx * g.dy * t)
    out = {
        "layout": prob.meta["layout"], "cooling": prob.meta["cooling"],
        "grid": f"{g.nx}x{g.ny}",
        "T_max_C": float(T.max()), "T_min_C": float(T.min()), "T_mean_C": float(T.mean()),
        "hotspot_xy_mm": [float(g.x[i] / MM), float(g.y[j] / MM)],
        "q_peak_xy_mm": [float(g.x[iq] / MM), float(g.y[jq] / MM)],
        "hotspot_to_q_peak_mm": float(np.hypot(g.x[i] - g.x[iq], g.y[j] - g.y[jq]) / MM),
        "q_peak_W_m3": float(q.max()), "q_peak_W_cm2": float(q.max() * t / 1e4),
        "P_total_W": P_total, "avg_flux_W_cm2": P_total / (g.Lx * g.Ly * 1e4),
        # how "visible" the hotspot is: peak rise vs the spread of the field
        "T_range_C": float(T.max() - T.min()),
        "solve_time_s": sol.info["t_total_s"], "n_unknowns": sol.info["n_unknowns"],
        "energy_balance_rel_err": sol.info["energy_balance_rel_err"],
    }
    if prob.meta["cooling"] == "backside":
        out["spreading_length_mm"] = float(np.sqrt(float(cfg.physics.materials.silicon) * t
                                                   / float(cfg.physics.h)) / MM)
    return out

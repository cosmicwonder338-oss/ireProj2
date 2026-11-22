"""Side-view cross-section of the package for defect detection (Stage 6).

Bottom → top:  copper spreader | thermal interface material (TIM) | silicon die.
Heat is generated in the die (uniform background + a Gaussian hotspot); it leaves through the
bottom of the copper into the heat sink (convective, h, T_amb); the top and sides are insulated.
A hidden void (air, k ≈ 0.03 W/mK) sits inside the TIM and blocks heat locally, so the die
surface above it runs hotter. Only the top-surface temperature is measured.

Layer thicknesses are exaggerated for the TIM (0.2 mm instead of ~25–50 µm) so a 2D model can
resolve it; the physics (5000× conductivity contrast at the void) is unchanged.
"""
from __future__ import annotations

import numpy as np

from .fd_solver import Grid, Problem, solve
from .layouts import gaussian

MM = 1e-3


def stack_geometry(cfg) -> dict:
    """Layer list [(material, y0, y1)] in metres, total height, void box."""
    st = cfg.stack
    y = 0.0
    layers = []
    for mat, t_mm in st.layers:
        layers.append((str(mat), y, y + float(t_mm) * MM))
        y += float(t_mm) * MM
    tim = next(l for l in layers if l[0] == "tim")
    v = st.void
    void = (float(v.x0) * MM, float(v.x1) * MM, tim[1], tim[2])
    return {"layers": layers, "H": y, "W": float(cfg.geometry.Lx), "void": void}


def stack_fields(cfg, X, Y, with_void=True):
    """k (W/mK) and relative q at points (X, Y) in metres."""
    geo = stack_geometry(cfg)
    mats = cfg.physics.materials
    k = np.zeros_like(X, dtype=float)
    q = np.zeros_like(X, dtype=float)
    hs = cfg.stack.hotspot
    for mat, y0, y1 in geo["layers"]:
        inside = (Y >= y0) & (Y < y1) if y1 < geo["H"] else (Y >= y0) & (Y <= y1)
        k[inside] = float(mats[mat])
        if mat == "silicon":
            q[inside] = (1.0 + float(hs.amp) * gaussian(X, 0 * X + 0, float(hs.x) * MM, 0,
                                                        float(hs.width) * MM))[inside]
    if with_void:
        x0, x1, y0, y1 = geo["void"]
        k[(X >= x0) & (X <= x1) & (Y >= y0) & (Y < y1)] = float(mats.void)
    return k, q


def build_stack(cfg, with_void=True, q_scale=None):
    """FV problem + solution for the stack. Power is scaled so the hottest point (with the void)
    sits `layouts.target_peak_rise` above ambient unless q_scale (W/m³ per unit q_rel) is given."""
    geo = stack_geometry(cfg)
    st = cfg.stack
    ny = int(round(geo["H"] / (float(st.dy_mm) * MM)))
    grid = Grid(geo["W"], geo["H"], int(st.nx), ny)
    X, Y = grid.mesh()
    k, q_rel = stack_fields(cfg, X, Y, with_void)
    T_amb, h = float(cfg.physics.T_amb), float(cfg.physics.h)
    bc = {"bottom": {"type": "convective", "h": h, "T_inf": T_amb}}
    if q_scale is None:
        k_v, _ = stack_fields(cfg, X, Y, True)
        unit = solve(Problem(grid, k_v, 1e8 * q_rel, bc))
        q_scale = 1e8 * float(cfg.layouts.target_peak_rise) / (unit.T.max() - T_amb)
    prob = Problem(grid, k, q_scale * q_rel, bc, meta={"q_scale": q_scale, **geo})
    return prob, solve(prob)


def top_profile(sol) -> np.ndarray:
    """Temperature along the top surface (extrapolated from the top cell row, insulated → ≈ row)."""
    return sol.T[-1, :]

"""The chip problem in nondimensional form, plus its finite-volume reference solution.

Everything the PINN sees is in hat-variables (see common/scales.py):
    ∇̂·(k̂ ∇̂T̂) + q̂ − ŝ T̂ = 0,   edges: insulated (k̂ ∂T̂/∂n̂ = 0) or convective (+ Bi T̂ = 0)
Metrics and plots convert back to °C / mm.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from common.scales import Scales
from solver.layouts import build_problem, cooling_setup, q_function
from solver.sensors import placement, read_sensors

MM = 1e-3


@dataclass
class ChipProblem:
    cfg: object
    scales: Scales
    layout: str
    cooling: str
    fv_prob: object            # solver.fd_solver.Problem (SI)
    fv_sol: object             # solver.fd_solver.Solution (°C)
    Lx: float                  # nondimensional domain size
    Ly: float
    k_hat: float               # uniform conductivity (top view: silicon)
    s_hat: float               # back-face sink coefficient
    edges: dict                # edge -> (kind, Bi)
    res_scale: float           # residual normaliser: max(|q̂|, ŝ·max T̂)
    _q_si: object = None

    # ---------- construction --------------------------------------------------------------------
    @classmethod
    def build(cls, cfg, layout: str | None = None, cooling: str | None = None) -> "ChipProblem":
        layout = layout or cfg.problem.layout
        cooling = cooling or cfg.physics.cooling
        sc = Scales.from_config(cfg)
        prob, sol = build_problem(cfg, layout, cooling)
        bc, sink = cooling_setup(cfg, cooling)
        edges = {e: (b["type"], sc.biot(b["h"]) if b["type"] == "convective" else 0.0)
                 for e, b in bc.items()}
        s_hat = float(sc.s_nd(sink))
        q_hat_max = float(sc.q_nd(prob.q.max()))
        T_hat_max = float(sc.T_nd(sol.T.max()))
        self = cls(cfg=cfg, scales=sc, layout=layout, cooling=cooling, fv_prob=prob, fv_sol=sol,
                   Lx=sc.x_nd(prob.grid.Lx), Ly=sc.x_nd(prob.grid.Ly),
                   k_hat=float(sc.k_nd(cfg.physics.materials.silicon)), s_hat=s_hat, edges=edges,
                   res_scale=max(q_hat_max, s_hat * T_hat_max, 1.0))
        self._q_si = q_function(cfg, layout, prob.meta["q_scale_W_m3"])
        return self

    # ---------- fields at arbitrary points ------------------------------------------------------
    def q_hat(self, xy: torch.Tensor) -> torch.Tensor:
        """Exact nondimensional power density at points xy (N, 2) -> (N, 1)."""
        p = xy.detach().cpu().numpy().astype(np.float64) * self.scales.L
        q = self.scales.q_nd(self._q_si(p[:, 0], p[:, 1]))
        return torch.tensor(q, dtype=xy.dtype, device=xy.device).reshape(-1, 1)

    # ---------- reference data on the FV grid ---------------------------------------------------
    @property
    def grid_nd(self):
        g = self.fv_prob.grid
        return self.scales.x_nd(g.x), self.scales.x_nd(g.y)

    @property
    def T_ref(self) -> np.ndarray:
        return self.fv_sol.T                                   # °C, (ny, nx)

    @property
    def q_ref(self) -> np.ndarray:
        return self.fv_prob.q                                  # W/m³, (ny, nx)

    def sensors(self, n=None, noise_rel=None, kind=None, seed=None) -> dict:
        """Synthetic sensors (positions in m, readings in °C) from the FV solution."""
        sc = self.cfg.sensors
        n = int(n if n is not None else sc.n)
        seed = int(seed if seed is not None else self.cfg.seed)
        g = self.fv_prob.grid
        xy = placement(n, g.Lx, g.Ly, kind or sc.placement, seed=seed)
        return read_sensors(self.fv_sol, xy, float(noise_rel if noise_rel is not None
                                                   else sc.noise_rel),
                            float(self.cfg.physics.T_amb), seed=seed)


def field_metrics(T_pred: np.ndarray, T_ref: np.ndarray, T_amb: float, grid) -> dict:
    """Error metrics in physical units. Relative L2 is taken on the temperature *rise* (T − T_amb);
    relative to absolute °C it would look 2–3× better than it is."""
    e = T_pred - T_ref
    jr, ir = np.unravel_index(np.argmax(T_ref), T_ref.shape)
    jp, ip = np.unravel_index(np.argmax(T_pred), T_pred.shape)
    return {
        "rel_L2_rise": float(np.linalg.norm(e) / np.linalg.norm(T_ref - T_amb)),
        "max_abs_err_C": float(np.abs(e).max()),
        "mean_abs_err_C": float(np.abs(e).mean()),
        "peak_T_err_C": float(T_pred.max() - T_ref.max()),
        "hotspot_loc_err_mm": float(np.hypot(grid.x[ip] - grid.x[ir], grid.y[jp] - grid.y[jr]) / MM),
    }

"""Pieces every stage script shares: collocation sets, physics losses, grid evaluation, snapshots."""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from . import sampling
from .losses import bc_residual, mse, pde_residual
from .problem import ChipProblem, field_metrics


def config_signature(cfg, **extra) -> str:
    """Fingerprint of everything that defines a run (device excluded) — resume only if equal."""
    d = {k: v for k, v in cfg.to_dict().items() if k != "device"}
    d.update(extra)
    return json.dumps(d, sort_keys=True, default=str)


class Collocation:
    """Random interior + edge points, with the exact q̂ at the interior points."""

    def __init__(self, cp: ChipProblem, tcfg, device):
        self.cp, self.t, self.device = cp, tcfg, device
        self.xy = self.q = None
        self.edges = {}

    def resample(self):
        cp, t = self.cp, self.t
        self.xy = sampling.interior(int(t.n_collocation), cp.Lx, cp.Ly, self.device)
        self.q = cp.q_hat(self.xy)
        self.edges = {e: sampling.edge(e, int(t.n_boundary), cp.Lx, cp.Ly, self.device)
                      for e in cp.edges}


def physics_losses(model, col: Collocation, cp: ChipProblem, k_hat, q_hat=None):
    """(mean squared normalised PDE residual, mean squared BC residual).

    k_hat: float, scalar tensor (Stage 4) or callable xy -> (N,1) (Stage 6).
    q_hat: None (use the known q̂ at the collocation points) or callable xy -> (N,1) (Stage 5).
    Insulated edges use ∂T/∂n = 0 directly (not k ∂T/∂n), so an unknown k cannot satisfy them by
    shrinking towards zero.
    """
    xy = col.xy
    k = k_hat(xy) if callable(k_hat) else k_hat
    q = q_hat(xy) if callable(q_hat) else col.q
    r = pde_residual(model(xy), xy, k, q, cp.s_hat) / cp.res_scale
    l_bc = 0.0
    for e, (xb, nb) in col.edges.items():
        kind, Bi = cp.edges[e]
        kb = 1.0 if kind == "insulated" else (k_hat(xb) if callable(k_hat) else k_hat)
        l_bc = l_bc + mse(bc_residual(model(xb), xb, kb, nb, kind, Bi))
    return mse(r), l_bc / max(len(col.edges), 1)


class GridEvaluator:
    """Predict on the FV cell centres (°C), compute metrics, keep training snapshots."""

    def __init__(self, cp: ChipProblem, model, device, snap_stride: int = 2):
        self.cp, self.model, self.device = cp, model, device
        xg, yg = cp.grid_nd
        self.shape = cp.T_ref.shape
        self.G = sampling.grid_points(xg, yg, device)
        self.s = snap_stride
        self.G_snap = sampling.grid_points(xg[::snap_stride], yg[::snap_stride], device)
        self.snap_shape = (len(yg[::snap_stride]), len(xg[::snap_stride]))
        self.snaps = {"step": [], "T": []}
        self.T_amb = float(cp.cfg.physics.T_amb)

    def predict(self, coarse=False) -> np.ndarray:
        G, shape = (self.G_snap, self.snap_shape) if coarse else (self.G, self.shape)
        with torch.no_grad():
            out = self.model(G).cpu().numpy().reshape(shape).astype(np.float64)
        return self.cp.scales.T_dim(out)

    def metrics(self) -> dict:
        return field_metrics(self.predict(), self.cp.T_ref, self.T_amb, self.cp.fv_prob.grid)

    def evaluate(self) -> dict:
        m = self.metrics()
        return {"rel_L2": m["rel_L2_rise"], "max_err_C": m["max_abs_err_C"]}

    def snapshot(self, step: int):
        self.snaps["step"].append(step)
        self.snaps["T"].append(self.predict(coarse=True).astype(np.float32))

    def save_snapshots(self, path, **extra):
        if not self.snaps["step"]:
            return
        xg, yg = self.cp.grid_nd
        L = self.cp.scales.L
        np.savez_compressed(path, step=np.array(self.snaps["step"]), T=np.stack(self.snaps["T"]),
                            x=xg[::self.s] * L, y=yg[::self.s] * L,
                            T_ref=self.cp.T_ref[::self.s, ::self.s], **extra)

    def inference_time(self, reps: int = 10) -> float:
        for _ in range(3):
            self.predict()
        if isinstance(self.device, torch.device) and self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            self.predict()
        return (time.perf_counter() - t0) / reps


def sensor_tensors(cp: ChipProblem, sens: dict, device, noisy: bool = True):
    """Sensor positions (nondimensional) and readings (T̂) as tensors."""
    sc = cp.scales
    xy = torch.tensor(sc.x_nd(sens["xy"]), dtype=torch.float32, device=device)
    T = sens["T_noisy"] if noisy else sens["T_clean"]
    return xy, torch.tensor(sc.T_nd(T), dtype=torch.float32, device=device).reshape(-1, 1)

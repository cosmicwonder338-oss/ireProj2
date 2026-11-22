"""Pieces every stage script shares: collocation sets, physics losses, grid evaluation, snapshots."""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from . import sampling
from .losses import bc_residual, mse, pde_residual
from .problem import ChipProblem, field_metrics, q_field_metrics


def config_signature(cfg, **extra) -> str:
    """Fingerprint of everything that defines a run (device excluded) — resume only if equal."""
    d = {k: v for k, v in cfg.to_dict().items() if k != "device"}
    d.update(extra)
    return json.dumps(d, sort_keys=True, default=str)


class Collocation:
    """Interior + edge points, with the exact q̂ at the interior points.

    mode 'uniform': fresh uniform random points at every resample.
    mode 'rad': residual-based adaptive distribution (RAD, Wu et al. 2023). Draw a pool of
      `pool_factor`·N uniform points, score them with |PDE residual| from `score_fn`, and sample N
      without replacement with probability ∝ |r|^k / mean(|r|^k) + c. Points concentrate where the
      network is wrong (e.g. power-block edges) while c keeps some uniform coverage.
    """

    def __init__(self, cp: ChipProblem, tcfg, device, score_fn=None):
        self.cp, self.t, self.device = cp, tcfg, device
        self.mode = str(tcfg.get("collocation", "uniform"))
        self.score_fn = score_fn
        self.xy = self.q = None
        self.edges = {}
        self.n_resample = 0

    def _scores(self, pool: torch.Tensor) -> torch.Tensor:
        out = []
        for chunk in torch.split(pool, 8192):
            c = chunk.detach().requires_grad_(True)
            out.append(self.score_fn(c).detach().abs().reshape(-1))
        return torch.cat(out)

    def resample(self):
        cp, t = self.cp, self.t
        n = int(t.n_collocation)
        if self.mode == "rad" and self.score_fn is not None and self.n_resample > 0:
            pool = sampling.interior(int(t.get("rad_pool_factor", 4)) * n, cp.Lx, cp.Ly, self.device)
            r = self._scores(pool) ** float(t.get("rad_k", 1.0))
            p = r / (r.mean() + 1e-12) + float(t.get("rad_c", 1.0))
            idx = torch.multinomial(p, n, replacement=False)
            self.xy = pool[idx].detach().requires_grad_(True)
        else:
            self.xy = sampling.interior(n, cp.Lx, cp.Ly, self.device)
        self.q = cp.q_hat(self.xy)
        self.edges = {e: sampling.edge(e, int(t.n_boundary), cp.Lx, cp.Ly, self.device)
                      for e in cp.edges}
        self.n_resample += 1


def residual_scorer(model, cp: ChipProblem, k_hat, q_hat=None):
    """xy -> PDE residual (for RAD). q_hat None uses the exact source."""
    def score(xy):
        k = k_hat(xy) if callable(k_hat) else k_hat
        q = q_hat(xy) if callable(q_hat) else cp.q_hat(xy)
        return pde_residual(model(xy), xy, k, q, cp.s_hat)
    return score


def physics_losses(model, col: Collocation, cp: ChipProblem, k_hat, q_hat=None, res_scale=None):
    """(mean squared normalised PDE residual, mean squared BC residual).

    k_hat: float, scalar tensor (Stage 4) or callable xy -> (N,1) (Stage 6).
    q_hat: None (use the known q̂ at the collocation points) or callable xy -> (N,1) (Stage 5).
    Insulated edges use ∂T/∂n = 0 directly (not k ∂T/∂n), so an unknown k cannot satisfy them by
    shrinking towards zero.
    """
    xy = col.xy
    k = k_hat(xy) if callable(k_hat) else k_hat
    q = q_hat(xy) if callable(q_hat) else col.q
    r = pde_residual(model(xy), xy, k, q, cp.s_hat) / (res_scale or cp.res_scale)
    l_bc = 0.0
    for e, (xb, nb) in col.edges.items():
        kind, Bi = cp.edges[e]
        kb = 1.0 if kind == "insulated" else (k_hat(xb) if callable(k_hat) else k_hat)
        l_bc = l_bc + mse(bc_residual(model(xb), xb, kb, nb, kind, Bi))
    return mse(r), l_bc / max(len(col.edges), 1)


class GridEvaluator:
    """Predict on the FV cell centres (°C), compute metrics, keep training snapshots."""

    def __init__(self, cp: ChipProblem, model, device, snap_stride: int = 2, q_model=None):
        self.cp, self.model, self.device, self.q_model = cp, model, device, q_model
        xg, yg = cp.grid_nd
        self.shape = cp.T_ref.shape
        self.G = sampling.grid_points(xg, yg, device)
        self.s = snap_stride
        self.G_snap = sampling.grid_points(xg[::snap_stride], yg[::snap_stride], device)
        self.snap_shape = (len(yg[::snap_stride]), len(xg[::snap_stride]))
        self.snaps = {"step": [], "T": [], "q": []}
        self.T_amb = float(cp.cfg.physics.T_amb)

    def predict(self, coarse=False) -> np.ndarray:
        G, shape = (self.G_snap, self.snap_shape) if coarse else (self.G, self.shape)
        with torch.no_grad():
            out = self.model(G).cpu().numpy().reshape(shape).astype(np.float64)
        return self.cp.scales.T_dim(out)

    def predict_q(self, coarse=False) -> np.ndarray:
        """Predicted power density (W/m³) from the q-network."""
        G, shape = (self.G_snap, self.snap_shape) if coarse else (self.G, self.shape)
        with torch.no_grad():
            out = self.q_model(G).cpu().numpy().reshape(shape).astype(np.float64)
        return self.cp.scales.q_dim(out)

    def metrics(self) -> dict:
        return field_metrics(self.predict(), self.cp.T_ref, self.T_amb, self.cp.fv_prob.grid)

    def q_metrics(self) -> dict:
        return q_field_metrics(self.predict_q(), self.cp.q_ref, self.cp.fv_prob.grid)

    def evaluate(self) -> dict:
        m = self.metrics()
        out = {"rel_L2": m["rel_L2_rise"], "max_err_C": m["max_abs_err_C"]}
        if self.q_model is not None:
            out["q_rel_L2"] = self.q_metrics()["q_rel_L2"]
        return out

    def snapshot(self, step: int):
        self.snaps["step"].append(step)
        self.snaps["T"].append(self.predict(coarse=True).astype(np.float32))
        if self.q_model is not None:
            self.snaps["q"].append(self.predict_q(coarse=True).astype(np.float32))

    def save_snapshots(self, path, **extra):
        if not self.snaps["step"]:
            return
        xg, yg = self.cp.grid_nd
        L = self.cp.scales.L
        if self.snaps["q"]:
            extra = {"q": np.stack(self.snaps["q"]), "q_ref": self.cp.q_ref[::self.s, ::self.s],
                     **extra}
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


def data_weight(cfg, sigma_K: float) -> float:
    """Weight of the sensor-misfit term (mean squared error in T̂ units).

    'noise' (default): Gaussian-likelihood weighting w = (σ_pde / σ_data)², σ_data = sensor noise
      std in T̂ units (floored at sigma_floor_K for noise-free data). Noisy sensors then pull less,
      so the optimiser cannot buy a better data fit by bending T away from the physics — the
      failure seen in Stage 4 v1 with a fixed weight (k collapsed at 5 % noise).
    'fixed': w = loss.w_data.
    """
    lc = cfg.loss
    if str(lc.get("data_weighting", "noise")) == "fixed":
        return float(lc.w_data)
    s_data = max(float(sigma_K), float(lc.get("sigma_floor_K", 0.1))) / float(cfg.scales.dT)
    return (float(lc.get("sigma_pde", 0.01)) / s_data) ** 2

"""Layered PINN for the side-view stack (Stage 6): copper | TIM | silicon, unknown k in the TIM.

Interface handling — per-layer networks with continuity conditions (chosen over one network with a
smoothed k):
  * At a material interface T is continuous but ∂T/∂y jumps by the conductivity ratio
    (silicon/TIM = 50×, TIM/void = 100×). A single smooth MLP cannot represent that kink without
    smearing it over many cells — and smearing the k-jump is exactly what would hide a void.
  * So each layer gets its own T-network on its own box (inputs normalised per layer, so the thin
    0.2 mm TIM gets the full input range), and the interfaces get two residuals:
        T_below − T_above = 0                     (temperature continuity)
        k_below ∂yT_below − k_above ∂yT_above = 0 (heat-flux continuity)
  * The layer geometry is known by design; only the TIM conductivity field is unknown. It is a
    separate network k(x, y) in the TIM only, bounded between k_void/2 and 2·k_TIM via a sigmoid
    on log k, initialised at the nominal TIM value (i.e. "assume no defect" until data disagree).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from common.scales import Scales
from solver.stack import stack_fields, stack_geometry
from .losses import bc_residual, grad, mse, pde_residual
from .models import build_mlp


class KField(nn.Module):
    """log k̂ = lo + (hi − lo) · sigmoid(MLP(x, y)), initialised at k̂_init."""

    def __init__(self, mcfg, lo_xy, hi_xy, k_min, k_max, k_init):
        super().__init__()
        self.lo, self.hi = float(np.log(k_min)), float(np.log(k_max))
        frac = (np.log(k_init) - self.lo) / (self.hi - self.lo)
        self.net = build_mlp(mcfg, lo_xy, hi_xy, out_bias=float(np.log(frac / (1 - frac))))
        with torch.no_grad():                       # start (almost) exactly at the nominal value
            last = [m for m in self.net.net if isinstance(m, nn.Linear)][-1]
            last.weight.mul_(0.01)

    def log_k(self, xy):
        return self.lo + (self.hi - self.lo) * torch.sigmoid(self.net(xy))

    def forward(self, xy):
        return self.log_k(xy).exp()


class StackPINN:
    def __init__(self, cfg, fv_prob, device):
        self.cfg, self.device = cfg, device
        self.sc = Scales.from_config(cfg)
        geo = stack_geometry(cfg)
        self.geo = geo
        sc = self.sc
        self.layers = [(m, sc.x_nd(y0), sc.x_nd(y1)) for m, y0, y1 in geo["layers"]]
        self.W, self.H = sc.x_nd(geo["W"]), sc.x_nd(geo["H"])
        mats = cfg.physics.materials
        self.k_known = {m: float(sc.k_nd(mats[m])) for m in ("copper", "silicon")}
        self.k_tim, self.k_void = float(sc.k_nd(mats.tim)), float(sc.k_nd(mats.void))
        self.Bi = float(sc.biot(cfg.physics.h))
        self.q_scale = fv_prob.meta["q_scale"]
        # Isotropic input scaling: x and y share the scale 2/W. Normalising each thin layer's y-range
        # to [−1, 1] separately (anisotropic) stretches y by W/thickness (20× for the die), which
        # amplifies network curvature in ∂²T/∂y² by (W/t)² and left the silicon PDE residual stuck
        # (first CPU test). With isotropic scaling a thin layer is a thin slice of input space and
        # T varies slowly across it — as the physics says.
        iso = cfg.get("stack_pinn", {}).get("isotropic_inputs", True)

        def box(y0, y1):
            if not iso:
                return (0.0, y0), (self.W, y1)
            ym = 0.5 * (y0 + y1)
            return (0.0, ym - 0.5 * self.W), (self.W, ym + 0.5 * self.W)

        self.nets = nn.ModuleDict({m: build_mlp(cfg.model, *box(y0, y1))
                                   for m, y0, y1 in self.layers}).to(device)
        tim = self.box("tim")
        self.kf = KField(cfg.get("kmodel") or cfg.model, *box(tim[1], tim[2]),
                         self.k_void / 2, 2 * self.k_tim, self.k_tim).to(device)
        # residual normaliser: peak source; flux normaliser: mean heat flux through the stack
        si = self.box("silicon")
        self.q_max = float(sc.q_nd(fv_prob.q.max()))
        self.flux_ref = float(sc.q_nd(fv_prob.q[fv_prob.q > 0].mean())) * (si[2] - si[1])
        self.pts = {}

    # ------------------------------------------------------------------ helpers
    def box(self, mat):
        return next(l for l in self.layers if l[0] == mat)

    def k_of(self, mat, xy):
        return self.kf(xy) if mat == "tim" else self.k_known[mat]

    def q_hat(self, mat, xy):
        if mat != "silicon":
            return 0.0
        p = xy.detach().cpu().numpy().astype(np.float64) * self.sc.L
        _, q_rel = stack_fields(self.cfg, p[:, 0], np.full(len(p), self.geo["layers"][-1][1] + 1e-9))
        q = self.sc.q_nd(self.q_scale * q_rel)
        return torch.tensor(q, dtype=xy.dtype, device=xy.device).reshape(-1, 1)

    def _rand_box(self, n, x0, x1, y0, y1):
        u = torch.rand(n, 2, device=self.device)
        return (u * torch.tensor([x1 - x0, y1 - y0], device=self.device)
                + torch.tensor([x0, y0], device=self.device)).requires_grad_(True)

    def _line(self, n, x=None, y=None, y0=0.0, y1=0.0):
        t = torch.rand(n, device=self.device)
        if y is not None:
            xy = torch.stack([t * self.W, torch.full_like(t, y)], 1)
        else:
            xy = torch.stack([torch.full_like(t, x), y0 + t * (y1 - y0)], 1)
        return xy.requires_grad_(True)

    # ------------------------------------------------------------------ sampling
    def resample(self):
        t = self.cfg.training
        n_l, n_b = int(t.n_collocation) // len(self.layers), int(t.n_boundary)
        P = {"int": {}, "sides": {}, "q": {}}
        for m, y0, y1 in self.layers:
            P["int"][m] = self._rand_box(n_l, 0.0, self.W, y0, y1)
            P["q"][m] = self.q_hat(m, P["int"][m])
            P["sides"][m] = [(self._line(n_b // 2, x=0.0, y0=y0, y1=y1), (-1.0, 0.0)),
                             (self._line(n_b // 2, x=self.W, y0=y0, y1=y1), (1.0, 0.0))]
        P["bottom"] = self._line(n_b, y=self.layers[0][1])
        P["top"] = self._line(n_b, y=self.layers[-1][2])
        P["ifaces"] = [(a[0], b[0], self._line(n_b, y=a[2]))
                       for a, b in zip(self.layers[:-1], self.layers[1:])]
        self.pts = P

    # ------------------------------------------------------------------ losses
    def losses(self, xs, Ts):
        P, dev = self.pts, self.device
        l_pde = 0.0
        for m, *_ in self.layers:
            xy = P["int"][m]
            r = pde_residual(self.nets[m](xy), xy, self.k_of(m, xy), P["q"][m], 0.0) / self.q_max
            l_pde = l_pde + mse(r)
        l_pde = l_pde / len(self.layers)
        # boundaries: sides + top insulated (∂T/∂n = 0), copper bottom convective
        l_bc = 0.0
        for m, *_ in self.layers:
            for xy, nrm in P["sides"][m]:
                n = torch.tensor(nrm, device=dev).expand(len(xy), 2)
                l_bc = l_bc + mse(bc_residual(self.nets[m](xy), xy, 1.0, n, "insulated"))
        top = P["top"]
        l_bc = l_bc + mse(bc_residual(self.nets["silicon"](top), top, 1.0,
                                      torch.tensor([0.0, 1.0], device=dev).expand(len(top), 2),
                                      "insulated"))
        bot = P["bottom"]
        l_bc = l_bc + mse(bc_residual(self.nets["copper"](bot), bot, self.k_known["copper"],
                                      torch.tensor([0.0, -1.0], device=dev).expand(len(bot), 2),
                                      "convective", Bi=self.Bi))
        # interfaces: continuity of T and of normal heat flux
        l_if = 0.0
        for a, b, xy in P["ifaces"]:
            Ta, Tb = self.nets[a](xy), self.nets[b](xy)
            fa = self.k_of(a, xy) * grad(Ta, xy)[:, 1:2]
            fb = self.k_of(b, xy) * grad(Tb, xy)[:, 1:2]
            l_if = l_if + mse(Ta - Tb) + mse((fa - fb) / self.flux_ref)
        # data: top-surface sensors
        l_data = mse(self.nets["silicon"](xs) - Ts)
        # total variation of log k in the TIM
        xt = P["int"]["tim"]
        l_tv = grad(self.kf.log_k(xt), xt).abs().sum(dim=1).mean()
        return {"pde": l_pde, "bc": l_bc, "iface": l_if, "data": l_data, "tv": l_tv}

    # ------------------------------------------------------------------ evaluation
    def modules(self):
        return {**{f"T_{m}": net for m, net in self.nets.items()}, "k": self.kf}

    def predict_grid(self, grid):
        """T (°C) on the FV cell centres, each cell evaluated by its own layer's network."""
        X, Y = grid.mesh()
        xy = np.column_stack([self.sc.x_nd(X.ravel()), self.sc.x_nd(Y.ravel())])
        out = np.zeros(len(xy))
        with torch.no_grad():
            for i, (m, y0, y1) in enumerate(self.layers):
                last = i == len(self.layers) - 1
                sel = (xy[:, 1] >= y0) & ((xy[:, 1] <= y1) if last else (xy[:, 1] < y1))
                if sel.any():
                    pts = torch.tensor(xy[sel], dtype=torch.float32, device=self.device)
                    out[sel] = self.nets[m](pts).cpu().numpy().ravel()
        return self.sc.T_dim(out.reshape(X.shape))

    def k_line(self, x_m: np.ndarray) -> np.ndarray:
        """Predicted TIM conductivity (W/mK) along the TIM mid-line."""
        _, y0, y1 = self.box("tim")
        xy = np.column_stack([self.sc.x_nd(x_m), np.full(len(x_m), 0.5 * (y0 + y1))])
        with torch.no_grad():
            k = self.kf(torch.tensor(xy, dtype=torch.float32, device=self.device))
        return self.sc.k_dim(k.cpu().numpy().ravel())

    def top_line(self, x_m: np.ndarray) -> np.ndarray:
        xy = np.column_stack([self.sc.x_nd(x_m), np.full(len(x_m), self.H)])
        with torch.no_grad():
            T = self.nets["silicon"](torch.tensor(xy, dtype=torch.float32, device=self.device))
        return self.sc.T_dim(T.cpu().numpy().ravel())


def void_metrics(x_m, k_pred, void_x, k_tim, k_void) -> dict:
    """Detect the void as the region where k < √(k_TIM · k_void) (geometric midpoint)."""
    thr = float(np.sqrt(k_tim * k_void))
    pred = k_pred < thr
    true = (x_m >= void_x[0]) & (x_m <= void_x[1])
    inter, union = (pred & true).sum(), (pred | true).sum()
    out = {"void_iou": float(inter / union) if union else 0.0, "void_detected": bool(pred.any()),
           "k_min_pred_W_mK": float(k_pred.min()), "x_kmin_mm": float(x_m[np.argmin(k_pred)] * 1e3),
           "k_outside_void_median_W_mK": float(np.median(k_pred[~true])),
           "threshold_W_mK": thr}
    if pred.any():
        xs = x_m[pred]
        out.update({"void_pred_x0_mm": float(xs.min() * 1e3), "void_pred_x1_mm": float(xs.max() * 1e3),
                    "void_center_err_mm": float(abs(xs.mean() - np.mean(void_x)) * 1e3),
                    "false_positive_mm": float((pred & ~true).sum() * (x_m[1] - x_m[0]) * 1e3)})
    return out

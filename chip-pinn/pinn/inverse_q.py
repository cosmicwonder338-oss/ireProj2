"""Inverse problem 2: unknown power map q(x, y) from sparse noisy temperature sensors.

Two models: a T-network and a power-map model q(x,y) ≥ 0 — either a free-form softplus network
(qmodel.type: mlp) or K learnable Gaussian sources (qmodel.type: sources). Loss terms:
  pde    |∇·(k∇T) + q − sT|² / R²      R = ŝ·max(T̂_obs) — set from the data, not from the truth
  bc     insulated edges
  data   (T(sensor) − T_obs)², weight (σ_pde/σ_noise)²            (pinn.experiment.data_weight)
  power  (mean q / q_mean_known − 1)²   total chip power is known (package power telemetry)
  tv     mean |∇q| / q_mean             total variation: favours piecewise-constant power blocks

Why priors are unavoidable: 15 point measurements cannot determine a whole field q — infinitely
many power maps fit them. The PDE narrows that set enormously (T must be a smoothed version of
q/s), and positivity + known total power + TV pick the most plausible member.
"""
from __future__ import annotations

import time

import torch

from common.seed import set_seed
from .experiment import (Collocation, GridEvaluator, config_signature, data_weight, physics_losses,
                         residual_scorer, sensor_tensors)
from .losses import LossBalancer, grad, mse
from .models import GaussianSources, build_mlp, count_params
from .train import Trainer

LN2 = 0.6931471805599453


def run_inverse_q(cfg, cp, sens: dict, out_dir, device, tag: str = "", snapshots: bool = True):
    """Train T- and q-networks on the sensor data. Returns a dict with metrics, predictions,
    history and timings."""
    t, sc, lc = cfg.training, cp.scales, cfg.loss
    set_seed(cfg.seed)
    lo, hi = (0.0, 0.0), (cp.Lx, cp.Ly)
    T_model = build_mlp(cfg.model, lo, hi).to(device)
    q_mean = float(sc.q_nd(cp.q_ref.mean()))               # known total power → mean density
    qcfg = cfg.get("qmodel") or {}
    if str(qcfg.get("type", "mlp")) == "sources":
        q_model = GaussianSources(int(qcfg.get("n_sources", 4)), cp.Lx, cp.Ly, q_mean).to(device)
    else:
        # softplus(0) = ln 2, so an untrained q-net starts near the uniform map with the right power
        q_model = build_mlp(qcfg if "hidden" in qcfg else cfg.model, lo, hi,
                            out_transform="softplus", out_scale=q_mean / LN2).to(device)

    xs, Ts = sensor_tensors(cp, sens, device)
    res_scale = max(cp.s_hat * float(Ts.max()), 1.0)
    col = Collocation(cp, t, device, score_fn=residual_scorer(T_model, cp, cp.k_hat, q_model))
    ev = GridEvaluator(cp, T_model, device, q_model=q_model)
    weights = {"pde": float(lc.w_pde), "bc": float(lc.w_bc), "data": data_weight(cfg, sens["sigma"]),
               "power": float(lc.get("w_power", 1.0)), "tv": float(lc.get("w_tv_q", 1e-4))}
    params = list(T_model.parameters()) + list(q_model.parameters())
    bal = LossBalancer(params, every=int(t.get("balance_every", 100)),
                       enabled=bool(t.get("adaptive_weights", False)))
    holder = {}

    def loss_fn():
        l_pde, l_bc = physics_losses(T_model, col, cp, cp.k_hat, q_hat=q_model, res_scale=res_scale)
        l_data = mse(T_model(xs) - Ts)
        qv = q_model(col.xy)
        l_power = (qv.mean() / q_mean - 1.0) ** 2
        terms = {"pde": l_pde, "bc": l_bc, "data": l_data, "power": l_power}
        if weights["tv"] > 0:
            terms["tv"] = grad(qv, col.xy).abs().sum(dim=1).mean() / q_mean
        bal.update = holder["trainer"].phase == "adam"
        return bal(terms, weights), {**terms, **bal.logged()}

    trainer = Trainer({"T": T_model, "q": q_model}, loss_fn, t, out_dir, device,
                      resample=col.resample, evaluate=ev.evaluate,
                      snapshot=ev.snapshot if snapshots else None,
                      signature=config_signature(cfg, run=tag, sensors=sens["xy"].round(9).tolist()))
    holder["trainer"] = trainer
    history = trainer.run()

    T_pred, q_pred = ev.predict(), ev.predict_q()
    m = {**ev.metrics(), **ev.q_metrics()}
    if snapshots:
        ev.save_snapshots(out_dir / "snapshots.npz")
    torch.save({"T": T_model.state_dict(), "q": q_model.state_dict()}, out_dir / "model.pt")
    return {"metrics": m, "T_pred": T_pred, "q_pred": q_pred, "history": history,
            "train_time_s": trainer.elapsed, "weights": weights,
            "n_params": count_params(T_model) + count_params(q_model)}


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t0

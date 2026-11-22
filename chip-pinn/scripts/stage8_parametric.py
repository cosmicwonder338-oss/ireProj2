"""Stage 8b: parametric PINN T(x, y, P) — one network for every total power P in a range — and a
speed comparison against the FV solver.

To make this non-trivial, silicon conductivity is temperature dependent, k(T) = k300·(T/300 K)^−α
with α ≈ 1.3 (real silicon: ~20 % lower k at 85 °C than at 27 °C). The problem is then nonlinear,
T no longer scales linearly with P, and the FV reference needs Picard iterations (several sparse
solves per query) — while the trained PINN answers any P with one forward pass.

    python scripts/stage8_parametric.py --config configs/stage8_parametric.yaml

Outputs in results/<name>/: metrics.json, per_power.csv/.md, peak_vs_power.png, history.csv
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataclasses

import numpy as np
import pandas as pd
import torch

from common.config import parse_args, setup_run
from common.report import md_table
from pinn import sampling
from pinn.experiment import config_signature
from pinn.losses import LossBalancer, bc_residual, mse, pde_residual
from pinn.models import build_mlp, count_params
from pinn.problem import ChipProblem, field_metrics
from pinn.train import Trainer
from solver.fd_solver import solve_nonlinear


def main():
    args = parse_args(__doc__, default_config="stage8_parametric.yaml")
    cfg, out_dir, device = setup_run(args.config, args.set)
    s8, t, T_amb = cfg.stage8b, cfg.training, float(cfg.physics.T_amb)
    cp = ChipProblem.build(cfg)                       # nominal layout; its q is the P = 1 shape
    sc = cp.scales
    k300, alpha = float(s8.k300), float(s8.alpha)
    p_lo, p_hi = float(s8.p_min), float(s8.p_max)

    def k_dim(T_C):                                   # W/mK, numpy or torch
        return k300 * ((T_C + 273.15) / 300.0) ** (-alpha)

    def k_hat_of(T_hat):
        return k_dim(sc.T_dim(T_hat)) / sc.k_ref

    # ---------------- FV reference: nonlinear solve per power level ----------------
    def fv_solve(p):
        prob = dataclasses.replace(cp.fv_prob, q=p * cp.fv_prob.q)
        return solve_nonlinear(prob, k_dim, tol=1e-5)

    test_p = [float(p) for p in s8.test_powers]
    P_nom = float(np.sum(cp.fv_prob.q) * cp.fv_prob.grid.dx * cp.fv_prob.grid.dy
                  * float(cfg.physics.thickness))
    fv = {}
    for p in test_p:
        sol = fv_solve(p)
        lin = sc.T_amb + p * (cp.T_ref - sc.T_amb)    # constant-k (linear) prediction
        fv[p] = (sol, lin)
        print(f"[stage8b] FV nonlinear P={p * P_nom:5.1f} W: Tmax={sol.T.max():.2f}C "
              f"({sol.info['picard_iters']} Picard iters, {sol.info['t_total_s']:.2f}s); "
              f"constant-k model would say {lin.max():.2f}C", flush=True)

    # ---------------- parametric PINN ----------------
    model = build_mlp(cfg.model, lo=(0.0, 0.0, p_lo), hi=(cp.Lx, cp.Ly, p_hi)).to(device)
    pts = {}

    def resample():
        n, nb = int(t.n_collocation), int(t.n_boundary)
        xy = sampling.interior(n, cp.Lx, cp.Ly, device).detach()
        p = p_lo + (p_hi - p_lo) * torch.rand(n, 1, device=device)
        pts["in"] = torch.cat([xy, p], 1).requires_grad_(True)
        pts["q"] = p * cp.q_hat(xy)
        pts["edges"] = []
        for e in cp.edges:
            exy, nrm = sampling.edge(e, nb, cp.Lx, cp.Ly, device)
            pe = p_lo + (p_hi - p_lo) * torch.rand(nb, 1, device=device)
            inp = torch.cat([exy.detach(), pe], 1).requires_grad_(True)
            pts["edges"].append((inp, torch.cat([nrm, torch.zeros(nb, 1, device=device)], 1)))

    res_scale = cp.res_scale * p_hi
    bal = LossBalancer(model.parameters(), enabled=bool(t.get("adaptive_weights", False)))
    holder = {}

    def loss_fn():
        inp = pts["in"]
        T = model(inp)
        r = pde_residual(T, inp, k_hat_of(T), pts["q"], cp.s_hat) / res_scale
        l_bc = sum(mse(bc_residual(model(e), e, 1.0, nrm, "insulated")) for e, nrm in pts["edges"])
        l_bc = l_bc / len(pts["edges"])
        bal.update = holder["trainer"].phase == "adam"
        terms = {"pde": mse(r), "bc": l_bc}
        return bal(terms, {"pde": 1.0, "bc": float(cfg.loss.w_bc)}), terms

    xg, yg = cp.grid_nd
    G = sampling.grid_points(xg, yg, device)

    def predict(p):
        inp = torch.cat([G, torch.full((len(G), 1), p, device=device)], 1)
        with torch.no_grad():
            return sc.T_dim(model(inp).cpu().numpy().reshape(cp.T_ref.shape).astype(np.float64))

    def evaluate():
        errs = [field_metrics(predict(p), fv[p][0].T, T_amb, cp.fv_prob.grid)["rel_L2_rise"]
                for p in test_p]
        return {"rel_L2": float(np.mean(errs)), "rel_L2_worst": float(np.max(errs))}

    trainer = Trainer({"T": model}, loss_fn, t, out_dir, device, resample=resample,
                      evaluate=evaluate, signature=config_signature(cfg))
    holder["trainer"] = trainer
    history = trainer.run()

    # ---------------- accuracy and speed ----------------
    rows = []
    for p in test_p:
        sol, lin = fv[p]
        m = field_metrics(predict(p), sol.T, T_amb, cp.fv_prob.grid)
        rows.append({"P_W": p * P_nom, "p_rel": p, "fv_Tmax_C": float(sol.T.max()),
                     "pinn_Tmax_C": float(predict(p).max()), "constant_k_Tmax_C": float(lin.max()),
                     "pinn_rel_L2": m["rel_L2_rise"], "pinn_max_err_C": m["max_abs_err_C"],
                     "fv_time_s": sol.info["t_total_s"], "picard_iters": sol.info["picard_iters"]})
    df = pd.DataFrame(rows)
    for _ in range(3):
        predict(1.0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for p in np.linspace(p_lo, p_hi, 50):
        predict(float(p))
    t_query = (time.perf_counter() - t0) / 50
    t_fv = float(df.fv_time_s.mean())
    metrics = {"train_time_s": trainer.elapsed, "pinn_query_time_s": t_query,
               "fv_nonlinear_time_s_mean": t_fv, "speedup_per_query": t_fv / t_query,
               "break_even_queries": trainer.elapsed / max(t_fv - t_query, 1e-9),
               "mean_rel_L2": float(df.pinn_rel_L2.mean()),
               "worst_max_err_C": float(df.pinn_max_err_C.max()),
               "P_nominal_W": P_nom, "alpha": alpha, "k300": k300, "n_params": count_params(model)}
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    df.to_csv(out_dir / "per_power.csv", index=False)
    (out_dir / "per_power.md").write_text(md_table(df) + "\n", encoding="utf-8")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    print("\n=== Stage 8b result ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    for k, v in metrics.items():
        print(f"{k:28s} {v:.4g}")

    try:
        from pinn import plotting as PP
        from common import plotting as P
        ps = np.linspace(p_lo, p_hi, 41)
        fig, ax = P.figure(1, 1, panel=(6.0, 3.8))
        a = ax[0, 0]
        a.plot(ps * P_nom, [predict(float(p)).max() for p in ps], color=PP.SERIES[0], lw=2,
               label="parametric PINN (one network, any P)")
        a.plot(ps * P_nom, [T_amb + p * (cp.T_ref.max() - T_amb) for p in ps], color=PP.SERIES[1],
               lw=2, ls="--", label="constant-k model (linear in P)")
        a.scatter(df.P_W, df.fv_Tmax_C, s=50, color="white", edgecolor=P.INK, zorder=5,
                  label="FV with k(T), Picard")
        a.set_xlabel("total chip power (W)"); a.set_ylabel("peak temperature (°C)")
        a.set_title("Peak temperature vs power — silicon k(T) makes it nonlinear")
        a.legend(frameon=False, fontsize=8); a.grid(True, color="#e6e6e6", lw=0.6)
        P.save(fig, out_dir / "peak_vs_power.png")
        PP.training_curves(history, out_dir / "training.png", title="Parametric PINN training")
    except ImportError as e:
        print(f"[plot] skipped: {e}")


if __name__ == "__main__":
    main()

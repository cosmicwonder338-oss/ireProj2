"""Stage 6: detect a hidden void in the TIM layer from top-surface temperatures only.

    python scripts/stage6_defect.py --config configs/stage6_defect.yaml

Side-view stack (copper / TIM / silicon), layered PINN with interface continuity, unknown k(x, y)
in the TIM. Runs with and without a total-variation (TV) penalty on log k.
Outputs in results/<name>/: summary.csv/.md, k_profile.png, top_surface.png, stack_fields.png,
per-variant sub-folders (history, metrics, checkpoint).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from common.config import parse_args, setup_run
from common.report import md_table
from common.seed import rng, set_seed
from pinn.experiment import config_signature, data_weight
from pinn.losses import LossBalancer
from pinn.problem import field_metrics
from pinn.stack_pinn import StackPINN, void_metrics
from pinn.train import Trainer
from solver.sensors import interpolator
from solver.stack import build_stack, top_profile


def top_sensors(cfg, sol, H):
    s6 = cfg.stage6
    n, margin = int(s6.n_sensors), 0.1e-3
    x = np.linspace(margin, sol.grid.Lx - margin, n)
    T = interpolator(sol)(np.column_stack([np.full(n, H), x]))
    sigma = float(s6.noise_rel) * (sol.T.max() - float(cfg.physics.T_amb))
    return x, T + rng(int(cfg.seed) + 1).normal(0, sigma, n), sigma


def run_variant(cfg, prob, sol, x_s, T_s, sigma, w_tv, out_dir, device):
    set_seed(cfg.seed)
    t = cfg.training
    sp = StackPINN(cfg, prob, device)
    sc = sp.sc
    xs = torch.tensor(np.column_stack([sc.x_nd(x_s), np.full(len(x_s), sp.H)]),
                      dtype=torch.float32, device=device)
    Ts = torch.tensor(sc.T_nd(T_s), dtype=torch.float32, device=device).reshape(-1, 1)
    weights = {"pde": float(cfg.loss.w_pde), "bc": float(cfg.loss.w_bc),
               "iface": float(cfg.loss.get("w_iface", 1.0)), "data": data_weight(cfg, sigma),
               "tv": float(w_tv)}
    params = [p for m in sp.modules().values() for p in m.parameters()]
    bal = LossBalancer(params, every=int(t.get("balance_every", 100)),
                       enabled=bool(t.get("adaptive_weights", False)))
    holder = {}
    grid, T_amb = prob.grid, float(cfg.physics.T_amb)
    void_x = prob.meta["void"][:2]
    mats = cfg.physics.materials

    def loss_fn():
        terms = sp.losses(xs, Ts)
        if weights["tv"] == 0:
            terms = {k: v for k, v in terms.items() if k != "tv"}
        bal.update = holder["trainer"].phase == "adam"
        return bal(terms, weights), {**terms, **bal.logged()}

    def evaluate():
        m = field_metrics(sp.predict_grid(grid), sol.T, T_amb, grid)
        v = void_metrics(grid.x, sp.k_line(grid.x), void_x, float(mats.tim), float(mats.void))
        return {"rel_L2": m["rel_L2_rise"], "max_err_C": m["max_abs_err_C"], "void_iou": v["void_iou"]}

    trainer = Trainer(sp.modules(), loss_fn, t, out_dir, device, resample=sp.resample,
                      evaluate=evaluate, signature=config_signature(cfg, w_tv=w_tv))
    holder["trainer"] = trainer
    history = trainer.run()
    T_pred = sp.predict_grid(grid)
    k_line = sp.k_line(grid.x)
    m = {**field_metrics(T_pred, sol.T, T_amb, grid),
         **void_metrics(grid.x, k_line, void_x, float(mats.tim), float(mats.void)),
         "train_time_s": trainer.elapsed, "w_tv": float(w_tv), "w_data": weights["data"]}
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    np.savez_compressed(out_dir / "fields.npz", T_pred=T_pred, k_line=k_line,
                        top=sp.top_line(grid.x))
    return m, T_pred, k_line, sp.top_line(grid.x), history


def main():
    args = parse_args(__doc__, default_config="stage6_defect.yaml")
    cfg, out_dir, device = setup_run(args.config, args.set)
    prob, sol = build_stack(cfg, with_void=True)
    _, sol0 = build_stack(cfg, with_void=False, q_scale=prob.meta["q_scale"])
    H = prob.meta["H"]
    x_s, T_s, sigma = top_sensors(cfg, sol, H)
    sig = top_profile(sol) - top_profile(sol0)
    print(f"[stage6] stack {prob.grid.nx}x{prob.grid.ny} cells, FV solve "
          f"{sol.info['t_total_s']:.2f}s; void signature on top surface: max {sig.max():.2f} K; "
          f"{len(x_s)} top sensors, noise sigma {sigma:.2f} K", flush=True)

    rows, res = [], {}
    for w_tv in cfg.stage6.tv_weights:
        tag = "no_tv" if float(w_tv) == 0 else f"tv_{float(w_tv):g}"
        sub = out_dir / tag
        sub.mkdir(parents=True, exist_ok=True)
        m, T_pred, k_line, top, hist = run_variant(cfg, prob, sol, x_s, T_s, sigma, w_tv, sub, device)
        res[tag] = (m, T_pred, k_line, top, hist)
        rows.append({"variant": tag, **{k: m.get(k, np.nan) for k in (
            "void_iou", "void_center_err_mm", "void_pred_x0_mm", "void_pred_x1_mm",
            "false_positive_mm", "k_min_pred_W_mK", "k_outside_void_median_W_mK",
            "rel_L2_rise", "max_abs_err_C", "train_time_s")}})
        print(f"[stage6] {tag:10s} IoU={m['void_iou']:.3f} kmin={m['k_min_pred_W_mK']:.3g} W/mK "
              f"at x={m['x_kmin_mm']:.2f}mm  k outside (median)={m['k_outside_void_median_W_mK']:.3g}  "
              f"T relL2={m['rel_L2_rise']:.2e} max={m['max_abs_err_C']:.2f}K", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.md").write_text(
        f"Void: x = {cfg.stack.void.x0}–{cfg.stack.void.x1} mm in the TIM; "
        f"top-surface signature {sig.max():.2f} K; {len(x_s)} sensors, σ = {sigma:.2f} K\n\n"
        + md_table(df) + "\n", encoding="utf-8")
    print("\n=== Stage 6 result ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    try:
        plot(cfg, prob, sol, sol0, x_s, T_s, res, out_dir)
    except ImportError as e:
        print(f"[plot] skipped: {e}")


def plot(cfg, prob, sol, sol0, x_s, T_s, res, out_dir):
    from common import plotting as P
    from pinn.plotting import SERIES
    g = prob.grid
    xm = g.x * 1e3
    mats = cfg.physics.materials
    v0, v1 = float(cfg.stack.void.x0), float(cfg.stack.void.x1)

    # 1) k along the TIM mid-line (log scale)
    fig, ax = P.figure(1, 1, panel=(6.4, 3.8))
    a = ax[0, 0]
    k_true = np.where((xm >= v0) & (xm <= v1), float(mats.void), float(mats.tim))
    a.semilogy(xm, k_true, color=P.INK, lw=2, label="true k (TIM with void)")
    for c, (tag, (m, _, k_line, _, _)) in zip(SERIES, res.items()):
        a.semilogy(xm, k_line, color=c, lw=2, label=f"PINN, {tag.replace('_', ' ')} "
                   f"(IoU {m['void_iou']:.2f})")
    a.axhline(np.sqrt(float(mats.tim) * float(mats.void)), color=P.MUTED, lw=1, ls=":")
    a.annotate("detection threshold", (0.2, np.sqrt(float(mats.tim) * float(mats.void))),
               xytext=(0, 4), textcoords="offset points", fontsize=8, color=P.MUTED)
    a.set_xlabel("x (mm)"); a.set_ylabel("TIM conductivity k (W/mK)")
    a.set_title("Recovered TIM conductivity along the layer")
    a.legend(frameon=False, fontsize=8); a.grid(True, color="#e6e6e6", lw=0.6)
    P.save(fig, out_dir / "k_profile.png")

    # 2) top-surface temperature: what was measured and what the PINN fits
    fig, ax = P.figure(1, 1, panel=(6.4, 3.8))
    a = ax[0, 0]
    a.plot(xm, top_profile(sol), color=P.INK, lw=2, label="true (with void)")
    a.plot(xm, top_profile(sol0), color=P.MUTED, lw=1.5, ls="--", label="same chip, no void")
    for c, (tag, (_, _, _, top, _)) in zip(SERIES, res.items()):
        a.plot(xm, top, color=c, lw=1.5, label=f"PINN, {tag.replace('_', ' ')}")
    a.scatter(x_s * 1e3, T_s, s=22, color="white", edgecolor=P.INK, zorder=5, label="sensors")
    a.axvspan(v0, v1, color=P.MUTED, alpha=0.12, lw=0)
    a.annotate("void below", (v0, a.get_ylim()[0]), xytext=(3, 4), textcoords="offset points",
               fontsize=8, color=P.INK)
    a.set_xlabel("x (mm)"); a.set_ylabel("top-surface T (°C)")
    a.set_title("Top-surface temperature — the only data")
    a.legend(frameon=False, fontsize=8); a.grid(True, color="#e6e6e6", lw=0.6)
    P.save(fig, out_dir / "top_surface.png")

    # 3) cross-section fields (y stretched ×3 for readability)
    tag = list(res)[-1]
    T_pred = res[tag][1]
    fig, ax = P.figure(3, 1, panel=(9.0, 2.3))
    ext = [0, g.Lx * 1e3, 0, g.Ly * 1e3]
    for a, F, title, kind, lab in ((ax[0, 0], sol.T, "True temperature (FV)", "T", "°C"),
                                   (ax[1, 0], T_pred, f"PINN ({tag.replace('_', ' ')})", "T", "°C"),
                                   (ax[2, 0], T_pred - sol.T, "PINN − true", "err", "K")):
        P.field(a, g, F, title, kind, lab, contours=False,
                vmin=float(sol.T.min()) if kind == "T" else None,
                vmax=float(sol.T.max()) if kind == "T" else None)
        a.set_aspect(3.0)
        for _, y0, y1 in prob.meta["layers"][1:]:
            a.axhline(y0 * 1e3, color=P.INK, lw=0.6, alpha=0.6)
        a.add_patch(P.plt.Rectangle((v0, prob.meta["void"][2] * 1e3), v1 - v0,
                                    (prob.meta["void"][3] - prob.meta["void"][2]) * 1e3,
                                    fill=False, ec=P.INK, lw=1, ls="--"))
        a.set_xlim(ext[0], ext[1])
    P.save(fig, out_dir / "stack_fields.png")
    print(f"[plot] wrote figures to {out_dir}")


if __name__ == "__main__":
    main()

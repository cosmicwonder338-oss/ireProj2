"""Stage 4: recover an unknown *constant* conductivity k from sparse noisy sensors.

    python scripts/stage4_inverse_k.py --config configs/stage4_inverse_k.yaml
    python scripts/stage4_inverse_k.py --set stage4.run_pinn=false     # FV baseline only (no torch)

For every noise level × noise draw (same sensor positions):
  * PINN: T-network + one trainable scalar k̂ = exp(θ), loss = PDE + BC + w_data · sensor misfit
  * FV least squares baseline: the uniform k whose FV solution best fits the same readings
Outputs in results/<name>/: summary.csv, summary.md, k_error_vs_noise.png, k_trajectory.png and
one sub-folder per PINN run (history.csv, metrics.json, checkpoint).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from common.config import parse_args, setup_run
from common.report import md_table
from common.seed import set_seed
from solver.inverse import fit_constant_k
from solver.layouts import build_problem
from solver.sensors import placement, read_sensors


def run_pinn(cfg, cp, sens, out_dir, device, k_init, tag):
    """One inverse PINN run. Returns (k_pred W/mK, field metrics, train seconds, history)."""
    from pinn.experiment import (Collocation, GridEvaluator, config_signature, data_weight,
                                 physics_losses, sensor_tensors)
    from pinn.losses import mse
    from pinn.models import PositiveScalar, build_mlp
    from pinn.train import Trainer

    set_seed(cfg.seed)                                  # same network init for every run
    k_ref, t = cp.scales.k_ref, cfg.training
    model = build_mlp(cfg.model, lo=(0.0, 0.0), hi=(cp.Lx, cp.Ly)).to(device)
    kmod = PositiveScalar(k_init / k_ref).to(device)
    col = Collocation(cp, t, device)
    ev = GridEvaluator(cp, model, device)
    xs, Ts = sensor_tensors(cp, sens, device)
    w_pde, w_bc = float(cfg.loss.w_pde), float(cfg.loss.w_bc)
    w_data = data_weight(cfg, sens["sigma"])
    print(f"[stage4] {tag}: data weight {w_data:.3g}")

    def loss_fn():
        k = kmod()
        l_pde, l_bc = physics_losses(model, col, cp, k)
        l_data = mse(model(xs) - Ts)
        return (w_pde * l_pde + w_bc * l_bc + w_data * l_data,
                {"pde": l_pde, "bc": l_bc, "data": l_data, "k_W_mK": k.detach() * k_ref})

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer({"T": model, "k": kmod}, loss_fn, t, out_dir, device, resample=col.resample,
                      evaluate=ev.evaluate, signature=config_signature(cfg, run=tag))
    history = trainer.run()
    k_pred = float(kmod().detach().cpu()) * k_ref
    m = ev.metrics()
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"k_pred": k_pred, "train_time_s": trainer.elapsed, "w_data": w_data, **m}, f,
                  indent=2)
    return k_pred, m, trainer.elapsed, history


def main():
    args = parse_args(__doc__, default_config="stage4_inverse_k.yaml")
    cfg, out_dir, device = setup_run(args.config, args.set)
    s4, T_amb = cfg.stage4, float(cfg.physics.T_amb)
    layout, cooling = cfg.problem.layout, cfg.physics.cooling
    k_true = float(cfg.physics.materials.silicon)

    prob, sol = build_problem(cfg, layout, cooling)                       # synthetic truth, 200²
    xy = placement(int(cfg.sensors.n), prob.grid.Lx, prob.grid.Ly, cfg.sensors.placement,
                   seed=int(cfg.seed))
    cp = None
    if bool(s4.run_pinn):
        from pinn.problem import ChipProblem
        cp = ChipProblem.build(cfg)

    rows, trajectories = [], {}
    for noise in s4.noise_levels:
        seeds = [s4.noise_seeds[0]] if float(noise) == 0.0 else list(s4.noise_seeds)
        for ns in seeds:
            sens = read_sensors(sol, xy, float(noise), T_amb, seed=int(cfg.seed) + 1000 * int(ns))
            tag = f"noise{100 * float(noise):g}pct_seed{ns}"
            fv = fit_constant_k(cfg, layout, cooling, prob.meta["q_scale_W_m3"], xy,
                                sens["T_noisy"], n=int(s4.baseline_grid))
            row = {"noise_pct": 100 * float(noise), "noise_seed": int(ns),
                   "sigma_K": sens["sigma"], "k_true": k_true,
                   "fv_k": fv["k"], "fv_k_err_pct": 100 * (fv["k"] / k_true - 1),
                   "fv_rms_misfit_K": fv["rms_misfit_K"], "fv_time_s": fv["time_s"],
                   "fv_n_solves": fv["n_solves"]}
            if cp is not None:
                k_pred, m, t_train, hist = run_pinn(cfg, cp, sens, out_dir / tag, device,
                                                    float(s4.k_init), tag)
                row.update({"pinn_k": k_pred, "pinn_k_err_pct": 100 * (k_pred / k_true - 1),
                            "pinn_T_rel_L2": m["rel_L2_rise"],
                            "pinn_T_max_err_C": m["max_abs_err_C"], "pinn_time_s": t_train})
                if ns == s4.noise_seeds[0]:
                    trajectories[float(noise)] = [(h["step"], h["k_W_mK"]) for h in hist
                                                  if "k_W_mK" in h]
            rows.append(row)
            print(f"[stage4] {tag:22s} sigma={row['sigma_K']:.2f}K  "
                  f"FV k={row['fv_k']:.2f} ({row['fv_k_err_pct']:+.2f}%)"
                  + (f"  PINN k={row['pinn_k']:.2f} ({row['pinn_k_err_pct']:+.2f}%) "
                     f"T relL2={row['pinn_T_rel_L2']:.2e}" if cp is not None else ""), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    err_cols = ["fv_k_err_pct"] + (["pinn_k_err_pct"] if cp is not None else [])
    agg = df.groupby("noise_pct")[err_cols].agg(
        lambda v: np.sqrt(np.mean(np.square(v)))).reset_index()          # RMS error over draws
    agg.columns = ["noise_pct"] + [c.replace("_err_pct", "_rms_err_pct") for c in err_cols]
    print("\n=== Stage 4 result: RMS k error (%) over the paired noise draws ===")
    print(agg.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Extra baseline-only draws: the paired draws are enough to compare methods on identical
    # data, but too few to estimate the error-vs-noise curve itself (one draw can be harmless).
    extra = []
    for noise in [n for n in s4.noise_levels if float(n) > 0]:
        for d in range(int(s4.baseline_draws)):
            sens = read_sensors(sol, xy, float(noise), T_amb, seed=int(cfg.seed) + 7919 * (d + 1))
            fv = fit_constant_k(cfg, layout, cooling, prob.meta["q_scale_W_m3"], xy,
                                sens["T_noisy"], n=int(s4.baseline_grid))
            extra.append({"noise_pct": 100 * float(noise), "draw": d,
                          "fv_k_err_pct": 100 * (fv["k"] / k_true - 1)})
    ext = pd.DataFrame(extra)
    ext_md = ""
    if len(ext):
        ext.to_csv(out_dir / "baseline_draws.csv", index=False)
        ext_agg = ext.groupby("noise_pct")["fv_k_err_pct"].agg(
            rms=lambda v: np.sqrt(np.mean(np.square(v))), mean="mean", std="std").reset_index()
        print(f"\n=== FV baseline, {int(s4.baseline_draws)} draws per level: k error (%) ===")
        print(ext_agg.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        ext_md = (f"\n\n### FV baseline, {int(s4.baseline_draws)} extra draws per noise level "
                  "(k error %)\n\n" + md_table(ext_agg))
    (out_dir / "summary.md").write_text(
        "### Recovered k — per run\n\n" + md_table(df) +
        "\n\n### RMS k error (%) over the paired noise draws\n\n" + md_table(agg) + ext_md + "\n",
        encoding="utf-8")

    try:
        plot(df, trajectories, k_true, k_init=float(s4.k_init), out_dir=out_dir, ext=ext)
    except ImportError as e:
        print(f"[plot] skipped: {e}")


def plot(df, trajectories, k_true, k_init, out_dir, ext=None):
    from common import plotting as P
    from pinn.plotting import SERIES

    fig, ax = P.figure(1, 1, panel=(5.6, 4.0))
    a = ax[0, 0]
    series = [("fv_k_err_pct", "FV least squares (100² grid)", SERIES[1], "s")]
    if "pinn_k_err_pct" in df:
        series.insert(0, ("pinn_k_err_pct", "PINN", SERIES[0], "o"))
    for col, label, c, mk in series:
        a.scatter(df["noise_pct"], df[col], color=c, marker=mk, s=36, alpha=0.55,
                  edgecolor="white", linewidth=1.5, zorder=3)
        mean = df.groupby("noise_pct")[col].mean()
        a.plot(mean.index, mean.values, color=c, lw=2, marker=mk, ms=8, label=label + " (mean)",
               zorder=4)
    if ext is not None and len(ext):
        band = ext.groupby("noise_pct")["fv_k_err_pct"].agg(
            lambda v: np.sqrt(np.mean(np.square(v))))
        x = np.concatenate([[0.0], band.index.values])
        r = np.concatenate([[0.0], band.values])
        a.fill_between(x, -r, r, color=SERIES[1], alpha=0.12, lw=0,
                       label=f"FV ± RMS error over {ext['draw'].nunique()} draws")
    a.axhline(0, color=P.MUTED, lw=1)
    a.set_xlabel("sensor noise (% of max temperature rise)")
    a.set_ylabel("recovered k error (%)")
    a.set_title(f"Constant-k recovery from 15 sensors (true k = {k_true:.0f} W/mK)")
    a.legend(frameon=False)
    a.grid(True, color="#e6e6e6", lw=0.6)
    P.save(fig, out_dir / "k_error_vs_noise.png")

    if trajectories:
        fig, ax = P.figure(1, 1, panel=(5.6, 4.0))
        a = ax[0, 0]
        for c, (noise, traj) in zip(SERIES, sorted(trajectories.items())):
            s, k = zip(*traj)
            a.plot(s, k, color=c, lw=2, label=f"{100 * noise:g}% noise")
        a.axhline(k_true, color=P.INK, lw=1, ls="--")
        a.annotate(f"true k = {k_true:.0f}", (0, k_true), xytext=(4, 4),
                   textcoords="offset points", fontsize=8, color=P.INK)
        a.set_xlabel("iteration"); a.set_ylabel("k (W/mK)")
        a.set_title(f"PINN estimate of k during training (start {k_init:.0f} W/mK)")
        a.legend(frameon=False)
        a.grid(True, color="#e6e6e6", lw=0.6)
        P.save(fig, out_dir / "k_trajectory.png")
    print(f"[plot] wrote figures to {out_dir}")


if __name__ == "__main__":
    main()

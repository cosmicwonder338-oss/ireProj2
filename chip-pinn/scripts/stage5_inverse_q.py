"""Stage 5 (main result): reconstruct the full temperature map AND the unknown power map from
10–15 noisy sensors; locate the hotspot. Baseline: thin-plate-spline interpolation of the sensors.

    python scripts/stage5_inverse_q.py --config configs/stage5_inverse_q.yaml

Outputs in results/<name>/<layout>/: metrics.json, history.csv, reconstruction.png, training.png,
snapshots.npz (for the GIF); results/<name>/summary.csv + summary.md across layouts.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from common.config import parse_args, setup_run
from common.report import md_table
from pinn.inverse_q import run_inverse_q
from pinn.problem import ChipProblem, field_metrics
from solver.inverse import rbf_interpolate

KEYS = ["rel_L2_rise", "max_abs_err_C", "mean_abs_err_C", "peak_T_err_C", "hotspot_loc_err_mm"]
SUMMARY_COLS = ["layout", "n_sensors", "pinn_rel_L2_rise", "pinn_max_abs_err_C",
                "pinn_hotspot_loc_err_mm", "pinn_peak_T_err_C", "pinn_q_rel_L2",
                "pinn_q_peak_loc_err_mm", "spline_rel_L2_rise", "spline_max_abs_err_C",
                "spline_hotspot_loc_err_mm", "spline_peak_T_err_C", "train_time_s"]


def run_layout(cfg, layout, out_dir, device, snapshots=True, sensor_xy=None, plots=True):
    """One inverse-q reconstruction + spline baseline on the same sensors. Returns a summary row."""
    cp = ChipProblem.build(cfg, layout=layout)
    sens = cp.sensors(xy=sensor_xy)
    out = run_inverse_q(cfg, cp, sens, out_dir, device, tag=layout, snapshots=snapshots)
    T_amb, grid = float(cfg.physics.T_amb), cp.fv_prob.grid
    T_rbf = rbf_interpolate(sens["xy"], sens["T_noisy"], grid)
    m_rbf = field_metrics(T_rbf, cp.T_ref, T_amb, grid)
    m = out["metrics"]
    row = {"layout": layout, "n_sensors": len(sens["xy"]), "noise_K": sens["sigma"],
           **{f"pinn_{k}": v for k, v in m.items()}, **{f"spline_{k}": m_rbf[k] for k in KEYS},
           "train_time_s": out["train_time_s"], "n_params": out["n_params"],
           "best_sensor_minus_true_peak_C": float(sens["T_clean"].max() - cp.T_ref.max())}
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({**row, "weights": out["weights"]}, f, indent=2)
    pd.DataFrame(out["history"]).to_csv(out_dir / "history.csv", index=False)
    if plots:
        try:
            from pinn import plotting as PP
            PP.reconstruction(grid, cp.T_ref, out["T_pred"], cp.q_ref, out["q_pred"], sens["xy"],
                              sens["T_noisy"], out_dir / "reconstruction.png", T_rbf=T_rbf,
                              title=f"Power-map & temperature reconstruction — {layout}",
                              m=m, m_rbf=m_rbf, thickness=float(cfg.physics.thickness))
            PP.training_curves(out["history"], out_dir / "training.png",
                               terms=("loss", "pde", "data", "tv", "power"), metric="rel_L2",
                               title=f"Inverse-q training — {layout}")
        except ImportError as e:
            print(f"[plot] skipped: {e}")
    print(f"[stage5] {layout:16s} n={row['n_sensors']:2d} PINN: T relL2={row['pinn_rel_L2_rise']:.3e} "
          f"max={row['pinn_max_abs_err_C']:.2f}K hotspot={row['pinn_hotspot_loc_err_mm']:.2f}mm "
          f"peakT={row['pinn_peak_T_err_C']:+.2f}K q relL2={row['pinn_q_rel_L2']:.3f} | "
          f"spline: max={row['spline_max_abs_err_C']:.2f}K "
          f"hotspot={row['spline_hotspot_loc_err_mm']:.2f}mm peakT={row['spline_peak_T_err_C']:+.2f}K",
          flush=True)
    return row


def main():
    args = parse_args(__doc__, default_config="stage5_inverse_q.yaml")
    cfg, out_dir, device = setup_run(args.config, args.set)
    rows = []
    for layout in cfg.stage5.layouts:
        sub = out_dir / layout
        sub.mkdir(parents=True, exist_ok=True)
        rows.append(run_layout(cfg, layout, sub, device))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    md = md_table(df[SUMMARY_COLS])
    print("\n=== Stage 5 result ===")
    print(df[SUMMARY_COLS].to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    # Prior sensitivity: once the sensors are fitted to noise level, the temperature between sensors
    # is decided by the priors (15 points cannot pin down a whole power map). Report how the answer
    # moves with the prior — every variant is shown, none is selected using the ground truth.
    variants = cfg.stage5.get("prior_variants") or {}
    if variants:
        from common.config import Config, apply_overrides
        srows = []
        for vname, sets in variants.items():
            c = Config(apply_overrides(cfg.to_dict(), list(sets)))
            sub = out_dir / "priors" / vname
            sub.mkdir(parents=True, exist_ok=True)
            r = run_layout(c, cfg.stage5.get("prior_layout", "quad_core"), sub, device,
                           snapshots=False, plots=True)
            srows.append({"prior": vname, **{k: r[k] for k in SUMMARY_COLS[2:8]}})
        sdf = pd.DataFrame(srows)
        sdf.to_csv(out_dir / "priors.csv", index=False)
        md += ("\n\n#### Sensitivity to the power-map prior (quad_core; main runs use a free-form "
               "q-network with TV weight 1e-4)\n\n" + md_table(sdf))
        print("\n=== Stage 5 prior sensitivity ===")
        print(sdf.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    (out_dir / "summary.md").write_text(md + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

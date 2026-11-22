"""Stage 1: finite-volume reference solutions + synthetic sensors for every layout × cooling model.

    python scripts/stage1_reference.py --config configs/stage1_reference.yaml [--set ...]

Outputs in results/stage1_reference/:
  <layout>_<cooling>.npz   T, k, q (SI, °C), cell centres, sensor xy / clean / noisy readings
  metrics.json, metrics.csv
  layouts_<cooling>.png    power map | temperature map + sensors, one row per layout
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from common.config import parse_args, setup_run
from solver.layouts import build_problem, summarize
from solver.sensors import placement, read_sensors


def main():
    args = parse_args(__doc__, default_config="stage1_reference.yaml")
    cfg, out_dir, _ = setup_run(args.config, args.set)
    sc, T_amb = cfg.sensors, float(cfg.physics.T_amb)
    rows, results = [], {}

    for cooling in cfg.stage1.coolings:
        for layout in cfg.stage1.layouts:
            prob, sol = build_problem(cfg, layout, cooling)
            xy = placement(int(sc.n), prob.grid.Lx, prob.grid.Ly, sc.placement, seed=cfg.seed)
            sens = read_sensors(sol, xy, float(sc.noise_rel), T_amb, seed=cfg.seed)
            m = summarize(cfg, prob, sol)
            m["sensor_sigma_K"] = sens["sigma"]
            m["sensor_T_range_C"] = [float(sens["T_clean"].min()), float(sens["T_clean"].max())]
            m["sensor_max_minus_true_max_K"] = float(sens["T_clean"].max() - sol.T.max())
            rows.append(m)
            results[(layout, cooling)] = (prob, sol, sens, m)
            np.savez_compressed(
                out_dir / f"{layout}_{cooling}.npz", T=sol.T, k=prob.k, q=prob.q,
                x=prob.grid.x, y=prob.grid.y, Lx=prob.grid.Lx, Ly=prob.grid.Ly,
                sink=np.broadcast_to(prob.sink, sol.T.shape), T_amb=T_amb,
                sensor_xy=sens["xy"], sensor_T_clean=sens["T_clean"],
                sensor_T_noisy=sens["T_noisy"], sensor_sigma=sens["sigma"],
                meta=json.dumps({"layout": layout, "cooling": cooling,
                                 "blocks": prob.meta["blocks"]}))
            print(f"{layout:16s} {cooling:9s} Tmax={m['T_max_C']:6.2f}C Tmin={m['T_min_C']:6.2f}C "
                  f"range={m['T_range_C']:5.2f}K  hotspot@({m['hotspot_xy_mm'][0]:.2f},"
                  f"{m['hotspot_xy_mm'][1]:.2f})mm  q_peak->hotspot={m['hotspot_to_q_peak_mm']:.2f}mm  "
                  f"P={m['P_total_W']:.2f}W  q_peak={m['q_peak_W_cm2']:.0f}W/cm2  "
                  f"solve={m['solve_time_s']:.3f}s  Ebal={m['energy_balance_rel_err']:.1e}")

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)

    try:
        plot(cfg, results, out_dir)
    except ImportError as e:
        print(f"[plot] skipped: {e}")


def plot(cfg, results, out_dir):
    from common import plotting as P

    for cooling in cfg.stage1.coolings:
        layouts = list(cfg.stage1.layouts)
        fig, axes = P.figure(len(layouts), 2)
        for r, layout in enumerate(layouts):
            prob, sol, sens, m = results[(layout, cooling)]
            P.field(axes[r, 0], prob.grid, prob.q * float(cfg.physics.thickness) / 1e4,
                    f"{layout}: power density", kind="q", label="W/cm²", contours=False)
            P.field(axes[r, 1], prob.grid, sol.T,
                    f"{layout}: temperature ({cooling} cooling)", kind="T", label="°C",
                    sensors_xy=sens["xy"])
            P.mark(axes[r, 1], tuple(m["hotspot_xy_mm"]), f"{m['T_max_C']:.1f} °C")
        P.save(fig, out_dir / f"layouts_{cooling}.png")
        print(f"[plot] wrote {out_dir / f'layouts_{cooling}.png'}")


if __name__ == "__main__":
    main()

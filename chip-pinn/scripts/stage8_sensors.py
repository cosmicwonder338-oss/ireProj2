"""Stage 8a: how many sensors, and where? Error vs number of sensors (5, 10, 15, 25) for random
placement vs data-driven QR placement (POD of random chips + QR column pivoting), on the inverse
power-map problem (quad_core). The spline-interpolation baseline is reported on the same sensors.

    python scripts/stage8_sensors.py --config configs/stage8_sensors.yaml

Outputs in results/<name>/: sensors.csv, sensors.md, sensor_count.png, placements.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from common.config import parse_args, setup_run
from common.report import md_table
from scripts.stage5_inverse_q import run_layout
from solver.sensors import placement, pod_library, qr_placement


def main():
    args = parse_args(__doc__, default_config="stage8_sensors.yaml")
    cfg, out_dir, device = setup_run(args.config, args.set)
    s8 = cfg.stage8
    Lx, Ly = float(cfg.geometry.Lx), float(cfg.geometry.Ly)
    lib_grid, snaps = pod_library(cfg, n_maps=int(s8.library_maps))
    layout = cfg.problem.layout
    rows, placements = [], {}
    for n in s8.n_sensors:
        for kind in s8.placements:
            if kind == "qr":
                xy = qr_placement(int(n), lib_grid, snaps)
            else:
                xy = placement(int(n), Lx, Ly, kind, seed=int(cfg.seed))
            placements[(int(n), kind)] = xy
            sub = out_dir / f"n{int(n)}_{kind}"
            sub.mkdir(parents=True, exist_ok=True)
            row = run_layout(cfg, layout, sub, device, snapshots=False, sensor_xy=xy, plots=False)
            rows.append({"n_sensors": int(n), "placement": kind, **row})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sensors.csv", index=False)
    cols = ["n_sensors", "placement", "pinn_rel_L2_rise", "pinn_max_abs_err_C",
            "pinn_hotspot_loc_err_mm", "pinn_peak_T_err_C", "pinn_q_rel_L2",
            "spline_max_abs_err_C", "spline_hotspot_loc_err_mm", "spline_peak_T_err_C"]
    (out_dir / "sensors.md").write_text(md_table(df[cols]) + "\n", encoding="utf-8")
    print("\n=== Stage 8a: sensor count x placement ===")
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    try:
        plot(df, placements, cfg, out_dir)
    except ImportError as e:
        print(f"[plot] skipped: {e}")


def plot(df, placements, cfg, out_dir):
    from common import plotting as P
    from pinn.plotting import SERIES
    metrics = [("pinn_max_abs_err_C", "spline_max_abs_err_C", "max |T error| (K)"),
               ("pinn_hotspot_loc_err_mm", "spline_hotspot_loc_err_mm", "hotspot location error (mm)"),
               ("pinn_peak_T_err_C", "spline_peak_T_err_C", "peak-T error (K)")]
    fig, ax = P.figure(1, 3, panel=(4.6, 3.6))
    for a, (pc, sc_, label) in zip(ax[0], metrics):
        i = 0
        for method, col, ls in (("PINN", pc, "-"), ("spline", sc_, "--")):
            for kind in df.placement.unique():
                d = df[df.placement == kind].sort_values("n_sensors")
                y = d[col].abs() if "peak" in col else d[col]
                a.plot(d.n_sensors, y, ls=ls, marker="o", ms=6, lw=2, color=SERIES[i],
                       label=f"{method}, {kind}")
                i += 1
        a.set_xlabel("number of sensors"); a.set_ylabel(label.replace("peak-T", "|peak-T|"))
        a.set_xticks(sorted(df.n_sensors.unique()))
        a.grid(True, color="#e6e6e6", lw=0.6)
    ax[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Sensor-count study — inverse power-map reconstruction", color=P.INK)
    P.save(fig, out_dir / "sensor_count.png")

    ns = sorted({n for n, _ in placements})
    kinds = sorted({k for _, k in placements})
    fig, ax = P.figure(len(kinds), len(ns), panel=(2.6, 2.6))
    for r, kind in enumerate(kinds):
        for c, n in enumerate(ns):
            a = ax[r, c]
            xy = placements[(n, kind)] * 1e3
            a.scatter(xy[:, 0], xy[:, 1], s=24, color="white", edgecolor=P.INK)
            a.set_xlim(0, 10); a.set_ylim(0, 10); a.set_aspect("equal")
            a.set_title(f"{kind}, n={n}", fontsize=9)
            a.set_xticks([0, 5, 10]); a.set_yticks([0, 5, 10])
    P.save(fig, out_dir / "placements.png")
    print(f"[plot] wrote figures to {out_dir}")


if __name__ == "__main__":
    main()

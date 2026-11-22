"""Stage 7: ablation — vanilla PINN vs Fourier features vs adaptive loss weighting vs residual-based
adaptive collocation (RAD), on the forward problem (Stage 3) and the inverse power-map problem
(Stage 5, quad_core). Each variant is a separate run of the stage script (own folder, checkpoint,
resume); this script launches them and collects one table.

    python scripts/stage7_ablation.py --config configs/stage7_ablation.yaml

Outputs in results/<name>/: ablation.csv, ablation.md, ablation.png
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from common.config import parse_args, setup_run
from common.paths import RESULTS_DIR
from common.report import md_table

VARIANTS = {
    "vanilla": [],
    "fourier_features": ["model.fourier.n=64", "model.fourier.sigma=2.0"],
    "adaptive_weights": ["training.adaptive_weights=true"],
    "rad_collocation": ["training.collocation=rad"],
}
PROBLEMS = {
    "forward (stage 3)": ("stage3_forward.py", "stage3_forward.yaml", "forward", []),
    "inverse q (stage 5)": ("stage5_inverse_q.py", "stage5_inverse_q.yaml", "inverse_q",
                            ["stage5.layouts=[quad_core]", "stage5.prior_variants={}"]),
}


def main():
    args = parse_args(__doc__, default_config="stage7_ablation.yaml")
    cfg, out_dir, _ = setup_run(args.config, args.set)
    t = cfg.training
    passthrough = [f"training.{k}={t[k]}" for k in
                   ("adam_iters", "lbfgs_iters", "n_collocation", "time_budget_min", "eval_every")]
    passthrough.append("training.snapshot_every=0")
    rows = []
    for pname, (script, conf, short, extra) in PROBLEMS.items():
        for vname, sets in VARIANTS.items():
            run_name = f"{cfg.name}/{short}_{vname}"
            cmd = [sys.executable, str(ROOT / "scripts" / script), "--config", conf,
                   "--set", f"name={run_name}", *extra, *passthrough, *sets]
            print(f"\n[stage7] ===== {pname} / {vname} =====", flush=True)
            subprocess.run(cmd, check=True, cwd=ROOT)
            mdir = RESULTS_DIR / run_name
            if short == "inverse_q":
                m = json.loads((mdir / "quad_core" / "metrics.json").read_text())
                row = {"T_rel_L2": m["pinn_rel_L2_rise"], "T_max_err_C": m["pinn_max_abs_err_C"],
                       "hotspot_err_mm": m["pinn_hotspot_loc_err_mm"],
                       "peak_T_err_C": m["pinn_peak_T_err_C"], "q_rel_L2": m["pinn_q_rel_L2"],
                       "train_time_s": m["train_time_s"]}
            else:
                m = json.loads((mdir / "metrics.json").read_text())
                row = {"T_rel_L2": m["rel_L2_rise"], "T_max_err_C": m["max_abs_err_C"],
                       "hotspot_err_mm": m["hotspot_loc_err_mm"], "peak_T_err_C": m["peak_T_err_C"],
                       "q_rel_L2": float("nan"), "train_time_s": m["train_time_s"]}
            rows.append({"problem": pname, "variant": vname, **row})
            print(f"[stage7] {pname} / {vname}: {row}", flush=True)

    df = pd.DataFrame(rows)
    for p in df["problem"].unique():                      # error relative to vanilla
        base = df[(df.problem == p) & (df.variant == "vanilla")]["T_rel_L2"].iloc[0]
        df.loc[df.problem == p, "T_rel_L2_vs_vanilla"] = df.loc[df.problem == p, "T_rel_L2"] / base
    df.to_csv(out_dir / "ablation.csv", index=False)
    (out_dir / "ablation.md").write_text(md_table(df) + "\n", encoding="utf-8")
    print("\n=== Stage 7 ablation ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    try:
        plot(df, out_dir)
    except ImportError as e:
        print(f"[plot] skipped: {e}")


def plot(df, out_dir):
    """Dot plot: T relative L2 error per variant, one panel per problem (log x)."""
    from common import plotting as P
    from pinn.plotting import SERIES
    probs = list(df["problem"].unique())
    fig, ax = P.figure(1, len(probs), panel=(4.8, 3.0))
    for a, p in zip(ax[0], probs):
        d = df[df.problem == p].reset_index(drop=True)
        for i, r in d.iterrows():
            a.plot([r.T_rel_L2], [i], "o", ms=9, color=SERIES[0], mec="white", mew=1.5)
            a.annotate(f"{r.T_rel_L2:.2e}", (r.T_rel_L2, i), xytext=(8, -3),
                       textcoords="offset points", fontsize=8, color=P.INK)
        a.set_xscale("log"); a.set_yticks(range(len(d)))
        a.set_yticklabels([v.replace("_", " ") for v in d.variant]); a.invert_yaxis()
        a.set_title(p); a.set_xlabel("temperature rel. L2 error (lower is better)")
        a.grid(True, axis="x", color="#e6e6e6", lw=0.6)
        a.margins(x=0.35)
    P.save(fig, out_dir / "ablation.png")


if __name__ == "__main__":
    main()

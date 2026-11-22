"""Stage 2: verification report — FV convergence on the manufactured solution, the layered-void
exactness check, and the autograd residual check (if torch is available).

    python scripts/stage2_verify.py --config configs/stage2_verify.yaml

Outputs in results/stage2_verify/: convergence.csv, metrics.json, convergence.png
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common.config import parse_args, setup_run
from solver.verification import MMS_S, mms_convergence, mms_fields, two_layer


def autograd_check():
    try:
        import torch
    except ImportError:
        return {"autograd": "skipped (torch not installed)"}
    from pinn.losses import pde_residual
    out = {}
    for dt in (torch.float64, torch.float32):
        torch.manual_seed(0)
        xy = torch.rand(4096, 2, dtype=dt).requires_grad_(True)
        f = mms_fields(xy[:, 0:1], xy[:, 1:2], lib=torch)
        r = pde_residual(f["T"], xy, f["k"], f["q"].detach(), s=MMS_S)
        out[f"autograd_mms_residual_max_{str(dt).split('.')[-1]}"] = r.abs().max().item()
    out["mms_q_max"] = float(f["q"].abs().max().item())
    return out


def main():
    args = parse_args(__doc__, default_config="stage2_verify.yaml")
    cfg, out_dir, _ = setup_run(args.config, args.set)
    rows = mms_convergence(tuple(cfg.stage2.ns))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "convergence.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3e}"))

    layered = [two_layer(n, 150.0, k2) for n in (20, 40) for k2 in (3.0, 0.03)]
    metrics = {"mms": rows, "two_layer": layered, **autograd_check()}
    for r in layered:
        print(f"two-layer n={r['n']}: max error vs exact = {r['err_max']:.2e}")
    for k, v in metrics.items():
        if k.startswith("autograd") or k == "mms_q_max":
            print(f"{k}: {v}")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    try:
        from common import plotting as P
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"[plot] skipped: {e}")
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    h = df["h"].to_numpy()
    ax.loglog(h, df["err_L2"], "o-", color=P.INK, lw=2, ms=8, label="L2 (RMS) error")
    ax.loglog(h, df["err_max"], "s--", color=P.MUTED, lw=2, ms=8, label="max error")
    ref = df["err_L2"].iloc[0] * (h / h[0]) ** 2
    ax.loglog(h, ref, ":", color=P.MUTED, lw=1, label="slope 2 reference")
    ax.set_xlabel("cell size h (nondimensional)"); ax.set_ylabel("error in T")
    ax.set_title("FV solver: manufactured-solution convergence")
    ax.legend(frameon=False)
    ax.grid(True, which="major", color="#e6e6e6", lw=0.6)
    P.save(fig, out_dir / "convergence.png")
    print(f"[plot] wrote {out_dir / 'convergence.png'}")


if __name__ == "__main__":
    main()

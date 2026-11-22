"""Stage 3: forward PINN — known k and q, predict T. Compared against the FV reference.

    python scripts/stage3_forward.py --config configs/stage3_forward.yaml [--set problem.layout=cpu_gpu]

Outputs in results/<name>/: metrics.json, history.csv, comparison.png, training.png,
snapshots.npz (predictions during training, for the GIF), model.pt
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from common.config import parse_args, setup_run
from common.device import env_report
from pinn import sampling
from pinn.losses import bc_residual, mse, pde_residual
from pinn.models import build_mlp, count_params
from pinn.problem import ChipProblem, field_metrics
from pinn.train import Trainer


def main():
    args = parse_args(__doc__, default_config="stage3_forward.yaml")
    cfg, out_dir, device = setup_run(args.config, args.set)
    cp = ChipProblem.build(cfg)
    sc, T_amb, t = cp.scales, float(cfg.physics.T_amb), cfg.training
    print(f"[stage3] layout={cp.layout} cooling={cp.cooling} device={device} "
          f"k̂={cp.k_hat:.3f} ŝ={cp.s_hat:.3f} q̂_max={sc.q_nd(cp.q_ref.max()):.3f} "
          f"residual scale={cp.res_scale:.3f} edges={cp.edges}")

    model = build_mlp(cfg.model, lo=(0.0, 0.0), hi=(cp.Lx, cp.Ly)).to(device)
    print(f"[stage3] MLP {cfg.model.layers}x{cfg.model.hidden} {cfg.model.activation}, "
          f"{count_params(model)} parameters")

    # ---------------- collocation sets ----------------
    pts = {}

    def resample():
        pts["xy"] = sampling.interior(int(t.n_collocation), cp.Lx, cp.Ly, device)
        pts["q"] = cp.q_hat(pts["xy"])
        pts["edges"] = {e: sampling.edge(e, int(t.n_boundary), cp.Lx, cp.Ly, device)
                        for e in cp.edges}

    w_pde, w_bc = float(cfg.loss.w_pde), float(cfg.loss.w_bc)

    def loss_fn():
        xy = pts["xy"]
        r = pde_residual(model(xy), xy, cp.k_hat, pts["q"], cp.s_hat) / cp.res_scale
        l_pde = mse(r)
        l_bc = sum(mse(bc_residual(model(xb), xb, cp.k_hat, nb, *cp.edges[e]))
                   for e, (xb, nb) in pts["edges"].items()) / len(pts["edges"])
        return w_pde * l_pde + w_bc * l_bc, {"pde": l_pde, "bc": l_bc}

    # ---------------- evaluation on the full FV grid ----------------
    xg, yg = cp.grid_nd
    G = sampling.grid_points(xg, yg, device)
    ny, nx = cp.T_ref.shape

    def predict(points, shape):
        with torch.no_grad():
            return sc.T_dim(model(points).cpu().numpy().reshape(shape).astype(np.float64))

    def evaluate():
        m = field_metrics(predict(G, (ny, nx)), cp.T_ref, T_amb, cp.fv_prob.grid)
        return {"rel_L2": m["rel_L2_rise"], "max_err_C": m["max_abs_err_C"]}

    G_snap = sampling.grid_points(xg[::2], yg[::2], device)
    snaps = {"step": [], "T": []}

    def snapshot(step):
        snaps["step"].append(step)
        snaps["T"].append(predict(G_snap, (len(yg[::2]), len(xg[::2]))).astype(np.float32))

    sig = json.dumps({k: v for k, v in cfg.to_dict().items() if k != "device"}, sort_keys=True)
    trainer = Trainer({"T": model}, loss_fn, t, out_dir, device, resample=resample,
                      evaluate=evaluate, snapshot=snapshot, signature=sig)
    history = trainer.run()

    # ---------------- final metrics ----------------
    T_pred = predict(G, (ny, nx))
    m = field_metrics(T_pred, cp.T_ref, T_amb, cp.fv_prob.grid)
    for _ in range(3):                                    # warm-up, then time inference
        predict(G, (ny, nx))
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        predict(G, (ny, nx))
    t_inf = (time.perf_counter() - t0) / 10
    lb = [h for h in history if "pde" in h]
    metrics = {
        "layout": cp.layout, "cooling": cp.cooling, **m,
        "train_time_s": trainer.elapsed, "adam_iters_done": trainer.adam_done,
        "lbfgs_iters_done": trainer.lbfgs_done,
        "fv_solve_time_s": cp.fv_sol.info["t_total_s"], "fv_grid": f"{nx}x{ny}",
        "pinn_inference_time_s_full_grid": t_inf,
        "final_loss": lb[-1]["loss"] if lb else None,
        "final_loss_pde": lb[-1]["pde"] if lb else None, "final_loss_bc": lb[-1]["bc"] if lb else None,
        "n_params": count_params(model), "residual_scale": cp.res_scale, "env": env_report(),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    if snaps["step"]:
        np.savez_compressed(out_dir / "snapshots.npz", step=np.array(snaps["step"]),
                            T=np.stack(snaps["T"]), x=xg[::2] * sc.L, y=yg[::2] * sc.L,
                            T_ref=cp.T_ref[::2, ::2])
    torch.save(model.state_dict(), out_dir / "model.pt")

    print("\n=== Stage 3 result ===")
    for k in ("rel_L2_rise", "max_abs_err_C", "mean_abs_err_C", "peak_T_err_C",
              "hotspot_loc_err_mm", "train_time_s", "fv_solve_time_s",
              "pinn_inference_time_s_full_grid"):
        print(f"{k:34s} {metrics[k]:.4g}")

    from pinn import plotting as PP
    PP.compare(cp.fv_prob.grid, cp.T_ref, T_pred, out_dir / "comparison.png",
               f"Forward PINN — {cp.layout}, {cp.cooling} cooling", metrics=m)
    PP.training_curves(history, out_dir / "training.png", title="Forward PINN training")
    print(f"[plot] wrote comparison.png, training.png to {out_dir}")


if __name__ == "__main__":
    main()

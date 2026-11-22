"""Stage 3: forward PINN — known k and q, predict T. Compared against the FV reference.

    python scripts/stage3_forward.py --config configs/stage3_forward.yaml [--set problem.layout=cpu_gpu]

Outputs in results/<name>/: metrics.json, history.csv, comparison.png, training.png,
snapshots.npz (predictions during training, for the GIF), model.pt
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from common.config import parse_args, setup_run
from common.device import env_report
from pinn.experiment import (Collocation, GridEvaluator, config_signature, physics_losses,
                             residual_scorer)
from pinn.losses import LossBalancer
from pinn.models import build_mlp, count_params
from pinn.problem import ChipProblem
from pinn.train import Trainer


def main():
    args = parse_args(__doc__, default_config="stage3_forward.yaml")
    cfg, out_dir, device = setup_run(args.config, args.set)
    cp = ChipProblem.build(cfg)
    sc, t = cp.scales, cfg.training
    print(f"[stage3] layout={cp.layout} cooling={cp.cooling} device={device} "
          f"k̂={cp.k_hat:.3f} ŝ={cp.s_hat:.3f} q̂_max={sc.q_nd(cp.q_ref.max()):.3f} "
          f"residual scale={cp.res_scale:.3f} edges={cp.edges}")

    model = build_mlp(cfg.model, lo=(0.0, 0.0), hi=(cp.Lx, cp.Ly)).to(device)
    print(f"[stage3] MLP {cfg.model.layers}x{cfg.model.hidden} {cfg.model.activation}, "
          f"{count_params(model)} parameters")

    # ablation switches (Stage 7): training.collocation = uniform | rad,
    # training.adaptive_weights = true | false, model.fourier.n > 0
    col = Collocation(cp, t, device, score_fn=residual_scorer(model, cp, cp.k_hat))
    ev = GridEvaluator(cp, model, device)
    weights = {"pde": float(cfg.loss.w_pde), "bc": float(cfg.loss.w_bc)}
    bal = LossBalancer(model.parameters(), every=int(t.get("balance_every", 100)),
                       enabled=bool(t.get("adaptive_weights", False)))
    holder = {}

    def loss_fn():
        l_pde, l_bc = physics_losses(model, col, cp, cp.k_hat)
        bal.update = holder["trainer"].phase == "adam"
        total = bal({"pde": l_pde, "bc": l_bc}, weights)
        return total, {"pde": l_pde, "bc": l_bc, **bal.logged()}

    trainer = Trainer({"T": model}, loss_fn, t, out_dir, device, resample=col.resample,
                      evaluate=ev.evaluate, snapshot=ev.snapshot,
                      signature=config_signature(cfg))
    holder["trainer"] = trainer
    history = trainer.run()

    # ---------------- final metrics ----------------
    T_pred = ev.predict()
    m = ev.metrics()
    lb = [h for h in history if "pde" in h]
    metrics = {
        "layout": cp.layout, "cooling": cp.cooling, **m,
        "train_time_s": trainer.elapsed, "adam_iters_done": trainer.adam_done,
        "lbfgs_iters_done": trainer.lbfgs_done,
        "fv_solve_time_s": cp.fv_sol.info["t_total_s"],
        "fv_grid": f"{cp.T_ref.shape[1]}x{cp.T_ref.shape[0]}",
        "pinn_inference_time_s_full_grid": ev.inference_time(),
        "final_loss": lb[-1]["loss"] if lb else None,
        "final_loss_pde": lb[-1]["pde"] if lb else None, "final_loss_bc": lb[-1]["bc"] if lb else None,
        "n_params": count_params(model), "residual_scale": cp.res_scale, "env": env_report(),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    ev.save_snapshots(out_dir / "snapshots.npz")
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

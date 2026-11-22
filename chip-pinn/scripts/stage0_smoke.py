"""Stage 0 smoke test: config, seeds, device, scales, checkpoint round-trip.

    python scripts/stage0_smoke.py --config configs/stage0_smoke.yaml
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make project importable

import numpy as np

from common.config import parse_args, setup_run
from common.device import env_report
from common.scales import Scales
from common.seed import set_seed


def main():
    args = parse_args(__doc__, default_config="stage0_smoke.yaml")
    cfg, out_dir, device = setup_run(args.config, args.set)
    report = {"env": env_report(), "device": str(device), "out_dir": str(out_dir)}

    # 1) seeds: the same seed must give the same draws
    set_seed(cfg.seed); a = np.random.rand(3)
    set_seed(cfg.seed); b = np.random.rand(3)
    report["seed_reproducible_numpy"] = bool(np.array_equal(a, b))

    # 2) nondimensional scales for the default chip
    s = Scales.from_config(cfg)
    T = np.array([40.0, 70.0, 100.0])
    report["scales"] = s.to_dict()
    report["biot"] = s.biot(cfg.physics.h)
    report["k_hat"] = {m: s.k_nd(k) for m, k in cfg.physics.materials.items()}
    report["T_roundtrip_max_err"] = float(np.abs(s.T_dim(s.T_nd(T)) - T).max())

    # 3) torch: device, seeded draws, checkpoint round-trip
    try:
        import torch
        from common.checkpoint import load_checkpoint, save_checkpoint

        set_seed(cfg.seed); ta = torch.rand(3)
        set_seed(cfg.seed); tb = torch.rand(3)
        report["seed_reproducible_torch"] = bool(torch.equal(ta, tb))

        net = torch.nn.Linear(2, 1).to(device)
        opt = torch.optim.Adam(net.parameters())
        ck = out_dir / "ckpt_smoke.pt"
        save_checkpoint(ck, model=net, optimizer=opt, step=7)
        net2 = torch.nn.Linear(2, 1).to(device)
        payload = load_checkpoint(ck, map_location=device, model=net2)
        report["checkpoint_ok"] = bool(payload["step"] == 7 and all(
            torch.equal(p, q) for p, q in zip(net.parameters(), net2.parameters())))

        x = torch.rand(4096, 4096, device=device)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter(); (x @ x).sum().item()
        report["matmul_4096_s"] = round(time.perf_counter() - t0, 4)
    except ImportError:
        report["torch"] = "NOT INSTALLED - PINN stages need torch (preinstalled on Kaggle)"

    with open(out_dir / "env.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

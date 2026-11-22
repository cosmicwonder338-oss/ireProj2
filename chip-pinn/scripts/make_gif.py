"""Stage 9: animated GIF of the reconstruction improving during training (from snapshots.npz).

    python scripts/make_gif.py [--run stage5_inverse_q/quad_core] [--out figures/reconstruction.gif]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="stage5_inverse_q/quad_core", help="run folder inside results/")
    ap.add_argument("--out", default="figures/reconstruction.gif", help="path inside results/")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--thickness", type=float, default=5e-4, help="die thickness (m) for W/cm²")
    a = ap.parse_args()
    from matplotlib.animation import FuncAnimation, PillowWriter

    from common import plotting as P
    from common.paths import RESULTS_DIR

    d = np.load(RESULTS_DIR / a.run / "snapshots.npz")
    steps, T = d["step"], d["T"]
    has_q = "q" in d.files
    idx = np.unique(np.linspace(0, len(steps) - 1, min(a.frames, len(steps))).astype(int))
    ext = [0, d["x"][-1] * 1e3 + (d["x"][1] - d["x"][0]) * 5e2, 0,
           d["y"][-1] * 1e3 + (d["y"][1] - d["y"][0]) * 5e2]
    ncol = 4 if has_q else 2
    fig, ax = P.plt.subplots(1, ncol, figsize=(3.3 * ncol, 3.4), layout="constrained")
    vT = (float(d["T_ref"].min()), float(d["T_ref"].max()))
    panels = [(ax[0], d["T_ref"], "True temperature (°C)", "T", vT),
              (ax[1], T[idx[0]], "PINN temperature", "T", vT)]
    if has_q:
        qs = a.thickness / 1e4
        vq = (0.0, float(d["q_ref"].max()) * qs)
        panels += [(ax[2], d["q_ref"] * qs, "True power (W/cm²)", "q", vq),
                   (ax[3], d["q"][idx[0]] * qs, "PINN power", "q", vq)]
    ims = []
    for axis, F, title, kind, (lo, hi) in panels:
        ims.append(axis.imshow(F, origin="lower", extent=ext, cmap=P.CMAP[kind], vmin=lo, vmax=hi))
        axis.set_title(title, fontsize=9); axis.set_xticks([]); axis.set_yticks([])
    txt = fig.suptitle("", color=P.INK)

    def update(f):
        i = idx[f]
        ims[1].set_data(T[i])
        if has_q:
            ims[3].set_data(d["q"][i] * a.thickness / 1e4)
        err = np.abs(T[i] - d["T_ref"]).max()
        txt.set_text(f"training iteration {int(steps[i])} — max |T error| {err:.2f} K")
        return ims + [txt]

    out = RESULTS_DIR / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, update, frames=len(idx), blit=False).save(out, writer=PillowWriter(fps=6))
    print(f"[gif] wrote {out} ({len(idx)} frames)")


if __name__ == "__main__":
    main()

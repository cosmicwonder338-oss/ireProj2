"""PINN figures: true | prediction | error panels, and training curves."""
from __future__ import annotations

import numpy as np

from common import plotting as P

plt = P.plt
# validated categorical palette (fixed order, never cycled)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def compare(grid, T_true, T_pred, path, title="", sensors_xy=None, metrics=None):
    """Three panels in °C: reference | PINN | signed error (diverging, zero-centred)."""
    fig, ax = P.figure(1, 3)
    vmin, vmax = float(min(T_true.min(), T_pred.min())), float(max(T_true.max(), T_pred.max()))
    P.field(ax[0, 0], grid, T_true, "Reference (finite volume)", "T", "°C", sensors_xy, vmin=vmin, vmax=vmax)
    P.field(ax[0, 1], grid, T_pred, "PINN", "T", "°C", sensors_xy, vmin=vmin, vmax=vmax)
    P.field(ax[0, 2], grid, T_pred - T_true, "PINN − reference", "err", "K", contours=False)
    for a, F in ((ax[0, 0], T_true), (ax[0, 1], T_pred)):
        j, i = np.unravel_index(np.argmax(F), F.shape)
        P.mark(a, (grid.x[i] * 1e3, grid.y[j] * 1e3), f"{F.max():.1f} °C")
    if metrics:
        ax[0, 2].text(0.02, 0.02, f"rel L2 {metrics['rel_L2_rise']:.2e}\nmax |err| "
                      f"{metrics['max_abs_err_C']:.3f} K", transform=ax[0, 2].transAxes,
                      fontsize=8, color=P.INK, va="bottom",
                      bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    if title:
        fig.suptitle(title, color=P.INK)
    P.save(fig, path)


def training_curves(history: list[dict], path, terms=("loss", "pde", "bc"), metric="rel_L2",
                    title=""):
    """Left: loss terms (log). Right: evaluation metric (log). Separate axes, never a dual axis.
    A vertical rule marks the Adam → L-BFGS switch."""
    steps = np.array([h["step"] for h in history])
    switch = next((h["step"] for h in history if h["phase"] == "lbfgs"), None)
    fig, ax = P.figure(1, 2, panel=(5.2, 3.6))
    a = ax[0, 0]
    for c, term in zip(SERIES, terms):
        vals = np.array([h.get(term, np.nan) for h in history], float)
        if np.all(np.isnan(vals)):
            continue
        a.semilogy(steps, vals, color=c, lw=2, label=term)
    a.set_title("Training loss"); a.set_xlabel("iteration"); a.set_ylabel("loss (nondimensional)")
    a.legend(frameon=False)
    b = ax[0, 1]
    pts = [(h["step"], h[metric]) for h in history if metric in h]
    if pts:
        s, v = zip(*pts)
        b.semilogy(s, v, color=SERIES[0], lw=2, marker="o", ms=4)
        b.annotate(f"{v[-1]:.2e}", (s[-1], v[-1]), xytext=(-6, 8), textcoords="offset points",
                   ha="right", fontsize=8, color=P.INK)
    b.set_title(f"Error vs reference ({metric})"); b.set_xlabel("iteration"); b.set_ylabel(metric)
    for x in (a, b):
        x.grid(True, which="major", color="#e6e6e6", lw=0.6)
        if switch is not None:
            x.axvline(switch, color=P.MUTED, lw=1, ls="--")
            x.annotate("L-BFGS", (switch, 1), xycoords=("data", "axes fraction"),
                       xytext=(4, -12), textcoords="offset points", fontsize=8, color=P.MUTED)
    if title:
        fig.suptitle(title, color=P.INK)
    P.save(fig, path)

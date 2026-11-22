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


def reconstruction(grid, T_true, T_pred, q_true, q_pred, sensors_xy, T_sens, path, T_rbf=None,
                   title="", m=None, m_rbf=None, thickness=5e-4):
    """Main Stage 5 figure.
    Row 1 (temperature, °C): true map | what the sensors see | PINN reconstruction | PINN error
    Row 2: true power | PINN power | interpolation baseline | baseline error"""
    fig, ax = P.figure(2, 4, panel=(4.0, 3.5))
    vmin, vmax = float(T_true.min()), float(T_true.max())
    P.field(ax[0, 0], grid, T_true, "True temperature", "T", "°C", sensors_xy, vmin=vmin, vmax=vmax)
    a = ax[0, 1]
    sc = a.scatter(sensors_xy[:, 0] * 1e3, sensors_xy[:, 1] * 1e3, c=T_sens, cmap=P.CMAP["T"],
                   vmin=vmin, vmax=vmax, s=90, edgecolor=P.INK, linewidth=1.0)
    a.set_xlim(0, grid.Lx * 1e3); a.set_ylim(0, grid.Ly * 1e3); a.set_aspect("equal")
    a.set_title(f"All the PINN sees: {len(T_sens)} sensors")
    a.set_xlabel("x (mm)"); a.set_ylabel("y (mm)")
    cb = plt.colorbar(sc, ax=a, fraction=0.046, pad=0.04)
    cb.set_label("°C"); cb.outline.set_visible(False)
    P.field(ax[0, 2], grid, T_pred, "PINN reconstruction", "T", "°C", sensors_xy, vmin=vmin, vmax=vmax)
    P.field(ax[0, 3], grid, T_pred - T_true, "PINN error", "err", "K", contours=False)
    for a, F in ((ax[0, 0], T_true), (ax[0, 2], T_pred)):
        j, i = np.unravel_index(np.argmax(F), F.shape)
        P.mark(a, (grid.x[i] * 1e3, grid.y[j] * 1e3), f"{F.max():.1f} °C")
    qmax = float(max(q_true.max(), q_pred.max())) * thickness / 1e4
    P.field(ax[1, 0], grid, q_true * thickness / 1e4, "True power density", "q", "W/cm²",
            contours=False, vmin=0, vmax=qmax)
    P.field(ax[1, 1], grid, q_pred * thickness / 1e4, "PINN power density", "q", "W/cm²",
            contours=False, vmin=0, vmax=qmax)
    if T_rbf is not None:
        P.field(ax[1, 2], grid, T_rbf, "Baseline: spline interpolation", "T", "°C", sensors_xy,
                vmin=vmin, vmax=vmax)
        P.field(ax[1, 3], grid, T_rbf - T_true, "Baseline error", "err", "K", contours=False)
        j, i = np.unravel_index(np.argmax(T_rbf), T_rbf.shape)
        P.mark(ax[1, 2], (grid.x[i] * 1e3, grid.y[j] * 1e3), f"{T_rbf.max():.1f} °C")
    for a, mm, name in ((ax[0, 3], m, "PINN"), (ax[1, 3], m_rbf, "spline")):
        if mm:
            a.text(0.02, 0.02, f"{name}: max |err| {mm['max_abs_err_C']:.2f} K\n"
                   f"hotspot off by {mm['hotspot_loc_err_mm']:.2f} mm\n"
                   f"peak T error {mm['peak_T_err_C']:+.2f} K", transform=a.transAxes, fontsize=8,
                   color=P.INK, va="bottom",
                   bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    if title:
        fig.suptitle(title, color=P.INK)
    P.save(fig, path)


def line_compare(x, series: list, path, xlabel, ylabel, title="", logy=False, hline=None):
    """Simple multi-series line chart. series: [(label, y, style)], style ∈ {'solid','dashed'}."""
    fig, ax = P.figure(1, 1, panel=(6.0, 3.8))
    a = ax[0, 0]
    for c, (label, y, style) in zip(SERIES, series):
        (a.semilogy if logy else a.plot)(x, y, color=c, lw=2, ls="--" if style == "dashed" else "-",
                                         label=label)
    if hline is not None:
        a.axhline(hline[0], color=P.INK, lw=1, ls=":")
        a.annotate(hline[1], (x[0], hline[0]), xytext=(4, 4), textcoords="offset points",
                   fontsize=8, color=P.INK)
    a.set_xlabel(xlabel); a.set_ylabel(ylabel)
    if title:
        a.set_title(title)
    a.legend(frameon=False)
    a.grid(True, color="#e6e6e6", lw=0.6)
    P.save(fig, path)

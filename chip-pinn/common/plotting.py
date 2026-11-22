"""Shared map plotting (matplotlib, headless). All axes in mm, temperatures in °C.

Colour rules: magnitude fields use a single-hue sequential map (light = low, dark = high);
signed errors use a diverging map with a neutral midpoint pinned at zero. No rainbow maps.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

CMAP = {"T": "Reds", "q": "Oranges", "k": "Blues", "err": "RdBu_r"}
INK, MUTED = "#222222", "#6b6b6b"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9, "axes.titlesize": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})


def extent_mm(grid):
    return [0, grid.Lx * 1e3, 0, grid.Ly * 1e3]


def field(ax, grid, F, title="", kind="T", label="", sensors_xy=None, contours=True,
          vmin=None, vmax=None):
    """Draw one field map. kind ∈ {T, q, k, err}. sensors_xy in metres, shape (n, 2)."""
    norm = None
    if kind == "err":
        lim = float(np.nanmax(np.abs(F))) or 1.0
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)
        vmin = vmax = None
    im = ax.imshow(F, origin="lower", extent=extent_mm(grid), cmap=CMAP[kind], norm=norm,
                   vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
    if contours and kind == "T":
        X, Y = grid.mesh()
        ax.contour(X * 1e3, Y * 1e3, F, levels=8, colors=INK, linewidths=0.4, alpha=0.45)
    if sensors_xy is not None:
        ax.scatter(sensors_xy[:, 0] * 1e3, sensors_xy[:, 1] * 1e3, s=28, facecolor="white",
                   edgecolor=INK, linewidth=1.0, zorder=5, label="sensors")
    ax.set_title(title, color=INK)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, color=INK); cb.outline.set_visible(False)
    return im


def mark(ax, xy_mm, text, marker="X"):
    """Mark a point (e.g. the hotspot) with a symbol + direct label, not colour alone.
    White fill + dark outline and a white label box keep it readable on the darkest colours."""
    ax.plot(*xy_mm, marker=marker, ms=10, mfc="white", mec=INK, mew=1.2, zorder=7, ls="none",
            clip_on=False)
    # keep the label inside the axes: flip it left / down near the right / top edge
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    fx = (xy_mm[0] - x0) / (x1 - x0)
    fy = (xy_mm[1] - y0) / (y1 - y0)
    dx, ha = (-7, "right") if fx > 0.7 else (7, "left")
    dy, va = (-7, "top") if fy > 0.85 else (7, "bottom")
    ax.annotate(text, xy_mm, xytext=(dx, dy), textcoords="offset points", fontsize=8, color=INK,
                ha=ha, va=va, zorder=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))


def figure(nrows=1, ncols=1, panel=(4.6, 3.9)):
    """Figure with constrained layout (keeps colourbar labels from colliding with neighbours)."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows),
                             layout="constrained", squeeze=False)
    return fig, axes


def save(fig, path):
    if not fig.get_constrained_layout():
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

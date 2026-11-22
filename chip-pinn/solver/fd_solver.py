"""Cell-centred finite-volume solver for steady 2D conduction with variable conductivity.

    ∇·(k ∇T) + q − s (T − T_s) = 0      in Ω = [0, Lx] × [0, Ly]

  k(x,y)  conductivity per cell (W/mK)
  q(x,y)  volumetric heat source per cell (W/m³)
  s(x,y)  optional volumetric sink coefficient (W/m³K), e.g. back-face cooling h/t of a thin die
  T_s     sink temperature (°C)

Boundary conditions, per edge (bottom: y=0, top: y=Ly, left: x=0, right: x=Lx):
  {"type": "insulated"}                                   −k ∂T/∂n = 0
  {"type": "convective", "h": h, "T_inf": T∞}             −k ∂T/∂n = h (T − T∞)  (T∞ scalar/callable)
  {"type": "dirichlet", "value": g}                       T = g   (g scalar or callable g(s))
where n is the outward normal and s is the coordinate along the edge.

Discretisation (why finite volume): fluxes are balanced exactly across every cell face, so energy is
conserved to round-off, and material jumps (silicon / TIM / copper / void) are handled correctly by
using the *harmonic* mean of k on faces (it is the series resistance of the two half-cells). An
arithmetic mean would let heat leak straight through a near-zero-k void.

Arrays are indexed [j, i] with j along y (rows) and i along x (columns); plot with origin="lower".
Second-order accurate for smooth k (verified by the manufactured-solution test in Stage 2).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

EDGES = ("bottom", "top", "left", "right")


@dataclass
class Grid:
    Lx: float
    Ly: float
    nx: int
    ny: int

    @property
    def dx(self) -> float: return self.Lx / self.nx
    @property
    def dy(self) -> float: return self.Ly / self.ny
    @property
    def x(self) -> np.ndarray: return (np.arange(self.nx) + 0.5) * self.dx   # cell centres
    @property
    def y(self) -> np.ndarray: return (np.arange(self.ny) + 0.5) * self.dy

    def mesh(self):
        """(X, Y) cell-centre coordinates, each shape (ny, nx)."""
        return np.meshgrid(self.x, self.y)


@dataclass
class Problem:
    grid: Grid
    k: np.ndarray                         # (ny, nx) W/mK
    q: np.ndarray                         # (ny, nx) W/m³
    bc: dict                              # edge -> bc dict (see module docstring)
    sink: np.ndarray | float = 0.0        # (ny, nx) or scalar, W/m³K
    T_sink: float = 0.0                   # °C
    meta: dict = field(default_factory=dict)


@dataclass
class Solution:
    T: np.ndarray                         # (ny, nx) °C at cell centres
    grid: Grid
    info: dict


def _harmonic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 2.0 * a * b / (a + b)


def _bc_value(bc: dict, s: np.ndarray) -> np.ndarray:
    v = bc.get("value", 0.0)
    return np.asarray(v(s) if callable(v) else np.full_like(s, float(v)), dtype=float)


def _boundary_faces(p: Problem):
    """Yield (edge, slice, G_b, T_b) for every non-insulated edge.

    G_b is the face conductance per unit depth between the boundary cell centre (half a cell from
    the face) and the boundary value T_b; heat leaving the cell through that face is G_b (T_P − T_b).
    """
    g = p.grid
    k = np.asarray(p.k, float)
    edge_sl = {"bottom": (0, slice(None)), "top": (g.ny - 1, slice(None)),
               "left": (slice(None), 0), "right": (slice(None), g.nx - 1)}
    for edge in EDGES:
        bc = p.bc.get(edge, {"type": "insulated"})
        t = bc["type"]
        if t == "insulated":
            continue
        sl = edge_sl[edge]
        kb = k[sl]
        if edge in ("bottom", "top"):
            area, half, s = g.dx, g.dy / 2, g.x
        else:
            area, half, s = g.dy, g.dx / 2, g.y
        if t == "convective":                      # half-cell conduction in series with the film
            G = area / (half / kb + 1.0 / float(bc["h"]))
            Tb = _bc_value({"value": bc.get("T_inf", 0.0)}, s)  # T∞ may vary along the edge
        elif t == "dirichlet":
            G = area * kb / half
            Tb = _bc_value(bc, s)
        else:
            raise ValueError(f"unknown bc type '{t}' on {edge}")
        yield edge, sl, G, Tb


def assemble(p: Problem):
    """Build sparse A and rhs so that A T = rhs.

    Row P is the energy balance of cell P (W per unit depth):
        Σ_f G_f (T_nb − T_P) + Σ_b G_b (T_b − T_P) + sV (T_s − T_P) + qV = 0
    """
    g = p.grid
    nx, ny, dx, dy = g.nx, g.ny, g.dx, g.dy
    V = dx * dy
    k = np.asarray(p.k, float)
    q = np.asarray(p.q, float)
    assert k.shape == (ny, nx) and q.shape == (ny, nx), "k and q must be (ny, nx)"
    idx = np.arange(nx * ny).reshape(ny, nx)
    diag = np.zeros((ny, nx))
    rhs = -q * V
    rows, cols, vals = [], [], []

    Gx = _harmonic(k[:, :-1], k[:, 1:]) * dy / dx      # interior x-faces
    Gy = _harmonic(k[:-1, :], k[1:, :]) * dx / dy      # interior y-faces
    for G, a, c in ((Gx, idx[:, :-1], idx[:, 1:]), (Gy, idx[:-1, :], idx[1:, :])):
        rows += [a.ravel(), c.ravel()]
        cols += [c.ravel(), a.ravel()]
        vals += [G.ravel(), G.ravel()]
    diag[:, :-1] -= Gx; diag[:, 1:] -= Gx
    diag[:-1, :] -= Gy; diag[1:, :] -= Gy

    for _, sl, G, Tb in _boundary_faces(p):
        diag[sl] -= G
        rhs[sl] -= G * Tb

    sink = np.broadcast_to(np.asarray(p.sink, float), (ny, nx))
    diag -= sink * V
    rhs -= sink * V * p.T_sink

    rows.append(idx.ravel()); cols.append(idx.ravel()); vals.append(diag.ravel())
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(nx * ny, nx * ny))
    return A, rhs.ravel()


def energy_balance(p: Problem, T: np.ndarray) -> dict:
    """Global check: heat generated = heat leaving through boundaries + sink (per unit depth, W/m)."""
    V = p.grid.dx * p.grid.dy
    q_in = float(np.sum(p.q) * V)
    out_b = sum(float(np.sum(G * (T[sl] - Tb))) for _, sl, G, Tb in _boundary_faces(p))
    sink = np.broadcast_to(np.asarray(p.sink, float), T.shape)
    out_s = float(np.sum(sink * V * (T - p.T_sink)))
    denom = max(abs(q_in), 1e-30)
    return {"heat_in_W_per_m": q_in, "heat_out_edges_W_per_m": out_b,
            "heat_out_sink_W_per_m": out_s,
            "energy_balance_rel_err": abs(q_in - out_b - out_s) / denom}


def solve(p: Problem) -> Solution:
    t0 = time.perf_counter()
    A, rhs = assemble(p)
    t_asm = time.perf_counter() - t0
    T = spla.spsolve(A.tocsc(), rhs)
    t_total = time.perf_counter() - t0
    info = {"n_unknowns": int(T.size), "t_assemble_s": t_asm, "t_solve_s": t_total - t_asm,
            "t_total_s": t_total,
            "linear_residual": float(np.linalg.norm(A @ T - rhs) / max(np.linalg.norm(rhs), 1e-30))}
    T = T.reshape(p.grid.ny, p.grid.nx)
    info.update(energy_balance(p, T))
    return Solution(T=T, grid=p.grid, info=info)

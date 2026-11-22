"""Stage 2: finite-volume solver verification (numpy/scipy only)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common.config import load_config
from common.scales import Scales
from solver.fd_solver import Grid, Problem, solve
from solver.layouts import build_problem
from solver.verification import mms_convergence, two_layer


def test_mms_second_order_convergence():
    rows = mms_convergence(ns=(16, 32, 64, 128))
    for r in rows[1:]:
        assert r["order_L2"] > 1.8, rows
        assert r["order_max"] > 1.7, rows
    assert rows[-1]["err_max"] < 1e-3, rows
    assert all(r["energy_balance_rel_err"] < 1e-9 for r in rows)


def test_two_layer_void_interface_exact():
    # silicon (150) under an air void (0.03): 5000x contrast; harmonic mean must be exact
    r = two_layer(n=40, k_bottom=150.0, k_top=0.03)
    assert r["err_max"] < 1e-9, r


def test_nondimensional_problem_matches_SI():
    """Solving in hat-variables and converting back must equal the SI solve (checks q̂, ŝ, k̂)."""
    cfg = load_config("stage1_reference.yaml")
    sc = Scales.from_config(cfg)
    prob, sol = build_problem(cfg, "quad_core", "backside", 64, 64)
    g = prob.grid
    g_nd = Grid(sc.x_nd(g.Lx), sc.x_nd(g.Ly), g.nx, g.ny)
    nd = Problem(g_nd, sc.k_nd(prob.k), sc.q_nd(prob.q), prob.bc,
                 sink=sc.s_nd(np.asarray(prob.sink)), T_sink=0.0)
    T_back = sc.T_dim(solve(nd).T)
    assert np.abs(T_back - sol.T).max() < 1e-8 * sol.T.max()


def test_uniform_backside_matches_q_over_s():
    """Uniform q, insulated edges, back-face sink: T − T_amb = q/s everywhere."""
    g = Grid(0.01, 0.01, 20, 20)
    q, s, Ta = 5e8, 2e7, 40.0
    p = Problem(g, np.full((20, 20), 150.0), np.full((20, 20), q), {}, sink=s, T_sink=Ta)
    assert np.allclose(solve(p).T, Ta + q / s, rtol=1e-12)

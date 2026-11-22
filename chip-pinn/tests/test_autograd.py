"""Stage 2: autograd PDE / BC residuals against analytic fields and finite differences (torch).

Skipped automatically when torch is not installed (it is preinstalled on Kaggle).
"""
import sys
from pathlib import Path
from unittest import SkipTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _need_torch():
    if torch is None:
        raise SkipTest("torch not installed")
    torch.manual_seed(0)
    return torch.float64


def _points(n=512, y=None):
    dt = torch.float64
    xy = torch.rand(n, 2, dtype=dt)
    if y is not None:
        xy[:, 1] = y
    return xy.requires_grad_(True)


def test_pde_residual_vanishes_on_manufactured_solution():
    _need_torch()
    from pinn.losses import pde_residual
    from solver.verification import MMS_S, mms_fields

    xy = _points()
    f = mms_fields(xy[:, 0:1], xy[:, 1:2], lib=torch)
    r = pde_residual(f["T"], xy, f["k"], f["q"].detach(), s=MMS_S)
    assert r.abs().max().item() < 1e-10, r.abs().max().item()
    # sensitivity check: ignoring ∇k (treating k as a constant 1) must break the residual
    r_bad = pde_residual(f["T"], xy, 1.0, f["q"].detach(), s=MMS_S)
    assert r_bad.abs().max().item() > 1e-2


def test_gradient_matches_analytic():
    _need_torch()
    from pinn.losses import grad
    from solver.verification import mms_fields

    xy = _points()
    f = mms_fields(xy[:, 0:1], xy[:, 1:2], lib=torch)
    g = grad(f["T"], xy)
    assert torch.allclose(g[:, 0:1], f["Tx"], atol=1e-12)
    assert torch.allclose(g[:, 1:2], f["Ty"], atol=1e-12)


def test_boundary_residuals():
    _need_torch()
    from pinn.losses import bc_residual, normal_flux
    from solver.verification import mms_fields

    # top edge y=1 is insulated in the MMS problem
    xy = _points(256, y=1.0)
    f = mms_fields(xy[:, 0:1], xy[:, 1:2], lib=torch)
    n_top = torch.tensor([[0.0, 1.0]], dtype=torch.float64).expand(256, 2)
    assert bc_residual(f["T"], xy, f["k"], n_top, "insulated").abs().max().item() < 1e-12
    # bottom edge y=0, outward normal −y: k ∂T/∂n = −k T_y
    xy = _points(256, y=0.0)
    f = mms_fields(xy[:, 0:1], xy[:, 1:2], lib=torch)
    n_bot = torch.tensor([[0.0, -1.0]], dtype=torch.float64).expand(256, 2)
    assert torch.allclose(normal_flux(f["T"], xy, f["k"], n_bot), -f["k"] * f["Ty"], atol=1e-12)
    # convective residual k∂T/∂n + Bi·T: for T = exp(Bi·y), k = 1, normal −y,
    # k∂T/∂n = −Bi·T, so the residual is exactly zero
    Bi = 0.7
    T = torch.exp(Bi * xy[:, 1:2])
    assert bc_residual(T, xy, 1.0, n_bot, "convective", Bi=Bi).abs().max().item() < 1e-12


def _mlp(width=32, out_softplus=False):
    layers = [torch.nn.Linear(2, width), torch.nn.Tanh(), torch.nn.Linear(width, width),
              torch.nn.Tanh(), torch.nn.Linear(width, 1)]
    if out_softplus:
        layers.append(torch.nn.Softplus())
    return torch.nn.Sequential(*layers).double()


def test_network_divergence_matches_finite_differences():
    """div(k∇T) through two real networks (T-net and a positive k-net) vs a central-difference
    stencil applied to the autograd flux — exercises the exact code path of Stage 6."""
    _need_torch()
    from pinn.losses import grad, pde_residual

    Tn, kn = _mlp(), _mlp(out_softplus=True)
    xy = (0.1 + 0.8 * torch.rand(64, 2, dtype=torch.float64)).requires_grad_(True)
    div_auto = pde_residual(Tn(xy), xy, kn(xy), 0.0, 0.0)

    def flux(p):
        p = p.detach().requires_grad_(True)
        return (kn(p) * grad(Tn(p), p)).detach()

    h = 1e-4
    ex = torch.tensor([[h, 0.0]], dtype=torch.float64)
    ey = torch.tensor([[0.0, h]], dtype=torch.float64)
    div_fd = ((flux(xy + ex)[:, 0:1] - flux(xy - ex)[:, 0:1]) +
              (flux(xy + ey)[:, 1:2] - flux(xy - ey)[:, 1:2])) / (2 * h)
    err = (div_auto - div_fd).abs().max().item()
    scale = div_fd.abs().max().item()
    assert err < 1e-6 * max(scale, 1.0), (err, scale)


def test_float32_residual_accuracy_is_adequate():
    """Training runs in float32: the MMS residual should still be ~1e-5, far below PINN errors."""
    _need_torch()
    from pinn.losses import pde_residual
    from solver.verification import MMS_S, mms_fields

    xy = torch.rand(512, 2).requires_grad_(True)
    f = mms_fields(xy[:, 0:1], xy[:, 1:2], lib=torch)
    r = pde_residual(f["T"], xy, f["k"], f["q"].detach(), s=MMS_S)
    rel = r.abs().max().item() / f["q"].abs().max().item()
    assert rel < 1e-4, rel

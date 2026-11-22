"""Physics residuals via autograd (all quantities nondimensional, see common/scales.py).

PDE      r = ∇·(k ∇T) + q − s T                       (= 0 in the domain)
BC       insulated:   r_b = k ∂T/∂n                    (= 0)
         convective:  r_b = k ∂T/∂n + Bi T             (= 0;  from −k ∂T/∂n = Bi (T − 0))
         dirichlet:   r_b = T − g                      (= 0)

Note the divergence form: we differentiate the *flux* k∇T, not k·ΔT. That way a k produced by a
network (Stage 6) contributes its own gradient ∇k·∇T automatically and correctly.
"""
from __future__ import annotations

import torch


def grad(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """du/dx for u: (N, 1) computed from x: (N, d). Keeps the graph for higher derivatives.
    Returns zeros (not None) when u does not depend on x (e.g. a constant k)."""
    g, = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True,
                             allow_unused=True)
    return torch.zeros_like(x) if g is None else g


def _as_col(v, like: torch.Tensor) -> torch.Tensor:
    """Scalars / (N,) / (N,1) -> (N,1) tensor on like's device & dtype."""
    v = torch.as_tensor(v, dtype=like.dtype, device=like.device)
    return v.reshape(-1, 1) if v.dim() > 0 else v.expand(like.shape[0], 1)


def pde_residual(T: torch.Tensor, xy: torch.Tensor, k, q, s=0.0) -> torch.Tensor:
    """r = ∇·(k ∇T) + q − s T at the points xy (requires_grad=True), all shapes (N, 1)."""
    k, q, s = _as_col(k, T), _as_col(q, T), _as_col(s, T)
    dT = grad(T, xy)                         # (N, 2)
    fx, fy = k * dT[:, 0:1], k * dT[:, 1:2]  # heat flux (sign: q_vec = −flux)
    div = grad(fx, xy)[:, 0:1] + grad(fy, xy)[:, 1:2]
    return div + q - s * T


def normal_flux(T: torch.Tensor, xy: torch.Tensor, k, normal: torch.Tensor) -> torch.Tensor:
    """k ∂T/∂n with outward unit normal (N, 2)."""
    return _as_col(k, T) * (grad(T, xy) * normal).sum(dim=1, keepdim=True)


def bc_residual(T, xy, k, normal, kind: str, Bi: float = 0.0, g=None) -> torch.Tensor:
    if kind == "insulated":
        return normal_flux(T, xy, k, normal)
    if kind == "convective":
        return normal_flux(T, xy, k, normal) + Bi * T
    if kind == "dirichlet":
        return T - _as_col(g, T)
    raise ValueError(f"unknown bc kind '{kind}'")


def mse(r: torch.Tensor) -> torch.Tensor:
    return (r ** 2).mean()


class LossBalancer:
    """Adaptive loss weights by gradient statistics ("learning-rate annealing", Wang, Teng &
    Perdikaris 2021). Every `every` Adam steps:
        λ̂_i = max_θ |∇θ L_ref| / mean_θ |∇θ (w_i L_i)|,     λ_i ← (1 − α) λ_i + α λ̂_i
    so that no term's gradient is drowned by the PDE term's. Weights are frozen when `update` is
    False (L-BFGS needs a fixed objective). Disabled → plain fixed weights.
    """

    def __init__(self, params, ref="pde", every=100, alpha=0.1, enabled=True, lam_max=1e4):
        self.params = [p for p in params if p.requires_grad]
        self.ref, self.every, self.alpha = ref, int(every), float(alpha)
        self.enabled, self.lam_max = bool(enabled), float(lam_max)
        self.lam: dict = {}
        self.calls = 0
        self.update = True

    def _grad_stat(self, term, reduce):
        gs = torch.autograd.grad(term, self.params, retain_graph=True, allow_unused=True)
        flat = torch.cat([g.abs().reshape(-1) for g in gs if g is not None])
        return reduce(flat) if flat.numel() else torch.zeros((), device=term.device)

    def __call__(self, terms: dict, weights: dict) -> torch.Tensor:
        """terms: name -> loss tensor; weights: name -> base weight. Returns the weighted total."""
        if self.enabled and self.update:
            if self.calls % self.every == 0:
                g_ref = self._grad_stat(weights.get(self.ref, 1.0) * terms[self.ref], torch.max)
                for k, v in terms.items():
                    if k == self.ref or not torch.is_tensor(v) or not v.requires_grad:
                        continue
                    g = self._grad_stat(weights.get(k, 1.0) * v, torch.mean)
                    hat = float(g_ref / (g + 1e-12))
                    hat = min(max(hat, 1.0 / self.lam_max), self.lam_max)
                    self.lam[k] = hat if k not in self.lam else (1 - self.alpha) * self.lam[k] + self.alpha * hat
            self.calls += 1
        total = 0.0
        for k, v in terms.items():
            lam = self.lam.get(k, 1.0) if self.enabled else 1.0
            total = total + lam * weights.get(k, 1.0) * v
        return total

    def logged(self) -> dict:
        return {f"lam_{k}": v for k, v in self.lam.items()} if self.enabled else {}

"""Nondimensionalization.

Dimensional problem:   ∇·(k ∇T) + q − s (T − T_amb) = 0,    -k ∂T/∂n = h (T - T_amb)  on convective edges
  (s = h/t is the back-face sink of a thin die; s = 0 for edge-only cooling)
With  x̂ = x/L,  T̂ = (T - T_amb)/ΔT,  k̂ = k/k_ref,  q̂ = q L² / (k_ref ΔT),  ŝ = s L² / k_ref:
                       ∇̂·(k̂ ∇̂T̂) + q̂ − ŝ T̂ = 0,       -k̂ ∂T̂/∂n̂ = Bi T̂,   Bi = h L / k_ref

All methods are plain arithmetic, so they work on floats, numpy arrays and torch tensors alike.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Scales:
    L: float        # length scale, m
    k_ref: float    # conductivity scale, W/mK
    dT: float       # temperature-rise scale, K
    T_amb: float    # ambient temperature, °C (the zero of T̂)

    @classmethod
    def from_config(cls, cfg) -> "Scales":
        return cls(L=float(cfg.scales.L), k_ref=float(cfg.scales.k_ref),
                   dT=float(cfg.scales.dT), T_amb=float(cfg.physics.T_amb))

    # --- derived scales -----------------------------------------------------------------------
    @property
    def q_ref(self) -> float:
        """Power-density scale, W/m³: q = q_ref * q̂."""
        return self.k_ref * self.dT / self.L ** 2

    def biot(self, h: float) -> float:
        return h * self.L / self.k_ref

    def s_nd(self, s):
        """Volumetric sink coefficient W/m³K -> ŝ = s L² / k_ref (backside: s = h/t)."""
        return s * self.L ** 2 / self.k_ref

    # --- conversions --------------------------------------------------------------------------
    def x_nd(self, x):   return x / self.L                    # m   -> x̂
    def x_dim(self, xh): return xh * self.L                   # x̂   -> m
    def T_nd(self, T):   return (T - self.T_amb) / self.dT    # °C  -> T̂
    def T_dim(self, Th): return Th * self.dT + self.T_amb     # T̂   -> °C
    def dT_dim(self, e): return e * self.dT                   # T̂ difference -> K (= °C difference)
    def k_nd(self, k):   return k / self.k_ref
    def k_dim(self, kh): return kh * self.k_ref
    def q_nd(self, q):   return q / self.q_ref
    def q_dim(self, qh): return qh * self.q_ref

    def to_dict(self) -> dict:
        d = asdict(self)
        d["q_ref"] = self.q_ref
        return d

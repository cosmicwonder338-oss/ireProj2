# Lab notes — diagnoses and fixes

Running log of what failed, why, and what fixed it. Every number here comes from a real run in
`results/<name>/`. These notes feed the README "Lessons learned" section.

## Stage 0 — setup
- **YAML float gotcha.** PyYAML implements YAML 1.1, where `1e4` (no dot, no sign on the exponent)
  is a *string*, not a float. A silently-string `h` would crash or, worse, be concatenated somewhere.
  Fix: configs use `1.0e+4`, and `common/config.py` coerces numeric-looking strings to float
  (covered by `tests/test_infra.py::test_bare_1e4_string_is_coerced`).
- **Scales chosen.** L = 10 mm, k_ref = 150 W/mK (silicon), ΔT = 50 K, T_amb = 40 °C, giving
  q_ref = k_ref·ΔT/L² = 7.5e7 W/m³ and Bi = hL/k_ref ≈ 0.67. With these, T̂ ∈ [0, ~1.2] for
  30–60 °C hotspots and k̂ spans 0.0002 (void) … 2.67 (copper) — the void contrast is the hard part
  for Stage 6.

## Stage 1 — reference solver
- **Finite volume, harmonic-mean face conductivity.** Exact flux balance per cell, so energy closes
  to round-off (measured energy-balance error 1e-12 on all six runs). The harmonic mean is the
  series resistance of two half-cells; an arithmetic mean would let heat leak through a void.
  Quick 1D check (uniform q, convective bottom): max error 8.3e-3 → 2.1e-3 → 5.2e-4 K for
  20→40→80 cells, i.e. 2nd order. Formal manufactured-solution test comes in Stage 2.
- **Grid independence (quad_core).** Power needed for a +45 K peak changes < 0.01 % from 100² to
  400² cells, so 200² (40 000 unknowns, ~0.35 s per solve with spsolve) is the reference grid.
- **Power is set by a target peak rise (45 K).** The problem is linear in q, so a unit solve fixes
  the scale exactly; the scaled problem is then re-solved and the resulting W, W/cm² are reported.
- **Cooling model matters a lot — and the literal brief has a problem.** With only the bottom
  *edge* convective (top-view 2D domain), the whole die sits ≥ 31 K above ambient, the in-die
  range is only ~13–14 K, the chip dissipates just ~1.6 W, and the hottest point is pulled
  toward the far insulated edge: for `random_hotspots` it is **5.0 mm** from the strongest power
  source. Temperature then mostly encodes "distance from the cooled edge", not "where the power
  is", which undermines hotspot localisation (Stage 5).
  The **backside** model (thin die cooled through its back face → distributed sink h/t, spreading
  length ℓ = √(k t/h) = 2.7 mm) gives realistic numbers: 25–29 W total, 106–168 W/cm² peak flux,
  29–31 K in-die range, and hotspots within 0.27–0.42 mm of the power peak.
- **Sparse-sensor difficulty is real.** With 15 random sensors on backside quad_core, the hottest
  sensor reads 8.5 K below the true peak: the hotspot falls between sensors, so it must be
  *inferred* — which is exactly what the PINN has to do.
- **Decision: backside cooling is the default** for all top-view stages (`physics.cooling: backside`).
  Edge cooling stays available for comparison.

## Stage 2 — verification
- **Manufactured solution** (variable k = 1 + ½ sin πx sin πy, sink s = 2, Dirichlet + insulated +
  convective-with-varying-T∞ edges, q derived analytically). FV errors, n = 16 → 256:
  L2 2.00e-3 → 7.80e-6, max 4.21e-3 → 1.69e-5; observed order 2.000 (L2) and 1.98–2.00 (max) at
  every refinement. Energy balance ≤ 2e-13. So all three BC implementations, the variable-k
  harmonic faces and the sink term are second-order correct.
- **Layered void check.** Silicon (150) under TIM (3) or an air void (0.03), interface on a cell
  face: max error vs. the exact piecewise-linear solution 6.7e-16 … 2.1e-15 — exact to round-off,
  even at 5000× contrast. This is the property Stage 6 relies on.
- **Nondimensional consistency.** Solving quad_core/backside in hat-variables (k̂, q̂, ŝ = sL²/k_ref)
  and converting back reproduces the SI solve to < 1e-8 relative — the same scaling the PINN uses.
- **Autograd residual** (`pinn/losses.py`, divergence form ∂x(k ∂xT) + ∂y(k ∂yT), so ∇k·∇T is
  included automatically): tests written for the MMS field, analytic gradients, all BC residuals,
  and a two-network (T-net, softplus k-net) divergence vs. central differences. They need torch,
  so they skip locally. **Kaggle run (GPU): all 15 tests pass**, including the 5 autograd tests
  (float64 MMS residual < 1e-10; two-network divergence matches central differences to < 1e-6
  relative; float32 residual < 1e-4 relative to max |q̂|).
- Plot fix after the first Kaggle render: the hotspot marker/label were dark-on-dark-red and the
  colourbar label collided with the neighbouring axis label → white-filled marker with a label box,
  constrained layout.

## Stage 3 — forward PINN
- **Loss scaling.** On backside quad_core, q̂_max ≈ 28 and ŝ ≈ 13.3 while T̂ ∈ [0.3, 0.9]. The raw
  residual is O(10), so its square would dominate the O(1) Neumann BC term by ~100×. The PDE
  residual is divided by `res_scale = max(q̂_max, ŝ·T̂_max)` so both terms start at comparable size.
- **Exact source at every collocation point.** q is evaluated analytically (`solver.layouts.q_function`)
  instead of interpolating the gridded map; matches the FV grid map to 1e-16.
- **Error metric.** Relative L2 is computed on the temperature *rise* T − T_amb. On absolute °C it
  would look 2–3× better than it really is.
- **Result (Kaggle GPU, quad_core / backside, 4×64 tanh, 12 737 params, 10k Adam + L-BFGS):**
  rel L2 (rise) 6.18e-4, max |err| 0.047 K, mean |err| 0.014 K, peak-T error −0.015 °C,
  hotspot location error 0.0 mm (same cell). Training 244 s vs. FV solve 0.29 s; PINN inference
  on the full 200² grid 2.0 ms. The 1-minute smoke run (300 Adam + 100 L-BFGS) already reached
  rel L2 4.7e-2, max 2.5 K — L-BFGS alone cut the error ~3× in 100 iterations.
- **Where the error lives: spectral bias at power-block edges.** The error map shows straight
  streaks at x, y ≈ 1, 4.5, 5.5, 9 mm — exactly the edges of the power blocks, where q̂ jumps from
  ~1.4 to ~28 over 0.15 mm. T itself is smooth there, but its second derivative (= −q̂ + ŝT̂) is
  not, and a tanh MLP fits high-frequency content last. Fourier features / residual-adaptive
  collocation (Stage 7) target exactly this.
- **Honest speed statement.** For a single *forward* problem with known k and q the PINN is
  ~800× slower than the sparse FV solve. Its value is in inverse problems (Stages 4–6) and in
  parametric surrogates queried many times (Stage 8).
- Refactor: collocation sets, physics losses, grid evaluation and snapshots moved to
  `pinn/experiment.py` so Stages 4–8 reuse them; Stage 3 behaviour (and checkpoint signature) unchanged.

## Stage 4 — inverse 1: constant k from 15 noisy sensors
- **Setup.** quad_core / backside, true k = 150 W/mK, initial guess 50 W/mK. k̂ = exp(θ) (always
  positive, Adam steps are relative). Loss = PDE + BC + 10 · mean (T̂(sensor) − T̂_obs)².
  Insulated BCs are imposed as ∂T/∂n = 0 rather than k ∂T/∂n = 0, otherwise the optimiser could
  satisfy them by driving k → 0.
- **Classical baseline (FV least squares).** Bounded 1-D minimisation over log k, ~10 FV solves,
  ~0.9 s per fit (local CPU). It inverts on a 100² grid while the data come from 200², to avoid
  the *inverse crime* (same model for data and inversion → spuriously perfect recovery).
  Noise-free: k error +0.017 % — the discretisation floor.
- **k is only weakly identifiable here — a physics finding, not a solver bug.** With back-face
  cooling the temperature is mostly set by the *local* balance q ≈ s(T − T_amb); k only controls
  lateral spreading (ℓ = √(kt/h) ≈ 2.7 mm). So small sensor noise maps to large k error.
  FV baseline over 20 noise draws per level (RMS / mean k error): 1 % noise → 4.4 % / +1.1 %,
  2 % → 9.2 % / +2.5 %, 5 % → 26.3 % / +7.9 %. Error grows ~linearly with noise, with a
  *positive bias* because T depends nonlinearly on k (overestimates are cheaper in misfit).
  On the 3 paired draws the RMS errors are 3.3 / 6.7 / 18.1 %; one draw (seed 1) happens to be
  nearly orthogonal to the k-sensitivity (error ≤ 0.08 % at every level), which is why 3 draws
  alone would understate the noise sensitivity.
- PINN results: pending the Kaggle run.

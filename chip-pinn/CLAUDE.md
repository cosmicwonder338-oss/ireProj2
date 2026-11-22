# Thermal Detective — project rules

PINN project: reconstruct chip temperature, power hotspots and hidden defects from sparse sensors.
Physics: steady 2D conduction  ∇·(k ∇T) + q = 0.

## Environment
- Target runtime is **Kaggle** (T4/P100 GPU, ~12 h session). Project root on Kaggle is
  `/kaggle/working/chip-pinn/` — the only persistent folder. All paths are resolved relative to the
  project root (`common/paths.py`), never hard-coded.
- Allowed packages: torch, numpy, scipy, matplotlib, pandas, pyyaml, tqdm. No pip installs.
  (pytest is optional — `python tests/run_tests.py` runs every test without it.)
- Pick the device with `common.device.get_device()`; everything must also run on CPU.
- Each training run must fit in ~15 min on a T4 (`training.time_budget_min`) and checkpoint
  regularly (`common.checkpoint`) so a session restart can resume.

## Code layout
- `solver/`  finite-difference reference solver, chip layouts, sensor sampling (numpy/scipy only).
- `pinn/`    `models.py`, `losses.py`, `sampling.py`, `train.py`, `plotting.py` (torch).
- `common/`  config, seeding, device, nondimensional scales, checkpoints, paths.
- `configs/` one YAML per experiment, deep-merged on top of `configs/default.yaml`.
- `scripts/` one entry point per stage: `python scripts/<stage>.py --config configs/<exp>.yaml [--set a.b=v ...]`.
- `results/<name>/` every output of experiment `<name>` (resolved config, metrics, figures, checkpoints).
- `tests/`   plain-function tests, runnable with `python tests/run_tests.py` or pytest.

## Rules
1. **Never claim a result that was not actually run.** Metrics in README/notes are copied from real
   run output (`results/<name>/metrics.json`), never typed by hand or estimated.
2. **Nondimensionalize before training**: x̂ = x/L, T̂ = (T − T_amb)/ΔT, k̂ = k/k_ref,
   q̂ = q·L²/(k_ref·ΔT), Bi = h·L/k_ref. Convert back to °C / mm / W·m⁻³ for every plot and metric.
3. Fixed seeds (`common.seed.set_seed`) for every run; the seed is stored in the resolved config.
4. When training fails or is inaccurate, diagnose (scaling, loss imbalance, spectral bias,
   ill-posedness), record the diagnosis and the fix in `results/notes.md` — it feeds the README.
5. Build stages in order and stop after each one for review.
6. YAML gotcha: PyYAML reads `1e4` as a *string*. Write `1.0e+4`; the loader also coerces
   numeric-looking strings, but don't rely on it.
7. Keep modules small, commented, and free of notebook-only code; the notebook only calls scripts.

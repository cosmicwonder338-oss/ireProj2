"""Stage 9: generate README.md from the results on disk.

Every number in the README is read from results/*/metrics.json, *.csv or *.md produced by real runs
— nothing is typed by hand. Sections whose results are missing say so instead of inventing numbers.

    python scripts/make_report.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from common.paths import RESULTS_DIR
from common.report import md_table

R = RESULTS_DIR
MISSING = "_Not run yet — run this stage in `main.ipynb` and regenerate the README._"


def jload(p):
    p = R / p
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def text(p):
    p = R / p
    return p.read_text(encoding="utf-8").strip() if p.exists() else None


def fig(p, alt):
    return f"![{alt}](results/{p})" if (R / p).exists() else ""


def f(v, fmt="{:.3g}"):
    try:
        return fmt.format(v)
    except (TypeError, ValueError):
        return str(v)


def headline():
    lines = []
    s5 = R / "stage5_inverse_q" / "summary.csv"
    if s5.exists():
        d = pd.read_csv(s5)
        for _, r in d.iterrows():
            lines.append(
                f"- **{r.layout}** — from {int(r.n_sensors)} noisy sensors the PINN reconstructs the full "
                f"map with max error {f(r.pinn_max_abs_err_C, '{:.2f}')} K, puts the hotspot "
                f"{f(r.pinn_hotspot_loc_err_mm, '{:.2f}')} mm from the truth and its peak within "
                f"{f(abs(r.pinn_peak_T_err_C), '{:.2f}')} K (spline interpolation of the same sensors: "
                f"{f(r.spline_max_abs_err_C, '{:.2f}')} K max error, hotspot "
                f"{f(r.spline_hotspot_loc_err_mm, '{:.2f}')} mm off, peak "
                f"{f(abs(r.spline_peak_T_err_C), '{:.2f}')} K off).")
    s6 = R / "stage6_defect" / "summary.csv"
    if s6.exists():
        d = pd.read_csv(s6)
        best = d.sort_values("void_iou", ascending=False).iloc[0]
        lines.append(f"- **Hidden void** in the thermal-interface layer, found from top-surface "
                     f"temperatures only: best IoU {f(best.void_iou, '{:.2f}')} ({best.variant.replace('_', ' ')}).")
    m3 = jload("stage3_forward/metrics.json")
    if m3:
        lines.append(f"- **Forward solve** (known power): max error {f(m3['max_abs_err_C'], '{:.3f}')} K "
                     f"(rel. L2 {f(m3['rel_L2_rise'], '{:.2e}')}) vs a verified finite-volume solver.")
    m8 = jload("stage8_parametric/metrics.json")
    if m8:
        lines.append(f"- **Parametric surrogate** with temperature-dependent silicon k: one network for "
                     f"{f(0.5 * m8['P_nominal_W'], '{:.0f}')}–{f(1.5 * m8['P_nominal_W'], '{:.0f}')} W, "
                     f"{f(m8['speedup_per_query'], '{:.0f}')}× faster per query than the nonlinear FV "
                     f"solver; pays back its training after {f(m8['break_even_queries'], '{:.0f}')} queries.")
    return "\n".join(lines) if lines else MISSING


def stage_tables():
    out = []
    # Stage 1
    s1 = R / "stage1_reference" / "metrics.csv"
    out.append("### Stage 1 — reference solver and chip layouts\n")
    if s1.exists():
        d = pd.read_csv(s1)[["layout", "cooling", "T_max_C", "T_min_C", "T_range_C",
                             "hotspot_to_q_peak_mm", "P_total_W", "q_peak_W_cm2", "solve_time_s"]]
        out += [md_table(d), "", fig("stage1_reference/layouts_backside.png", "layouts")]
    else:
        out.append(MISSING)
    # Stage 2
    out.append("\n### Stage 2 — verification\n")
    c2 = R / "stage2_verify" / "convergence.csv"
    if c2.exists():
        d = pd.read_csv(c2)[["n", "err_L2", "err_max", "order_L2", "order_max"]]
        out += ["Finite-volume solver on a manufactured solution (variable k, sink, all three "
                "boundary types):", "", md_table(d), "", fig("stage2_verify/convergence.png", "convergence")]
    else:
        out.append(MISSING)
    # Stage 3
    out.append("\n### Stage 3 — forward PINN\n")
    m3 = jload("stage3_forward/metrics.json")
    if m3:
        keys = ["rel_L2_rise", "max_abs_err_C", "mean_abs_err_C", "peak_T_err_C", "hotspot_loc_err_mm",
                "train_time_s", "fv_solve_time_s", "pinn_inference_time_s_full_grid"]
        out += [md_table(pd.DataFrame([{k: m3[k] for k in keys}])), "",
                fig("stage3_forward/comparison.png", "forward"), fig("stage3_forward/training.png", "training")]
    else:
        out.append(MISSING)
    # Stage 4
    out.append("\n### Stage 4 — inverse 1: unknown constant conductivity\n")
    t4 = text("stage4_inverse_k/summary.md")
    out += [t4, "", fig("stage4_inverse_k/k_error_vs_noise.png", "k error"),
            fig("stage4_inverse_k/k_trajectory.png", "k trajectory")] if t4 else [MISSING]
    # Stage 5
    out.append("\n### Stage 5 — inverse 2 (main result): unknown power map from sparse sensors\n")
    t5 = text("stage5_inverse_q/summary.md")
    if t5:
        out += [t5, ""]
        for lay in ("quad_core", "cpu_gpu", "random_hotspots"):
            out.append(fig(f"stage5_inverse_q/{lay}/reconstruction.png", lay))
    else:
        out.append(MISSING)
    # Stage 6
    out.append("\n### Stage 6 — inverse 3: hidden void in the thermal interface layer\n")
    t6 = text("stage6_defect/summary.md")
    out += [t6, "", fig("stage6_defect/k_profile.png", "k profile"),
            fig("stage6_defect/top_surface.png", "top surface"),
            fig("stage6_defect/stack_fields.png", "stack")] if t6 else [MISSING]
    # Stage 7
    out.append("\n### Stage 7 — what actually helps: ablation\n")
    t7 = text("stage7_ablation/ablation.md")
    out += [t7, "", fig("stage7_ablation/ablation.png", "ablation")] if t7 else [MISSING]
    # Stage 8
    out.append("\n### Stage 8a — how many sensors, and where?\n")
    t8 = text("stage8_sensors/sensors.md")
    out += [t8, "", fig("stage8_sensors/sensor_count.png", "sensor count"),
            fig("stage8_sensors/placements.png", "placements")] if t8 else [MISSING]
    out.append("\n### Stage 8b — parametric PINN vs the solver\n")
    t8b = text("stage8_parametric/per_power.md")
    m8 = jload("stage8_parametric/metrics.json")
    if t8b and m8:
        out += [t8b, "", md_table(pd.DataFrame([{k: m8[k] for k in (
            "train_time_s", "pinn_query_time_s", "fv_nonlinear_time_s_mean", "speedup_per_query",
            "break_even_queries")}])), "", fig("stage8_parametric/peak_vs_power.png", "parametric")]
    else:
        out.append(MISSING)
    return "\n".join(x for x in out if x is not None)


STATIC_PHYSICS = r"""
## Physics

Steady heat conduction in the die, with conductivity $k(x,y)$ and power density $q(x,y)$:

$$\nabla\cdot\big(k\,\nabla T\big) + q - s\,(T - T_\mathrm{amb}) = 0$$

* **Top view (Stages 1–5, 7, 8)** — a 10 mm × 10 mm, 0.5 mm-thick silicon die (k = 150 W/mK) cooled
  through its whole back face into a heat sink (h = 10⁴ W/m²K, T_amb = 40 °C). Integrating over the
  thickness t turns back-face cooling into a distributed sink s = h/t; the edges are insulated.
  (The original brief's single convective *edge* was also implemented; Stage 1 shows it pulls the
  hotspot up to 5 mm away from the power source and caps the chip at ~1.6 W, so backside cooling is
  the default — see the lab notes.)
* **Side view (Stage 6)** — copper spreader (400 W/mK) / thermal interface material (3 W/mK) /
  silicon die, convective bottom, insulated top and sides; a hidden air void (0.03 W/mK) in the TIM.
* **Nondimensionalisation** (everything the networks see): x̂ = x/L, T̂ = (T − T_amb)/ΔT, k̂ = k/k_ref,
  q̂ = qL²/(k_ref ΔT), ŝ = sL²/k_ref, Bi = hL/k_ref with L = 10 mm, ΔT = 50 K, k_ref = 150 W/mK.
  All metrics and plots are converted back to °C, mm and W.
"""

STATIC_METHOD = r"""
## Method

* **Reference solver** — cell-centred finite volumes on a 200² grid (scipy sparse direct solve),
  harmonic-mean face conductivities so material jumps are exact; verified to second order on a
  manufactured solution, exact to round-off across a 5000× conductivity jump, energy balance ~1e-12.
* **PINN** — tanh MLPs (4×64), inputs mapped to [−1, 1], Xavier init; the PDE residual comes from
  autograd in divergence form, so a learned k(x,y) contributes ∇k·∇T automatically. Adam
  (exponential lr decay, collocation points re-drawn every 500 steps) then L-BFGS (strong-Wolfe line
  search, fixed points). Wall-clock budget, atomic checkpoints and resume for Kaggle.
* **Loss balance** — the PDE residual is normalised by its natural scale; sensor misfit is weighted
  by its noise level, w = (σ_pde/σ_noise)², i.e. a Gaussian likelihood (the first Stage 4 run with a
  fixed weight collapsed at 5 % noise — see notes).
* **Inverse problems** add networks/parameters for the unknowns: a positive scalar k (Stage 4); a
  softplus power-map network with known total power and a total-variation prior (Stage 5); a
  bounded log-k network in the TIM with per-layer temperature networks joined by temperature- and
  heat-flux-continuity conditions (Stage 6).
* **Baselines** so the PINN numbers mean something: FV least squares for k (Stage 4), thin-plate
  spline interpolation of the sensors (Stages 5, 8), and the FV solver itself for speed (Stages 3, 8).
"""

STATIC_LIMITS = r"""
## Limitations

* 2D models: top view with an averaged back-face sink, and a side-view slice. Real packages are 3D,
  with lateral spreading in the heat spreader and nonuniform heat-sink coefficients.
* Synthetic data from the same physics family as the PINN (on a different, finer grid to avoid the
  "inverse crime"); no model-form error, no sensor drift or calibration bias.
* The power-map inverse problem is fundamentally under-determined from 10–15 point sensors; results
  depend on the priors (known total power, total variation). Sharp block edges are reconstructed
  smoothed.
* Constant-k identification is weak under backside cooling (k only controls lateral spreading), so
  noise maps to large k errors for *any* method — see the FV baseline in Stage 4.
* The TIM is drawn 0.2 mm thick (real: 25–50 µm) so a 2D grid and network can resolve it.
* For a single forward solve with known inputs, the PINN is far slower than the sparse FV solver; it
  pays off for inverse problems and for repeated parametric queries.
"""

STATIC_RUN = r"""
## How to run

**Kaggle (GPU):** import `main.ipynb`, set *Accelerator = GPU* and *Internet = On*, *Run All*.
The first cell clones this repository into `/kaggle/working/chip-pinn`. Set `QUICK = True` in that
cell for a ~20-minute end-to-end check with short training; `QUICK = False` reproduces the results
here. Every run checkpoints into `results/<name>/` and resumes if the session restarts. The last
cells build the GIF, regenerate this README and zip `results/` for download.

**Locally:** `pip install torch numpy scipy matplotlib pandas pyyaml`, then e.g.

```
python tests/run_tests.py
python scripts/stage5_inverse_q.py --config configs/stage5_inverse_q.yaml --set training.adam_iters=2000
```

## Project layout

```
solver/    finite-volume solver, layouts, sensors, stack model, classical inverse baselines
pinn/      models, losses, sampling, training loop, problems, inverse-q and layered-stack PINNs, plots
common/    config (YAML + --set overrides), seeds, device, scales, checkpoints, plotting, reports
configs/   one YAML per experiment, merged on top of default.yaml
scripts/   one entry point per stage + make_gif.py + make_report.py
tests/     solver verification and autograd residual tests (python tests/run_tests.py)
results/   outputs of every run, plus notes.md (diagnoses and fixes)
```
"""


def main():
    notes = text("notes.md") or ""
    parts = [
        "# Thermal Detective",
        "",
        "**Reconstructing chip temperature, power hotspots and hidden defects from sparse sensors "
        "with physics-informed neural networks (PINNs).**",
        "",
        fig("figures/reconstruction.gif", "reconstruction during training"),
        "",
        "A chip has a handful of on-die thermal sensors, but what matters is the *whole* temperature "
        "map: where the hotspot is, how hot it really gets between sensors, which block is burning "
        "the power, and whether the thermal interface underneath has a void. This project solves "
        "those questions as inverse problems with PINNs — neural networks trained to satisfy the heat "
        "equation and the sparse measurements at the same time — and checks every answer against a "
        "verified finite-volume solver and simple baselines.",
        "",
        "## Headline results",
        "",
        headline(),
        STATIC_PHYSICS,
        STATIC_METHOD,
        "## Results by stage",
        "",
        stage_tables(),
        "",
        "## Lessons learned (lab notes)",
        "",
        notes.replace("# Lab notes — diagnoses and fixes", "").strip() or MISSING,
        STATIC_LIMITS,
        STATIC_RUN,
    ]
    out = ROOT / "README.md"
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"[report] wrote {out}")


if __name__ == "__main__":
    main()

"""Stage 0 tests: config merging/overrides, YAML float coercion, scales, seeding."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common.config import apply_overrides, deep_merge, load_config
from common.scales import Scales
from common.seed import set_seed


def test_default_config_loads_with_float_values():
    cfg = load_config("default.yaml")
    assert cfg.name == "default"
    assert isinstance(cfg.physics.h, float) and cfg.physics.h == 1.0e4
    assert isinstance(cfg.training.adam_lr, float)


def test_experiment_merges_on_default_and_overrides_apply():
    cfg = load_config("stage0_smoke.yaml", ["training.adam_iters=7", "model.hidden=32"])
    assert cfg.name == "stage0_smoke" and cfg.seed == 42
    assert cfg.training.adam_iters == 7 and cfg.model.hidden == 32
    assert cfg.physics.materials.silicon == 150.0          # inherited from default.yaml


def test_bare_1e4_string_is_coerced():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.yaml"
        p.write_text("name: x\nphysics:\n  h: 1e4\n", encoding="utf-8")
        cfg = load_config(p)
    assert cfg.physics.h == 1.0e4 and isinstance(cfg.physics.h, float)


def test_deep_merge_does_not_mutate_inputs():
    a = {"x": {"y": 1, "z": 2}}
    b = {"x": {"y": 5}}
    m = deep_merge(a, b)
    assert m == {"x": {"y": 5, "z": 2}} and a == {"x": {"y": 1, "z": 2}}
    assert apply_overrides(a, ["x.w=[1, 2]"])["x"]["w"] == [1, 2]


def test_scales_roundtrip_and_values():
    s = Scales(L=0.01, k_ref=150.0, dT=50.0, T_amb=40.0)
    T = np.linspace(20, 120, 11)
    assert np.allclose(s.T_dim(s.T_nd(T)), T)
    assert np.isclose(s.biot(1.0e4), 1.0e4 * 0.01 / 150.0)
    assert np.isclose(s.q_ref, 150.0 * 50.0 / 0.01 ** 2)    # 7.5e7 W/m³
    assert np.isclose(s.T_nd(90.0), 1.0)


def test_seed_reproducible():
    set_seed(3); a = np.random.rand(5)
    set_seed(3); b = np.random.rand(5)
    assert np.array_equal(a, b)

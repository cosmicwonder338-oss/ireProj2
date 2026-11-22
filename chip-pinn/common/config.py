"""YAML config system.

Each experiment is one YAML file. It is deep-merged on top of configs/default.yaml (or on top of the
file named in its optional `base:` key), then command-line overrides `--set a.b=value` are applied.
The fully resolved config is saved to results/<name>/config.yaml so every run is reproducible.
"""
from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path
from typing import Any, Iterable

import yaml

from .paths import CONFIG_DIR, results_dir

_FLOAT_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)$")


class Config(dict):
    """dict with attribute access (cfg.training.adam_lr); nested dicts are converted recursively."""

    def __init__(self, data: dict | None = None):
        super().__init__()
        for k, v in (data or {}).items():
            self[k] = Config(v) if isinstance(v, dict) else v

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(f"config has no key '{key}'") from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = Config(value) if isinstance(value, dict) else value

    def to_dict(self) -> dict:
        return {k: v.to_dict() if isinstance(v, Config) else v for k, v in self.items()}


def _coerce(v: Any) -> Any:
    """PyYAML (YAML 1.1) parses '1e4' as a string; turn such strings into floats."""
    if isinstance(v, str) and _FLOAT_RE.match(v.strip()):
        return float(v)
    if isinstance(v, dict):
        return {k: _coerce(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_coerce(x) for x in v]
    return v


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base` (override wins)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return _coerce(yaml.safe_load(f) or {})


def _resolve_chain(path: Path, depth: int = 0) -> dict:
    if depth > 8:
        raise RecursionError("config `base:` chain too deep (cycle?)")
    data = _read_yaml(path)
    base_name = data.pop("base", None)
    if base_name is not None:
        base = _resolve_chain((path.parent / base_name).resolve(), depth + 1)
    elif path.resolve() != (CONFIG_DIR / "default.yaml").resolve():
        base = _read_yaml(CONFIG_DIR / "default.yaml")
    else:
        base = {}
    return deep_merge(base, data)


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    """Apply 'a.b.c=value' strings; values are parsed as YAML (so 3, 1.0e-3, true, [1,2] work)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.sub=value, got '{item}'")
        key, raw = item.split("=", 1)
        value = _coerce(yaml.safe_load(raw))
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return cfg


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> Config:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path
    cfg = apply_overrides(_resolve_chain(path.resolve()), overrides or [])
    if not cfg.get("name"):
        raise ValueError(f"config {path} has no `name`")
    return Config(cfg)


def save_config(cfg: Config, out_dir: Path) -> Path:
    p = Path(out_dir) / "config.yaml"
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
    return p


def parse_args(description: str = "", default_config: str | None = None) -> argparse.Namespace:
    """Standard CLI for every stage script: --config <yaml> [--set key=value ...]."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=default_config, required=default_config is None)
    ap.add_argument("--set", nargs="*", default=[], action="extend", metavar="KEY=VALUE",
                    help="override config entries, e.g. --set training.adam_iters=200 "
                         "(may be repeated; later values win)")
    return ap.parse_args()


def setup_run(config_path: str | Path, overrides: Iterable[str] | None = None):
    """Load config, seed everything, create results/<name>/, save the resolved config.

    Returns (cfg, out_dir, device).
    """
    import sys

    from .device import get_device
    from .seed import set_seed

    for stream in (sys.stdout, sys.stderr):       # unicode symbols (k̂, °C) on Windows consoles
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    cfg = load_config(config_path, overrides)
    set_seed(cfg.seed)
    out_dir = results_dir(cfg.name)
    save_config(cfg, out_dir)
    device = get_device(cfg.get("device", "auto"))
    return cfg, out_dir, device

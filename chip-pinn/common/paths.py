"""Project-relative paths.

The project root is the folder that contains this package, so the same code works locally and at
/kaggle/working/chip-pinn/. Set CHIP_PINN_RESULTS to redirect outputs elsewhere.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = Path(os.environ.get("CHIP_PINN_RESULTS", PROJECT_ROOT / "results"))


def results_dir(name: str) -> Path:
    """Return (and create) results/<name>/."""
    d = RESULTS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d

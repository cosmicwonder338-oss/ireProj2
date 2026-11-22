"""Minimal test runner (no pytest needed): runs every test_* function in tests/test_*.py.

    python tests/run_tests.py            # all tests
    python tests/run_tests.py config     # only files whose name contains 'config'
"""
import importlib.util
import sys
import time
import traceback
from unittest import SkipTest  # also understood by pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(filters):
    failed, passed, skipped = [], 0, 0
    for f in sorted((ROOT / "tests").glob("test_*.py")):
        if filters and not any(s in f.stem for s in filters):
            continue
        spec = importlib.util.spec_from_file_location(f.stem, f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            t0 = time.perf_counter()
            try:
                getattr(mod, name)()
                passed += 1
                status = "PASS"
            except SkipTest as e:
                skipped += 1
                status = f"SKIP ({e})"
            except Exception:
                failed.append(f"{f.stem}.{name}")
                status = "FAIL"
                traceback.print_exc()
            print(f"{status:5s} {f.stem}.{name}  [{time.perf_counter() - t0:.2f}s]")
    print(f"\n{passed} passed, {len(failed)} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

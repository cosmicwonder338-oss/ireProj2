"""Device selection and an environment report."""
import platform
import sys


def get_device(pref: str = "auto"):
    """Return a torch.device: 'cuda' if available (and requested), else 'cpu'.

    Returns the string 'cpu' when torch isn't installed, so solver-only code still runs.
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda" and not torch.cuda.is_available():
        print("[device] cuda requested but not available -> cpu")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def env_report() -> dict:
    """Versions + hardware, saved with every run."""
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "scipy", "matplotlib", "pandas", "yaml", "torch"):
        try:
            info[mod] = __import__(mod).__version__
        except ImportError:
            info[mod] = None
    try:
        import torch
        info["cuda"] = torch.cuda.is_available()
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["cuda"], info["gpu"] = False, None
    return info

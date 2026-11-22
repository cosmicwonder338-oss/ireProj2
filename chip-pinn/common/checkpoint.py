"""Crash-safe checkpoints so a Kaggle session restart doesn't lose training progress.

Writes go to a temp file first and are then atomically renamed, so a kill mid-write never leaves a
corrupt checkpoint behind.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def save_checkpoint(path: str | Path, **state: Any) -> None:
    """save_checkpoint(path, model=..., optimizer=..., step=..., history=...).

    nn.Module / optimizer objects are stored via their state_dict(); everything else as-is.
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: (v.state_dict() if hasattr(v, "state_dict") else v) for k, v in state.items()}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location="cpu", **targets: Any) -> dict | None:
    """Load a checkpoint if it exists. Objects passed as keyword args (model=net, optimizer=opt)
    get load_state_dict() called on them. Returns the raw payload, or None if no checkpoint."""
    import torch

    path = Path(path)
    if not path.exists():
        return None
    payload = torch.load(path, map_location=map_location, weights_only=False)
    for k, obj in targets.items():
        if k in payload and obj is not None:
            obj.load_state_dict(payload[k])
    return payload

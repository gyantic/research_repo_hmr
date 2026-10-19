"""Checkpoint helpers shared by the export_*_romp.py scripts in this directory."""
from __future__ import annotations

import pathlib

import torch


def file_size_mb(path: pathlib.Path) -> float:
    return path.stat().st_size / 1024 ** 2


def load_checkpoint(path: pathlib.Path):
    """Return (raw_checkpoint, state_dict).

    ROMP checkpoints are either a bare state dict or a dict holding it under
    'model_state_dict'.
    """
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        state_dict = raw.get("model_state_dict", raw)
    else:
        state_dict = raw
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(state_dict)!r}")
    return raw, state_dict


def strip_module_prefix(state_dict: dict) -> dict:
    """Drop the 'module.' prefix that (Distributed)DataParallel adds to every key."""
    return {k.removeprefix("module."): v for k, v in state_dict.items()}

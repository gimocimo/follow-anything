"""Configuration loading and device selection."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml


def load_config(path: Union[str, Path] = "configs/default.yaml") -> dict:
    """Load a YAML config file into a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def get_device(pref: str = "auto") -> str:
    """Resolve the compute device.

    `auto` picks CUDA if available, then Apple MPS, then CPU. torch is imported
    lazily so the rest of the package (e.g. splits) works without it installed.
    """
    if pref and pref != "auto":
        return pref
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"

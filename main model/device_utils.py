"""Device selection helpers for long EEG experiments."""

from __future__ import annotations

import os

import torch


def select_device(prefer_mps: bool = True) -> str:
    """Prefer CUDA, then Apple MPS, then CPU."""
    forced = os.environ.get("SSL_DEVICE", "").strip().lower()
    if forced:
        if forced not in {"cpu", "mps", "cuda"}:
            raise ValueError("SSL_DEVICE must be one of: cpu, mps, cuda")
        if forced == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("SSL_DEVICE=cuda requested but CUDA is unavailable")
        mps = getattr(torch.backends, "mps", None)
        if forced == "mps" and (mps is None or not mps.is_available()):
            raise RuntimeError("SSL_DEVICE=mps requested but MPS is unavailable")
        return forced
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if prefer_mps and mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def seed_accelerator(device: str, seed: int) -> None:
    """Apply accelerator-specific reproducibility/performance settings."""
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

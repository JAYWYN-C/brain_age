"""EEG augmentation transforms (B-4 in §10.2).

Each transform is a callable `t(x: Tensor) -> Tensor` where x has shape
(C, L). Chain via `Compose([...])`. Augmentations are applied on the CPU
inside the Dataset, only when training (caller-controlled).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch


class Compose:
    def __init__(self, transforms: Sequence):
        self.transforms = list(transforms)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self) -> str:
        return "Compose(" + ", ".join(repr(t) for t in self.transforms) + ")"


class ChannelDropout:
    """Zero out a random subset of channels (per epoch). p = drop probability per channel."""

    def __init__(self, p: float = 0.1):
        if not 0.0 <= p < 1.0:
            raise ValueError("p must be in [0, 1)")
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0:
            return x
        C = x.shape[0]
        mask = torch.rand(C, device=x.device) >= self.p
        # never drop ALL channels
        if not mask.any():
            mask[torch.randint(0, C, (1,))] = True
        return x * mask.view(C, 1).to(x.dtype)


class TimeShift:
    """Roll the time axis by a uniform offset in [-max_shift, max_shift]."""

    def __init__(self, max_shift: int = 40):
        if max_shift < 0:
            raise ValueError("max_shift must be >= 0")
        self.max_shift = max_shift

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_shift == 0:
            return x
        shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)))
        if shift == 0:
            return x
        return torch.roll(x, shifts=shift, dims=-1)


class GaussianNoise:
    """Additive zero-mean Gaussian noise; sigma is in input units."""

    def __init__(self, sigma: float = 0.05):
        if sigma < 0:
            raise ValueError("sigma must be >= 0")
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.sigma <= 0.0:
            return x
        return x + torch.randn_like(x) * self.sigma


def build_transforms(spec: Optional[dict]) -> Optional[Compose]:
    """Build a Compose from a dict like:
        {"channel_dropout": 0.1, "time_shift": 40, "gaussian_noise": 0.05}
    Missing/zero entries are skipped. Returns None if nothing to do.
    """
    if not spec:
        return None
    parts: List = []
    p_drop = float(spec.get("channel_dropout", 0.0))
    if p_drop > 0:
        parts.append(ChannelDropout(p_drop))
    max_shift = int(spec.get("time_shift", 0))
    if max_shift > 0:
        parts.append(TimeShift(max_shift))
    sigma = float(spec.get("gaussian_noise", 0.0))
    if sigma > 0:
        parts.append(GaussianNoise(sigma))
    if not parts:
        return None
    return Compose(parts)

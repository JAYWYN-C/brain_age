"""V0/V1 downstream model builder for the final 0702 main experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ssl_model import (
    BackboneConfig,
    BrainAgeRegressor,
    DownstreamClassifier,
    build_backbone,
)


VARIANTS = ("V0", "V1")


@dataclass
class VariantSpec:
    name: str
    pretext_init: bool
    freeze_backbone: bool


VARIANT_SPECS = {
    "V0": VariantSpec("V0", pretext_init=False, freeze_backbone=False),
    "V1": VariantSpec("V1", pretext_init=True, freeze_backbone=False),
}


def _load_pretext_backbone(checkpoint_path: str, device: str = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg_dict = ckpt["backbone_cfg"]
    # Final 0702 checkpoints use BasicDeepCNN.
    cfg_dict.setdefault("backbone_type", "basic_deep_cnn")
    cfg_dict.setdefault("seq_length", 800)
    cfg_dict.setdefault("extra_input_channels", 0)
    cfg = BackboneConfig(**cfg_dict)
    pretext = BrainAgeRegressor(cfg=cfg)
    pretext.load_state_dict(ckpt["state_dict"])
    return pretext.backbone


def build_downstream(
    variant: str,
    *,
    backbone_cfg: Optional[BackboneConfig] = None,
    pretext_checkpoint: Optional[str] = None,
    n_classes: int = 4,
    head_hidden: Optional[int] = None,
    dropout: float = 0.3,
    device: str = "cpu",
) -> DownstreamClassifier:
    if variant not in VARIANT_SPECS:
        raise ValueError(f"unknown variant {variant}, expected one of {VARIANTS}")
    spec = VARIANT_SPECS[variant]

    if spec.pretext_init:
        if not pretext_checkpoint:
            raise ValueError(f"variant {variant} needs pretext_checkpoint")
        backbone = _load_pretext_backbone(pretext_checkpoint, device=device)
    else:
        backbone = build_backbone(backbone_cfg)

    return DownstreamClassifier(
        backbone=backbone,
        n_classes=n_classes,
        side_dim=0,
        freeze_backbone=spec.freeze_backbone,
        head_hidden=head_hidden,
        dropout=dropout,
    )

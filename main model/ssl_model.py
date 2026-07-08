"""Brain-age SSL backbone + heads for the final 0702 V0/V1 main experiment.

Architecture:

    EEG (B, 19, 800)
      -> BasicDeepCNN1D backbone
      -> Global Average + Max Pooling
      -> Linear -> 128-d embedding
      -> head:
           pretext: Linear(128 -> 1)        # age regression
           downstream: Linear(128 -> 4)     # 4-class
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


@dataclass
class BackboneConfig:
    backbone_type: str = "basic_deep_cnn"
    in_channels: int = 19
    seq_length: int = 800
    base_channels: int = 32
    embedding_dim: int = 128
    # Legacy fields kept so old config dictionaries still load.
    kernel_sizes: tuple = (15, 31, 63)
    pools: tuple = (4, 4, 4)
    dropout: float = 0.3
    extra_input_channels: int = 0
    deep4_final_conv_length: str | int = "auto"
    deep4_n_filters_time: int = 25
    deep4_n_filters_spat: int = 25
    deep4_filter_time_length: int = 10
    deep4_pool_time_length: int = 3
    deep4_pool_time_stride: int = 3
    deep4_n_filters_2: int = 50
    deep4_filter_length_2: int = 10
    deep4_n_filters_3: int = 100
    deep4_filter_length_3: int = 10
    deep4_n_filters_4: int = 200
    deep4_filter_length_4: int = 10
    deep4_first_pool_mode: str = "max"
    deep4_later_pool_mode: str = "max"
    deep4_split_first_layer: bool = True
    deep4_batch_norm: bool = True
    deep4_batch_norm_alpha: float = 0.1
    deep4_stride_before_pool: bool = False
    # LayerNorm on the final embedding to stabilize feature scale when BN is
    # disabled or running stats are unreliable on small CN-only datasets.
    add_output_layernorm: bool = False
    drop_path: float = 0.1


EEGBackbone = nn.Module


class _BasicDeepCNNBackbone(nn.Module):
    """Wrapper around basic_deep_cnn.BasicDeepCNN1D — 0702 paper main model
    (deep plain 1-D CNN). Exposes the SSL backbone interface. Implementation
    lives in basic_deep_cnn.py."""

    def __init__(self, cfg: Optional[BackboneConfig] = None):
        super().__init__()
        from basic_deep_cnn import BasicDeepCNN1D  # local import: avoid cycles
        cfg = cfg or BackboneConfig(backbone_type="basic_deep_cnn")
        self.cfg = cfg
        self.net = BasicDeepCNN1D(
            in_channels=cfg.in_channels + cfg.extra_input_channels,
            embedding_dim=cfg.embedding_dim,
            base_channels=cfg.base_channels,
            dropout=cfg.dropout,
            add_output_layernorm=cfg.add_output_layernorm,
        )

    @property
    def embedding_dim(self) -> int:
        return int(self.cfg.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_backbone(cfg: Optional[BackboneConfig] = None) -> nn.Module:
    cfg = cfg or BackboneConfig()
    if cfg.backbone_type == "basic_deep_cnn":
        return _BasicDeepCNNBackbone(cfg)
    raise ValueError("final 0702 main model only supports backbone_type='basic_deep_cnn'")


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


class AgeRegressionHead(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.fc(e).squeeze(-1)


class ClassificationHead(nn.Module):
    """Linear classifier for V0/V1 downstream diagnosis."""

    def __init__(self, embedding_dim: int, n_classes: int, side_dim: int = 0,
                 hidden: Optional[int] = None, dropout: float = 0.3):
        super().__init__()
        in_dim = embedding_dim + side_dim
        layers = []
        if hidden:
            layers += [
                nn.Linear(in_dim, hidden, bias=False),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, n_classes))
        self.fc = nn.Sequential(*layers)

    def forward(self, e: torch.Tensor, side: Optional[torch.Tensor] = None) -> torch.Tensor:
        if side is not None:
            e = torch.cat([e, side], dim=1)
        return self.fc(e)


# ---------------------------------------------------------------------------
# Composed models
# ---------------------------------------------------------------------------


class BrainAgeRegressor(nn.Module):
    """Pretext model: backbone + age head."""

    def __init__(self, backbone: Optional[EEGBackbone] = None,
                 cfg: Optional[BackboneConfig] = None):
        super().__init__()
        self.backbone = backbone or build_backbone(cfg)
        self.head = AgeRegressionHead(self.backbone.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.backbone(x)
        return self.head(e)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class DownstreamClassifier(nn.Module):
    """4-class classifier for the final V0/V1 comparison."""

    def __init__(
        self,
        backbone: EEGBackbone,
        n_classes: int = 4,
        side_dim: int = 0,
        freeze_backbone: bool = False,
        head_hidden: Optional[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if side_dim != 0:
            raise ValueError("final 0702 V0/V1 classifier does not use extra tabular inputs")
        self.backbone = backbone
        self.head = ClassificationHead(
            embedding_dim=backbone.embedding_dim,
            n_classes=n_classes,
            side_dim=side_dim,
            hidden=head_hidden,
            dropout=dropout,
        )
        self.side_dim = side_dim

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
            self._frozen = True
        else:
            self._frozen = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self._frozen:
            self.backbone.eval()
        return self

    def forward(
        self,
        x: torch.Tensor,
        side: Optional[torch.Tensor] = None,
        age: Optional[torch.Tensor] = None,
        pred_age: Optional[torch.Tensor] = None,
    ):
        """Return diagnosis logits."""
        if side is not None:
            raise ValueError("final 0702 V0/V1 classifier does not use extra tabular inputs")
        if self._frozen:
            with torch.no_grad():
                e = self.backbone(x)
        else:
            e = self.backbone(x)
        return self.head(e)

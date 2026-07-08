"""BasicDeepCNN1D — 0702 paper main experimental backbone.

Design rationale
----------------
This is the final 0702 paper main backbone. The paper comparison is only V0
(random-init downstream) versus V1 (SSL pretext-init downstream). The model is
a deep, plain convolutional stack designed to make pretext initialization
meaningful on the small CAUEEG cohort.

BasicDeepCNN1D is a plain deep 1-D CNN built on that principle but is NOT a
VGG clone:
  * single (not double) conv per stage,
  * wide EEG-appropriate temporal kernels (25 -> 5) instead of VGG's fixed 3,
  * strided-conv + max-pool hybrid downsampling,
  * mean+std temporal statistics-pooling head (VGG uses flatten/global-avg),
  * a small projection MLP + optional output LayerNorm to match the rest of
    the SSL codebase.
It is reported in the paper as the "deep basic CNN" main model.

Interface used by the 0702 V0/V1 pipeline:
forward(x: [B, C, T]) -> [B, embedding_dim].
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _PlainConvBlock(nn.Module):
    """Plain conv -> BatchNorm -> GELU -> (optional) MaxPool -> Dropout.

    Deliberately NO residual shortcut and NO depthwise-separable path: this is
    what keeps the block hard to optimise from scratch, which is precisely the
    regime where SSL pretext initialisation (V1) helps.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        pool: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size=kernel_size,
            stride=stride, padding=kernel_size // 2, bias=False,
        )
        self.norm = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pool(x)
        return self.drop(x)


class _TemporalStatsPool1d(nn.Module):
    """Global mean+std over time -> [B, 2*C]. Length-agnostic."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1)
        std = x.std(dim=-1, unbiased=False)
        return torch.cat([mean, std], dim=1)


class BasicDeepCNN1D(nn.Module):
    """Deep plain 1-D CNN main model (0702).

    base_channels=32 gives the channel ladder [32, 64, 128, 128, 128].
    Six conv stages, plain (no residual), progressively narrower temporal
    kernels, hybrid stride/pool downsampling, mean+std stat-pool head.
    """

    def __init__(
        self,
        in_channels: int,
        embedding_dim: int,
        base_channels: int = 32,
        dropout: float = 0.3,
        add_output_layernorm: bool = True,
    ) -> None:
        super().__init__()
        c1 = base_channels          # 32
        c2 = base_channels * 2      # 64
        c3 = base_channels * 4      # 128
        c4 = base_channels * 4      # 128

        # Deep plain stack. Stem does an aggressive stride-2 + pool-2 temporal
        # decimation; the middle stages are stride-1 conv + pool-2 (VGG-style
        # "conv then pool"); the last two stages are stride-1 holds with no
        # pool so the network is genuinely deep (6 conv layers) before the
        # statistics head.
        self.features = nn.Sequential(
            _PlainConvBlock(in_channels, c1, kernel_size=25, stride=2, pool=2, dropout=0.0),
            _PlainConvBlock(c1, c2, kernel_size=15, stride=1, pool=2, dropout=dropout),
            _PlainConvBlock(c2, c3, kernel_size=11, stride=1, pool=2, dropout=dropout),
            _PlainConvBlock(c3, c4, kernel_size=9,  stride=1, pool=2, dropout=dropout),
            _PlainConvBlock(c4, c4, kernel_size=7,  stride=1, pool=1, dropout=dropout),
            _PlainConvBlock(c4, c4, kernel_size=5,  stride=1, pool=1, dropout=dropout),
        )

        self.pool = _TemporalStatsPool1d()
        self.proj = nn.Sequential(
            nn.Linear(c4 * 2, embedding_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim, bias=False),
        )
        self.out_norm = (
            nn.LayerNorm(embedding_dim) if add_output_layernorm else nn.Identity()
        )

        self._embedding_dim = int(embedding_dim)
        self.reset_weights()

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def reset_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.proj(x)
        return self.out_norm(x)

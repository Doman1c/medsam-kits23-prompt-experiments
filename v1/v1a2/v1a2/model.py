from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups != 0:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class DualKidneyPromptNet(nn.Module):
    """Lightweight 2.5D CenterNet-style dual-kidney prompt generator."""

    def __init__(self, in_channels: int = 3, base_channels: int = 24) -> None:
        super().__init__()
        b = int(base_channels)
        self.stem = ConvBlock(in_channels, b, stride=2)      # 128 x 128
        self.stage1 = ConvBlock(b, b * 2, stride=2)          # 64 x 64
        self.stage2 = ConvBlock(b * 2, b * 4, stride=2)      # 32 x 32
        self.stage3 = ConvBlock(b * 4, b * 6, stride=2)      # 16 x 16

        self.lat2 = nn.Conv2d(b * 4, b * 3, 1)
        self.lat1 = nn.Conv2d(b * 2, b * 3, 1)
        self.up2 = ConvBlock(b * 6 + b * 3, b * 3)
        self.up1 = ConvBlock(b * 3 + b * 3, b * 3)
        self.shared = ConvBlock(b * 3, b * 3)

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(b * 3, b * 2, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(b * 2, 2, 1),
        )
        self.size_head = nn.Sequential(
            nn.Conv2d(b * 3, b * 2, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(b * 2, 4, 1),
        )
        self.offset_head = nn.Sequential(
            nn.Conv2d(b * 3, b * 2, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(b * 2, 4, 1),
        )
        self.presence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(b * 6, b * 2),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(b * 2, 2),
        )
        self._initialize_heads()

    def _initialize_heads(self) -> None:
        nn.init.constant_(self.heatmap_head[-1].bias, -2.19)
        # Typical kidney boxes occupy roughly 10-25% of a 2D slice dimension.
        nn.init.constant_(self.size_head[-1].bias, math.log(0.18))
        nn.init.constant_(self.offset_head[-1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)

        d2 = F.interpolate(x3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.up2(torch.cat([d2, self.lat2(x2)], dim=1))
        d1 = F.interpolate(d2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.up1(torch.cat([d1, self.lat1(x1)], dim=1))
        feature = self.shared(d1)

        return {
            "presence_logits": self.presence_head(x3),
            "heatmap_logits": self.heatmap_head(feature),
            "log_size_map": self.size_head(feature),
            "offset_logits": self.offset_head(feature),
        }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

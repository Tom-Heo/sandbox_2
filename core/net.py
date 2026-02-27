from __future__ import annotations

import torch
import torch.nn as nn
from .heo import Heo
from .block import Bottleneck, Stage, Block, SimpleGate


class SandNet(nn.Module):
    def __init__(self, bottleneck_dim=2048):
        super().__init__()
        self.base_dim = bottleneck_dim // 8

        self.stem = nn.Conv2d(3, self.base_dim, kernel_size=1, padding=0)

        self.encoder_stage1 = Stage(self.base_dim)
        self.downsample1 = nn.Conv2d(
            self.base_dim, self.base_dim * 2, kernel_size=2, stride=2, padding=0
        )

        self.encoder_stage2 = Stage(self.base_dim * 2)
        self.downsample2 = nn.Conv2d(
            self.base_dim * 2, self.base_dim * 4, kernel_size=2, stride=2, padding=0
        )
        self.encoder_stage3 = Stage(self.base_dim * 4)
        self.downsample3 = nn.Conv2d(
            self.base_dim * 4, self.base_dim * 8, kernel_size=2, stride=2, padding=0
        )

        self.bottleneck = Bottleneck(self.base_dim * 8)

        self.upsample3 = nn.ConvTranspose2d(
            self.base_dim * 8, self.base_dim * 4, kernel_size=2, stride=2
        )
        self.reduce3 = nn.Conv2d(
            self.base_dim * 8, self.base_dim * 4, kernel_size=1, padding=0
        )
        self.decoder_stage3 = Stage(self.base_dim * 4)

        self.upsample2 = nn.ConvTranspose2d(
            self.base_dim * 4, self.base_dim * 2, kernel_size=2, stride=2
        )
        self.reduce2 = nn.Conv2d(
            self.base_dim * 4, self.base_dim * 2, kernel_size=1, padding=0
        )
        self.decoder_stage2 = Stage(self.base_dim * 2)

        self.upsample1 = nn.ConvTranspose2d(
            self.base_dim * 2, self.base_dim, kernel_size=2, stride=2
        )
        self.reduce1 = nn.Conv2d(
            self.base_dim * 2, self.base_dim, kernel_size=1, padding=0
        )
        self.decoder_stage1 = Stage(self.base_dim)

        self.head = nn.Conv2d(self.base_dim, 3, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B, 3, H, W) - OklabP
        Output: (B, 3, H, W) - Refined OklabP
        """
        raw = x

        x = self.stem(x)

        x = self.encoder_stage1(x)
        stage1 = x
        x = self.downsample1(x)

        x = self.encoder_stage2(x)
        stage2 = x
        x = self.downsample2(x)

        x = self.encoder_stage3(x)
        stage3 = x
        x = self.downsample3(x)

        x = self.bottleneck(x)

        x = self.upsample3(x)
        x = torch.cat([x, stage3], dim=1)
        x = self.reduce3(x)
        x = self.decoder_stage3(x)

        x = self.upsample2(x)
        x = torch.cat([x, stage2], dim=1)
        x = self.reduce2(x)
        x = self.decoder_stage2(x)

        x = self.upsample1(x)
        x = torch.cat([x, stage1], dim=1)
        x = self.reduce1(x)
        x = self.decoder_stage1(x)

        x = self.head(x)

        return raw + x

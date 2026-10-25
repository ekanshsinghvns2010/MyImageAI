
"""MyImageAI v2: residual U-Net for RGB diffusion."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps, dim):
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000)
        * torch.arange(
            half, device=timesteps.device, dtype=torch.float32
        )
        / max(half - 1, 1)
    )
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([angles.sin(), angles.cos()], dim=1)

    if dim % 2:
        embedding = F.pad(embedding, (0, 1))

    return embedding


class ResBlock(nn.Module):
    """Residual convolution block conditioned on diffusion time."""

    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, padding=1
        )

        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1
        )

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x, time_features):
        residual = self.skip(x)

        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_projection(time_features)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))

        return h + residual


class MyImageAIModelV2(nn.Module):
    """Compact residual U-Net for 32x32 RGB images."""

    def __init__(
        self,
        in_channels=3,
        base_channels=32,
        time_dim=128,
    ):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 3

        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_conv = nn.Conv2d(
            in_channels, c1, 3, padding=1
        )

        self.enc1 = ResBlock(c1, c1, time_dim)
        self.down1 = nn.Conv2d(c1, c2, 4, stride=2, padding=1)

        self.enc2 = ResBlock(c2, c2, time_dim)
        self.down2 = nn.Conv2d(c2, c3, 4, stride=2, padding=1)

        self.mid1 = ResBlock(c3, c3, time_dim)
        self.mid2 = ResBlock(c3, c3, time_dim)

        self.dec2 = ResBlock(c3 + c2, c2, time_dim)
        self.dec1 = ResBlock(c2 + c1, c1, time_dim)

        self.output_norm = nn.GroupNorm(8, c1)
        self.output_conv = nn.Conv2d(
            c1, in_channels, 3, padding=1
        )

    def forward(self, images, timesteps):
        t = self.time_mlp(
            timestep_embedding(timesteps, self.time_dim)
        )

        x0 = self.input_conv(images)
        x1 = self.enc1(x0, t)

        x2 = self.down1(x1)
        x2 = self.enc2(x2, t)

        x3 = self.down2(x2)
        x3 = self.mid1(x3, t)
        x3 = self.mid2(x3, t)

        x = F.interpolate(
            x3, size=x2.shape[-2:], mode="nearest"
        )
        x = self.dec2(torch.cat([x, x2], dim=1), t)

        x = F.interpolate(
            x, size=x1.shape[-2:], mode="nearest"
        )
        x = self.dec1(torch.cat([x, x1], dim=1), t)

        return self.output_conv(F.silu(self.output_norm(x)))

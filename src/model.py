
"""Small diffusion neural network for MyImageAI.

This model predicts the noise added to an image at a given timestep.
It is designed for our first 28x28 grayscale training experiment.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Convert diffusion timesteps into sinusoidal embeddings."""
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


class MyImageAIModel(nn.Module):
    """A compact convolutional noise-prediction model."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        time_dim: int = 128,
    ):
        super().__init__()

        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_conv = nn.Conv2d(
            in_channels, base_channels, kernel_size=3, padding=1
        )

        self.down_conv = nn.Conv2d(
            base_channels,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.down_norm = nn.GroupNorm(8, base_channels * 2)

        self.time_projection = nn.Linear(
            time_dim, base_channels * 2
        )

        self.middle_conv = nn.Sequential(
            nn.Conv2d(
                base_channels * 2,
                base_channels * 2,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
        )

        self.up_conv = nn.Sequential(
            nn.Conv2d(
                base_channels * 3,
                base_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )

        self.output_conv = nn.Conv2d(
            base_channels, in_channels, kernel_size=3, padding=1
        )

    def forward(
        self, images: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Predict the noise present in images at the given timesteps."""
        time_features = timestep_embedding(
            timesteps, self.time_dim
        )
        time_features = self.time_mlp(time_features)

        skip = F.silu(self.input_conv(images))

        hidden = self.down_conv(skip)
        hidden = self.down_norm(hidden)

        time_features = self.time_projection(time_features)
        hidden = hidden + time_features[:, :, None, None]
        hidden = F.silu(hidden)

        hidden = self.middle_conv(hidden)

        hidden = F.interpolate(
            hidden,
            size=skip.shape[-2:],
            mode="nearest",
        )

        hidden = torch.cat([skip, hidden], dim=1)
        hidden = self.up_conv(hidden)

        return self.output_conv(hidden)


"""Diffusion training and image sampling utilities for MyImageAI."""

import torch
import torch.nn as nn


class Diffusion:
    """Basic DDPM noise schedule and reverse sampling process."""

    def __init__(
        self,
        timesteps: int = 300,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cpu",
    ):
        self.timesteps = timesteps
        self.device = torch.device(device)

        self.betas = torch.linspace(
            beta_start, beta_end, timesteps, device=self.device
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(
            1.0 - self.alpha_bars
        )

    def add_noise(self, images, timesteps, noise=None):
        """Add noise to clean images at selected timesteps."""
        if noise is None:
            noise = torch.randn_like(images)

        signal = self.sqrt_alpha_bars[timesteps].view(-1, 1, 1, 1)
        noise_scale = self.sqrt_one_minus_alpha_bars[
            timesteps
        ].view(-1, 1, 1, 1)

        noisy_images = signal * images + noise_scale * noise
        return noisy_images, noise

    @torch.no_grad()
    def sample(self, model, image_size=28, batch_size=16, channels=1):
        """Generate images by iteratively removing predicted noise."""
        model.eval()

        images = torch.randn(
            batch_size,
            channels,
            image_size,
            image_size,
            device=self.device,
        )

        for step in reversed(range(self.timesteps)):
            t = torch.full(
                (batch_size,), step, device=self.device, dtype=torch.long
            )

            predicted_noise = model(images, t)

            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            beta = self.betas[step]

            images = (
                images
                - beta / torch.sqrt(1.0 - alpha_bar) * predicted_noise
            ) / torch.sqrt(alpha)

            if step > 0:
                images += torch.sqrt(beta) * torch.randn_like(images)

        return images.clamp(-1, 1)

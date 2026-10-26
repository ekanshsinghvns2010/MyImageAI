
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        frequencies = torch.exp(
            torch.arange(half, device=timesteps.device, dtype=torch.float32)
            * -scale
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        if embedding.shape[1] < self.dim:
            embedding = F.pad(embedding, (0, self.dim - embedding.shape[1]))
        return embedding


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim, dropout=0.05):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return (h + self.skip(x)) / math.sqrt(2.0)


class SelfAttention2d(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        assert channels % heads == 0
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(
            channels, heads, batch_first=True
        )
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        tokens = self.norm(x).flatten(2).transpose(1, 2)
        tokens, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = tokens.transpose(1, 2).reshape(b, c, h, w)
        return residual + self.proj(tokens) / math.sqrt(2.0)


class MyImageAIModelV5(nn.Module):
    """
    128x128 class-conditioned noise-prediction U-Net.
    Classes: cat=0, dog=1, wild=2, null/unconditional=3.
    """
    def __init__(
        self,
        image_channels=3,
        base_channels=48,
        time_dim=192,
        num_classes=3,
        dropout=0.05,
    ):
        super().__init__()
        self.image_channels = image_channels
        self.base_channels = base_channels
        self.num_classes = num_classes
        self.null_class = num_classes

        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 3,
            base_channels * 4,
        )

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.class_embedding = nn.Embedding(num_classes + 1, time_dim)
        self.input = nn.Conv2d(image_channels, c1, 3, padding=1)

        self.enc1 = ResBlock(c1, c1, time_dim, dropout)
        self.down1 = nn.Conv2d(c1, c2, 4, stride=2, padding=1)
        self.enc2 = ResBlock(c2, c2, time_dim, dropout)
        self.down2 = nn.Conv2d(c2, c3, 4, stride=2, padding=1)
        self.enc3 = ResBlock(c3, c3, time_dim, dropout)
        self.down3 = nn.Conv2d(c3, c4, 4, stride=2, padding=1)
        self.enc4 = ResBlock(c4, c4, time_dim, dropout)
        self.down4 = nn.Conv2d(c4, c4, 4, stride=2, padding=1)

        self.mid1 = ResBlock(c4, c4, time_dim, dropout)
        self.attention = SelfAttention2d(c4)
        self.mid2 = ResBlock(c4, c4, time_dim, dropout)

        self.up4 = nn.Conv2d(c4, c4, 3, padding=1)
        self.dec4 = ResBlock(c4 + c4, c4, time_dim, dropout)
        self.up3 = nn.Conv2d(c4, c3, 3, padding=1)
        self.dec3 = ResBlock(c3 + c3, c3, time_dim, dropout)
        self.up2 = nn.Conv2d(c3, c2, 3, padding=1)
        self.dec2 = ResBlock(c2 + c2, c2, time_dim, dropout)
        self.up1 = nn.Conv2d(c2, c1, 3, padding=1)
        self.dec1 = ResBlock(c1 + c1, c1, time_dim, dropout)

        self.output = nn.Sequential(
            nn.GroupNorm(8, c1),
            nn.SiLU(),
            nn.Conv2d(c1, image_channels, 3, padding=1),
        )

    def forward(self, x, timesteps, labels):
        if x.ndim != 4 or x.shape[1] != self.image_channels:
            raise ValueError("Expected input shaped [B, 3, H, W].")
        if labels.ndim != 1 or labels.shape[0] != x.shape[0]:
            raise ValueError("labels must have shape [B].")

        emb = self.time_mlp(timesteps) + self.class_embedding(labels)

        e1 = self.enc1(self.input(x), emb)
        e2 = self.enc2(self.down1(e1), emb)
        e3 = self.enc3(self.down2(e2), emb)
        e4 = self.enc4(self.down3(e3), emb)

        h = self.down4(e4)
        h = self.mid1(h, emb)
        h = self.attention(h)
        h = self.mid2(h, emb)

        h = F.interpolate(h, size=e4.shape[-2:], mode="nearest")
        h = self.dec4(torch.cat([self.up4(h), e4], dim=1), emb)
        h = F.interpolate(h, size=e3.shape[-2:], mode="nearest")
        h = self.dec3(torch.cat([self.up3(h), e3], dim=1), emb)
        h = F.interpolate(h, size=e2.shape[-2:], mode="nearest")
        h = self.dec2(torch.cat([self.up2(h), e2], dim=1), emb)
        h = F.interpolate(h, size=e1.shape[-2:], mode="nearest")
        h = self.dec1(torch.cat([self.up1(h), e1], dim=1), emb)

        return self.output(h)

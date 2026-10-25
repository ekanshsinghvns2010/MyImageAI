"""MyImageAI V4: class-conditional residual U-Net with attention."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps, dim):
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(
            half, device=timesteps.device, dtype=torch.float32
        ) / max(half - 1, 1)
    )
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    emb = torch.cat([angles.sin(), angles.cos()], dim=1)
    return F.pad(emb, (0, 1)) if dim % 2 else emb


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, dropout=0.05):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.condition = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, condition):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.condition(condition)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        self.heads = heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(
            b, 3, self.heads, c // self.heads, h * w
        )
        q, k, v = qkv.unbind(1)
        q = q.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        weights = torch.matmul(q, k) * ((c // self.heads) ** -0.5)
        weights = weights.softmax(dim=-1)
        out = torch.matmul(weights, v).permute(0, 1, 3, 2)
        return x + self.proj(out.reshape(b, c, h, w))


class MyImageAIModelV4(nn.Module):
    NUM_CLASSES = 10
    NULL_CLASS = 10

    def __init__(self, in_channels=3, base_channels=64, time_dim=256,
                 num_classes=10, dropout=0.05):
        super().__init__()
        if base_channels % 8:
            raise ValueError("base_channels must be divisible by 8")
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.time_dim = time_dim
        self.num_classes = num_classes
        c1, c2, c3 = base_channels, 2 * base_channels, 3 * base_channels

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.class_embedding = nn.Embedding(num_classes + 1, time_dim)
        self.input_conv = nn.Conv2d(in_channels, c1, 3, padding=1)
        self.enc1 = ResBlock(c1, c1, time_dim, dropout)
        self.down1 = nn.Conv2d(c1, c2, 4, stride=2, padding=1)
        self.enc2 = ResBlock(c2, c2, time_dim, dropout)
        self.attn2 = AttentionBlock(c2)
        self.down2 = nn.Conv2d(c2, c3, 4, stride=2, padding=1)
        self.mid1 = ResBlock(c3, c3, time_dim, dropout)
        self.mid_attn = AttentionBlock(c3)
        self.mid2 = ResBlock(c3, c3, time_dim, dropout)
        self.dec2 = ResBlock(c3 + c2, c2, time_dim, dropout)
        self.dec_attn2 = AttentionBlock(c2)
        self.dec1 = ResBlock(c2 + c1, c1, time_dim, dropout)
        self.output_norm = nn.GroupNorm(8, c1)
        self.output_conv = nn.Conv2d(c1, in_channels, 3, padding=1)

    def forward(self, images, timesteps, labels):
        if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
            raise ValueError("labels must have shape [batch_size]")
        cond = self.time_mlp(timestep_embedding(timesteps, self.time_dim))
        cond = cond + self.class_embedding(labels.long())
        x1 = self.enc1(self.input_conv(images), cond)
        x2 = self.attn2(self.enc2(self.down1(x1), cond))
        x3 = self.mid1(self.down2(x2), cond)
        x3 = self.mid_attn(x3)
        x3 = self.mid2(x3, cond)
        x = F.interpolate(x3, size=x2.shape[-2:], mode="nearest")
        x = self.dec_attn2(self.dec2(torch.cat([x, x2], 1), cond))
        x = F.interpolate(x, size=x1.shape[-2:], mode="nearest")
        x = self.dec1(torch.cat([x, x1], 1), cond)
        return self.output_conv(F.silu(self.output_norm(x)))

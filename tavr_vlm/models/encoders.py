from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_positions(length: int, dim: int, device: torch.device) -> torch.Tensor:
    pe = torch.zeros(length, dim, device=device)
    position = torch.arange(0, length, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe


class CT3DEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, patch_size: tuple[int, int, int] = (8, 16, 16),
                 embed_dim: int = 512, depth: int = 4, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.patch = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B,C,D,H,W]
        tokens = self.patch(x).flatten(2).transpose(1, 2).contiguous()
        pos = sinusoidal_positions(tokens.shape[1], tokens.shape[2], tokens.device).unsqueeze(0)
        tokens = self.encoder(tokens + pos)
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        return tokens, pooled


class EchoTemporalEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, embed_dim: int = 256, temporal_layers: int = 2,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, embed_dim), nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=temporal_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C,H,W]
        b, t, c, h, w = x.shape
        frames = self.frame_encoder(x.reshape(b * t, c, h, w)).reshape(b, t, -1)
        pos = sinusoidal_positions(t, frames.shape[-1], frames.device).unsqueeze(0)
        out = self.temporal(frames + pos)
        return self.norm(out.mean(dim=1))


class TabularEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.extend([nn.LayerNorm(dims[i + 1]), nn.GELU(), nn.Dropout(dropout)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class FusionEncoder(nn.Module):
    def __init__(self, ct_dim: int, echo_dim: int, tab_dim: int, hidden_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ct_dim + echo_dim + tab_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )

    def forward(self, ct: torch.Tensor, echo: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([ct, echo, tab], dim=-1))

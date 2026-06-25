from __future__ import annotations

import torch
from torch import nn

from tavr_vlm.models.encoders import sinusoidal_positions


class TransformerReportDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 512, num_layers: int = 4,
                 num_heads: int = 8, ff_dim: int = 2048, dropout: float = 0.1,
                 pad_token_id: int = 0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    @staticmethod
    def causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, input_ids: torch.Tensor, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # input_ids: [B,T], memory: [B,N,D]
        t = input_ids.shape[1]
        x = self.embedding(input_ids)
        x = x + sinusoidal_positions(t, x.shape[-1], x.device).unsqueeze(0)
        padding = input_ids.eq(self.pad_token_id)
        hidden = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=self.causal_mask(t, input_ids.device),
            tgt_key_padding_mask=padding,
        )
        hidden = self.norm(hidden)
        return self.lm_head(hidden), hidden

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class RCGAModule(nn.Module):
    def __init__(self, hidden_dim: int, visual_dim: int, risk_classes: int = 3, prototype_dim: int = 512,
                 topk_ratio: float = 0.2, disable_risk_prototypes: bool = False,
                 disable_purification: bool = False, stop_gradient_global: bool = True) -> None:
        super().__init__()
        self.risk_classes = risk_classes
        self.topk_ratio = float(topk_ratio)
        self.disable_risk_prototypes = disable_risk_prototypes
        self.disable_purification = disable_purification
        self.stop_gradient_global = stop_gradient_global
        self.risk_head = nn.Linear(hidden_dim, risk_classes)
        self.risk_prototypes = nn.Parameter(torch.randn(risk_classes, prototype_dim) * 0.02)
        self.direct_risk = nn.Linear(hidden_dim, prototype_dim)
        self.q = nn.Linear(prototype_dim, prototype_dim)
        self.k = nn.Linear(visual_dim, prototype_dim)
        self.visual_to_proto = nn.Linear(visual_dim, prototype_dim) if visual_dim != prototype_dim else nn.Identity()
        self.proto_to_visual = nn.Linear(prototype_dim, visual_dim) if visual_dim != prototype_dim else nn.Identity()

    def compute_bottleneck(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.risk_head(h)
        p = F.softmax(logits, dim=-1)
        if self.disable_risk_prototypes:
            z = self.direct_risk(h)
        else:
            z = p @ self.risk_prototypes
        return logits, z

    def risk_to_region(self, z_risk: torch.Tensor, v_visual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q(z_risk).unsqueeze(1)  # [B,1,D]
        k = self.k(v_visual)             # [B,N,D]
        scores = torch.matmul(q, k.transpose(1, 2)).squeeze(1) / (k.shape[-1] ** 0.5)
        m_global = F.softmax(scores, dim=-1)
        b_global = self.topk_support(m_global, self.topk_ratio)
        if self.disable_purification:
            v_act = v_visual
        else:
            v_act = v_visual * m_global.unsqueeze(-1)
        return m_global, b_global, v_act

    @staticmethod
    def topk_support(mask: torch.Tensor, ratio: float) -> torch.Tensor:
        b, n = mask.shape
        k = max(1, int(round(n * ratio)))
        idx = torch.topk(mask, k=k, dim=-1).indices
        out = torch.zeros_like(mask)
        out.scatter_(1, idx, 1.0)
        return out

    def token_grounding(self, hidden: torch.Tensor, v_visual: torch.Tensor, b_global: torch.Tensor,
                        eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden: [B,T,D_dec], v_visual: [B,N,D_vis]
        h = hidden
        if h.shape[-1] != self.risk_prototypes.shape[-1]:
            # Lazy projection would be brittle; enforce dimensions at model construction.
            raise RuntimeError("Decoder hidden dim must match R-CGA prototype dim for token grounding.")
        q = self.q(h)
        k = self.k(v_visual)
        scores = torch.matmul(q, k.transpose(1, 2)) / (k.shape[-1] ** 0.5)
        m_t = F.softmax(scores, dim=-1)
        support = b_global.detach() if self.stop_gradient_global else b_global
        m_tilde = m_t * support.unsqueeze(1)
        m_tilde = m_tilde / m_tilde.sum(dim=-1, keepdim=True).clamp_min(eps)
        return m_t, m_tilde

    def forward(self, h: torch.Tensor, v_visual: torch.Tensor) -> dict[str, torch.Tensor]:
        risk_logits, z_risk = self.compute_bottleneck(h)
        m_global, b_global, v_act = self.risk_to_region(z_risk, v_visual)
        return {
            "risk_logits": risk_logits,
            "z_risk": z_risk,
            "m_global": m_global,
            "b_global": b_global,
            "v_act": v_act,
        }

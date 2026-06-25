from __future__ import annotations

import torch
import torch.nn.functional as F


def support_projected_causal_loss(m_t: torch.Tensor, m_tilde: torch.Tensor, m_global: torch.Tensor,
                                  b_global: torch.Tensor, entity_token_mask: torch.Tensor,
                                  outside_penalty_alpha: float = 1.0, eps: float = 1e-8,
                                  stop_gradient_global: bool = True) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Support-projected causal consistency loss.

    Args:
        m_t: Raw token attention, [B,T,N].
        m_tilde: Support-projected token attention, [B,T,N].
        m_global: Global risk mask, [B,N].
        b_global: Binary TopK support mask, [B,N].
        entity_token_mask: Boolean mask over decoder input positions, [B,T].
    """
    if entity_token_mask.shape[1] != m_t.shape[1]:
        entity_token_mask = entity_token_mask[:, : m_t.shape[1]]
    ent = entity_token_mask.bool()
    if ent.sum() == 0:
        zero = m_t.sum() * 0.0
        return zero, {"kl": zero.detach(), "outside": zero.detach(), "entity_tokens": torch.tensor(0, device=m_t.device)}

    global_target = m_global.detach() if stop_gradient_global else m_global
    support_target = b_global.detach() if stop_gradient_global else b_global
    log_mtilde = torch.log(m_tilde.clamp_min(eps))
    log_global = torch.log(global_target.unsqueeze(1).clamp_min(eps))
    kl_per_token = (m_tilde * (log_mtilde - log_global)).sum(dim=-1)
    kl = kl_per_token[ent].mean()

    outside_mass = (m_t * (1.0 - support_target.unsqueeze(1))).sum(dim=-1)
    outside = outside_mass[ent].mean()
    loss = kl + outside_penalty_alpha * outside
    return loss, {"kl": kl.detach(), "outside": outside.detach(), "entity_tokens": ent.sum().detach()}

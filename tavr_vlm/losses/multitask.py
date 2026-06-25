from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from tavr_vlm.losses.causal import support_projected_causal_loss


class TAVRMultiTaskLoss(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        lcfg = cfg["loss"]
        mcfg = cfg["model"]
        self.lambda_risk = float(lcfg.get("lambda_risk", 1.0))
        self.lambda_lm = float(lcfg.get("lambda_lm", 1.0))
        self.lambda_rec = float(lcfg.get("lambda_rec", 0.0))
        self.lambda_causal = float(lcfg.get("lambda_causal", 0.0))
        self.outside_alpha = float(lcfg.get("outside_penalty_alpha", 1.0))
        self.pad_token_id = int(mcfg.get("pad_token_id", 0))
        self.stop_gradient_global = bool(mcfg.get("rcga", {}).get("stop_gradient_global", True))
        self.label_smoothing = float(lcfg.get("label_smoothing", 0.0))

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        labels = batch["report_ids"][:, 1:].contiguous()
        lm_logits = outputs["lm_logits"]
        lm_loss = F.cross_entropy(
            lm_logits.reshape(-1, lm_logits.shape[-1]), labels.reshape(-1),
            ignore_index=self.pad_token_id, label_smoothing=self.label_smoothing,
        )
        risk_loss = F.cross_entropy(outputs["risk_logits"], batch["risk"])
        rec_loss = outputs["recommendation_logits"].sum() * 0.0
        if batch.get("recommendation") is not None and outputs["recommendation_logits"].numel() > 0:
            rec_loss = F.binary_cross_entropy_with_logits(outputs["recommendation_logits"], batch["recommendation"])
        ent_mask = batch["entity_token_mask"][:, :-1].contiguous()
        causal_loss, causal_stats = support_projected_causal_loss(
            outputs["m_t"], outputs["m_tilde"], outputs["m_global"], outputs["b_global"], ent_mask,
            outside_penalty_alpha=self.outside_alpha, stop_gradient_global=self.stop_gradient_global,
        )
        total = (self.lambda_lm * lm_loss + self.lambda_risk * risk_loss +
                 self.lambda_rec * rec_loss + self.lambda_causal * causal_loss)
        stats = {
            "loss": float(total.detach().cpu()),
            "loss_lm": float(lm_loss.detach().cpu()),
            "loss_risk": float(risk_loss.detach().cpu()),
            "loss_rec": float(rec_loss.detach().cpu()),
            "loss_causal": float(causal_loss.detach().cpu()),
            "loss_causal_kl": float(causal_stats["kl"].detach().cpu()),
            "loss_causal_outside": float(causal_stats["outside"].detach().cpu()),
        }
        return total, stats

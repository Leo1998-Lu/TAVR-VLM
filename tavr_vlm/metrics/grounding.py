from __future__ import annotations

import torch


def _to_support(mask: torch.Tensor, topk_ratio: float = 0.2) -> torch.Tensor:
    if mask.dtype == torch.bool or set(torch.unique(mask.detach().cpu()).tolist()).issubset({0, 1}):
        return mask.bool()
    flat = mask.flatten()
    k = max(1, int(round(flat.numel() * topk_ratio)))
    idx = torch.topk(flat, k).indices
    out = torch.zeros_like(flat, dtype=torch.bool)
    out[idx] = True
    return out.reshape_as(mask)


def miou_from_masks(pred_masks: torch.Tensor, roi_masks: torch.Tensor, topk_ratio: float = 0.2) -> float:
    """Compute mean IoU for predicted token masks and ROI masks.

    pred_masks: [B,T,N] or [B,N]. roi_masks: [B,N] or broadcastable spatial supports.
    """
    if pred_masks.ndim == 3:
        pred_masks = pred_masks.max(dim=1).values
    scores = []
    for pred, roi in zip(pred_masks, roi_masks):
        p = _to_support(pred, topk_ratio).flatten()
        r = _to_support(roi, topk_ratio).flatten()
        if p.numel() != r.numel():
            r = torch.nn.functional.interpolate(r.float().view(1, 1, -1), size=p.numel(), mode="nearest").view(-1).bool()
        inter = (p & r).sum().float()
        union = (p | r).sum().float().clamp_min(1.0)
        scores.append((inter / union).item())
    return float(sum(scores) / max(1, len(scores)))

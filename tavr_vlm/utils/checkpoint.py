from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None,
                    epoch: int, metrics: dict[str, Any], config: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "config": config,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None,
                    map_location: str | torch.device = "cpu") -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt

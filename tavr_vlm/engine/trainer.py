from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tavr_vlm.losses import TAVRMultiTaskLoss
from tavr_vlm.utils.checkpoint import save_checkpoint


def build_optimizer(model: torch.nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    ocfg = cfg.get("optimizer", {})
    if ocfg.get("name", "adamw").lower() != "adamw":
        raise ValueError("Only AdamW is implemented in the reference training code")
    return torch.optim.AdamW(
        model.parameters(), lr=float(ocfg.get("lr", 1e-4)), weight_decay=float(ocfg.get("weight_decay", 1e-2)),
        betas=tuple(ocfg.get("betas", [0.9, 0.999])),
    )


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict[str, Any], steps_per_epoch: int):
    scfg = cfg.get("scheduler", {})
    if scfg.get("name", "cosine").lower() != "cosine":
        return None
    epochs = int(cfg.get("run", {}).get("epochs", 80))
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = int(scfg.get("warmup_epochs", 5) * steps_per_epoch)
    min_lr = float(scfg.get("min_lr", 1e-6))
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        import math
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        # LambdaLR cannot express absolute min_lr per group exactly; approximate via ratio to first group.
        return max(min_lr / max(base_lrs[0], 1e-12), cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    def __init__(self, model: torch.nn.Module, cfg: dict[str, Any], output_dir: str | Path, device: torch.device) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.loss_fn = TAVRMultiTaskLoss(cfg)
        self.optimizer = build_optimizer(model, cfg)
        self.scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("amp", True)) and device.type == "cuda")
        self.grad_clip = float(cfg.get("optimizer", {}).get("grad_clip_norm", 0.0))

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def train_epoch(self, loader: DataLoader, scheduler=None) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}
        count = 0
        pbar = tqdm(loader, desc="train", leave=False)
        for batch in pbar:
            batch = self._to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                outputs = self.model(batch)
                loss, stats = self.loss_fn(outputs, batch)
            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if scheduler is not None:
                scheduler.step()
            for k, v in stats.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            count += 1
            pbar.set_postfix(loss=stats.get("loss", 0.0))
        return {k: v / max(1, count) for k, v in totals.items()}

    @torch.no_grad()
    def validate_loss(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        totals: dict[str, float] = {}
        count = 0
        for batch in tqdm(loader, desc="val", leave=False):
            batch = self._to_device(batch)
            outputs = self.model(batch)
            _, stats = self.loss_fn(outputs, batch)
            for k, v in stats.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            count += 1
        return {"val_" + k: v / max(1, count) for k, v in totals.items()}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> None:
        scheduler = build_scheduler(self.optimizer, self.cfg, len(train_loader))
        run_cfg = self.cfg.get("run", {})
        epochs = int(run_cfg.get("epochs", 80))
        monitor = run_cfg.get("monitor", "val_loss")
        mode = run_cfg.get("monitor_mode", "min")
        best = None
        for epoch in range(1, epochs + 1):
            train_stats = self.train_epoch(train_loader, scheduler)
            metrics = {"epoch": epoch, **train_stats}
            if val_loader is not None:
                metrics.update(self.validate_loss(val_loader))
            score = metrics.get(monitor)
            is_best = False
            if score is not None:
                is_best = best is None or (score > best if mode == "max" else score < best)
                if is_best:
                    best = score
                    save_checkpoint(self.output_dir / "best.pt", self.model, self.optimizer, epoch, metrics, self.cfg)
            if epoch % int(run_cfg.get("save_every", 10)) == 0:
                save_checkpoint(self.output_dir / f"epoch_{epoch:04d}.pt", self.model, self.optimizer, epoch, metrics, self.cfg)
            save_checkpoint(self.output_dir / "last.pt", self.model, self.optimizer, epoch, metrics, self.cfg)
            print(metrics)

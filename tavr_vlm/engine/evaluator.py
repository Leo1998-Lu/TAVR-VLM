from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from tavr_vlm.data.vocab import WordVocab
from tavr_vlm.metrics import hallucination_rate, miou_from_masks, report_metrics, risk_metrics
from tavr_vlm.utils.io import load_json, save_json


class Evaluator:
    def __init__(self, model: torch.nn.Module, cfg: dict[str, Any], vocab: WordVocab, device: torch.device) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.vocab = vocab
        self.device = device
        ontology_path = cfg.get("paths", {}).get("entity_ontology")
        self.ontology = []
        if ontology_path and Path(ontology_path).exists():
            self.ontology = load_json(ontology_path).get("entities", [])

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, Any]:
        self.model.eval()
        preds: list[str] = []
        refs: list[str] = []
        y_true: list[int] = []
        y_prob: list[list[float]] = []
        evidence_entities: list[list[str]] = []
        miou_scores: list[float] = []
        records: list[dict[str, Any]] = []
        for batch in tqdm(loader, desc="eval", leave=False):
            batch_dev = self._to_device(batch)
            out = self.model.generate(batch_dev)
            ids = out["generated_ids"].detach().cpu().tolist()
            batch_preds = [self.vocab.decode(seq) for seq in ids]
            preds.extend(batch_preds)
            refs.extend(batch["report_text"])
            probs = F.softmax(out["risk_logits"], dim=-1).detach().cpu().tolist()
            y_prob.extend(probs)
            y_true.extend(batch["risk"].cpu().tolist())
            for ev, ex, ref in zip(batch["evidence_entities"], batch["expert_entities"], batch["report_text"]):
                support = ev or ex
                if not support and self.ontology:
                    from tavr_vlm.metrics.hallucination import extract_entities
                    support = list(extract_entities(ref, self.ontology))
                evidence_entities.append(support)
            if "roi" in batch:
                miou_scores.append(miou_from_masks(out["m_tilde"].detach().cpu(), batch["roi"].cpu(),
                                                   topk_ratio=self.cfg["model"]["rcga"].get("topk_ratio", 0.2)))
            for pid, pred in zip(batch["patient_id"], batch_preds):
                records.append({"patient_id": pid, "prediction": pred})
        metrics: dict[str, Any] = {}
        metrics.update(risk_metrics(y_true, y_prob))
        metrics.update(report_metrics(preds, refs))
        if self.ontology:
            hr, hr_counts = hallucination_rate(preds, evidence_entities, self.ontology)
            metrics["hallucination_rate"] = hr
            metrics.update({"hallucination_" + k: v for k, v in hr_counts.items()})
        if miou_scores:
            metrics["miou"] = float(sum(miou_scores) / len(miou_scores))
        metrics["num_cases"] = len(preds)
        metrics["predictions"] = records
        return metrics

    def save(self, metrics: dict[str, Any], path: str | Path) -> None:
        save_json(metrics, path)

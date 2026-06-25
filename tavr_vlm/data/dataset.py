from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from tavr_vlm.data.vocab import WordVocab
from tavr_vlm.utils.io import load_array, load_json, load_text


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s.startswith("["):
            return [str(v).strip() for v in json.loads(s)]
        sep = ";" if ";" in s else ","
        return [v.strip() for v in s.split(sep) if v.strip()]
    return [str(value)]


def _center_crop_or_pad(x: torch.Tensor, target: Sequence[int]) -> torch.Tensor:
    target = tuple(int(t) for t in target)
    if tuple(x.shape) == target:
        return x
    # Pad first to at least target size.
    pads: list[int] = []
    for cur, tgt in zip(reversed(x.shape), reversed(target)):
        total = max(0, tgt - cur)
        pads.extend([total // 2, total - total // 2])
    if any(pads):
        x = F.pad(x, pads)
    # Center crop.
    slices = []
    for cur, tgt in zip(x.shape, target):
        start = max(0, (cur - tgt) // 2)
        slices.append(slice(start, start + tgt))
    return x[tuple(slices)]


def _normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.float()
    mean = x.mean()
    std = x.std().clamp_min(eps)
    return (x - mean) / std


class TAVRDataset(Dataset):
    def __init__(self, manifest: str | Path, split: str, vocab: WordVocab, config: dict[str, Any],
                 entity_terms: Sequence[str] | None = None) -> None:
        self.rows = [r for r in read_manifest(manifest) if str(r.get("split", "")).lower() == split.lower()]
        if not self.rows:
            raise ValueError(f"No rows found for split={split} in {manifest}")
        self.vocab = vocab
        self.config = config
        self.max_len = int(config["model"].get("max_report_len", 160))
        self.risk_label_map = config.get("data", {}).get("risk_label_map", {"l":0,"m":1,"h":2,"low":0,"medium":1,"high":2})
        self.entity_token_ids = vocab.ids_for_terms(entity_terms or [])
        self.recommendation_labels = []
        ontology_path = config.get("paths", {}).get("entity_ontology")
        if ontology_path and Path(ontology_path).exists():
            self.recommendation_labels = load_json(ontology_path).get("recommendation_labels", [])
        self.rec_to_id = {name: i for i, name in enumerate(self.recommendation_labels)}

    def __len__(self) -> int:
        return len(self.rows)

    def _load_ct(self, row: dict[str, Any]) -> torch.Tensor:
        x = load_array(row["ct_path"]).float()
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError(f"CT must have shape [D,H,W] or [C,D,H,W], got {tuple(x.shape)}")
        target = self.config.get("data", {}).get("ct_shape")
        if target:
            x = _center_crop_or_pad(x, target)
        if self.config.get("data", {}).get("normalize_ct", True):
            x = _normalize(x)
        return x

    def _load_echo(self, row: dict[str, Any]) -> torch.Tensor:
        x = load_array(row["echo_path"]).float()
        if x.ndim == 3:  # [T,H,W]
            x = x.unsqueeze(1)
        if x.ndim == 4 and x.shape[0] in (1, 3) and x.shape[1] > 4:  # [C,T,H,W]
            x = x.permute(1, 0, 2, 3).contiguous()
        if x.ndim != 4:
            raise ValueError(f"Echo must have shape [T,H,W] or [T,C,H,W], got {tuple(x.shape)}")
        target = self.config.get("data", {}).get("echo_shape")
        if target:
            x = _center_crop_or_pad(x, target)
        if self.config.get("data", {}).get("normalize_echo", True):
            x = _normalize(x)
        return x

    def _load_tabular(self, row: dict[str, Any]) -> torch.Tensor:
        if row.get("tabular_path"):
            path = Path(row["tabular_path"])
            if path.suffix == ".json":
                obj = load_json(path)
                if isinstance(obj, dict):
                    values = [float(v) for _, v in sorted(obj.items())]
                else:
                    values = [float(v) for v in obj]
                return torch.tensor(values, dtype=torch.float32)
            return load_array(path).float().flatten()
        columns = self.config.get("data", {}).get("tabular_columns", [])
        if not columns:
            raise ValueError("tabular_path is missing and data.tabular_columns is not configured")
        return torch.tensor([float(row[c]) for c in columns], dtype=torch.float32)

    def _report_text(self, row: dict[str, Any]) -> str:
        if row.get("report_text"):
            return str(row["report_text"])
        if row.get("report_path"):
            return load_text(row["report_path"])
        raise ValueError("Each row must contain report_text or report_path")

    def _rec_labels(self, row: dict[str, Any]) -> torch.Tensor:
        dim = int(self.config["model"].get("recommendation_dim", len(self.rec_to_id) or 0))
        y = torch.zeros(dim, dtype=torch.float32)
        for label in _parse_list(row.get("recommendation_labels")):
            if label in self.rec_to_id and self.rec_to_id[label] < dim:
                y[self.rec_to_id[label]] = 1.0
        return y

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        text = self._report_text(row)
        report_ids = torch.tensor(self.vocab.encode(text, add_special=True, max_len=self.max_len), dtype=torch.long)
        entity_mask = torch.tensor([int(i.item() in self.entity_token_ids) for i in report_ids], dtype=torch.bool)
        label = str(row["risk_label"]).lower()
        if label not in self.risk_label_map:
            raise ValueError(f"Unknown risk label {label}")
        item = {
            "patient_id": row.get("patient_id", str(idx)),
            "ct": self._load_ct(row),
            "echo": self._load_echo(row),
            "tabular": self._load_tabular(row),
            "risk": torch.tensor(int(self.risk_label_map[label]), dtype=torch.long),
            "report_ids": report_ids,
            "entity_token_mask": entity_mask,
            "recommendation": self._rec_labels(row),
            "report_text": text,
            "evidence_entities": _parse_list(row.get("evidence_entities")),
            "expert_entities": _parse_list(row.get("expert_entities")),
        }
        if row.get("roi_path"):
            item["roi"] = load_array(row["roi_path"]).float()
        return item


def collate_tavr(batch: list[dict[str, Any]], pad_id: int = 0) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["patient_id"] = [b["patient_id"] for b in batch]
    out["ct"] = torch.stack([b["ct"] for b in batch])
    out["echo"] = torch.stack([b["echo"] for b in batch])
    out["tabular"] = torch.stack([b["tabular"] for b in batch])
    out["risk"] = torch.stack([b["risk"] for b in batch])
    out["recommendation"] = torch.stack([b["recommendation"] for b in batch])
    max_len = max(b["report_ids"].numel() for b in batch)
    ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    entity_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["report_ids"].numel()
        ids[i, :n] = b["report_ids"]
        entity_mask[i, :n] = b["entity_token_mask"]
    out["report_ids"] = ids
    out["entity_token_mask"] = entity_mask
    out["report_text"] = [b["report_text"] for b in batch]
    out["evidence_entities"] = [b["evidence_entities"] for b in batch]
    out["expert_entities"] = [b["expert_entities"] for b in batch]
    if all("roi" in b for b in batch):
        out["roi"] = torch.stack([b["roi"] for b in batch])
    return out

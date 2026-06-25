#!/usr/bin/env python
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tavr_vlm.data.dataset import TAVRDataset, collate_tavr
from tavr_vlm.data.vocab import WordVocab
from tavr_vlm.engine import Trainer
from tavr_vlm.models import TAVRVLM
from tavr_vlm.utils.config import load_config, save_config
from tavr_vlm.utils.io import load_json
from tavr_vlm.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TAVR-VLM R-CGA on a real M3TAVR-style manifest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    vocab = WordVocab.load(args.vocab)
    cfg["model"]["vocab_size"] = len(vocab)
    cfg["model"]["pad_token_id"] = vocab.pad_id
    cfg["model"]["bos_token_id"] = vocab.bos_id
    cfg["model"]["eos_token_id"] = vocab.eos_id
    set_seed(int(cfg.get("seed", 2026)))

    entity_terms = []
    ontology_path = cfg.get("paths", {}).get("entity_ontology")
    if ontology_path and Path(ontology_path).exists():
        entity_terms = load_json(ontology_path).get("entities", [])

    train_ds = TAVRDataset(args.manifest, "train", vocab, cfg, entity_terms=entity_terms)
    val_ds = TAVRDataset(args.manifest, "val", vocab, cfg, entity_terms=entity_terms)
    collate = partial(collate_tavr, pad_id=vocab.pad_id)
    loader_cfg = cfg.get("data", {})
    train_loader = DataLoader(train_ds, batch_size=int(loader_cfg.get("batch_size", 2)), shuffle=True,
                              num_workers=int(cfg.get("num_workers", 8)), pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=int(loader_cfg.get("batch_size", 2)), shuffle=False,
                            num_workers=int(cfg.get("num_workers", 8)), pin_memory=True, collate_fn=collate)
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = TAVRVLM(cfg)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "resolved_config.yaml")
    trainer = Trainer(model, cfg, out_dir, device)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()

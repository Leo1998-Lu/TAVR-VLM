#!/usr/bin/env python
from __future__ import annotations

import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader

from tavr_vlm.data.dataset import TAVRDataset, collate_tavr
from tavr_vlm.data.vocab import WordVocab
from tavr_vlm.engine import Evaluator
from tavr_vlm.models import TAVRVLM
from tavr_vlm.utils.checkpoint import load_checkpoint
from tavr_vlm.utils.config import load_config
from tavr_vlm.utils.io import load_json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained TAVR-VLM checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    vocab = WordVocab.load(args.vocab)
    cfg["model"]["vocab_size"] = len(vocab)
    cfg["model"]["pad_token_id"] = vocab.pad_id
    cfg["model"]["bos_token_id"] = vocab.bos_id
    cfg["model"]["eos_token_id"] = vocab.eos_id
    entity_terms = []
    ontology_path = cfg.get("paths", {}).get("entity_ontology")
    if ontology_path and Path(ontology_path).exists():
        entity_terms = load_json(ontology_path).get("entities", [])
    ds = TAVRDataset(args.manifest, args.split, vocab, cfg, entity_terms=entity_terms)
    loader = DataLoader(ds, batch_size=int(cfg.get("data", {}).get("batch_size", 2)), shuffle=False,
                        num_workers=int(cfg.get("num_workers", 8)), pin_memory=True,
                        collate_fn=partial(collate_tavr, pad_id=vocab.pad_id))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = TAVRVLM(cfg)
    load_checkpoint(args.checkpoint, model, map_location=device)
    evaluator = Evaluator(model, cfg, vocab, device)
    metrics = evaluator.evaluate(loader)
    evaluator.save(metrics, args.output)
    printable = {k: v for k, v in metrics.items() if k != "predictions"}
    print(printable)


if __name__ == "__main__":
    main()

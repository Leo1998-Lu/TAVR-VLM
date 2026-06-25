#!/usr/bin/env python
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tavr_vlm.data.dataset import TAVRDataset, collate_tavr
from tavr_vlm.data.vocab import WordVocab
from tavr_vlm.models import TAVRVLM
from tavr_vlm.utils.checkpoint import load_checkpoint
from tavr_vlm.utils.config import load_config
from tavr_vlm.utils.io import load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Export R-CGA grounding masks for real patient records.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", required=True)
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
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=int(cfg.get("num_workers", 4)),
                        collate_fn=partial(collate_tavr, pad_id=vocab.pad_id))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = TAVRVLM(cfg)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.to(device).eval()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for batch in loader:
            pid = batch["patient_id"][0]
            batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model.generate(batch_dev)
            np.savez_compressed(
                out_dir / f"{pid}.npz",
                m_global=out["m_global"].detach().cpu().numpy(),
                b_global=out["b_global"].detach().cpu().numpy(),
                m_t=out["m_t"].detach().cpu().numpy(),
                m_tilde=out["m_tilde"].detach().cpu().numpy(),
                generated_ids=out["generated_ids"].detach().cpu().numpy(),
            )


if __name__ == "__main__":
    main()

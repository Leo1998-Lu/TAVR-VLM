#!/usr/bin/env python
from __future__ import annotations

import argparse
from tavr_vlm.data.dataset import read_manifest
from tavr_vlm.data.vocab import WordVocab
from tavr_vlm.utils.io import load_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Build report vocabulary from a training manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-size", type=int, default=None)
    args = parser.parse_args()
    texts = []
    for row in read_manifest(args.manifest):
        if str(row.get("split", "train")).lower() != "train":
            continue
        if row.get("report_text"):
            texts.append(row["report_text"])
        elif row.get("report_path"):
            texts.append(load_text(row["report_path"]))
    vocab = WordVocab.build(texts, min_freq=args.min_freq, max_size=args.max_size)
    vocab.save(args.output)
    print({"vocab_size": len(vocab), "output": args.output})


if __name__ == "__main__":
    main()

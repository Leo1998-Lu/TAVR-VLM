#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from tavr_vlm.data.dataset import read_manifest

REQUIRED = {"patient_id", "split", "ct_path", "echo_path", "tabular_path", "risk_label"}
TEXT_FIELDS = {"report_text", "report_path"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a real M3TAVR-style manifest without loading image tensors.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--check-paths", action="store_true")
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    if not rows:
        raise SystemExit("Manifest is empty")
    missing = []
    for i, row in enumerate(rows):
        miss = [k for k in REQUIRED if not row.get(k)]
        if not any(row.get(k) for k in TEXT_FIELDS):
            miss.append("report_text_or_report_path")
        if miss:
            missing.append((i, miss))
        if args.check_paths:
            for key in ["ct_path", "echo_path", "tabular_path", "report_path", "roi_path"]:
                if row.get(key) and not Path(row[key]).exists():
                    missing.append((i, [f"missing_path:{key}={row[key]}"]))
    if missing:
        for idx, miss in missing[:20]:
            print(f"row {idx}: {miss}")
        raise SystemExit(f"Manifest validation failed with {len(missing)} row issues")
    splits = sorted(set(str(r.get("split", "")).lower() for r in rows))
    print({"rows": len(rows), "splits": splits, "status": "ok"})


if __name__ == "__main__":
    main()

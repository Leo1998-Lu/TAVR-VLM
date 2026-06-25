#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tavr_vlm.data.dataset import read_manifest
from tavr_vlm.utils.io import load_array, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tabular risk-only baselines on real M3TAVR tabular data.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", choices=["xgboost", "random_forest"], default="random_forest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    label_map = {"l": 0, "low": 0, "m": 1, "mid": 1, "medium": 1, "h": 2, "high": 2}

    def make_xy(split: str):
        xs, ys = [], []
        for r in rows:
            if str(r.get("split", "")).lower() != split:
                continue
            xs.append(load_array(r["tabular_path"]).float().flatten().numpy())
            ys.append(label_map[str(r["risk_label"]).lower()])
        return np.stack(xs), np.array(ys)

    x_train, y_train = make_xy("train")
    x_test, y_test = make_xy("test")
    if args.model == "xgboost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss")
    else:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=500, random_state=2026, class_weight="balanced")
    clf.fit(x_train, y_train)
    prob = clf.predict_proba(x_test)
    from tavr_vlm.metrics.risk import risk_metrics
    metrics = risk_metrics(y_test, prob)
    save_json(metrics, args.output)
    print(metrics)


if __name__ == "__main__":
    main()

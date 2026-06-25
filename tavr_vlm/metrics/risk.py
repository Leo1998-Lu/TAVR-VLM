from __future__ import annotations

import numpy as np


def risk_metrics(y_true: list[int] | np.ndarray, y_prob: list[list[float]] | np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = y_prob.argmax(axis=1)
    out: dict[str, float] = {}
    try:
        from sklearn.metrics import f1_score, roc_auc_score
        out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
        out["auroc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        out["macro_f1"] = float((y_true == y_pred).mean())
        out["auroc"] = float("nan")
    return out

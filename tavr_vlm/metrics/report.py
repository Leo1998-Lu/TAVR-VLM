from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable

from tavr_vlm.data.vocab import WordVocab


def _lcs(a: list[str], b: list[str]) -> int:
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def rouge_l(preds: Iterable[str], refs: Iterable[str]) -> float:
    scores = []
    for p, r in zip(preds, refs):
        pt = WordVocab.tokenize(p)
        rt = WordVocab.tokenize(r)
        if not pt or not rt:
            scores.append(0.0)
            continue
        l = _lcs(pt, rt)
        prec = l / len(pt)
        rec = l / len(rt)
        scores.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(sum(scores) / max(1, len(scores)))


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1)))


def cider(preds: list[str], refs: list[str], max_n: int = 4) -> float:
    """Self-contained CIDEr-style TF-IDF n-gram similarity.

    This is intended for reproducible in-repository scoring. For exact COCO CIDEr,
    users may substitute pycocoevalcap with the same prediction/reference files.
    """
    ref_tokens = [WordVocab.tokenize(r) for r in refs]
    pred_tokens = [WordVocab.tokenize(p) for p in preds]
    scores = []
    for n in range(1, max_n + 1):
        df: defaultdict[tuple[str, ...], int] = defaultdict(int)
        for toks in ref_tokens:
            for ng in set(_ngrams(toks, n)):
                df[ng] += 1
        num_docs = max(1, len(ref_tokens))
        for pt, rt in zip(pred_tokens, ref_tokens):
            pc = _ngrams(pt, n)
            rc = _ngrams(rt, n)
            if not pc or not rc:
                scores.append(0.0)
                continue
            vocab = set(pc) | set(rc)
            pv, rv = [], []
            for ng in vocab:
                idf = math.log((num_docs + 1.0) / (df.get(ng, 0) + 1.0)) + 1.0
                pv.append(pc.get(ng, 0) * idf)
                rv.append(rc.get(ng, 0) * idf)
            dot = sum(a*b for a, b in zip(pv, rv))
            pn = math.sqrt(sum(a*a for a in pv))
            rn = math.sqrt(sum(a*a for a in rv))
            scores.append(0.0 if pn == 0 or rn == 0 else dot / (pn * rn))
    return float(sum(scores) / max(1, len(scores)))


def report_metrics(preds: list[str], refs: list[str]) -> dict[str, float]:
    return {"rouge_l": rouge_l(preds, refs), "cider": cider(preds, refs)}

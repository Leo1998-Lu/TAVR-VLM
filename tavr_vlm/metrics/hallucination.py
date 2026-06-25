from __future__ import annotations

import regex as re


def extract_entities(text: str, ontology: list[str]) -> set[str]:
    text_l = text.lower()
    found = set()
    for ent in ontology:
        ent_l = ent.lower()
        pattern = r"(?<!\w)" + re.escape(ent_l) + r"(?!\w)"
        if re.search(pattern, text_l):
            found.add(ent)
    return found


def hallucination_rate(preds: list[str], evidence_entities: list[list[str]], ontology: list[str]) -> tuple[float, dict[str, int]]:
    unsupported = 0
    total = 0
    for pred, evidence in zip(preds, evidence_entities):
        gen = extract_entities(pred, ontology)
        support = {e.lower() for e in evidence}
        for ent in gen:
            total += 1
            if ent.lower() not in support:
                unsupported += 1
    hr = unsupported / total if total else 0.0
    return float(hr), {"unsupported": unsupported, "total_entities": total}

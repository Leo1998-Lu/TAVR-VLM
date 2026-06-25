# Evaluation protocol

## Risk prediction

`AUROC` and `Macro-F1` are computed over low/medium/high risk classes.

## Report generation

`ROUGE-L` and `CIDEr` are computed against expert structured reports. The local CIDEr implementation is self-contained; if `pycocoevalcap` is installed, users may replace it for compatibility with external benchmarks.

## Hallucination rate

The hallucination rate is computed as:

```text
HR = unsupported generated clinical entities / all generated clinical entities
```

Entities are extracted with `metadata/tavr_entity_ontology.json`. A generated entity is supported if it appears in `evidence_entities`, `expert_entities`, or the expert report entity set, depending on the chosen evaluation protocol.

## Grounding

For R-CGA, mIoU uses the constrained support-projected token mask `M_tilde`. Closed-source models that do not expose comparable grounding maps should be reported as `N/A` for mIoU.

# TAVR-VLM Reproducibility Repository

This repository provides a PyTorch implementation of **TAVR-VLM: Risk-Conditioned Causal Grounding for Hallucination-Resistant Report Generation**. It implements the paper's Risk-Conditioned Causal Grounding Attention (R-CGA) pipeline:

1. **Risk bottleneck construction**: multimodal encoders produce a joint patient embedding, a classifier predicts `P_risk`, and a learnable risk prototype matrix forms `Z_risk = P_risk^T W`.
2. **Risk-to-region grounding**: `Z_risk` queries dense CT visual tokens to produce `M_global`; visual tokens are activated as `V_act,i = M_global,i * V_visual,i`.
3. **Region-to-word generation**: an autoregressive decoder generates a structured report while raw token grounding `M_t` is support-projected into `B_global = TopK_rho(M_global)` to obtain `M_tilde`.
4. **Support-projected causal consistency**: the loss aligns `M_tilde` with `M_global` and penalizes attention leakage outside `B_global`.


## Repository layout

```text
tavr_vlm_repro/
├── configs/                      # Reproduction and ablation configs
├── docs/                         # Data, training, evaluation, reproducibility notes
├── metadata/                     # Entity ontology template
├── scripts/                      # Training/evaluation utilities requiring real data
├── tavr_vlm/                     # Python package
│   ├── data/                     # Manifest dataset and vocabulary
│   ├── engine/                   # Trainer and evaluator
│   ├── losses/                   # Multi-task and causal losses
│   ├── metrics/                  # Risk/report/grounding/hallucination metrics
│   ├── models/                   # Encoders, R-CGA, decoder, full model
│   └── utils/                    # Config, I/O, checkpointing, reproducibility
└── pyproject.toml
```

## Installation

```bash
conda create -n tavr-vlm python=3.10 -y
conda activate tavr-vlm
pip install -e .
pip install -r requirements.txt
```

The implementation uses only open-source PyTorch components. If your institution uses Qwen-VL or InternVL backbones, integrate them through the model interface in `tavr_vlm/models/model.py` while keeping the R-CGA module unchanged.

## Data manifest

Create CSV or JSONL manifests for train/validation/test records. Each row must include:

```text
patient_id,split,ct_path,echo_path,tabular_path,report_text_or_report_path,risk_label
```

Optional columns:

```text
roi_path,recommendation_labels,evidence_entities,expert_entities
```

See `docs/DATA_FORMAT.md` for the required array shapes and accepted file formats.

## Reproducing the proposed model

Build the vocabulary from the training manifest:

```bash
python scripts/build_vocab.py \
  --manifest /path/to/train_manifest.csv \
  --output /path/to/vocab.json \
  --min-freq 2
```

Train R-CGA:

```bash
python scripts/train.py \
  --config configs/tavr_vlm_rcga.yaml \
  --manifest /path/to/all_manifest.csv \
  --vocab /path/to/vocab.json \
  --output-dir /path/to/experiments/tavr_vlm_rcga
```

Evaluate on the held-out test set:

```bash
python scripts/evaluate.py \
  --config configs/tavr_vlm_rcga.yaml \
  --manifest /path/to/all_manifest.csv \
  --vocab /path/to/vocab.json \
  --checkpoint /path/to/experiments/tavr_vlm_rcga/best.pt \
  --split test \
  --output /path/to/experiments/tavr_vlm_rcga/test_metrics.json
```

Export grounding masks for qualitative figures:

```bash
python scripts/export_grounding.py \
  --config configs/tavr_vlm_rcga.yaml \
  --manifest /path/to/all_manifest.csv \
  --vocab /path/to/vocab.json \
  --checkpoint /path/to/experiments/tavr_vlm_rcga/best.pt \
  --split test \
  --output-dir /path/to/experiments/tavr_vlm_rcga/grounding
```

## Ablations

The ablation configs correspond to the paper's ablation table:

```bash
configs/ablations/no_risk_prototypes.yaml
configs/ablations/no_purification.yaml
configs/ablations/no_causal_loss.yaml
configs/ablations/no_stop_gradient.yaml
```

Run them with the same `scripts/train.py` and `scripts/evaluate.py` entry points.

## Reproducibility notes

* All random seeds are set from the config.
* The train/validation/test split is read from the manifest and never created inside training code.
* ROI masks are used only for evaluation unless explicitly enabled in a config.
* Closed-source model evaluation is intentionally not implemented in code, because it depends on external APIs and non-reproducible model versions. Store those outputs in the same manifest format and evaluate them with `tavr_vlm.metrics` if needed.


# M3TAVR-style data format

The repository does not include data. Provide patient records through a CSV or JSONL manifest.

## Required columns

| Column | Description |
| --- | --- |
| `patient_id` | Unique patient identifier. |
| `split` | One of `train`, `val`, `test`. |
| `ct_path` | Path to 3D CT volume stored as `.npy`, `.npz`, `.pt`, `.nii`, or `.nii.gz`. |
| `echo_path` | Path to echocardiography clip stored as `.npy`, `.npz`, or `.pt`. |
| `tabular_path` | Path to tabular vector stored as `.npy`, `.pt`, or `.json`. |
| `risk_label` | One of `low`, `medium`, `high` or `l`, `m`, `h`. |
| `report_text` or `report_path` | Structured expert report text or path to `.txt`. |

## Optional columns

| Column | Description |
| --- | --- |
| `roi_path` | Expert ROI mask for grounding evaluation. It may be a token-support vector of shape `[N]` or a spatial mask that is resized/tokenized by downstream code. |
| `recommendation_labels` | JSON list, comma-separated list, or semicolon-separated list of structured recommendation labels. |
| `evidence_entities` | JSON list or semicolon-separated list of expert-verified evidence entities. Used for hallucination-rate computation. |
| `expert_entities` | Optional entity list extracted from the expert report. |

## Expected tensors

* CT: `[D, H, W]` or `[1, D, H, W]`.
* Echo clip: `[T, H, W]`, `[T, 1, H, W]`, or `[T, C, H, W]`.
* Tabular vector: `[F]`.
* Report: tokenized with `scripts/build_vocab.py`.

Input arrays are center-cropped/padded to the shapes specified in the config.

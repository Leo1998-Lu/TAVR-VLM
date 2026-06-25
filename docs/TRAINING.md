# Training pipline

1. Prepare patient-level manifest files with fixed `split` assignments.
2. Build the report vocabulary from the training split only.
3. Train using `scripts/train.py` with `configs/tavr_vlm_rcga.yaml`.
4. Select the checkpoint by validation CIDEr or validation hallucination rate as configured.
5. Evaluate once on the test split using `scripts/evaluate.py`.


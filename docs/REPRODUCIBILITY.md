# Reproducibility checklist

* Patient-level train/validation/test split is provided by the manifest.
* Random seed is fixed through `tavr_vlm.utils.seed.set_seed`.
* Mixed precision can be disabled by setting `amp: false`.
* Grounding ROIs are loaded only when present and are not used for training unless a custom loss is added.
* The model stores the exact config and vocabulary hash with each checkpoint.


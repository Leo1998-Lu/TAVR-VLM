from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_array(path: str | Path) -> torch.Tensor:
    path = Path(path)
    suffixes = "".join(path.suffixes)
    if suffixes.endswith(".npy"):
        arr = np.load(path)
        return torch.as_tensor(arr)
    if suffixes.endswith(".npz"):
        data = np.load(path)
        key = "arr_0" if "arr_0" in data else list(data.keys())[0]
        return torch.as_tensor(data[key])
    if suffixes.endswith(".pt") or suffixes.endswith(".pth"):
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            for key in ("array", "image", "tensor", "data"):
                if key in obj:
                    obj = obj[key]
                    break
        return obj if isinstance(obj, torch.Tensor) else torch.as_tensor(obj)
    if suffixes.endswith(".nii") or suffixes.endswith(".nii.gz"):
        import nibabel as nib
        return torch.as_tensor(nib.load(str(path)).get_fdata(dtype=np.float32))
    raise ValueError(f"Unsupported array file: {path}")


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

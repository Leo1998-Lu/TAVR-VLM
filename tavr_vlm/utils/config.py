from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def _deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in override.items():
        if k == "_base_":
            continue
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_update(dict(out[k]), v)
        else:
            out[k] = deepcopy(v)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "_base_" in cfg:
        base_path = (path.parent / cfg["_base_"]).resolve()
        base = load_config(base_path)
        cfg = _deep_update(base, cfg)
    return cfg


def save_config(cfg: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(cfg), f, sort_keys=False)


def get_by_path(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur

from __future__ import annotations

import os
from typing import Any, Dict, Iterable


def _resolve_path(value: str, *, base_dir: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(value))
    if not os.path.isabs(expanded):
        expanded = os.path.join(base_dir, expanded)
    return os.path.normpath(expanded)


def resolve_config_paths(
    raw: Dict[str, Any],
    *,
    config_path: str,
    keys: Iterable[str],
) -> Dict[str, Any]:
    out = dict(raw)
    base_dir = os.path.dirname(_resolve_path(config_path, base_dir=os.getcwd()))
    for key in keys:
        value = out.get(key)
        if isinstance(value, str) and value.strip():
            out[key] = _resolve_path(value, base_dir=base_dir)
    return out

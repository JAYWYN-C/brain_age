"""Small config loader used by the final main-model scripts.

PyYAML is preferred when installed. The fallback parser intentionally supports
only the simple YAML subset used by this repo's configs: flat key/value pairs,
booleans, null, numbers, strings, and one-line lists.
"""

from __future__ import annotations

import ast
import os
from typing import Any, Dict


def _strip_inline_comment(line: str) -> str:
    in_quote = False
    quote_char = ""
    for i, ch in enumerate(line):
        if ch in ("'", '"'):
            if not in_quote:
                in_quote = True
                quote_char = ch
            elif quote_char == ch:
                in_quote = False
        elif ch == "#" and not in_quote:
            return line[:i].rstrip()
    return line.rstrip()


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if value == "":
        return ""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    if (value.startswith("[") and value.endswith("]")) or (
        value.startswith("{") and value.endswith("}")
    ):
        return ast.literal_eval(value)
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("'\"")


def load_yaml_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        try:
            from public_configs import CONFIG_BY_BASENAME
        except ImportError:
            CONFIG_BY_BASENAME = {}
        fallback = CONFIG_BY_BASENAME.get(os.path.basename(path))
        if fallback is not None:
            return dict(fallback)

    try:
        import yaml  # type: ignore
    except ImportError:
        out: Dict[str, Any] = {}
        with open(path, encoding="utf-8-sig") as f:
            for raw_line in f:
                line = _strip_inline_comment(raw_line).strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                out[key.strip()] = _parse_scalar(value)
        return out

    with open(path, encoding="utf-8-sig") as f:
        loaded = yaml.safe_load(f)
    return loaded or {}

"""Small progress helpers for CAUEEG experiments.

The training scripts run on macOS, Windows terminals, and PyCharm consoles.
Keep tqdm optional so experiments still run in minimal Python environments.
"""

from __future__ import annotations

import os
from typing import Iterable, Mapping, Optional


_DISABLE_VALUES = {"0", "false", "off", "no"}


class _NoOpProgress:
    def __init__(self, total: Optional[int] = None, desc: str = "", unit: str = "it"):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.n = 0

    def update(self, n: int = 1) -> None:
        self.n += int(n)

    def set_description(self, desc: str) -> None:
        self.desc = desc

    def set_postfix(self, values: Optional[Mapping[str, object]] = None, **kwargs) -> None:
        return None

    def refresh(self) -> None:
        return None

    def close(self) -> None:
        return None


def progress_enabled() -> bool:
    value = os.environ.get("CAUEEG_PROGRESS", "1").strip().lower()
    return value not in _DISABLE_VALUES


def progress_verbose() -> bool:
    value = os.environ.get("CAUEEG_PROGRESS_VERBOSE", "0").strip().lower()
    return value not in _DISABLE_VALUES


def planned_epoch_steps(folds: Iterable[int], epochs: int) -> int:
    return len(tuple(folds)) * max(int(epochs), 0)


def _tqdm_kwargs(total: Optional[int], desc: str, unit: str) -> dict:
    return {
        "total": total,
        "desc": desc,
        "unit": unit,
        "dynamic_ncols": True,
        "ascii": os.name == "nt",
        "leave": True,
    }


def progress_bar(total: Optional[int], desc: str, unit: str = "it"):
    if not progress_enabled():
        return _NoOpProgress(total=total, desc=desc, unit=unit)
    try:
        from tqdm.auto import tqdm
    except Exception:
        return _NoOpProgress(total=total, desc=desc, unit=unit)
    return tqdm(**_tqdm_kwargs(total=total, desc=desc, unit=unit))


def progress_write(message: str) -> None:
    if progress_enabled():
        try:
            from tqdm.auto import tqdm

            tqdm.write(message)
            return
        except Exception:
            pass
    print(message, flush=True)


def progress_log(message: str) -> None:
    if progress_verbose():
        progress_write(message)

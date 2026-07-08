"""pid -> age lookup utilities.

The CAUEEG dataset npy (`caueeg_dataset_final.npy`) does not carry age, so age
is attached as a sidecar file: `caueeg_dataset_final_age_lookup.npy`.

Layout (structured array):
    dtype = [('pid','<i8'), ('age','<f4')]

This module deliberately does NOT touch the original dataset npy. The lookup
file is built once from the CAUEEG annotation source and read-only afterwards.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np


AGE_LOOKUP_DTYPE = np.dtype([("pid", "<i8"), ("age", "<f4")])


def lookup_path(dataset_path: str) -> str:
    """Default sidecar path next to the dataset npy."""
    base, _ = os.path.splitext(dataset_path)
    return base + "_age_lookup.npy"


def save_lookup(out_path: str, pid_to_age: Dict[int, float]) -> None:
    """Persist a pid->age mapping as a structured npy."""
    items = sorted(pid_to_age.items(), key=lambda kv: int(kv[0]))
    arr = np.zeros(len(items), dtype=AGE_LOOKUP_DTYPE)
    for i, (pid, age) in enumerate(items):
        arr[i]["pid"] = int(pid)
        arr[i]["age"] = float(age)
    np.save(out_path, arr)


def load_lookup(path: str) -> Dict[int, float]:
    """Load lookup npy into a plain dict for O(1) access."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"age lookup file not found: {path}. "
            "Build it once from CAUEEG annotation (see save_lookup)."
        )
    arr = np.load(path, allow_pickle=False)
    if arr.dtype != AGE_LOOKUP_DTYPE:
        raise ValueError(
            f"unexpected lookup dtype {arr.dtype}, expected {AGE_LOOKUP_DTYPE}"
        )
    return {int(r["pid"]): float(r["age"]) for r in arr}


def broadcast_to_epochs(
    pid_array: np.ndarray,
    pid_to_age: Dict[int, float],
    missing: str = "error",
) -> np.ndarray:
    """Map an epoch-level pid array to epoch-level age array.

    missing: 'error' (raise) or 'nan' (fill with NaN) or 'drop_mask'
        (returns (ages, valid_mask)).
    """
    pid_array = np.asarray(pid_array, dtype=np.int64)
    unique_pids, inv = np.unique(pid_array, return_inverse=True)
    age_by_unique = np.empty(unique_pids.size, dtype=np.float32)
    valid_by_unique = np.ones(unique_pids.size, dtype=bool)

    missing_pids = []
    for i, pid in enumerate(unique_pids):
        age = pid_to_age.get(int(pid))
        if age is None:
            valid_by_unique[i] = False
            age_by_unique[i] = np.nan
            missing_pids.append(int(pid))
        else:
            age_by_unique[i] = age

    if missing_pids and missing == "error":
        raise KeyError(
            f"{len(missing_pids)} pid(s) without age in lookup, e.g. {missing_pids[:5]}"
        )

    out = age_by_unique[inv]
    valid = valid_by_unique[inv]
    if missing == "drop_mask":
        return out, valid
    return out


def lookup_summary(pid_to_age: Dict[int, float]) -> Dict[str, float]:
    ages = np.fromiter(pid_to_age.values(), dtype=np.float32)
    return {
        "n_subjects": int(ages.size),
        "age_min": float(np.min(ages)) if ages.size else float("nan"),
        "age_max": float(np.max(ages)) if ages.size else float("nan"),
        "age_mean": float(np.mean(ages)) if ages.size else float("nan"),
        "age_std": float(np.std(ages)) if ages.size else float("nan"),
    }

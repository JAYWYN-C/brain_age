"""Dataset registry for the CAUEEG SSL experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple


HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DATASET_ROOT = os.path.join(WORKSPACE_ROOT, "데이터셋")
COMMON_AGE_LOOKUP = os.path.join(DATASET_ROOT, "caueeg_dataset_final_age_lookup.npy")


@dataclass(frozen=True)
class DatasetSpec:
    alias: str
    filename: str
    window_sec: float
    overlap_sec: float
    low_cut: float
    high_cut: float

    @property
    def path(self) -> str:
        return os.path.join(DATASET_ROOT, self.filename)

    @property
    def age_lookup(self) -> str:
        return COMMON_AGE_LOOKUP


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "final": DatasetSpec(
        alias="final",
        filename="caueeg_dataset_final.npy",
        window_sec=4.0,
        overlap_sec=0.0,
        low_cut=0.5,
        high_cut=45.0,
    ),
}


def dataset_aliases() -> Tuple[str, ...]:
    return tuple(DATASET_SPECS.keys())


def get_dataset(alias: str) -> DatasetSpec:
    if alias not in DATASET_SPECS:
        raise ValueError(f"unknown dataset alias {alias}; expected one of {dataset_aliases()}")
    return DATASET_SPECS[alias]

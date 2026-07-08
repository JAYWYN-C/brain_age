"""Datasets for brain-age SSL pretext + downstream 4-class.

The CAUEEG npy is loaded with mmap so per-epoch __getitem__ is cheap. Age is
joined via the sidecar lookup (see age_lookup.py).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from age_lookup import broadcast_to_epochs, load_lookup, lookup_path


# Label mapping: y==0 -> CN, 1 -> MCI, 2 -> AD, 3 -> non-AD Dementia
CN_LABEL = 0
INPUT_NORM_MODES = {"none", "channel_zscore", "channel_robust_zscore"}


@dataclass
class DatasetBundle:
    X: np.ndarray         # (N, C, L)  float32 mmap
    y: np.ndarray         # (N,)       int64
    pid: np.ndarray       # (N,)       int64
    age: np.ndarray       # (N,)       float32 (epoch-level, broadcast from pid lookup)
    age_valid: np.ndarray # (N,)       bool

    @property
    def n_channels(self) -> int:
        return int(self.X.shape[1])

    @property
    def seq_length(self) -> int:
        return int(self.X.shape[2])


def _read_structured(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(data_path, mmap_mode="r")
    if data.dtype.names is None:
        raise ValueError(
            f"expected structured array with fields (X,y,pid), got dtype {data.dtype}"
        )
    X = data["X"]
    y = np.asarray(data["y"]).astype(np.int64, copy=False)
    pid = np.asarray(data["pid"]).astype(np.int64, copy=False)
    return X, y, pid


def load_bundle(
    data_path: str,
    age_lookup_file: Optional[str] = None,
    require_age: bool = True,
) -> DatasetBundle:
    """Load X/y/pid + epoch-level age.

    require_age=True: missing age -> raise.
    require_age=False: missing age -> NaN; downstream code must filter.
    """
    X, y, pid = _read_structured(data_path)
    age_path = age_lookup_file or lookup_path(data_path)

    if not os.path.exists(age_path) and not require_age:
        # Allow caller to inspect dataset before lookup is built.
        age = np.full(len(y), np.nan, dtype=np.float32)
        valid = np.zeros(len(y), dtype=bool)
        return DatasetBundle(X=X, y=y, pid=pid, age=age, age_valid=valid)

    pid_to_age = load_lookup(age_path)
    age, valid = broadcast_to_epochs(
        pid, pid_to_age, missing="drop_mask"
    )
    if require_age and not valid.all():
        n_missing = int((~valid).sum())
        n_subj_missing = len(set(int(p) for p in pid[~valid]))
        raise KeyError(
            f"{n_missing} epochs across {n_subj_missing} subjects have no age"
        )
    return DatasetBundle(X=X, y=y, pid=pid, age=age, age_valid=valid)


def cn_indices(bundle: DatasetBundle) -> np.ndarray:
    return np.where((bundle.y == CN_LABEL) & bundle.age_valid)[0].astype(np.int64)


def select_indices(
    bundle: DatasetBundle,
    fold_idx: np.ndarray,
    *,
    cn_only: bool,
) -> np.ndarray:
    """Restrict a fold's index array to (optionally CN, age-valid) epochs."""
    mask = bundle.age_valid[fold_idx]
    if cn_only:
        mask &= (bundle.y[fold_idx] == CN_LABEL)
    return fold_idx[mask]


def normalize_eeg_input(
    x: torch.Tensor,
    mode: str = "none",
    clip_value: float = 0.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize one EEG epoch shaped (C, L).

    `channel_robust_zscore` uses per-channel median/MAD and falls back to
    standard deviation when a channel is nearly constant. Clipping happens in
    normalized units to keep rare spikes from destabilizing early conv layers.
    """
    mode = (mode or "none").lower()
    if mode not in INPUT_NORM_MODES:
        raise ValueError(f"input_norm must be one of {sorted(INPUT_NORM_MODES)}, got {mode!r}")
    if mode == "none":
        out = x
    elif mode == "channel_zscore":
        center = x.mean(dim=-1, keepdim=True)
        scale = x.std(dim=-1, unbiased=False, keepdim=True).clamp_min(eps)
        out = (x - center) / scale
    else:
        center = x.median(dim=-1, keepdim=True).values
        abs_dev = (x - center).abs()
        robust_scale = abs_dev.median(dim=-1, keepdim=True).values * 1.4826
        std_scale = x.std(dim=-1, unbiased=False, keepdim=True)
        scale = torch.where(robust_scale >= eps, robust_scale, std_scale).clamp_min(eps)
        out = (x - center) / scale

    if clip_value and clip_value > 0:
        out = out.clamp(min=-float(clip_value), max=float(clip_value))
    return out


# ---------------------------------------------------------------------------
# Torch datasets
# ---------------------------------------------------------------------------


class _IndexedEEG(Dataset):
    """Common base: holds index array + bundle, materializes torch tensors lazily.

    `transform` (callable Tensor->Tensor, applied to (C,L) signal) is applied
    inside __getitem__ after materialization. Pass None for eval/test loaders.
    """

    def __init__(
        self,
        bundle: DatasetBundle,
        indices: np.ndarray,
        transform=None,
        input_norm: str = "none",
        input_clip: float = 0.0,
    ):
        if indices.dtype != np.int64:
            indices = indices.astype(np.int64, copy=False)
        self.bundle = bundle
        self.indices = indices
        self.transform = transform
        self.input_norm = input_norm
        self.input_clip = float(input_clip or 0.0)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _x(self, i: int) -> torch.Tensor:
        # mmap row -> contiguous owned copy -> torch tensor
        x = np.array(self.bundle.X[i], dtype=np.float32, copy=True)
        x = torch.from_numpy(x)
        x = normalize_eeg_input(x, mode=self.input_norm, clip_value=self.input_clip)
        if self.transform is not None:
            x = self.transform(x)
        return x


class AgeRegressionDataset(_IndexedEEG):
    """Pretext: returns (x, age). Caller controls CN-only / fold filtering."""

    def __getitem__(self, k: int):
        i = int(self.indices[k])
        x = self._x(i)
        age = float(self.bundle.age[i])
        pid = int(self.bundle.pid[i])
        # Safety check: skip NaN ages (should not happen if age_valid mask is correct)
        if not np.isfinite(age):
            raise ValueError(
                f"Dataset __getitem__: epoch index {i} (k={k}) has non-finite age {age} "
                f"(pid={pid}). This should be filtered by age_valid mask earlier."
            )
        return x, torch.tensor(age, dtype=torch.float32), torch.tensor(pid, dtype=torch.long)


class FullAgeDataset(_IndexedEEG):
    """For pretext evaluation: returns (x, age, pid, y)."""

    def __getitem__(self, k: int):
        i = int(self.indices[k])
        x = self._x(i)
        return (
            x,
            torch.tensor(float(self.bundle.age[i]), dtype=torch.float32),
            torch.tensor(int(self.bundle.pid[i]), dtype=torch.long),
            torch.tensor(int(self.bundle.y[i]), dtype=torch.long),
        )


class DownstreamDataset(_IndexedEEG):
    """4-class. Returns (x, age, pid, y, pred_age, bag).

    pred_age and bag come from a precomputed epoch-level cache (np.ndarray
    aligned with bundle row index). When cache is None they are zero so V0/V1
    can share the same dataset class.
    """

    def __init__(
        self,
        bundle: DatasetBundle,
        indices: np.ndarray,
        pred_age_cache: Optional[np.ndarray] = None,
        transform=None,
        input_norm: str = "none",
        input_clip: float = 0.0,
    ):
        super().__init__(
            bundle,
            indices,
            transform=transform,
            input_norm=input_norm,
            input_clip=input_clip,
        )
        if pred_age_cache is not None:
            if pred_age_cache.shape[0] != bundle.X.shape[0]:
                raise ValueError(
                    f"pred_age_cache len {pred_age_cache.shape[0]} != bundle len {bundle.X.shape[0]}"
                )
            self.pred_age = pred_age_cache.astype(np.float32, copy=False)
        else:
            self.pred_age = None

    def __getitem__(self, k: int):
        i = int(self.indices[k])
        x = self._x(i)
        age = float(self.bundle.age[i])
        if self.pred_age is not None:
            pred = float(self.pred_age[i])
            bag = pred - age
        else:
            pred = 0.0
            bag = 0.0
        return (
            x,
            torch.tensor(age, dtype=torch.float32),
            torch.tensor(pred, dtype=torch.float32),
            torch.tensor(bag, dtype=torch.float32),
            torch.tensor(int(self.bundle.pid[i]), dtype=torch.long),
            torch.tensor(int(self.bundle.y[i]), dtype=torch.long),
        )


class SubjectBalancedBatchSampler(Sampler):
    """Yield batches with `subjects_per_batch` distinct subjects each.

    Why: with O(200) epochs/subject, uniform shuffling produces batches
    dominated by a single subject. BatchNorm running stats then track that
    subject's distribution and collapse at eval. This sampler picks K
    subjects per batch and draws batch_size/K positions from each.
    """

    def __init__(
        self,
        pids: np.ndarray,
        batch_size: int,
        subjects_per_batch: int,
        seed: int = 0,
    ):
        self.batch_size = int(batch_size)
        self.subjects_per_batch = int(subjects_per_batch)
        if self.subjects_per_batch <= 0:
            raise ValueError("subjects_per_batch must be > 0")
        if self.batch_size % self.subjects_per_batch != 0:
            raise ValueError(
                f"batch_size {self.batch_size} must be divisible by "
                f"subjects_per_batch {self.subjects_per_batch}"
            )
        self.per_subject = self.batch_size // self.subjects_per_batch

        groups: Dict[int, list] = {}
        for k, p in enumerate(pids):
            groups.setdefault(int(p), []).append(int(k))
        if len(groups) < self.subjects_per_batch:
            raise ValueError(
                f"only {len(groups)} subjects available, "
                f"need >= {self.subjects_per_batch}"
            )
        self.subjects = np.asarray(sorted(groups.keys()), dtype=np.int64)
        self.groups = {s: np.asarray(groups[s], dtype=np.int64) for s in self.subjects.tolist()}
        self.n_batches = int(len(pids) // self.batch_size)
        self._epoch = 0
        self._seed = int(seed)

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        rng = np.random.default_rng(self._seed + self._epoch)
        self._epoch += 1
        for _ in range(self.n_batches):
            chosen = rng.choice(self.subjects, size=self.subjects_per_batch, replace=False)
            batch = []
            for s in chosen:
                pool = self.groups[int(s)]
                replace = pool.size < self.per_subject
                pick = rng.choice(pool, size=self.per_subject, replace=replace)
                batch.extend(int(i) for i in pick)
            yield batch

"""5-fold subject-level splits + inner validation split for 0702 V0/V1.

Loading order:
  1. If `splits_path` is provided and exists -> use that npz.
  2. Else build deterministic subject-stratified folds locally.

Expected splits.npz layout (compatible with §2 spec):
    keys are 'fold0_train', 'fold0_test', 'fold1_train', 'fold1_test', ...
    values are int64 arrays of EPOCH indices.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def make_subject_stratified_splits(
    pid: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build epoch-index folds stratified by each subject's majority label."""
    pid = np.asarray(pid)
    y = np.asarray(y)
    subjects = np.unique(pid)
    rng = np.random.default_rng(seed)

    subject_labels = {}
    for subject in subjects:
        labels = y[pid == subject].astype(np.int64)
        subject_labels[int(subject)] = int(np.bincount(labels).argmax())

    fold_subjects = [[] for _ in range(n_splits)]
    for label in sorted(set(subject_labels.values())):
        label_subjects = np.asarray(
            [subject for subject in subjects if subject_labels[int(subject)] == label],
            dtype=np.int64,
        )
        rng.shuffle(label_subjects)
        for i, subject in enumerate(label_subjects):
            fold_subjects[i % n_splits].append(int(subject))

    all_subjects = set(int(subject) for subject in subjects)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for subjects_for_test in fold_subjects:
        test_subjects = set(subjects_for_test)
        train_subjects = all_subjects - test_subjects
        train_idx = np.where(np.isin(pid, list(train_subjects)))[0].astype(np.int64)
        test_idx = np.where(np.isin(pid, list(test_subjects)))[0].astype(np.int64)
        splits.append((train_idx, test_idx))
    return splits


def load_or_build_splits(
    pid: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
    splits_path: Optional[str] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return list of (train_idx, test_idx) per fold."""
    if splits_path and os.path.exists(splits_path):
        data = np.load(splits_path)
        out = []
        for k in range(n_splits):
            tr_key = f"fold{k}_train"
            te_key = f"fold{k}_test"
            if tr_key not in data or te_key not in data:
                raise KeyError(f"missing {tr_key}/{te_key} in {splits_path}")
            out.append((
                np.asarray(data[tr_key], dtype=np.int64),
                np.asarray(data[te_key], dtype=np.int64),
            ))
        return out
    return make_subject_stratified_splits(pid, y, n_splits=n_splits, seed=seed)


def save_splits(
    splits_path: str,
    splits: List[Tuple[np.ndarray, np.ndarray]],
) -> None:
    payload = {}
    for k, (tr, te) in enumerate(splits):
        payload[f"fold{k}_train"] = np.asarray(tr, dtype=np.int64)
        payload[f"fold{k}_test"] = np.asarray(te, dtype=np.int64)
    np.savez(splits_path, **payload)


def split_train_into_train_val(
    train_idx: np.ndarray,
    pid: np.ndarray,
    y: np.ndarray,
    val_frac: float,
    seed: int = 42,
    cn_only: bool = False,
    cn_label: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Carve a SUBJECT-level val split out of train.

    val_frac is fraction of subjects (not epochs). For pretext (cn_only=True)
    the split is over CN subjects only and the val subjects are guaranteed CN.
    """
    train_idx = np.asarray(train_idx, dtype=np.int64)
    if val_frac <= 0.0:
        return train_idx, np.array([], dtype=np.int64)

    pid_tr = pid[train_idx]
    if cn_only:
        mask_cn = (y[train_idx] == cn_label)
        cand_subjects = np.unique(pid_tr[mask_cn])
    else:
        cand_subjects = np.unique(pid_tr)

    if cand_subjects.size < 2:
        return train_idx, np.array([], dtype=np.int64)

    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(cand_subjects.size * val_frac)))
    n_val = min(n_val, cand_subjects.size - 1)
    val_subjects = rng.choice(cand_subjects, size=n_val, replace=False)
    val_set = set(int(s) for s in val_subjects)

    val_mask = np.array([int(p) in val_set for p in pid_tr], dtype=bool)
    val_idx = train_idx[val_mask]
    inner_train_idx = train_idx[~val_mask]
    return inner_train_idx, val_idx

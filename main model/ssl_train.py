"""Pretext training loop: CN-only brain-age regression, fold-aware.

Per spec §3.2:
  - CN(Normal) subjects only
  - Same 5-fold subject-independent split as downstream (no leakage)
  - Epoch-level training, subject-level evaluation
  - Loss: Huber + lambda * (subject-mean(pred) - age)^2

Per spec §3.4 (gate):
  subject-level Pearson r >= 0.5 and MAE <= 8 yrs to proceed.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from contextlib import nullcontext

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pathing import resolve_config_paths  # noqa: E402

import windows_patch  # noqa: F401,E402

from ssl_dataset import (  # noqa: E402
    AgeRegressionDataset,
    DatasetBundle,
    load_bundle,
    select_indices,
)
from ssl_model import BackboneConfig, BrainAgeRegressor  # noqa: E402
from splits import load_or_build_splits, split_train_into_train_val  # noqa: E402
from transforms import build_transforms  # noqa: E402
from config_io import load_yaml_config  # noqa: E402
from device_utils import select_device, seed_accelerator  # noqa: E402
from progress import planned_epoch_steps, progress_bar, progress_log, progress_write  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PretextConfig:
    data_path: str = ""
    age_lookup_path: str = ""
    out_dir: str = ""
    sweep: str = "main_basicdeep"
    splits_path: str = ""             # optional fixed splits.npz (E-11)
    folds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    n_splits: int = 5
    seed: int = 42

    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    huber_delta: float = 1.0
    subject_reg_lambda: float = 0.1
    grad_clip_norm: float = 0.0
    max_batch_loss_abort: float = 0.0
    log_sanity: bool = True
    sanity_batch_size: int = 32
    sanity_abort_mae: float = 0.0

    num_workers: int = 2
    pin_memory: bool = True

    # backbone
    backbone_type: str = "basic_deep_cnn"
    base_channels: int = 32
    embedding_dim: int = 128
    dropout: float = 0.3
    # Legacy fields kept so old config dictionaries still load.
    deep4_batch_norm: bool = True
    add_output_layernorm: bool = False
    drop_path: float = 0.1

    # validation + early stopping (C-5)
    val_frac: float = 0.1            # subject-level fraction inside train fold
    epoch_sample_frac: float = 0.0    # subject-wise epoch/window fraction for quick runs
    early_stop_patience: int = 0     # 0 disables
    select_by: str = "val"           # "val" | "test" — what determines `best`

    # augmentation (B-4) — dict of {channel_dropout, time_shift, gaussian_noise}
    augment: Optional[Dict] = None
    input_norm: str = "none"
    input_clip: float = 0.0

    # gate (read-only metadata, used by reporting)
    gate_pearson_r: float = 0.5
    gate_mae: float = 8.0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _aggregate_subject_level(
    pred_epoch: np.ndarray,
    age_epoch: np.ndarray,
    pid_epoch: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average epoch-level predictions per subject."""
    subjects, inv = np.unique(pid_epoch, return_inverse=True)
    sums = np.bincount(inv, weights=pred_epoch, minlength=subjects.size)
    counts = np.bincount(inv, minlength=subjects.size).astype(np.float64)
    pred_subj = sums / np.maximum(counts, 1.0)

    age_sums = np.bincount(inv, weights=age_epoch, minlength=subjects.size)
    age_subj = age_sums / np.maximum(counts, 1.0)
    return subjects, pred_subj.astype(np.float32), age_subj.astype(np.float32)


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom < 1e-12:
        return float("nan")
    return float((a * b).sum() / denom)


def regression_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    err = pred - true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "pearson_r": _pearson_r(pred.astype(np.float64), true.astype(np.float64)),
        "n": int(pred.size),
    }


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


def _make_loader(
    dataset, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )


def _representative_subject_indices(
    bundle: DatasetBundle,
    indices: np.ndarray,
    max_items: int,
) -> np.ndarray:
    if max_items <= 0 or indices.size == 0:
        return np.empty(0, dtype=np.int64)

    selected: List[int] = []
    seen = set()
    for raw_idx in indices:
        idx = int(raw_idx)
        pid = int(bundle.pid[idx])
        if pid in seen:
            continue
        seen.add(pid)
        selected.append(idx)
        if len(selected) >= max_items:
            break
    return np.asarray(selected, dtype=np.int64)


def _sample_epochs_per_subject(
    indices: np.ndarray,
    pid: np.ndarray,
    sample_frac: float,
    seed: int,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    sample_frac = float(sample_frac or 0.0)
    if sample_frac <= 0.0 or indices.size == 0:
        return indices

    pid = np.asarray(pid)
    indexed_pid = pid[indices]
    rng = np.random.default_rng(int(seed))
    selected: List[int] = []
    for subject in np.unique(indexed_pid):
        subj_idx = indices[indexed_pid == subject]
        keep_n = max(1, int(np.ceil(subj_idx.size * sample_frac)))
        if subj_idx.size > keep_n:
            subj_idx = rng.choice(subj_idx, size=keep_n, replace=False)
        selected.extend(int(i) for i in subj_idx)
    return np.asarray(sorted(selected), dtype=np.int64)


@torch.no_grad()
def _predict_epochs(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    age_mean: float = 0.0,
    age_std: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predicts in z-score space then denormalizes back to year scale."""
    model.eval()
    preds, ages, pids = [], [], []
    for x, age, pid in loader:
        x = x.to(device, non_blocking=True)
        out_z = model(x).detach().cpu().numpy()
        preds.append(out_z * age_std + age_mean)
        ages.append(age.numpy())
        pids.append(pid.numpy())
        del x
    result = (
        np.concatenate(preds).astype(np.float32),
        np.concatenate(ages).astype(np.float32),
        np.concatenate(pids).astype(np.int64),
    )
    del preds, ages, pids
    gc.collect()
    return result


def _huber_loss_parts(
    pred: torch.Tensor,
    age: torch.Tensor,
    pid: torch.Tensor,
    delta: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # MPS gives garbage loss (1e7..1e34) for this layout. Move pred/age to CPU
    # via contiguous().cpu() (forces a sync read) and compute MSE there to
    # diagnose. autograd will route gradient back to MPS params automatically.
    pred_cpu = pred.contiguous().float().cpu()
    age_cpu = age.contiguous().float().cpu()
    if not torch.isfinite(pred_cpu).all() or not torch.isfinite(age_cpu).all():
        bad_pred = int((~torch.isfinite(pred_cpu)).sum().item())
        bad_age = int((~torch.isfinite(age_cpu)).sum().item())
        progress_log(
            f"[debug] non-finite in pred={bad_pred}/{pred_cpu.numel()}, "
            f"age={bad_age}/{age_cpu.numel()}; pred dtype={pred.dtype}, "
            f"age dtype={age.dtype}"
        )
    diff = pred_cpu - age_cpu
    abs_diff = diff.abs()
    inner = abs_diff.clamp(max=delta)
    base = (0.5 * inner * inner + delta * (abs_diff - inner)).mean()
    progress_log(
        f"[debug] huber: pred[{pred_cpu.shape}]={pred_cpu.mean():.3f}±{pred_cpu.std():.3f} "
        f"age[{age_cpu.shape}]={age_cpu.mean():.3f}±{age_cpu.std():.3f} "
        f"diff_abs_max={abs_diff.max():.3f} base={base.item():.4g}"
    )
    if lam <= 0.0:
        # Keep base on CPU; .backward() crosses devices fine. Moving back to
        # MPS could re-trigger the same kernel bug.
        return base, base, torch.zeros((), device=base.device, dtype=base.dtype)

    # subject-level mean(pred) vs age penalty inside the batch
    uniq, inv = torch.unique(pid, return_inverse=True)
    n_subj = uniq.numel()
    sums = torch.zeros(n_subj, device=pred.device, dtype=pred.dtype).scatter_add_(
        0, inv, pred
    )
    cnts = torch.zeros(n_subj, device=pred.device, dtype=pred.dtype).scatter_add_(
        0, inv, torch.ones_like(pred)
    )
    age = age.to(device=pred.device, dtype=pred.dtype)

    age_sums = torch.zeros(n_subj, device=pred.device,
                           dtype=pred.dtype).scatter_add_(
        0, inv, age
    )
    pred_mean = sums / cnts.clamp(min=1.0)
    age_mean = age_sums / cnts.clamp(min=1.0)
    reg = ((pred_mean - age_mean) ** 2).mean()
    total = base + lam * reg
    return total, base, reg


def _huber_with_subject_reg(
    pred: torch.Tensor,
    age: torch.Tensor,
    pid: torch.Tensor,
    delta: float,
    lam: float,
) -> torch.Tensor:
    total, _, _ = _huber_loss_parts(pred, age, pid, delta, lam)
    return total


def _tensor_stats(t: torch.Tensor) -> Dict[str, float]:
    t = t.detach().float()
    return {
        "mean": float(t.mean().cpu().item()),
        "std": float(t.std(unbiased=False).cpu().item()),
        "min": float(t.min().cpu().item()),
        "max": float(t.max().cpu().item()),
    }


@torch.no_grad()
def _age_sanity_snapshot(
    model: nn.Module,
    batch,
    device: str,
    age_mean: float,
    age_std: float,
    delta: float,
    lam: float,
) -> Dict[str, Dict[str, float]]:
    """Compare train/eval predictions on one fixed non-aug batch.

    The model predicts age z-scores, but diagnostics are reported both in
    z-score space and denormalized year space so scale collapse is visible.
    """
    x, age, pid = batch
    was_training = model.training
    snapshots: Dict[str, Dict[str, float]] = {}

    age_z_cpu = (age.float() - age_mean) / age_std
    x = x.to(device, non_blocking=True)
    age = age.to(device, non_blocking=True)
    pid = pid.to(device, non_blocking=True)
    age_z = age_z_cpu.to(device, non_blocking=True)

    for mode in ("train", "eval"):
        model.train(mode == "train")
        pred_z = model(x)
        pred_age = pred_z * age_std + age_mean
        total, base, reg = _huber_loss_parts(pred_z, age_z, pid, delta, lam)
        pred_z_stats = _tensor_stats(pred_z)
        pred_age_stats = _tensor_stats(pred_age)
        true_age_stats = _tensor_stats(age)
        snapshots[mode] = {
            "z_loss": float(total.detach().cpu().item()),
            "base_loss": float(base.detach().cpu().item()),
            "subject_reg": float(reg.detach().cpu().item()),
            "pred_z_mean": pred_z_stats["mean"],
            "pred_z_std": pred_z_stats["std"],
            "pred_z_min": pred_z_stats["min"],
            "pred_z_max": pred_z_stats["max"],
            "pred_age_mean": pred_age_stats["mean"],
            "pred_age_std": pred_age_stats["std"],
            "pred_age_min": pred_age_stats["min"],
            "pred_age_max": pred_age_stats["max"],
            "true_age_mean": true_age_stats["mean"],
            "true_age_std": true_age_stats["std"],
            "true_age_min": true_age_stats["min"],
            "true_age_max": true_age_stats["max"],
            "mae_years": float((pred_age - age).abs().mean().cpu().item()),
        }

    model.train(was_training)
    return snapshots


def _format_sanity(mode: str, stats: Dict[str, float]) -> str:
    return (
        f"{mode}: z_loss={stats['z_loss']:.4f} "
        f"base={stats['base_loss']:.4f} reg={stats['subject_reg']:.4f} "
        f"pred_z={stats['pred_z_mean']:.2f}+/-{stats['pred_z_std']:.2f} "
        f"pred_age={stats['pred_age_mean']:.1f}+/-{stats['pred_age_std']:.1f} "
        f"[{stats['pred_age_min']:.1f},{stats['pred_age_max']:.1f}] "
        f"true_age={stats['true_age_mean']:.1f}+/-{stats['true_age_std']:.1f} "
        f"mae={stats['mae_years']:.1f}"
    )


def _raise_if_sanity_failed(
    sanity: Dict[str, Dict[str, float]],
    threshold_mae: float,
    fold_id: int,
    epoch: int,
) -> None:
    if threshold_mae <= 0.0 or not sanity:
        return
    eval_stats = sanity.get("eval", {})
    mae = float(eval_stats.get("mae_years", 0.0))
    if mae <= threshold_mae:
        return
    raise RuntimeError(
        f"[fold{fold_id}] sanity failed @ ep{epoch}: "
        f"eval MAE={mae:.2f} > {threshold_mae:.2f}, "
        f"pred_age={eval_stats.get('pred_age_mean'):.2f}+/-"
        f"{eval_stats.get('pred_age_std'):.2f}, "
        f"z_loss={eval_stats.get('z_loss'):.4f}. "
        "Stopping before downstream because pretext collapsed."
    )


def _raise_if_batch_loss_invalid(
    loss: torch.Tensor,
    pred: torch.Tensor,
    x: torch.Tensor,
    threshold: float,
    fold_id: int,
    epoch: int,
    batch_idx: int,
) -> None:
    loss_value = float(loss.detach().float().cpu().item())
    loss_bad = not np.isfinite(loss_value)
    if threshold > 0.0:
        loss_bad = loss_bad or loss_value > threshold
    if not loss_bad:
        return

    pred_stats = _tensor_stats(pred)
    x_absmax = float(x.detach().float().abs().max().cpu().item())
    raise RuntimeError(
        f"[fold{fold_id}] batch loss failed @ ep{epoch} batch{batch_idx}: "
        f"loss={loss_value:.4g}, threshold={threshold:.4g}, "
        f"pred_z={pred_stats['mean']:.3f}+/-{pred_stats['std']:.3f} "
        f"[{pred_stats['min']:.3f},{pred_stats['max']:.3f}], "
        f"x_absmax={x_absmax:.3f}. "
        "Stopping before optimizer update because pretext became unstable."
    )


def train_one_fold(
    cfg: PretextConfig,
    bundle: DatasetBundle,
    fold_id: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    device: str,
    progress=None,
) -> Dict:
    train_idx = select_indices(bundle, train_idx, cn_only=True)
    test_idx = select_indices(bundle, test_idx, cn_only=True)

    if train_idx.size == 0 or test_idx.size == 0:
        raise RuntimeError(
            f"Fold {fold_id}: empty CN split (train={train_idx.size}, test={test_idx.size})"
        )

    # MPS/CPU doesn't support pin_memory
    pin_memory = cfg.pin_memory and (device == "cuda")
    inner_train_idx, val_idx = split_train_into_train_val(
        train_idx, bundle.pid, bundle.y,
        val_frac=cfg.val_frac, seed=cfg.seed + fold_id, cn_only=True,
    )
    n_inner_raw = int(inner_train_idx.size)
    n_val_raw = int(val_idx.size)
    n_test_raw = int(test_idx.size)
    inner_train_idx = _sample_epochs_per_subject(
        inner_train_idx,
        bundle.pid,
        cfg.epoch_sample_frac,
        seed=cfg.seed + fold_id,
    )
    val_idx = _sample_epochs_per_subject(
        val_idx,
        bundle.pid,
        cfg.epoch_sample_frac,
        seed=cfg.seed + 10_000 + fold_id,
    )
    test_idx = _sample_epochs_per_subject(
        test_idx,
        bundle.pid,
        cfg.epoch_sample_frac,
        seed=cfg.seed + 20_000 + fold_id,
    )
    if cfg.epoch_sample_frac > 0.0:
        progress_log(
            f"[fold{fold_id}] epoch sampling: frac={cfg.epoch_sample_frac:.3f} "
            f"train {n_inner_raw}->{inner_train_idx.size}, "
            f"val {n_val_raw}->{val_idx.size}, "
            f"test {n_test_raw}->{test_idx.size}"
        )
    use_val = val_idx.size > 0
    if cfg.select_by == "val" and not use_val:
        # gracefully fall back when val_frac=0
        cfg_select = "test"
    else:
        cfg_select = cfg.select_by

    transform = build_transforms(cfg.augment)

    train_ds = AgeRegressionDataset(
        bundle,
        inner_train_idx,
        transform=transform,
        input_norm=cfg.input_norm,
        input_clip=cfg.input_clip,
    )
    eval_train_ds = AgeRegressionDataset(
        bundle,
        inner_train_idx,
        input_norm=cfg.input_norm,
        input_clip=cfg.input_clip,
    )  # no aug for eval
    test_ds = AgeRegressionDataset(
        bundle,
        test_idx,
        input_norm=cfg.input_norm,
        input_clip=cfg.input_clip,
    )

    train_loader = _make_loader(
        train_ds, cfg.batch_size, True, cfg.num_workers, pin_memory
    )
    eval_train_loader = _make_loader(
        eval_train_ds, cfg.batch_size, False, cfg.num_workers, pin_memory
    )
    test_loader = _make_loader(
        test_ds, cfg.batch_size, False, cfg.num_workers, pin_memory
    )
    val_loader = None
    if use_val:
        val_ds = AgeRegressionDataset(
            bundle,
            val_idx,
            input_norm=cfg.input_norm,
            input_clip=cfg.input_clip,
        )
        val_loader = _make_loader(
            val_ds, cfg.batch_size, False, cfg.num_workers, pin_memory
        )

    # Age z-score: estimated only from inner_train so val/test stay leakage-free.
    age_train = bundle.age[inner_train_idx].astype(np.float64)
    age_mean = float(np.mean(age_train))
    age_std = float(np.std(age_train))
    if age_std < 1e-3:
        age_std = 1.0
    progress_log(
        f"[fold{fold_id}] age stats from inner_train: "
        f"mean={age_mean:.2f}, std={age_std:.2f}, n={age_train.size} | "
        f"input_norm={cfg.input_norm}, input_clip={cfg.input_clip}, "
        f"grad_clip_norm={cfg.grad_clip_norm}, "
        f"max_batch_loss_abort={cfg.max_batch_loss_abort}, "
        f"code={os.path.abspath(__file__)}"
    )
    sanity_batch = None
    if cfg.log_sanity and cfg.sanity_batch_size > 0:
        sanity_idx = _representative_subject_indices(
            bundle,
            inner_train_idx,
            max_items=min(int(cfg.sanity_batch_size), int(cfg.batch_size)),
        )
        if sanity_idx.size > 0:
            sanity_ds = AgeRegressionDataset(
                bundle,
                sanity_idx,
                input_norm=cfg.input_norm,
                input_clip=cfg.input_clip,
            )
            sanity_loader = _make_loader(
                sanity_ds,
                batch_size=int(sanity_idx.size),
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
            sanity_batch = next(iter(sanity_loader))
            progress_log(
                f"[fold{fold_id}] sanity batch: "
                f"{sanity_idx.size} epoch(s), "
                f"{np.unique(bundle.pid[sanity_idx]).size} subject(s)"
            )

    backbone_cfg = BackboneConfig(
        backbone_type=cfg.backbone_type,
        in_channels=bundle.n_channels,
        seq_length=bundle.seq_length,
        base_channels=cfg.base_channels,
        embedding_dim=cfg.embedding_dim,
        dropout=cfg.dropout,
        deep4_batch_norm=cfg.deep4_batch_norm,
        add_output_layernorm=cfg.add_output_layernorm,
        drop_path=cfg.drop_path,
    )
    model = BrainAgeRegressor(cfg=backbone_cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Only CUDA supports AMP (MPS/CPU don't)
    use_amp = (device == "cuda")
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    history: List[Dict] = []
    best = {"epoch": -1, "test_subj_mae": float("inf"), "select_by": cfg_select}
    best_select_metric = float("inf")
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        running_base = 0.0
        running_reg = 0.0
        n_batches = 0
        _dbg_count = 0
        for x, age, pid in train_loader:
            if _dbg_count < 2:
                age_cpu_pre = age.detach().float().cpu()
                pid_cpu_pre = pid.detach().long().cpu()
                progress_log(
                    f"[debug-loader] raw age stats: mean={age_cpu_pre.mean():.3f} "
                    f"std={age_cpu_pre.std():.3f} min={age_cpu_pre.min():.3f} "
                    f"max={age_cpu_pre.max():.3f} | first 8 ages={age_cpu_pre[:8].tolist()} "
                    f"first 8 pids={pid_cpu_pre[:8].tolist()} | "
                    f"age_mean_var={age_mean:.3f} age_std_var={age_std:.3f}"
                )
                _dbg_count += 1
            # MPS scalar arithmetic corrupts (age - age_mean) / age_std for some
            # batches (observed -1e21 garbage). Normalize on CPU first, transfer.
            age_z_cpu = (age.float() - age_mean) / age_std
            x = x.to(device, non_blocking=True)
            age = age.to(device, non_blocking=True)
            pid = pid.to(device, non_blocking=True)
            age_z = age_z_cpu.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            amp_ctx = torch.amp.autocast(device_type="cuda") if use_amp else nullcontext()
            with amp_ctx:
                pred = model(x)
                loss, base_loss, subject_reg = _huber_loss_parts(
                    pred, age_z, pid,
                    delta=cfg.huber_delta,
                    lam=cfg.subject_reg_lambda,
                )
                _raise_if_batch_loss_invalid(
                    loss,
                    pred,
                    x,
                    threshold=cfg.max_batch_loss_abort,
                    fold_id=fold_id,
                    epoch=epoch,
                    batch_idx=n_batches + 1,
                )
            if use_amp:
                scaler.scale(loss).backward()
                if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optim.step()
            running += float(loss.item())
            running_base += float(base_loss.item())
            running_reg += float(subject_reg.item())
            n_batches += 1
            del x, age, pid, loss, base_loss, subject_reg, pred
        train_loss = running / max(n_batches, 1)
        train_base_loss = running_base / max(n_batches, 1)
        train_subject_reg = running_reg / max(n_batches, 1)
        gc.collect()

        # eval (epoch + subject level) — _predict_epochs denormalizes back to years
        tr_pred, tr_age, tr_pid = _predict_epochs(
            model, eval_train_loader, device, age_mean, age_std
        )
        te_pred, te_age, te_pid = _predict_epochs(
            model, test_loader, device, age_mean, age_std
        )

        tr_epoch_m = regression_metrics(tr_pred, tr_age)
        te_epoch_m = regression_metrics(te_pred, te_age)
        _, tr_pred_s, tr_age_s = _aggregate_subject_level(tr_pred, tr_age, tr_pid)
        _, te_pred_s, te_age_s = _aggregate_subject_level(te_pred, te_age, te_pid)
        tr_subj_m = regression_metrics(tr_pred_s, tr_age_s)
        te_subj_m = regression_metrics(te_pred_s, te_age_s)

        val_epoch_m = val_subj_m = None
        if val_loader is not None:
            v_pred, v_age, v_pid = _predict_epochs(
                model, val_loader, device, age_mean, age_std
            )
            val_epoch_m = regression_metrics(v_pred, v_age)
            _, v_pred_s, v_age_s = _aggregate_subject_level(v_pred, v_age, v_pid)
            val_subj_m = regression_metrics(v_pred_s, v_age_s)

        sanity = None
        if sanity_batch is not None:
            sanity = _age_sanity_snapshot(
                model,
                sanity_batch,
                device=device,
                age_mean=age_mean,
                age_std=age_std,
                delta=cfg.huber_delta,
                lam=cfg.subject_reg_lambda,
            )

        ep_sec = time.time() - t0
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_base_loss": train_base_loss,
            "train_subject_reg": train_subject_reg,
            "tr_epoch": tr_epoch_m,
            "te_epoch": te_epoch_m,
            "tr_subj": tr_subj_m,
            "te_subj": te_subj_m,
            "val_epoch": val_epoch_m,
            "val_subj": val_subj_m,
            "sanity": sanity,
            "sec": ep_sec,
        })
        val_str = ""
        if val_subj_m is not None:
            val_str = (
                f"va_subj_mae={val_subj_m['mae']:.2f} r={val_subj_m['pearson_r']:.3f} | "
            )
        progress_log(
            f"[fold{fold_id}] ep{epoch:>3}/{cfg.epochs} loss={train_loss:.4f} "
            f"base={train_base_loss:.4f} reg={train_subject_reg:.4f} "
            f"tr_subj_mae={tr_subj_m['mae']:.2f} r={tr_subj_m['pearson_r']:.3f} | "
            f"{val_str}"
            f"te_subj_mae={te_subj_m['mae']:.2f} r={te_subj_m['pearson_r']:.3f} "
            f"({ep_sec:.1f}s)"
        )
        if progress is not None:
            progress.set_description(f"Pretext fold{fold_id} ep{epoch}/{cfg.epochs}")
            progress.set_postfix({
                "loss": f"{train_loss:.4f}",
                "teMAE": f"{te_subj_m['mae']:.2f}",
                "r": f"{te_subj_m['pearson_r']:.3f}",
            })
            progress.update(1)
        if sanity is not None:
            progress_log(
                f"[fold{fold_id}] sanity "
                f"{_format_sanity('train', sanity['train'])} | "
                f"{_format_sanity('eval', sanity['eval'])}"
            )
            _raise_if_sanity_failed(
                sanity,
                threshold_mae=cfg.sanity_abort_mae,
                fold_id=fold_id,
                epoch=epoch,
            )

        # checkpoint selection: by val (preferred) or test
        select_metric = (
            val_subj_m["mae"] if (cfg_select == "val" and val_subj_m is not None)
            else te_subj_m["mae"]
        )
        if select_metric < best_select_metric:
            best_select_metric = select_metric
            no_improve = 0
            best = {
                "epoch": epoch,
                "select_by": cfg_select,
                "select_metric": float(select_metric),
                "val_subj_mae": (val_subj_m["mae"] if val_subj_m else None),
                "val_subj_r": (val_subj_m["pearson_r"] if val_subj_m else None),
                "test_subj_mae": te_subj_m["mae"],
                "test_subj_r": te_subj_m["pearson_r"],
                "test_epoch_mae": te_epoch_m["mae"],
                "test_epoch_r": te_epoch_m["pearson_r"],
            }
            os.makedirs(cfg.out_dir, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "backbone_cfg": asdict(backbone_cfg),
                    "fold": fold_id,
                    "epoch": epoch,
                    "metrics": best,
                    "age_mean": age_mean,
                    "age_std": age_std,
                    "input_norm": cfg.input_norm,
                    "input_clip": cfg.input_clip,
                    "grad_clip_norm": cfg.grad_clip_norm,
                    "max_batch_loss_abort": cfg.max_batch_loss_abort,
                },
                os.path.join(cfg.out_dir, f"pretext_fold{fold_id}_best.pt"),
            )
            # Save subject-level test predictions at the best epoch so we can
            # analyse age-band accuracy of the brain-age pretext later
            # (age_band_analysis.py). pred_age vs true_age, one row per subject.
            np.savez(
                os.path.join(cfg.out_dir, f"pretext_fold{fold_id}_test_predictions.npz"),
                pred_age=np.asarray(te_pred_s, dtype=np.float32),
                true_age=np.asarray(te_age_s, dtype=np.float32),
                fold=np.int64(fold_id),
                epoch=np.int64(epoch),
            )
            gc.collect()
        else:
            no_improve += 1
            if cfg.early_stop_patience > 0 and no_improve >= cfg.early_stop_patience:
                progress_write(f"[fold{fold_id}] early stop @ ep{epoch} "
                               f"(no improve for {no_improve} epochs)")
                break

    if progress is not None and len(history) < cfg.epochs:
        progress.update(cfg.epochs - len(history))

    # last-epoch checkpoint as well (for repro)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "backbone_cfg": asdict(backbone_cfg),
            "fold": fold_id,
            "epoch": history[-1]["epoch"] if history else 0,
            "age_mean": age_mean,
            "age_std": age_std,
            "input_norm": cfg.input_norm,
            "input_clip": cfg.input_clip,
            "grad_clip_norm": cfg.grad_clip_norm,
            "max_batch_loss_abort": cfg.max_batch_loss_abort,
        },
        os.path.join(cfg.out_dir, f"pretext_fold{fold_id}_last.pt"),
    )

    # cleanup model and caches
    del model, optim, scaler, train_loader, eval_train_loader, test_loader, val_loader
    gc.collect()

    return {
        "fold": fold_id,
        "history": history,
        "best": best,
        "epoch_sample_frac": float(cfg.epoch_sample_frac),
        "n_inner_train_epochs": int(inner_train_idx.size),
        "n_val_epochs": int(val_idx.size),
        "n_test_epochs": int(test_idx.size),
        "n_train_subjects": int(np.unique(bundle.pid[train_idx]).size),
        "n_val_subjects": int(np.unique(bundle.pid[val_idx]).size) if val_idx.size else 0,
        "n_test_subjects": int(np.unique(bundle.pid[test_idx]).size),
    }


def run(cfg: PretextConfig) -> Dict:
    device = select_device()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    seed_accelerator(device, cfg.seed)

    os.makedirs(cfg.out_dir, exist_ok=True)
    # require_age=False so that age-missing epochs (e.g., placeholder pid==0)
    # are not a hard error — select_indices() filters them via age_valid mask.
    bundle = load_bundle(cfg.data_path, cfg.age_lookup_path or None, require_age=False)
    splits = load_or_build_splits(
        bundle.pid, bundle.y, n_splits=cfg.n_splits, seed=cfg.seed,
        splits_path=(cfg.splits_path or None),
    )

    fold_reports = []
    bar = progress_bar(
        total=planned_epoch_steps(cfg.folds, cfg.epochs),
        desc="Pretext overall",
        unit="ep",
    )
    try:
        for fold_id, (tr, te) in enumerate(splits):
            if fold_id not in cfg.folds:
                continue
            progress_write(f"\n===== Pretext fold {fold_id} =====")
            rep = train_one_fold(cfg, bundle, fold_id, tr, te, device, progress=bar)
            fold_reports.append(rep)
            with open(os.path.join(cfg.out_dir, f"pretext_fold{fold_id}_history.json"), "w") as f:
                json.dump(rep, f, indent=2)
    finally:
        bar.close()

    summary = {
        "config": asdict(cfg),
        "device": device,
        "folds": fold_reports,
    }
    if fold_reports:
        maes = [r["best"]["test_subj_mae"] for r in fold_reports]
        rs = [r["best"]["test_subj_r"] for r in fold_reports]
        summary["mean_test_subj_mae"] = float(np.mean(maes))
        summary["mean_test_subj_r"] = float(np.mean(rs))
        summary["gate_pass_mae"] = bool(np.mean(maes) <= cfg.gate_mae)
        summary["gate_pass_r"] = bool(np.mean(rs) >= cfg.gate_pearson_r)

    with open(os.path.join(cfg.out_dir, "pretext_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--folds", default=None, help="comma-separated fold ids (override)")
    args = p.parse_args()

    raw = resolve_config_paths(
        load_yaml_config(args.config),
        config_path=args.config,
        keys=("data_path", "age_lookup_path", "out_dir", "splits_path"),
    )
    cfg = PretextConfig(**raw)
    if args.folds is not None:
        cfg.folds = tuple(int(x) for x in args.folds.split(",") if x.strip())

    summary = run(cfg)
    progress_write("\nPretext done.")
    if "mean_test_subj_mae" in summary:
        progress_write(
            f"  test subj-MAE = {summary['mean_test_subj_mae']:.3f}, "
            f"r = {summary['mean_test_subj_r']:.3f}, "
            f"gate(MAE<= {cfg.gate_mae})={summary['gate_pass_mae']}, "
            f"gate(r>= {cfg.gate_pearson_r})={summary['gate_pass_r']}"
        )


if __name__ == "__main__":
    main()

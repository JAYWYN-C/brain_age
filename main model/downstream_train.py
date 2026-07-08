"""Downstream 4-class training for the final 0702 V0/V1 main experiment."""

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

try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pathing import resolve_config_paths  # noqa: E402
import windows_patch  # noqa: F401,E402

from ssl_dataset import DownstreamDataset, SubjectBalancedBatchSampler, load_bundle, select_indices  # noqa: E402
from ssl_model import BackboneConfig  # noqa: E402
from downstream_model import VARIANT_SPECS, build_downstream  # noqa: E402
from splits import load_or_build_splits, split_train_into_train_val  # noqa: E402
from transforms import build_transforms  # noqa: E402
from config_io import load_yaml_config  # noqa: E402
from device_utils import select_device, seed_accelerator  # noqa: E402
from progress import planned_epoch_steps, progress_bar, progress_log, progress_write  # noqa: E402


CLASS_NAMES = ["CN", "MCI", "AD", "non-AD Dementia"]
N_CLASSES = len(CLASS_NAMES)


@dataclass
class DownstreamConfig:
    data_path: str = ""
    age_lookup_path: str = ""
    out_dir: str = ""
    sweep: str = "main_basicdeep"
    pretext_dir: str = ""             # required for V1
    splits_path: str = ""             # optional fixed splits.npz
    variant: str = "V0"
    folds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    n_splits: int = 5
    seed: int = 42

    epochs: int = 50
    batch_size: int = 64
    lr: float = 2e-4
    weight_decay: float = 0.0
    dropout: float = 0.3
    head_hidden: Optional[int] = None

    backbone_type: str = "custom"
    backbone_base_channels: int = 32
    backbone_embedding_dim: int = 128
    backbone_add_output_layernorm: bool = False
    backbone_drop_path: float = 0.1

    num_workers: int = 2
    pin_memory: bool = True
    use_class_weight: bool = True

    # Subject-balanced batches for BN stability (0 disables, falls back to plain shuffle)
    subjects_per_batch: int = 0

    # Optional subject-wise epoch caps for quick experiments (0 disables).
    epoch_sample_frac: float = 0.0
    max_train_epochs_per_subject: int = 0
    max_eval_epochs_per_subject: int = 0

    require_age: bool = False

    # validation + early stopping
    val_frac: float = 0.1
    early_stop_patience: int = 0
    select_by: str = "val"            # "val" | "test"
    select_metric: str = "macro_f1"   # "acc" | "macro_f1" | "macro_sens"

    # augmentation
    augment: Optional[Dict] = None
    input_norm: str = "none"
    input_clip: float = 0.0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _confusion(pred: np.ndarray, true: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(true, pred):
        cm[int(t), int(p)] += 1
    return cm


def _per_class(cm: np.ndarray) -> List[Dict[str, float]]:
    total = cm.sum()
    out = []
    for i in range(cm.shape[0]):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = total - TP - FN - FP
        sens = TP / (TP + FN + 1e-12)
        spec = TN / (TN + FP + 1e-12)
        prec = TP / (TP + FP + 1e-12)
        f1 = 2 * prec * sens / (prec + sens + 1e-12)
        out.append({"sens": float(sens), "spec": float(spec),
                    "precision": float(prec), "f1": float(f1),
                    "TP": int(TP), "FN": int(FN), "FP": int(FP), "TN": int(TN)})
    return out


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def _metrics(pred: np.ndarray, true: np.ndarray, prob: Optional[np.ndarray] = None) -> Dict:
    cm = _confusion(pred, true, N_CLASSES)
    pc = _per_class(cm)
    acc = float(np.mean(pred == true))
    macro_f1 = float(np.mean([c["f1"] for c in pc]))
    macro_sens = float(np.mean([c["sens"] for c in pc]))
    macro_spec = float(np.mean([c["spec"] for c in pc]))

    macro_auc = float("nan")
    if HAS_SKLEARN and prob is not None:
        try:
            prob = np.asarray(prob, dtype=np.float64)
            if prob.shape == (true.size, N_CLASSES):
                macro_auc = float(roc_auc_score(true, prob, multi_class="ovr", average="macro"))
        except Exception:
            macro_auc = float("nan")

    return {"acc": acc, "macro_f1": macro_f1, "macro_sens": macro_sens, "macro_spec": macro_spec,
            "macro_auc": macro_auc, "cm": cm.tolist(), "per_class": pc}


def _subject_metrics_from_logits(
    logits: np.ndarray,
    true: np.ndarray,
    pid: np.ndarray,
) -> Dict:
    """Average epoch logits per subject, then compute subject-level metrics."""
    logits = np.asarray(logits, dtype=np.float64)
    true = np.asarray(true, dtype=np.int64)
    pid = np.asarray(pid, dtype=np.int64)
    if logits.ndim != 2 or logits.shape[0] != true.size or true.size != pid.size:
        raise ValueError("logits, true, and pid must have matching first dimensions")
    subjects, inv = np.unique(pid, return_inverse=True)
    sums = np.zeros((subjects.size, logits.shape[1]), dtype=np.float64)
    counts = np.bincount(inv, minlength=subjects.size).astype(np.float64)
    np.add.at(sums, inv, logits)
    subj_logits = sums / np.maximum(counts[:, None], 1.0)

    subj_true = np.zeros(subjects.size, dtype=np.int64)
    for k in range(subjects.size):
        labels = true[inv == k]
        subj_true[k] = int(np.bincount(labels, minlength=N_CLASSES).argmax())

    subj_prob = _softmax_np(subj_logits)
    subj_pred = subj_prob.argmax(axis=1).astype(np.int64)
    out = _metrics(subj_pred, subj_true, subj_prob)
    out["n_subjects"] = int(subjects.size)
    return out


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


def _make_loader(ds, bs, shuffle, nw, pm):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=nw,
                      pin_memory=pm, persistent_workers=(nw > 0))


def _limit_epochs_per_subject(
    indices: np.ndarray,
    pid: np.ndarray,
    max_per_subject: int,
    seed: int,
) -> np.ndarray:
    """Return a deterministic subject-balanced subset of epoch indices."""
    indices = np.asarray(indices, dtype=np.int64)
    max_per_subject = int(max_per_subject or 0)
    if max_per_subject <= 0 or indices.size == 0:
        return indices

    pid = np.asarray(pid)
    indexed_pid = pid[indices]
    rng = np.random.default_rng(int(seed))
    selected: List[int] = []
    for subject in np.unique(indexed_pid):
        subj_idx = indices[indexed_pid == subject]
        if subj_idx.size > max_per_subject:
            subj_idx = rng.choice(subj_idx, size=max_per_subject, replace=False)
        selected.extend(int(i) for i in subj_idx)
    return np.asarray(sorted(selected), dtype=np.int64)


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


def _class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
    w = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    w = w * (N_CLASSES / w.sum())
    return torch.tensor(w, dtype=torch.float32)


def requires_pred_age_cache(spec) -> bool:
    """V0/V1 downstream training does not use epoch-level pretext predictions."""
    return False


def forward_downstream_model(
    model,
    spec,
    x: torch.Tensor,
    *,
    side: Optional[torch.Tensor],
    age: torch.Tensor,
    pred_age: torch.Tensor,
):
    """Run the final V0/V1 classifier."""
    return model(x)


def compute_downstream_loss(
    outputs,
    y: torch.Tensor,
    age: torch.Tensor,
    loss_fn,
    *,
    aux_alpha: float = 0.0,
    aux_cn_only: bool = True,
    cn_label: int = 0,
    huber_delta: float = 1.0,
    force_cpu_ce: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute cross-entropy loss."""
    if isinstance(outputs, tuple):
        logits, aux_age_pred = outputs
    else:
        logits, aux_age_pred = outputs, None

    if force_cpu_ce:
        target = y.long().cpu()
        if target.numel() and (int(target.min()) < 0 or int(target.max()) >= int(logits.shape[1])):
            raise ValueError(
                f"class targets must be in [0, {int(logits.shape[1]) - 1}], "
                f"got min={int(target.min())}, max={int(target.max())}"
            )
        ce = loss_fn(logits.float().cpu(), target)
    else:
        target = y
        ce = loss_fn(logits, y)
    total = ce
    return total, {
        "ce": float(ce.detach().cpu().item()),
        "aux_age": 0.0,
        "aux_n": 0,
    }


def _move_targets_for_loss(
    y: torch.Tensor,
    device: str,
    force_cpu_ce: bool,
) -> torch.Tensor:
    """Keep CE targets on CPU when MPS loss is computed on CPU."""
    y = y.long()
    if force_cpu_ce:
        return y.cpu()
    return y.to(device, non_blocking=True)


def _select_metric(metrics: Dict, name: str) -> float:
    if name not in {"acc", "macro_f1", "macro_sens"}:
        raise ValueError("select_metric must be one of acc, macro_f1, macro_sens")
    return float(metrics[name])


@torch.no_grad()
def _evaluate(
    model,
    loader,
    device,
    spec,
    debug_tag: str = "",
) -> Dict:
    model.eval()
    preds, trues, logits_all, pids_all = [], [], [], []
    logit_means = []
    logit_stds = []
    with torch.no_grad():
        for x, age, pred_age, bag, pid, y in loader:
            x = x.to(device, non_blocking=True)
            outputs = forward_downstream_model(model, spec, x, side=None, age=age, pred_age=pred_age)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            logits_np = logits.detach().cpu().numpy()
            preds.append(logits_np.argmax(1))
            logits_all.append(logits_np)
            trues.append(y.numpy())
            pids_all.append(pid.numpy())
            logit_means.append(logits.detach().cpu().numpy().mean(axis=0))
            logit_stds.append(logits.detach().cpu().numpy().std(axis=0))
            del x, outputs, logits
    pred_arr = np.concatenate(preds)
    true_arr = np.concatenate(trues)
    logits_arr = np.concatenate(logits_all, axis=0)
    pid_arr = np.concatenate(pids_all)
    if debug_tag:
        pred_hist = np.bincount(pred_arr, minlength=N_CLASSES)
        true_hist = np.bincount(true_arr, minlength=N_CLASSES)
        lm = np.stack(logit_means).mean(axis=0)
        ls = np.stack(logit_stds).mean(axis=0)
        progress_log(f"[debug:{debug_tag}] pred_hist={pred_hist.tolist()} "
                     f"true_hist={true_hist.tolist()} "
                     f"logit_mean={np.array2string(lm, precision=3)} "
                     f"logit_std={np.array2string(ls, precision=3)}")
    epoch_result = _metrics(pred_arr, true_arr, _softmax_np(logits_arr))
    subject_result = _subject_metrics_from_logits(logits_arr, true_arr, pid_arr)
    result = {
        **subject_result,
        "subject": subject_result,
        "epoch": epoch_result,
        "epoch_acc": epoch_result["acc"],
        "epoch_macro_f1": epoch_result["macro_f1"],
        "epoch_macro_sens": epoch_result["macro_sens"],
    }
    del preds, trues, logits_all, pids_all, pred_arr, true_arr, logits_arr, pid_arr
    gc.collect()
    return result


def train_one_fold(
    cfg: DownstreamConfig,
    bundle,
    pred_age_cache: Optional[np.ndarray],
    fold_id: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    device: str,
    progress=None,
) -> Dict:
    spec = VARIANT_SPECS[cfg.variant]
    needs_pred_age = requires_pred_age_cache(spec)
    if needs_pred_age and pred_age_cache is None:
        raise ValueError(f"variant {cfg.variant} needs --bag-cache")

    train_idx = select_indices(bundle, train_idx, cn_only=False)
    test_idx = select_indices(bundle, test_idx, cn_only=False)
    if needs_pred_age:
        # restrict to epochs that have a pretext prediction
        valid = ~np.isnan(pred_age_cache)
        train_idx = train_idx[valid[train_idx]]
        test_idx = test_idx[valid[test_idx]]
        if train_idx.size == 0 or test_idx.size == 0:
            raise RuntimeError(f"{cfg.variant}/fold{fold_id}: no usable epochs remain.")

    inner_train_idx, val_idx = split_train_into_train_val(
        train_idx, bundle.pid, bundle.y,
        val_frac=cfg.val_frac, seed=cfg.seed + fold_id, cn_only=False,
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
    inner_train_idx = _limit_epochs_per_subject(
        inner_train_idx,
        bundle.pid,
        cfg.max_train_epochs_per_subject,
        seed=cfg.seed + fold_id,
    )
    val_idx = _limit_epochs_per_subject(
        val_idx,
        bundle.pid,
        cfg.max_eval_epochs_per_subject,
        seed=cfg.seed + 10_000 + fold_id,
    )
    test_idx = _limit_epochs_per_subject(
        test_idx,
        bundle.pid,
        cfg.max_eval_epochs_per_subject,
        seed=cfg.seed + 20_000 + fold_id,
    )
    if cfg.max_train_epochs_per_subject > 0 or cfg.max_eval_epochs_per_subject > 0:
        progress_log(
            f"[{cfg.variant}/fold{fold_id}] epoch caps: "
            f"train {n_inner_raw}->{inner_train_idx.size}, "
            f"val {n_val_raw}->{val_idx.size}, "
            f"test {n_test_raw}->{test_idx.size}"
        )
    elif cfg.epoch_sample_frac > 0.0:
        progress_log(
            f"[{cfg.variant}/fold{fold_id}] epoch sampling: "
            f"frac={cfg.epoch_sample_frac:.3f} "
            f"train {n_inner_raw}->{inner_train_idx.size}, "
            f"val {n_val_raw}->{val_idx.size}, "
            f"test {n_test_raw}->{test_idx.size}"
        )
    use_val = val_idx.size > 0
    cfg_select = cfg.select_by
    if cfg_select == "val" and not use_val:
        cfg_select = "test"
    if cfg_select not in {"val", "test"}:
        raise ValueError("select_by must be 'val' or 'test'")

    # MPS/CPU doesn't support pin_memory
    pin_memory = cfg.pin_memory and (device == "cuda")

    transform = build_transforms(cfg.augment)
    cache = pred_age_cache if needs_pred_age else None

    train_ds = DownstreamDataset(
        bundle,
        inner_train_idx,
        cache,
        transform=transform,
        input_norm=cfg.input_norm,
        input_clip=cfg.input_clip,
    )
    eval_train_ds = DownstreamDataset(
        bundle,
        inner_train_idx,
        cache,
        input_norm=cfg.input_norm,
        input_clip=cfg.input_clip,
    )
    test_ds = DownstreamDataset(
        bundle,
        test_idx,
        cache,
        input_norm=cfg.input_norm,
        input_clip=cfg.input_clip,
    )
    if cfg.subjects_per_batch and cfg.subjects_per_batch > 0:
        sampler = SubjectBalancedBatchSampler(
            pids=bundle.pid[inner_train_idx],
            batch_size=cfg.batch_size,
            subjects_per_batch=cfg.subjects_per_batch,
            seed=cfg.seed + fold_id,
        )
        progress_log(
            f"[{cfg.variant}/fold{fold_id}] subject-balanced sampler: "
            f"{cfg.subjects_per_batch} subj/batch, "
            f"{sampler.per_subject} epochs/subj, "
            f"{len(sampler)} batches"
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
            persistent_workers=(cfg.num_workers > 0),
        )
    else:
        train_loader = _make_loader(train_ds, cfg.batch_size, True, cfg.num_workers, pin_memory)
    eval_train_loader = _make_loader(eval_train_ds, cfg.batch_size, False, cfg.num_workers, pin_memory)
    test_loader = _make_loader(test_ds, cfg.batch_size, False, cfg.num_workers, pin_memory)
    val_loader = None
    if use_val:
        val_ds = DownstreamDataset(
            bundle,
            val_idx,
            cache,
            input_norm=cfg.input_norm,
            input_clip=cfg.input_clip,
        )
        val_loader = _make_loader(val_ds, cfg.batch_size, False, cfg.num_workers, pin_memory)

    backbone_cfg = BackboneConfig(
        backbone_type=cfg.backbone_type,
        in_channels=bundle.n_channels,
        seq_length=bundle.seq_length,
        base_channels=cfg.backbone_base_channels,
        embedding_dim=cfg.backbone_embedding_dim,
        dropout=cfg.dropout,
        add_output_layernorm=cfg.backbone_add_output_layernorm,
        drop_path=cfg.backbone_drop_path,
    )

    pretext_ckpt = None
    if spec.pretext_init:
        pretext_ckpt = os.path.join(cfg.pretext_dir, f"pretext_fold{fold_id}_best.pt")
        if not os.path.exists(pretext_ckpt):
            pretext_ckpt = os.path.join(cfg.pretext_dir, f"pretext_fold{fold_id}_last.pt")
        if not os.path.exists(pretext_ckpt):
            raise FileNotFoundError(f"pretext checkpoint missing: fold{fold_id}")

    model = build_downstream(
        cfg.variant,
        backbone_cfg=backbone_cfg,
        pretext_checkpoint=pretext_ckpt,
        n_classes=N_CLASSES,
        head_hidden=cfg.head_hidden,
        dropout=cfg.dropout,
        device=device,
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.use_class_weight:
        loss_device = "cpu" if device == "mps" else device
        cw = _class_weights(bundle.y[inner_train_idx]).to(loss_device)
    else:
        cw = None
    loss_fn = nn.CrossEntropyLoss(weight=cw)

    # Only CUDA supports AMP (MPS/CPU don't)
    use_amp = (device == "cuda")
    if use_amp:
        scaler = torch.amp.GradScaler(device_type="cuda")
    else:
        scaler = None

    history = []
    best = {"epoch": -1, "test_acc": -1.0, "select_by": cfg_select}
    best_select_metric = -float("inf")
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        running_ce = 0.0
        running_aux = 0.0
        n_batches = 0
        for x, age, pred_age, bag, pid, y in train_loader:
            x = x.to(device, non_blocking=True)
            age_d = age.to(device, non_blocking=True)
            pred_age_d = pred_age.to(device, non_blocking=True)
            y = _move_targets_for_loss(
                y,
                device=device,
                force_cpu_ce=(device == "mps"),
            )
            optim.zero_grad(set_to_none=True)
            amp_ctx = torch.amp.autocast(device_type="cuda") if use_amp else nullcontext()
            with amp_ctx:
                outputs = forward_downstream_model(model, spec, x, side=None, age=age_d, pred_age=pred_age_d)
                loss, loss_parts = compute_downstream_loss(
                    outputs,
                    y,
                    age_d,
                    loss_fn,
                    force_cpu_ce=(device == "mps"),
                )
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()
            running += float(loss.item())
            running_ce += loss_parts["ce"]
            running_aux += loss_parts["aux_age"]
            n_batches += 1
            del x, age_d, pred_age_d, y, outputs, loss
        train_loss = running / max(n_batches, 1)
        train_ce = running_ce / max(n_batches, 1)
        train_aux = running_aux / max(n_batches, 1)
        gc.collect()

        train_m = _evaluate(
            model, eval_train_loader, device, spec,
            debug_tag=f"{cfg.variant}/fold{fold_id}/ep{epoch}/train",
        )
        test_m = _evaluate(
            model, test_loader, device, spec,
            debug_tag=f"{cfg.variant}/fold{fold_id}/ep{epoch}/test",
        )
        val_m = (
            _evaluate(
                model, val_loader, device, spec,
                debug_tag=f"{cfg.variant}/fold{fold_id}/ep{epoch}/val",
            )
            if val_loader is not None else None
        )
        ep_sec = time.time() - t0

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ce": train_ce,
            "train_aux_age": train_aux,
            "train": {k: train_m[k] for k in ("acc", "macro_f1", "macro_sens")},
            "test": {k: test_m[k] for k in ("acc", "macro_f1", "macro_sens")},
            "val": ({k: val_m[k] for k in ("acc", "macro_f1", "macro_sens")} if val_m else None),
            "train_epoch": {k: train_m[f"epoch_{k}"] for k in ("acc", "macro_f1", "macro_sens")},
            "test_epoch": {k: test_m[f"epoch_{k}"] for k in ("acc", "macro_f1", "macro_sens")},
            "val_epoch": ({k: val_m[f"epoch_{k}"] for k in ("acc", "macro_f1", "macro_sens")} if val_m else None),
            "sec": ep_sec,
        })
        val_str = ""
        if val_m is not None:
            val_str = f"va_acc={val_m['acc']:.3f} va_f1={val_m['macro_f1']:.3f} | "
        progress_log(
            f"[{cfg.variant}/fold{fold_id}] ep{epoch:>3}/{cfg.epochs} "
            f"loss={train_loss:.4f} tr_acc={train_m['acc']:.3f} tr_f1={train_m['macro_f1']:.3f} | "
            f"{val_str}"
            f"te_acc={test_m['acc']:.3f} te_f1={test_m['macro_f1']:.3f} ({ep_sec:.1f}s)"
        )
        if progress is not None:
            progress.set_description(f"{cfg.variant} fold{fold_id} ep{epoch}/{cfg.epochs}")
            progress.set_postfix({
                "loss": f"{train_loss:.4f}",
                "te_acc": f"{test_m['acc']:.3f}",
                "te_f1": f"{test_m['macro_f1']:.3f}",
            })
            progress.update(1)

        chosen_m = val_m if (cfg_select == "val" and val_m is not None) else test_m
        select_metric = _select_metric(chosen_m, cfg.select_metric)
        if select_metric > best_select_metric:
            best_select_metric = select_metric
            no_improve = 0
            best = {
                "epoch": epoch,
                "select_by": cfg_select,
                "select_metric": cfg.select_metric,
                "select_value": float(select_metric),
                "val_acc": (val_m["acc"] if val_m else None),
                "val_macro_f1": (val_m["macro_f1"] if val_m else None),
                "val_macro_sens": (val_m["macro_sens"] if val_m else None),
                "test_acc": test_m["acc"],
                "test_macro_f1": test_m["macro_f1"],
                "test_macro_sens": test_m["macro_sens"],
                "test_macro_spec": test_m.get("macro_spec", 0.0),
                "test_macro_auc": test_m.get("macro_auc", 0.0),
                "test_cm": test_m["cm"],
                "test_per_class": test_m["per_class"],
                "test_epoch_acc": test_m["epoch_acc"],
                "test_epoch_macro_f1": test_m["epoch_macro_f1"],
                "test_epoch_macro_sens": test_m["epoch_macro_sens"],
                "test_epoch_cm": test_m["epoch"]["cm"],
                "test_epoch_per_class": test_m["epoch"]["per_class"],
            }
            os.makedirs(cfg.out_dir, exist_ok=True)
            torch.save(
                {"state_dict": model.state_dict(),
                 "fold": fold_id, "epoch": epoch, "variant": cfg.variant},
                os.path.join(cfg.out_dir, f"{cfg.variant}_fold{fold_id}_best.pt"),
            )
        else:
            no_improve += 1
            if cfg.early_stop_patience > 0 and no_improve >= cfg.early_stop_patience:
                progress_write(f"[{cfg.variant}/fold{fold_id}] early stop @ ep{epoch} "
                               f"(no improve for {no_improve} epochs)")
                break

    if progress is not None and len(history) < cfg.epochs:
        progress.update(cfg.epochs - len(history))

    torch.save(
        {"state_dict": model.state_dict(),
         "fold": fold_id, "epoch": history[-1]["epoch"] if history else 0,
         "variant": cfg.variant},
        os.path.join(cfg.out_dir, f"{cfg.variant}_fold{fold_id}_last.pt"),
    )

    # cleanup model and caches
    del model, optim, scaler, train_loader, eval_train_loader, test_loader, val_loader
    gc.collect()

    return {
        "fold": fold_id,
        "variant": cfg.variant,
        "history": history,
        "best": best,
        "epoch_sample_frac": float(cfg.epoch_sample_frac),
        "n_train": int(inner_train_idx.size),
        "n_val": int(val_idx.size),
        "n_test": int(test_idx.size),
        "n_train_subjects": int(np.unique(bundle.pid[inner_train_idx]).size),
        "n_val_subjects": int(np.unique(bundle.pid[val_idx]).size) if val_idx.size else 0,
        "n_test_subjects": int(np.unique(bundle.pid[test_idx]).size),
    }


def run(cfg: DownstreamConfig) -> Dict:
    device = select_device()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    seed_accelerator(device, cfg.seed)

    os.makedirs(cfg.out_dir, exist_ok=True)
    bundle = load_bundle(cfg.data_path, cfg.age_lookup_path or None,
                         require_age=cfg.require_age)

    spec = VARIANT_SPECS[cfg.variant]
    pred_age_cache = None

    splits = load_or_build_splits(
        bundle.pid, bundle.y, n_splits=cfg.n_splits, seed=cfg.seed,
        splits_path=(cfg.splits_path or None),
    )

    fold_reports = []
    bar = progress_bar(
        total=planned_epoch_steps(cfg.folds, cfg.epochs),
        desc=f"{cfg.variant} overall",
        unit="ep",
    )
    try:
        for fold_id, (tr, te) in enumerate(splits):
            if fold_id not in cfg.folds:
                continue
            progress_write(f"\n===== Downstream {cfg.variant} fold {fold_id} =====")
            rep = train_one_fold(
                cfg, bundle, pred_age_cache, fold_id, tr, te, device, progress=bar
            )
            fold_reports.append(rep)
            with open(os.path.join(cfg.out_dir, f"{cfg.variant}_fold{fold_id}_history.json"), "w") as f:
                json.dump(rep, f, indent=2)
            gc.collect()
    finally:
        bar.close()

    summary = {"config": asdict(cfg), "device": device, "folds": fold_reports}
    if fold_reports:
        accs = [r["best"]["test_acc"] for r in fold_reports]
        f1s = [r["best"]["test_macro_f1"] for r in fold_reports]
        sens = [r["best"]["test_macro_sens"] for r in fold_reports]
        specs = [r["best"]["test_per_class"][i]["spec"] for r in fold_reports for i in range(N_CLASSES)]
        
        # Collect all metrics with mean and std
        metrics_data = {}
        
        # Overall metrics
        metrics_data["acc"] = f"{float(np.mean(accs)):.4f}±{float(np.std(accs)):.4f}"
        metrics_data["macro_f1"] = f"{float(np.mean(f1s)):.4f}±{float(np.std(f1s)):.4f}"
        metrics_data["macro_sens"] = f"{float(np.mean(sens)):.4f}±{float(np.std(sens)):.4f}"
        
        # Per-class metrics
        if "test_macro_auc" in fold_reports[0]["best"]:
            aucs = [r["best"]["test_macro_auc"] for r in fold_reports]
            metrics_data["macro_auc"] = f"{float(np.nanmean(aucs)):.4f}±{float(np.nanstd(aucs)):.4f}"
        
        # Per-class sensitivity and specificity
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_sens = [r["best"]["test_per_class"][class_idx]["sens"] for r in fold_reports]
            class_spec = [r["best"]["test_per_class"][class_idx]["spec"] for r in fold_reports]
            metrics_data[f"{class_name}_sens"] = f"{float(np.mean(class_sens)):.4f}±{float(np.std(class_sens)):.4f}"
            metrics_data[f"{class_name}_spec"] = f"{float(np.mean(class_spec)):.4f}±{float(np.std(class_spec)):.4f}"
        
        # Average confusion matrix across folds
        cms = [np.array(r["best"]["test_cm"]) for r in fold_reports]
        avg_cm = np.mean(cms, axis=0)
        
        summary["metrics"] = metrics_data
        summary["avg_confusion_matrix"] = avg_cm.tolist()
        
        # Keep backward compatibility
        summary["mean_test_acc"] = float(np.mean(accs))
        summary["std_test_acc"] = float(np.std(accs))
        summary["mean_test_macro_f1"] = float(np.mean(f1s))
        summary["std_test_macro_f1"] = float(np.std(f1s))
        if "test_macro_auc" in fold_reports[0]["best"]:
            summary["mean_test_macro_auc"] = float(np.nanmean(aucs))
            summary["std_test_macro_auc"] = float(np.nanstd(aucs))
    with open(os.path.join(cfg.out_dir, f"{cfg.variant}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save confusion matrix as PNG
    if fold_reports and "avg_confusion_matrix" in summary:
        save_confusion_matrix_png(
            cfg.variant, 
            np.array(summary["avg_confusion_matrix"]), 
            cfg.out_dir
        )
    
    return summary


def save_confusion_matrix_png(variant: str, cm: np.ndarray, out_dir: str) -> None:
    """Save confusion matrix as PNG visualization."""
    if not HAS_MATPLOTLIB or cm.size == 0:
        return
    
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Round for display
        cm_display = np.round(cm, 2)
        
        # Create heatmap
        im = ax.imshow(cm_display, interpolation='nearest', cmap=plt.cm.Blues)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(CLASS_NAMES)))
        ax.set_yticks(np.arange(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, fontsize=12)
        ax.set_yticklabels(CLASS_NAMES, fontsize=12)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Count', rotation=270, labelpad=15)
        
        # Add text annotations
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                text = ax.text(j, i, f'{cm_display[i, j]:.1f}',
                             ha="center", va="center", color="black" if cm_display[i, j] < cm_display.max()/2 else "white",
                             fontsize=12, fontweight='bold')
        
        # Labels and title
        ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
        ax.set_title(f'{variant} - Average Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save PNG
        output_path = os.path.join(out_dir, f"{variant}_confusion_matrix.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved confusion matrix: {output_path}", flush=True)
    except Exception as e:
        print(f"⚠ Failed to save confusion matrix PNG: {e}", flush=True)


def print_per_fold_results(variant: str, fold_reports: List[Dict]) -> None:
    """Print results for each fold."""
    print(f"\n{'='*80}", flush=True)
    print(f"  {variant} - Per-Fold Results", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    for fold_idx, report in enumerate(fold_reports):
        best = report.get("best", {})
        print(f"{'─'*80}", flush=True)
        print(f"  FOLD {fold_idx}", flush=True)
        print(f"{'─'*80}", flush=True)
        
        # Overall metrics
        print(f"\n  Overall Metrics:", flush=True)
        print(f"    Accuracy:           {best.get('test_acc', 0):.4f}", flush=True)
        print(f"    Macro F1-Score:     {best.get('test_macro_f1', 0):.4f}", flush=True)
        print(f"    Macro Sensitivity:  {best.get('test_macro_sens', 0):.4f}", flush=True)
        print(f"    Macro Specificity:  {best.get('test_macro_spec', 0):.4f}", flush=True)
        if "test_macro_auc" in best:
            print(f"    Macro AUC:          {best['test_macro_auc']:.4f}", flush=True)
        
        # Class-wise metrics
        per_class = best.get("test_per_class", [])
        if per_class:
            print(f"\n  Class-wise Metrics:", flush=True)
            for class_idx, class_name in enumerate(CLASS_NAMES):
                if class_idx < len(per_class):
                    pc = per_class[class_idx]
                    print(f"    {class_name}:", flush=True)
                    print(f"      Sensitivity:  {pc.get('sens', 0):.4f}", flush=True)
                    print(f"      Specificity:  {pc.get('spec', 0):.4f}", flush=True)
        
        print()


def print_final_results_table(variant: str, metrics: Dict[str, str], cm: Optional[np.ndarray] = None) -> None:
    """Print final results in a nicely formatted table with confusion matrix."""
    print(f"\n{'='*60}", flush=True)
    print(f"  {variant} - Final Results", flush=True)
    print(f"{'='*60}", flush=True)
    
    # Overall Performance
    print(f"\n{'Metric':<30} {'Mean±Std':>25}", flush=True)
    print(f"{'-'*60}", flush=True)
    print(f"{'Overall Performance':<30}", flush=True)
    print(f"{'-'*60}", flush=True)
    
    overall_keys = ['acc', 'macro_f1', 'macro_sens', 'macro_spec', 'macro_auc']
    metric_names = {
        'acc': 'Accuracy',
        'macro_f1': 'Macro F1-Score',
        'macro_sens': 'Macro Sensitivity',
        'macro_spec': 'Macro Specificity',
        'macro_auc': 'Macro AUC',
    }
    
    for key in overall_keys:
        if key in metrics:
            name = metric_names.get(key, key)
            value = metrics[key]
            print(f"  {name:<28} {value:>25}", flush=True)
    
    # Class-wise Performance
    print(f"\n{'-'*60}", flush=True)
    print(f"{'Class-wise Performance':<30}", flush=True)
    print(f"{'-'*60}", flush=True)
    
    for class_name in CLASS_NAMES:
        sens_key = f"{class_name}_sens"
        spec_key = f"{class_name}_spec"
        
        if sens_key in metrics or spec_key in metrics:
            print(f"\n  {class_name}:")
            if sens_key in metrics:
                print(f"    {'Sensitivity':<25} {metrics[sens_key]:>25}", flush=True)
            if spec_key in metrics:
                print(f"    {'Specificity':<25} {metrics[spec_key]:>25}", flush=True)
    
    # Confusion Matrix
    if cm is not None and cm.size > 0:
        print(f"\n{'-'*60}", flush=True)
        print(f"{'Average Confusion Matrix':<30}", flush=True)
        print(f"{'-'*60}", flush=True)
        
        cm_int = np.round(cm).astype(int)
        header = "         " + "  ".join(f"{name:>8}" for name in CLASS_NAMES)
        print(f"\n{header}", flush=True)
        print(f"{'-'*60}", flush=True)
        
        for i, true_label in enumerate(CLASS_NAMES):
            row_str = f"{true_label:>7} │ " + "  ".join(f"{int(cm_int[i, j]):>8}" for j in range(N_CLASSES))
            print(row_str, flush=True)
        
        print(f"{'-'*60}", flush=True)
        print(f"{'(rows: true, cols: predicted)':<30}", flush=True)
    
    print(f"\n{'='*60}\n", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--variant", default=None, help="V0/V1 override")
    p.add_argument("--folds", default=None)
    args = p.parse_args()

    raw = resolve_config_paths(
        load_yaml_config(args.config),
        config_path=args.config,
        keys=(
            "data_path",
            "age_lookup_path",
            "out_dir",
            "pretext_dir",
            "splits_path",
        ),
    )
    cfg = DownstreamConfig(**raw)
    if args.variant is not None:
        cfg.variant = args.variant
    if args.folds is not None:
        cfg.folds = tuple(int(x) for x in args.folds.split(",") if x.strip())

    summary = run(cfg)
    if "metrics" in summary:
        # Print per-fold results
        fold_reports = summary.get("folds", [])
        if fold_reports:
            print_per_fold_results(cfg.variant, fold_reports)
        
        # Print final summary
        cm = np.array(summary.get("avg_confusion_matrix", [])) if "avg_confusion_matrix" in summary else None
        print_final_results_table(cfg.variant, summary["metrics"], cm)
    elif "mean_test_acc" in summary:
        print(
            f"\n{cfg.variant} done: ACC={summary['mean_test_acc']:.4f}+-"
            f"{summary['std_test_acc']:.4f}  macroF1={summary['mean_test_macro_f1']:.4f}+-"
            f"{summary['std_test_macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()

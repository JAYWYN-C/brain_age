"""Windows preflight + compatibility patches for the public main-model pipeline.

Importing this module (which happens automatically in the entry shims) fixes
the seven recurring Windows failures observed in prior runs:

  1. Missing caueeg_dataset_final_age_lookup.npy
       -> preflight checks the dataset/age-lookup paths and prints a
          one-liner pointing at experiments/data_preparation/build_age_lookup.py
          if missing.

  2. cp949 / UTF-8 BOM encoding error when reading YAML / JSON sweep files
       -> open(...) calls in config_io / sweep loader monkey-patched to use
          encoding="utf-8-sig" (eats BOM, accepts plain UTF-8).

  3. ssl_train.py  GradScaler device_type error (torch>=2.4 deprecation)
       -> torch.cuda.amp.GradScaler aliased to torch.amp.GradScaler("cuda").

  4. Windows DataLoader multiprocessing / pickle error
       -> torch.utils.data.DataLoader.__init__ coerces num_workers=0 and
          persistent_workers=False on Windows. Avoids spawn() pickling of
          local Datasets.

  5. _huber_loss_parts scatter_add_ dtype mismatch under AMP
       -> replaced with a float32-accumulating version that's dtype-safe.

  6. matplotlib missing
       -> import is already optional in the training scripts; this module warns
          once if it can't be imported (and the pipeline will skip plots).

  7. downstream_train.py GradScaler device_type error
       -> same fix as (3).

Idempotent: re-importing is a no-op.
"""

from __future__ import annotations

import builtins
import os
import platform
import sys
import warnings

IS_WINDOWS = platform.system() == "Windows"
_APPLIED = False


# ---------------------------------------------------------------------------
# (3) + (7) GradScaler shim
# ---------------------------------------------------------------------------

def _patch_gradscaler() -> None:
    import torch
    # New unified API exists in torch>=2.0
    if not hasattr(torch, "amp") or not hasattr(torch.amp, "GradScaler"):
        return

    class _CudaGradScalerShim(torch.amp.GradScaler):  # type: ignore[misc]
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("device", "cuda")
            super().__init__(*args, **kwargs)

    try:
        torch.cuda.amp.GradScaler = _CudaGradScalerShim  # type: ignore[assignment]
    except Exception:
        pass

    # Some torch builds want device_type kw; accept it transparently.
    _orig_init = torch.amp.GradScaler.__init__

    def _init_compat(self, *args, **kwargs):
        if "device_type" in kwargs and "device" not in kwargs:
            kwargs["device"] = kwargs.pop("device_type")
        return _orig_init(self, *args, **kwargs)

    torch.amp.GradScaler.__init__ = _init_compat  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# (4) DataLoader: force num_workers=0 on Windows
# ---------------------------------------------------------------------------

def _patch_dataloader() -> None:
    if not IS_WINDOWS:
        return
    import torch.utils.data as _td

    _orig_dl_init = _td.DataLoader.__init__

    def _init_no_workers(self, *args, **kwargs):
        kwargs["num_workers"] = 0
        kwargs["persistent_workers"] = False
        kwargs.pop("prefetch_factor", None)
        return _orig_dl_init(self, *args, **kwargs)

    _td.DataLoader.__init__ = _init_no_workers  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# (2) Encoding: ensure config files open as UTF-8-with-BOM
# ---------------------------------------------------------------------------

def _patch_config_encoding() -> None:
    """Patch config_io.load_yaml_config to open files with encoding='utf-8-sig'."""
    # main_model must already be on sys.path.
    try:
        import config_io  # type: ignore
    except ImportError:
        return

    _orig = config_io.load_yaml_config

    def _safe_load(path: str):
        # Try the original first under utf-8-sig by temporarily patching `open`
        real_open = builtins.open

        def _open_utf8sig(file, mode="r", *a, **kw):
            if (isinstance(file, (str, bytes, os.PathLike))
                    and (str(file).endswith((".yaml", ".yml", ".json")))
                    and "b" not in mode
                    and "encoding" not in kw):
                kw["encoding"] = "utf-8-sig"
            return real_open(file, mode, *a, **kw)

        builtins.open = _open_utf8sig
        try:
            return _orig(path)
        finally:
            builtins.open = real_open

    config_io.load_yaml_config = _safe_load


# ---------------------------------------------------------------------------
# (5) Huber loss: dtype-safe scatter_add
# ---------------------------------------------------------------------------

def _patch_huber_loss() -> None:
    try:
        import ssl_train  # type: ignore
        import torch
    except ImportError:
        return

    def _huber_loss_parts_safe(pred, age, pid, delta, lam):
        # Always accumulate in float32, on the same device as `pred`.
        pred32 = pred.float()
        age32 = age.float() if age.dtype != torch.float32 else age
        # Base Huber on CPU is fine; keep parity with original behaviour but
        # in float32 throughout.
        pred_cpu = pred32.contiguous().cpu()
        age_cpu = age32.contiguous().cpu()
        diff = pred_cpu - age_cpu
        abs_diff = diff.abs()
        inner = abs_diff.clamp(max=delta)
        base = (0.5 * inner * inner + delta * (abs_diff - inner)).mean()
        if lam <= 0.0:
            return base, base, torch.zeros((), device=base.device, dtype=base.dtype)

        pid_long = pid.long()
        uniq, inv = torch.unique(pid_long, return_inverse=True)
        n_subj = uniq.numel()
        dev = pred32.device
        sums = torch.zeros(n_subj, device=dev, dtype=torch.float32).scatter_add_(
            0, inv.to(dev), pred32
        )
        cnts = torch.zeros(n_subj, device=dev, dtype=torch.float32).scatter_add_(
            0, inv.to(dev), torch.ones_like(pred32)
        )
        age_dev = age32.to(device=dev, dtype=torch.float32)
        age_sums = torch.zeros(n_subj, device=dev, dtype=torch.float32).scatter_add_(
            0, inv.to(dev), age_dev
        )
        pred_mean = sums / cnts.clamp(min=1.0)
        age_mean = age_sums / cnts.clamp(min=1.0)
        reg = ((pred_mean - age_mean) ** 2).mean()
        total = base + lam * reg
        return total, base, reg

    ssl_train._huber_loss_parts = _huber_loss_parts_safe


# ---------------------------------------------------------------------------
# (1) Preflight: dataset + age lookup
# ---------------------------------------------------------------------------

def preflight_paths(data_path: str, age_lookup_path: str) -> None:
    missing = []
    if not os.path.exists(data_path):
        missing.append(("dataset", data_path))
    if not os.path.exists(age_lookup_path):
        missing.append(("age lookup", age_lookup_path))
    if not missing:
        return

    msg_lines = ["[preflight] missing required input file(s):"]
    for name, p in missing:
        msg_lines.append(f"  - {name}: {p}")
    if any(n == "age lookup" for n, _ in missing):
        msg_lines.append(
            "  Build the age lookup with:  python experiments\\data_preparation\\build_age_lookup.py"
        )
    raise FileNotFoundError("\n".join(msg_lines))


# ---------------------------------------------------------------------------
# (6) matplotlib
# ---------------------------------------------------------------------------

def _check_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        warnings.warn(
            "matplotlib not installed; plots will be skipped. "
            "pip install matplotlib to enable.",
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def apply_all() -> None:
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True
    _patch_gradscaler()
    _patch_dataloader()
    _patch_config_encoding()
    _patch_huber_loss()
    _check_matplotlib()


apply_all()

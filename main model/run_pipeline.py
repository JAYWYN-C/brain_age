"""Orchestrate the final 0702 main V0/V1 experiment pipeline.

The default mode is `--dry-run`, which prints the exact commands without
running long training jobs. Remove `--dry-run` when ready.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from config_io import load_yaml_config
from dataset_catalog import DATASET_SPECS, DatasetSpec, dataset_aliases, get_dataset
from progress import progress_bar, progress_write


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


VARIANT_CONFIG = {
    "V0": "downstream_v0.yaml",
    "V1": "downstream_v1.yaml",
}

DEFAULT_STAGES = ("pretext", "downstream", "compare")
DEFAULT_SWEEP_CONFIG = os.path.join(HERE, "configs", "hparam_sweeps.json")
DEFAULT_SWEEPS = ("main_basicdeep",)


@dataclass(frozen=True)
class SweepSpec:
    name: str
    description: str = ""
    variants: Tuple[str, ...] = ()
    pretext: Dict[str, Any] = field(default_factory=dict)
    downstream: Dict[str, Any] = field(default_factory=dict)
    downstream_by_variant: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineConfig:
    python: str = sys.executable
    configs_dir: str = os.path.join(HERE, "configs")
    out_root: str = os.path.join(HERE, "outputs")
    datasets: Tuple[str, ...] = dataset_aliases()
    folds: str = ""
    variants: Tuple[str, ...] = ("BEST",)
    stages: Tuple[str, ...] = DEFAULT_STAGES
    sweeps: Tuple[str, ...] = DEFAULT_SWEEPS
    sweep_config_path: str = DEFAULT_SWEEP_CONFIG
    skip_existing: bool = False


def _append_folds(cmd: List[str], folds: str) -> List[str]:
    if folds:
        return [*cmd, "--folds", folds]
    return cmd


def _config_path(configs_dir: str, filename: str) -> str:
    return os.path.join(configs_dir, filename)


def _uses_sweep_subdirs(cfg: PipelineConfig) -> bool:
    return True


def dataset_out_root(
    cfg: PipelineConfig,
    dataset: DatasetSpec,
    sweep: Optional[SweepSpec] = None,
) -> str:
    root = os.path.join(cfg.out_root, dataset.alias)
    if sweep is not None and _uses_sweep_subdirs(cfg):
        return os.path.join(root, sweep.name)
    return root


def dataset_runtime_config_dir(
    cfg: PipelineConfig,
    dataset: DatasetSpec,
    sweep: Optional[SweepSpec] = None,
) -> str:
    return os.path.join(dataset_out_root(cfg, dataset, sweep), "configs")


def _dump_flat_yaml(path: str, values: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for key, value in values.items():
            if isinstance(value, (list, tuple)):
                rendered = "[" + ", ".join(str(v) for v in value) + "]"
            elif value is None:
                rendered = "null"
            elif isinstance(value, bool):
                rendered = "true" if value else "false"
            else:
                rendered = str(value)
            f.write(f"{key}: {rendered}\n")


def _override_dict(raw: Any, sweep_name: str, field_name: str) -> Dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"sweep {sweep_name}.{field_name} must be an object")
    return dict(raw)


def _load_sweep_catalog(path: str) -> Dict[str, Dict[str, Any]]:
    if not path:
        from public_configs import SWEEP_CATALOG
        return {name: dict(spec) for name, spec in SWEEP_CATALOG.items()}
    if not os.path.exists(path):
        if os.path.basename(path) == "hparam_sweeps.json":
            from public_configs import SWEEP_CATALOG
            return {name: dict(spec) for name, spec in SWEEP_CATALOG.items()}
        raise FileNotFoundError(f"--sweep-config path does not exist: {path}")
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and isinstance(raw.get("sweeps"), dict):
        raw = raw["sweeps"]
    if not isinstance(raw, dict):
        raise ValueError(f"sweep config {path} must contain a JSON object")
    catalog: Dict[str, Dict[str, Any]] = {}
    for name, spec in raw.items():
        if spec is None:
            spec = {}
        if not isinstance(spec, dict):
            raise ValueError(f"sweep {name} must be an object")
        catalog[str(name)] = dict(spec)
    return catalog


def _make_sweep(name: str, raw: Dict[str, Any]) -> SweepSpec:
    downstream_by_variant = _override_dict(raw.get("downstream_by_variant"), name, "downstream_by_variant")
    normalized_by_variant = {
        str(variant): _override_dict(overrides, name, f"downstream_by_variant.{variant}")
        for variant, overrides in downstream_by_variant.items()
    }
    variants = raw.get("variants", ())
    if isinstance(variants, str):
        variants = tuple(part.strip() for part in variants.split(",") if part.strip())
    elif isinstance(variants, list):
        variants = tuple(str(part).strip() for part in variants if str(part).strip())
    elif variants is None:
        variants = ()
    elif not isinstance(variants, tuple):
        raise ValueError(f"sweep {name}.variants must be a string or list")
    return SweepSpec(
        name=name,
        description=str(raw.get("description", "")),
        variants=tuple(str(v).upper() for v in variants),
        pretext=_override_dict(raw.get("pretext"), name, "pretext"),
        downstream=_override_dict(raw.get("downstream"), name, "downstream"),
        downstream_by_variant=normalized_by_variant,
    )


def load_sweep_specs(path: str, requested: Sequence[str]) -> Tuple[SweepSpec, ...]:
    catalog = _load_sweep_catalog(path)
    names = tuple(requested) or DEFAULT_SWEEPS
    if any(name.lower() == "all" for name in names):
        names = tuple(catalog.keys())

    missing = [name for name in names if name not in catalog]
    if missing:
        raise ValueError(
            f"unknown sweep(s) {missing}; expected one of {tuple(catalog.keys())} or all"
        )
    return tuple(_make_sweep(name, catalog[name]) for name in names)


def _variants_for_sweep(cfg: PipelineConfig, sweep: SweepSpec) -> Tuple[str, ...]:
    if any(variant.upper() == "BEST" for variant in cfg.variants):
        if not sweep.variants:
            raise ValueError(f"sweep {sweep.name} does not define variants for --variants BEST")
        return sweep.variants
    return tuple(variant.upper() for variant in cfg.variants)


def _needs_pretext(variants: Sequence[str]) -> bool:
    return "V1" in variants


def write_runtime_configs(
    cfg: PipelineConfig,
    dataset: DatasetSpec,
    sweep: Optional[SweepSpec] = None,
) -> Dict[str, str]:
    """Create dataset-specific configs under outputs/<dataset>/configs."""
    sweep = sweep or SweepSpec(name="main_basicdeep")
    runtime_dir = dataset_runtime_config_dir(cfg, dataset, sweep)
    out_root = dataset_out_root(cfg, dataset, sweep)
    pretext_dir = os.path.join(out_root, "pretext")

    pretext = load_yaml_config(_config_path(cfg.configs_dir, "ssl_pretext.yaml"))
    pretext.update(sweep.pretext)
    pretext.update({
        "data_path": dataset.path,
        "age_lookup_path": dataset.age_lookup,
        "out_dir": pretext_dir,
        "sweep": sweep.name,
    })
    pretext_path = os.path.join(runtime_dir, "ssl_pretext.yaml")
    _dump_flat_yaml(pretext_path, pretext)

    paths = {"pretext": pretext_path}
    for variant, filename in VARIANT_CONFIG.items():
        raw = load_yaml_config(_config_path(cfg.configs_dir, filename))
        raw.update(sweep.downstream)
        raw.update(sweep.downstream_by_variant.get(variant, {}))
        raw.update({
            "data_path": dataset.path,
            "age_lookup_path": dataset.age_lookup,
            "out_dir": os.path.join(out_root, f"downstream_{variant.lower()}"),
            "pretext_dir": pretext_dir,
            "variant": variant,
            "sweep": sweep.name,
        })
        out_path = os.path.join(runtime_dir, filename)
        _dump_flat_yaml(out_path, raw)
        paths[variant] = out_path
    return paths


def _dataset_commands(cfg: PipelineConfig, dataset: DatasetSpec) -> List[List[str]]:
    commands: List[List[str]] = []
    stages = set(cfg.stages)
    sweeps = load_sweep_specs(cfg.sweep_config_path, cfg.sweeps)
    all_summaries: List[str] = []

    for sweep in sweeps:
        variants = _variants_for_sweep(cfg, sweep)
        runtime_configs = write_runtime_configs(cfg, dataset, sweep)
        out_root = dataset_out_root(cfg, dataset, sweep)

        if "pretext" in stages and _needs_pretext(variants):
            commands.append(_append_folds([
                cfg.python,
                os.path.join(HERE, "ssl_train.py"),
                "--config",
                runtime_configs["pretext"],
            ], cfg.folds))

        summaries = [
            os.path.join(
                out_root,
                f"downstream_{variant.lower()}",
                f"{variant}_summary.json",
            )
            for variant in variants
        ]
        all_summaries.extend(summaries)

        if "downstream" in stages:
            for variant in variants:
                if variant not in VARIANT_CONFIG:
                    raise ValueError(f"unknown variant {variant}; expected one of {sorted(VARIANT_CONFIG)}")
                commands.append(_append_folds([
                    cfg.python,
                    os.path.join(HERE, "downstream_train.py"),
                    "--config",
                    runtime_configs[variant],
                ], cfg.folds))

        if "compare" in stages:
            commands.append([
                cfg.python,
                os.path.join(HERE, "compare_variants.py"),
                "--summaries",
                *summaries,
                "--out-dir",
                os.path.join(out_root, "compare"),
            ])

    if "compare" in stages and len(sweeps) > 1:
        commands.append([
            cfg.python,
            os.path.join(HERE, "compare_variants.py"),
            "--summaries",
            *all_summaries,
            "--out-dir",
            os.path.join(cfg.out_root, dataset.alias, "compare_sweeps"),
        ])

    return commands


def build_pipeline_commands(cfg: PipelineConfig) -> List[List[str]]:
    commands: List[List[str]] = []
    for alias in cfg.datasets:
        commands.extend(_dataset_commands(cfg, get_dataset(alias)))
    return commands


def missing_dataset_inputs(cfg: PipelineConfig) -> List[str]:
    missing: List[str] = []
    for alias in cfg.datasets:
        dataset = get_dataset(alias)
        if not os.path.exists(dataset.path):
            missing.append(dataset.path)
        if not os.path.exists(dataset.age_lookup):
            missing.append(dataset.age_lookup)
    return sorted(set(missing))


def _shell_join(cmd: Sequence[str]) -> str:
    return " ".join(cmd)


def _is_ssl_train_command(cmd: Sequence[str]) -> bool:
    return any(os.path.basename(str(part)) == "ssl_train.py" for part in cmd)


def _command_script_name(cmd: Sequence[str]) -> str:
    for part in cmd:
        base = os.path.basename(str(part))
        if base.endswith(".py"):
            return base
    return ""


def _command_config_path(cmd: Sequence[str]) -> Optional[str]:
    try:
        idx = list(cmd).index("--config")
    except ValueError:
        return None
    if idx + 1 >= len(cmd):
        return None
    return str(cmd[idx + 1])


def _pretext_gate_failure_message(cmd: Sequence[str]) -> Optional[str]:
    if not _is_ssl_train_command(cmd):
        return None

    config_path = _command_config_path(cmd)
    if not config_path or not os.path.exists(config_path):
        return None

    cfg = load_yaml_config(config_path)
    out_dir = str(cfg.get("out_dir", ""))
    if not out_dir:
        return None

    summary_path = os.path.join(out_dir, "pretext_summary.json")
    if not os.path.exists(summary_path):
        return None

    with open(summary_path) as f:
        summary = json.load(f)

    pass_mae = bool(summary.get("gate_pass_mae", False))
    pass_r = bool(summary.get("gate_pass_r", False))
    if pass_mae and pass_r:
        return None

    mae = summary.get("mean_test_subj_mae")
    r = summary.get("mean_test_subj_r")
    gate_mae = cfg.get("gate_mae")
    gate_r = cfg.get("gate_pearson_r")
    return (
        "pretext gate failed; stopping downstream pipeline "
        f"(MAE={mae}, gate_mae={gate_mae}, r={r}, gate_r={gate_r}, "
        f"summary={summary_path})"
    )


def check_pretext_gate_after_command(cmd: Sequence[str]) -> None:
    message = _pretext_gate_failure_message(cmd)
    if message:
        raise RuntimeError(message)


def _command_arg(cmd: Sequence[str], name: str) -> Optional[str]:
    parts = list(cmd)
    try:
        idx = parts.index(name)
    except ValueError:
        return None
    if idx + 1 >= len(parts):
        return None
    return str(parts[idx + 1])


def _parse_folds(raw: Any) -> Tuple[int, ...]:
    if raw is None:
        return ()
    if isinstance(raw, int):
        return (int(raw),)
    if isinstance(raw, (list, tuple)):
        return tuple(int(part) for part in raw)
    text = str(raw).strip().strip("[]")
    if not text:
        return ()
    return tuple(int(part.strip()) for part in text.replace(",", " ").split() if part.strip())


def _requested_command_folds(cmd: Sequence[str], cfg: Dict[str, Any]) -> Tuple[int, ...]:
    override = _command_arg(cmd, "--folds")
    if override is not None:
        return _parse_folds(override)
    return _parse_folds(cfg.get("folds"))


def _summary_completed_folds(summary: Dict[str, Any]) -> Tuple[int, ...]:
    folds = summary.get("folds") or summary.get("fold_results") or []
    out: List[int] = []
    if isinstance(folds, list):
        for row in folds:
            if isinstance(row, dict) and row.get("fold") is not None:
                out.append(int(row["fold"]))
            elif isinstance(row, int):
                out.append(int(row))
    if out:
        return tuple(sorted(set(out)))
    return _parse_folds((summary.get("config") or {}).get("folds"))


def _normalized_config_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalized_config_value(v) for v in value]
    if isinstance(value, list):
        return [_normalized_config_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalized_config_value(v) for k, v in sorted(value.items())}
    return value


def _summary_config_matches(expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
    ignored = {"folds"}
    for key, expected_value in expected.items():
        if key in ignored:
            continue
        if key not in actual:
            return False
        if _normalized_config_value(actual[key]) != _normalized_config_value(expected_value):
            return False
    return True


def _completed_training_summary_path(cmd: Sequence[str], cfg: Dict[str, Any]) -> Optional[str]:
    script = _command_script_name(cmd)
    out_dir = str(cfg.get("out_dir") or "")
    if script in {"ssl_train.py", "ssl_train_entry.py"} and out_dir:
        return os.path.join(out_dir, "pretext_summary.json")
    if script in {"downstream_train.py", "downstream_train_entry.py"} and out_dir:
        variant = str(_command_arg(cmd, "--variant") or cfg.get("variant") or "").upper()
        if variant:
            return os.path.join(out_dir, f"{variant}_summary.json")
    return None


def _completed_command_reason(cmd: Sequence[str]) -> Optional[str]:
    config_path = _command_config_path(cmd)
    if not config_path or not os.path.exists(config_path):
        return None
    cfg = load_yaml_config(config_path)
    summary_path = _completed_training_summary_path(cmd, cfg)
    if not summary_path or not os.path.exists(summary_path):
        return None

    with open(summary_path, encoding="utf-8-sig") as f:
        summary = json.load(f)
    if _command_script_name(cmd) in {"ssl_train.py", "ssl_train_entry.py"}:
        if not bool(summary.get("gate_pass_mae", False)) or not bool(summary.get("gate_pass_r", False)):
            return None
    actual_cfg = summary.get("config") or {}
    if not _summary_config_matches(cfg, actual_cfg):
        return None

    requested = set(_requested_command_folds(cmd, cfg))
    completed = set(_summary_completed_folds(summary))
    if requested and not requested.issubset(completed):
        return None
    return f"found matching summary at {summary_path}"


def _relativize_arg(arg: str) -> str:
    """cwd-relative if possible; absolute as fallback (cross-drive on Windows)."""
    if not isinstance(arg, str):
        return arg
    if os.sep not in arg and "/" not in arg:
        return arg
    try:
        rel = os.path.relpath(arg)
    except ValueError:
        return arg
    return rel if not rel.startswith("..") else arg


def _relativize_cmd(cmd: Sequence[str]) -> List[str]:
    return [_relativize_arg(a) for a in cmd]


def run_commands(commands: Iterable[Sequence[str]], dry_run: bool, skip_existing: bool = False) -> None:
    command_list = [_relativize_cmd(c) for c in commands]
    bar = progress_bar(total=len(command_list), desc="Pipeline overall", unit="cmd")
    try:
        for idx, cmd in enumerate(command_list, start=1):
            progress_write(f"\n[{idx}/{len(command_list)}] {_shell_join(cmd)}")
            bar.set_description(f"Pipeline command {idx}/{len(command_list)}")
            skip_reason = _completed_command_reason(cmd) if skip_existing else None
            if skip_reason:
                progress_write(f"    [skip-existing] {skip_reason}")
                bar.update(1)
                continue
            if not dry_run:
                subprocess.run(cmd, check=True)
                check_pretext_gate_after_command(cmd)
            bar.update(1)
    finally:
        bar.close()


def _split_csv(value: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--folds", default="", help="comma-separated fold ids, e.g. 0 or 0,1")
    p.add_argument("--datasets", default=",".join(dataset_aliases()))
    p.add_argument("--variants", default="BEST")
    p.add_argument("--stages", default=",".join(DEFAULT_STAGES))
    p.add_argument("--sweeps", default=",".join(DEFAULT_SWEEPS), help="comma-separated sweep names, or all")
    p.add_argument("--sweep-config", default=DEFAULT_SWEEP_CONFIG)
    p.add_argument("--configs-dir", default=os.path.join(HERE, "configs"))
    p.add_argument("--out-root", default=os.path.join(HERE, "outputs"))
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip pretext/downstream commands whose matching outputs already cover the requested folds.",
    )
    args = p.parse_args()

    cfg = PipelineConfig(
        python=args.python,
        configs_dir=args.configs_dir,
        out_root=args.out_root,
        datasets=_split_csv(args.datasets),
        folds=args.folds,
        variants=_split_csv(args.variants),
        stages=_split_csv(args.stages),
        sweeps=_split_csv(args.sweeps),
        sweep_config_path=args.sweep_config,
        skip_existing=args.skip_existing,
    )
    if not args.dry_run:
        missing = missing_dataset_inputs(cfg)
        if missing:
            msg = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(f"missing dataset inputs:\n{msg}")
    run_commands(build_pipeline_commands(cfg), dry_run=args.dry_run, skip_existing=cfg.skip_existing)


if __name__ == "__main__":
    main()

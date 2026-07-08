"""Compare downstream variant summary JSON files.

Example:
    python compare_variants.py \
      --summaries outputs/downstream_v0/V0_summary.json \
                  outputs/downstream_v1/V1_summary.json \
      --out-dir outputs/compare
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


CLASS_NAMES = ("CN", "MCI", "AD", "non-AD Dementia")


METRIC_FIELDS = (
    "mean_test_acc",
    "std_test_acc",
    "mean_test_macro_f1",
    "std_test_macro_f1",
)


def _variant_name(summary: Dict, path: str) -> str:
    cfg = summary.get("config") or {}
    if cfg.get("variant"):
        return str(cfg["variant"])
    base = os.path.basename(path)
    return base.replace("_summary.json", "")


def _sweep_name(summary: Dict) -> str:
    cfg = summary.get("config") or {}
    return str(cfg.get("sweep") or "main_basicdeep")


def _mean(values: List[float]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return float(np.mean(clean))


def _extract_per_class(summary: Dict) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    folds = summary.get("folds") or []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        for metric in ("f1", "sens", "spec", "precision"):
            values = []
            for fold in folds:
                per_class = ((fold.get("best") or {}).get("test_per_class") or [])
                if class_idx < len(per_class):
                    values.append(per_class[class_idx].get(metric))
            out[f"{class_name}_{metric}_mean"] = _mean(values)
    return out


def _load_row(path: str) -> Dict:
    with open(path) as f:
        summary = json.load(f)
    row = {
        "sweep": _sweep_name(summary),
        "variant": _variant_name(summary, path),
        "path": path,
    }
    for field in METRIC_FIELDS:
        row[field] = summary.get(field)
    row["n_folds"] = len(summary.get("folds") or [])
    row.update(_extract_per_class(summary))
    return row


def _fmt_mean_std(mean, std) -> str:
    if mean is None:
        return ""
    if std is None:
        return f"{float(mean):.4f}"
    return f"{float(mean):.4f}+-{float(std):.4f}"


def render_markdown(rows: List[Dict]) -> str:
    lines = [
        "| Sweep | Variant | Acc | Macro-F1 | CN F1 | MCI F1 | AD F1 | non-AD Dementia F1 | Folds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        variant = row["variant"]
        if row.get("is_best_macro_f1"):
            variant = f"**{variant}**"
        lines.append(
            "| {sweep} | {variant} | {acc} | {f1} | {cn_f1} | {mci_f1} | {ad_f1} | {dem_f1} | {folds} |".format(
                sweep=row.get("sweep", "main_basicdeep"),
                variant=variant,
                acc=_fmt_mean_std(row.get("mean_test_acc"), row.get("std_test_acc")),
                f1=_fmt_mean_std(
                    row.get("mean_test_macro_f1"), row.get("std_test_macro_f1")
                ),
                cn_f1=_fmt_mean_std(row.get("CN_f1_mean"), None),
                mci_f1=_fmt_mean_std(row.get("MCI_f1_mean"), None),
                ad_f1=_fmt_mean_std(row.get("AD_f1_mean"), None),
                dem_f1=_fmt_mean_std(row.get("non-AD Dementia_f1_mean"), None),
                folds=row.get("n_folds", 0),
            )
        )
    return "\n".join(lines) + "\n"


def compare_summaries(summary_paths: Iterable[str]) -> Tuple[List[Dict], str]:
    # Skip (with a clear warning) any summary that does not exist yet, instead
    # of crashing the whole pipeline with a raw FileNotFoundError. This was the
    # old failure mode when a V0/V1 summary was missing (e.g. a copied-in V0
    # that never landed). With the single-sweep V0+V1 design both summaries are
    # produced in the same run, so this is just a safety net.
    paths = list(summary_paths)
    existing = [p for p in paths if os.path.exists(p)]
    for p in paths:
        if p not in existing:
            print(f"[compare_variants] WARNING: summary not found, skipping: {p}",
                  flush=True)
    if not existing:
        raise FileNotFoundError(
            "compare_variants: none of the requested summary files exist:\n  "
            + "\n  ".join(paths)
            + "\n  -> run the 'downstream' stage first so V0_summary.json and "
              "V1_summary.json are created."
        )
    rows = [_load_row(path) for path in existing]
    rows.sort(key=lambda r: (r["sweep"], r["variant"]))
    best = max(
        (r for r in rows if r.get("mean_test_macro_f1") is not None),
        key=lambda r: float(r["mean_test_macro_f1"]),
        default=None,
    )
    for row in rows:
        row["is_best_macro_f1"] = bool(best is not None and row is best)
    return rows, render_markdown(rows)


def write_outputs(rows: List[Dict], markdown: str, out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "variant_comparison.csv")
    md_path = os.path.join(out_dir, "variant_comparison.md")
    with open(csv_path, "w", newline="") as f:
        per_class_fields = tuple(
            f"{class_name}_{metric}_mean"
            for class_name in CLASS_NAMES
            for metric in ("f1", "sens", "spec", "precision")
        )
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "sweep",
                "variant",
                *METRIC_FIELDS,
                *per_class_fields,
                "is_best_macro_f1",
                "n_folds",
                "path",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    with open(md_path, "w") as f:
        f.write(markdown)
    return {"csv": csv_path, "markdown": md_path}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summaries", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    rows, markdown = compare_summaries(args.summaries)
    paths = write_outputs(rows, markdown, args.out_dir)
    print(markdown)
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()

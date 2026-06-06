#!/usr/bin/env python3
"""Plot accuracy metrics from evaluation summary CSV files.

Supported summary styles:
1) pseudo-GT summary (eval_pseudo_gt_sequence.py)
   - precision/recall/f1/detection_rate
   - Pseudo_MPJPE_mm / Pseudo_PA_MPJPE_mm / Pseudo_PVE_mm / Pseudo_PA_PVE_mm
2) GT summary (eval_3dpw_trt.py etc.)
   - precision/recall/f1/detection_rate
   - MPJPE_mm / PA_MPJPE_mm / PVE_mm / PA_PVE_mm
3) compare CSV (left/right columns)
   - precision_left/right, ... , Pseudo_MPJPE_mm_left/right, ...
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class EvalRun:
    run_name: str
    run_label: str
    framework: str
    source_csv: str
    precision: float
    recall: float
    f1: float
    detection_rate: float
    mpjpe_mm: float
    pa_mpjpe_mm: float
    pve_mm: float
    pa_pve_mm: float
    gt_instances: float = math.nan
    tp_instances: float = math.nan
    fn_instances: float = math.nan
    fp_instances: float = math.nan
    details_csv: str = ""
    precision_std: float = math.nan
    precision_sem: float = math.nan
    recall_std: float = math.nan
    recall_sem: float = math.nan
    f1_std: float = math.nan
    f1_sem: float = math.nan
    detection_rate_std: float = math.nan
    detection_rate_sem: float = math.nan
    mpjpe_mm_std: float = math.nan
    mpjpe_mm_sem: float = math.nan
    pa_mpjpe_mm_std: float = math.nan
    pa_mpjpe_mm_sem: float = math.nan
    pve_mm_std: float = math.nan
    pve_mm_sem: float = math.nan
    pa_pve_mm_std: float = math.nan
    pa_pve_mm_sem: float = math.nan


def _to_float(text: object) -> float:
    if text is None:
        return math.nan
    s = str(text).strip()
    if not s:
        return math.nan
    try:
        return float(s)
    except ValueError:
        return math.nan


def _first_float(row: Dict[str, str], keys: Sequence[str]) -> float:
    for k in keys:
        if k in row:
            v = _to_float(row[k])
            if not math.isnan(v):
                return v
    return math.nan


def _first_text(row: Dict[str, str], keys: Sequence[str]) -> str:
    for k in keys:
        if k in row:
            s = (row.get(k) or "").strip()
            if s:
                return s
    return ""


def calc_std_sem(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return math.nan, math.nan
    std = float(np.std(arr))
    sem = float(std / math.sqrt(arr.size))
    return std, sem


def infer_framework(*texts: str) -> str:
    blob = " ".join(t.lower() for t in texts if t)
    if "romp" in blob:
        return "ROMP"
    if "sat_hmr" in blob or "sathmr" in blob or "trt" in blob or "tensorrt" in blob:
        return "SAT-HMR"
    return "Unknown"


def make_run_label(run_name: str, framework: str) -> str:
    if not framework or framework == "Unknown":
        return run_name
    norm_run = run_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    norm_fw = framework.lower().replace("-", "").replace("_", "").replace(" ", "")
    if norm_run.startswith(norm_fw):
        return run_name
    return f"{framework}:{run_name}"


def parse_compare_row(row: Dict[str, str], csv_path: Path) -> List[EvalRun]:
    out: List[EvalRun] = []
    if "left_run" not in row or "right_run" not in row:
        return out

    for side in ("left", "right"):
        run_name = (row.get(f"{side}_run", "") or "").strip()
        if not run_name:
            run_name = f"{csv_path.stem}_{side}"
        pred_npz = (row.get(f"{side}_pred_npz", "") or "").strip()
        framework = infer_framework(run_name, pred_npz, str(csv_path))
        out.append(
            EvalRun(
                run_name=run_name,
                run_label=make_run_label(run_name, framework),
                framework=framework,
                source_csv=str(csv_path),
                precision=_first_float(row, [f"precision_{side}"]),
                recall=_first_float(row, [f"recall_{side}"]),
                f1=_first_float(row, [f"f1_{side}"]),
                detection_rate=_first_float(row, [f"detection_rate_{side}"]),
                mpjpe_mm=_first_float(row, [f"Pseudo_MPJPE_mm_{side}", f"MPJPE_mm_{side}"]),
                pa_mpjpe_mm=_first_float(row, [f"Pseudo_PA_MPJPE_mm_{side}", f"PA_MPJPE_mm_{side}"]),
                pve_mm=_first_float(row, [f"Pseudo_PVE_mm_{side}", f"PVE_mm_{side}"]),
                pa_pve_mm=_first_float(row, [f"Pseudo_PA_PVE_mm_{side}", f"PA_PVE_mm_{side}"]),
                gt_instances=_first_float(row, [f"gt_instances_{side}"]),
                tp_instances=_first_float(row, [f"tp_instances_{side}"]),
                fn_instances=_first_float(row, [f"fn_instances_{side}"]),
                fp_instances=_first_float(row, [f"fp_instances_{side}"]),
            )
        )
    return out


def parse_summary_row(row: Dict[str, str], csv_path: Path, row_index: int) -> EvalRun:
    run_name = (
        (row.get("run_name", "") or "").strip()
        or (row.get("backend", "") or "").strip()
        or Path((row.get("pred_npz", "") or "").strip()).stem
        or f"{csv_path.stem}_row{row_index}"
    )
    framework = infer_framework(
        run_name,
        (row.get("pred_npz", "") or "").strip(),
        (row.get("engine", "") or "").strip(),
        str(csv_path),
    )
    return EvalRun(
        run_name=run_name,
        run_label=make_run_label(run_name, framework),
        framework=framework,
        source_csv=str(csv_path),
        precision=_first_float(row, ["precision"]),
        recall=_first_float(row, ["recall"]),
        f1=_first_float(row, ["f1"]),
        detection_rate=_first_float(row, ["detection_rate"]),
        mpjpe_mm=_first_float(row, ["Pseudo_MPJPE_mm", "MPJPE_mm"]),
        pa_mpjpe_mm=_first_float(row, ["Pseudo_PA_MPJPE_mm", "PA_MPJPE_mm"]),
        pve_mm=_first_float(row, ["Pseudo_PVE_mm", "PVE_mm"]),
        pa_pve_mm=_first_float(row, ["Pseudo_PA_PVE_mm", "PA_PVE_mm"]),
        gt_instances=_first_float(row, ["gt_instances"]),
        tp_instances=_first_float(row, ["tp_instances"]),
        fn_instances=_first_float(row, ["fn_instances"]),
        fp_instances=_first_float(row, ["fp_instances"]),
        details_csv=_first_text(row, ["details_csv"]),
    )


def load_eval_runs(csv_paths: Sequence[Path]) -> List[EvalRun]:
    runs: List[EvalRun] = []
    for path in csv_paths:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if not row:
                    continue
                if "left_run" in row and "right_run" in row:
                    runs.extend(parse_compare_row(row, path))
                    continue
                runs.append(parse_summary_row(row, path, i))

    dedup: Dict[str, EvalRun] = {}
    for r in runs:
        key = f"{r.source_csv}::{r.run_label}"
        dedup[key] = r
    return list(dedup.values())


def _as_count(v: float) -> Optional[int]:
    if math.isnan(v):
        return None
    x = int(round(v))
    if x < 0:
        return None
    return x


def _binomial_std_sem(p: float, n: Optional[int]) -> Tuple[float, float]:
    if math.isnan(p) or n is None or n <= 0:
        return math.nan, math.nan
    var = max(p * (1.0 - p), 0.0)
    std = math.sqrt(var)
    sem = std / math.sqrt(n)
    return std, sem


def _find_details_csv(source_csv: Path) -> Optional[Path]:
    cands: List[Path] = []
    src_name = source_csv.name
    if "summary" in src_name:
        cands.append(source_csv.with_name(src_name.replace("summary", "details")))
    stem = source_csv.stem
    if "_summary" in stem:
        cands.append(source_csv.with_name(stem.replace("_summary", "_details") + source_csv.suffix))
    cands.append(source_csv.with_name(stem + "_details" + source_csv.suffix))

    seen: Dict[str, Path] = {}
    for p in cands:
        seen[str(p)] = p
    for p in seen.values():
        if p.is_file():
            return p
    return None


def _load_details_error_map(details_csv: Path) -> Dict[str, Tuple[float, float]]:
    values: Dict[str, List[float]] = {
        "mpjpe_mm": [],
        "pa_mpjpe_mm": [],
        "pve_mm": [],
        "pa_pve_mm": [],
    }
    key_map = {
        "mpjpe_mm": ("pseudo_mpjpe_mm", "mpjpe_mm"),
        "pa_mpjpe_mm": ("pseudo_pa_mpjpe_mm", "pa_mpjpe_mm"),
        "pve_mm": ("pseudo_pve_mm", "pve_mm"),
        "pa_pve_mm": ("pseudo_pa_pve_mm", "pa_pve_mm"),
    }
    with details_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for dst, keys in key_map.items():
                val = _first_float(row, keys)
                if not math.isnan(val):
                    values[dst].append(val)
    return {k: calc_std_sem(v) for k, v in values.items()}


def _attach_errors(runs: Sequence[EvalRun]) -> None:
    details_cache: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for r in runs:
        tp = _as_count(r.tp_instances)
        fp = _as_count(r.fp_instances)
        fn = _as_count(r.fn_instances)
        gt = _as_count(r.gt_instances)

        n_precision = (tp + fp) if tp is not None and fp is not None else None
        n_recall = (tp + fn) if tp is not None and fn is not None else gt

        r.precision_std, r.precision_sem = _binomial_std_sem(r.precision, n_precision)
        r.recall_std, r.recall_sem = _binomial_std_sem(r.recall, n_recall)
        r.detection_rate_std, r.detection_rate_sem = _binomial_std_sem(r.detection_rate, n_recall)

        if not math.isnan(r.f1) and not math.isnan(r.precision) and not math.isnan(r.recall):
            if n_precision and n_recall and n_precision > 0 and n_recall > 0 and (r.precision + r.recall) > 0:
                p = r.precision
                q = r.recall
                den = p + q
                dfdp = (2.0 * q * q) / (den * den)
                dfdq = (2.0 * p * p) / (den * den)
                var_p = max(p * (1.0 - p), 0.0) / n_precision
                var_q = max(q * (1.0 - q), 0.0) / n_recall
                var_f1 = max(dfdp * dfdp * var_p + dfdq * dfdq * var_q, 0.0)
                r.f1_sem = math.sqrt(var_f1)
                r.f1_std = r.f1_sem * math.sqrt(min(n_precision, n_recall))

        details_path = Path(r.details_csv) if r.details_csv else None
        if details_path and not details_path.is_file():
            details_path = None
        if details_path is None:
            guessed = _find_details_csv(Path(r.source_csv))
            if guessed is not None:
                details_path = guessed

        if details_path is None:
            continue

        r.details_csv = str(details_path)
        cache_key = str(details_path.resolve())
        if cache_key not in details_cache:
            details_cache[cache_key] = _load_details_error_map(details_path)
        err_map = details_cache[cache_key]
        r.mpjpe_mm_std, r.mpjpe_mm_sem = err_map.get("mpjpe_mm", (math.nan, math.nan))
        r.pa_mpjpe_mm_std, r.pa_mpjpe_mm_sem = err_map.get("pa_mpjpe_mm", (math.nan, math.nan))
        r.pve_mm_std, r.pve_mm_sem = err_map.get("pve_mm", (math.nan, math.nan))
        r.pa_pve_mm_std, r.pa_pve_mm_sem = err_map.get("pa_pve_mm", (math.nan, math.nan))


def _get_error_value(run: EvalRun, value_key: str, error_bar: str) -> float:
    if error_bar == "none":
        return math.nan
    suffix = "std" if error_bar == "std" else "sem"
    attr_map = {
        "precision": f"precision_{suffix}",
        "recall": f"recall_{suffix}",
        "f1": f"f1_{suffix}",
        "detection_rate": f"detection_rate_{suffix}",
        "mpjpe_mm": f"mpjpe_mm_{suffix}",
        "pa_mpjpe_mm": f"pa_mpjpe_mm_{suffix}",
        "pve_mm": f"pve_mm_{suffix}",
        "pa_pve_mm": f"pa_pve_mm_{suffix}",
    }
    attr = attr_map.get(value_key, "")
    if not attr:
        return math.nan
    return float(getattr(run, attr, math.nan))


def _bar_plot(
    runs: Sequence[EvalRun],
    value_key: str,
    title: str,
    ylabel: str,
    outpath: Path,
    descending: bool,
    error_bar: str,
) -> None:
    rows = []
    for r in runs:
        v = float(getattr(r, value_key))
        if math.isnan(v):
            continue
        err = _get_error_value(r, value_key, error_bar)
        rows.append((r.run_label, v, err))
    if not rows:
        return
    if len(rows) < 2:
        print(f"[SKIP] single-series bar plot: {outpath.name}")
        return

    rows.sort(key=lambda x: x[1], reverse=descending)
    labels = [x[0] for x in rows]
    values = [x[1] for x in rows]
    errors = [0.0 if math.isnan(x[2]) else max(0.0, x[2]) for x in rows]
    has_error = error_bar != "none" and any(e > 0.0 for e in errors)

    width = max(9.0, 0.7 * len(labels))
    fig, ax = plt.subplots(figsize=(width, 5.0))
    bars = ax.bar(
        np.arange(len(labels)),
        values,
        yerr=errors if has_error else None,
        capsize=4 if has_error else 0,
        ecolor="black",
        error_kw={"elinewidth": 1.0},
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    if has_error:
        ax.text(
            0.01,
            0.99,
            f"Error bar: {error_bar}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="dimgray",
        )
    for b, v, e in zip(bars, values, errors):
        y = v + e if e > 0 else v
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            y,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def write_summary(runs: Sequence[EvalRun], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_name",
        "run_label",
        "framework",
        "source_csv",
        "details_csv",
        "gt_instances",
        "tp_instances",
        "fn_instances",
        "fp_instances",
        "precision",
        "precision_std",
        "precision_sem",
        "recall",
        "recall_std",
        "recall_sem",
        "f1",
        "f1_std",
        "f1_sem",
        "detection_rate",
        "detection_rate_std",
        "detection_rate_sem",
        "mpjpe_mm",
        "mpjpe_mm_std",
        "mpjpe_mm_sem",
        "pa_mpjpe_mm",
        "pa_mpjpe_mm_std",
        "pa_mpjpe_mm_sem",
        "pve_mm",
        "pve_mm_std",
        "pve_mm_sem",
        "pa_pve_mm",
        "pa_pve_mm_std",
        "pa_pve_mm_sem",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in runs:
            writer.writerow({k: getattr(r, k) for k in fields})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot accuracy metrics from evaluation summary CSVs.")
    p.add_argument("--inputs", nargs="+", required=True, help="Input evaluation summary CSV paths/globs.")
    p.add_argument("--outdir", default="plots/accuracy_plots", help="Output directory for plots and merged summary.")
    p.add_argument(
        "--error-bar",
        choices=["none", "std", "sem"],
        default="sem",
        help="Error bar type for bar plots.",
    )
    return p.parse_args()


def expand_inputs(inputs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_file():
            paths.append(p)
            continue
        for m in sorted(Path().glob(raw)):
            if m.is_file():
                paths.append(m)
    dedup: Dict[str, Path] = {}
    for p in paths:
        dedup[str(p.resolve())] = p
    return sorted(dedup.values(), key=lambda x: str(x))


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_paths = expand_inputs(args.inputs)
    if not csv_paths:
        print("[ERROR] no input CSV found")
        return 2

    runs = load_eval_runs(csv_paths)
    if not runs:
        print("[ERROR] no evaluable rows found in inputs")
        return 2

    _attach_errors(runs)
    write_summary(runs, outdir / "accuracy_comparison_summary.csv")

    _bar_plot(
        runs,
        "mpjpe_mm",
        "MPJPE (mm) by Run",
        "MPJPE (mm)",
        outdir / "runs_mpjpe_mm.png",
        descending=False,
        error_bar=args.error_bar,
    )
    _bar_plot(
        runs,
        "pa_mpjpe_mm",
        "PA-MPJPE (mm) by Run",
        "PA-MPJPE (mm)",
        outdir / "runs_pa_mpjpe_mm.png",
        descending=False,
        error_bar=args.error_bar,
    )
    _bar_plot(
        runs,
        "pve_mm",
        "PVE (mm) by Run",
        "PVE (mm)",
        outdir / "runs_pve_mm.png",
        descending=False,
        error_bar=args.error_bar,
    )
    _bar_plot(
        runs,
        "pa_pve_mm",
        "PA-PVE (mm) by Run",
        "PA-PVE (mm)",
        outdir / "runs_pa_pve_mm.png",
        descending=False,
        error_bar=args.error_bar,
    )

    _bar_plot(
        runs,
        "precision",
        "Precision by Run",
        "Precision",
        outdir / "runs_precision.png",
        descending=True,
        error_bar=args.error_bar,
    )
    _bar_plot(
        runs,
        "recall",
        "Recall by Run",
        "Recall",
        outdir / "runs_recall.png",
        descending=True,
        error_bar=args.error_bar,
    )
    _bar_plot(
        runs,
        "f1",
        "F1 by Run",
        "F1",
        outdir / "runs_f1.png",
        descending=True,
        error_bar=args.error_bar,
    )
    _bar_plot(
        runs,
        "detection_rate",
        "Detection Rate by Run",
        "Detection Rate",
        outdir / "runs_detection_rate.png",
        descending=True,
        error_bar=args.error_bar,
    )

    print(f"[DONE] runs: {len(runs)}")
    print(f"[DONE] error_bar: {args.error_bar}")
    print(f"[DONE] outdir: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

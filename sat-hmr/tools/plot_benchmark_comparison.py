#!/usr/bin/env python3
"""
Benchmark CSV comparison plotter for SAT-HMR / ROMP style outputs.

Supported input examples:
- inference_stats_*.csv
- TCP_inference_stats_*.csv

The script parses:
1) Summary section ("# 統計サマリー")
2) Frame section ("# 全フレームデータ")

and generates:
- consolidated summary CSV
- long-format frame CSV
- run-level plots
- group-level plots for arbitrary axes (framework/backbone/transport/etc.)
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RunResult:
    path: Path
    run_name: str
    run_label: str
    framework: str
    backbone: str
    transport: str
    model: str
    tags: str
    frame_count: int
    detected_count: int
    inference_times: List[float]
    total_times: List[float]
    frame_series: List[Tuple[int, Optional[float], Optional[float]]]
    summary_metrics: Dict[str, float]


VALID_META_KEYS = {"run_name", "framework", "backbone", "transport", "model", "tags"}


def make_run_label(run_name: str, framework: str) -> str:
    if not framework or framework == "Unknown":
        return run_name
    norm_run = run_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    norm_fw = framework.lower().replace("-", "").replace("_", "").replace(" ", "")
    if norm_run.startswith(norm_fw):
        return run_name
    return f"{framework}:{run_name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare benchmark CSV files and generate plots."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input paths/globs/directories containing benchmark CSV files.",
    )
    parser.add_argument(
        "--outdir",
        default="plots",
        help="Output directory for plots and merged CSVs.",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help=(
            "Optional metadata CSV override. Columns: "
            "path,run_name,framework,backbone,transport,model,tags"
        ),
    )
    parser.add_argument(
        "--group-by",
        nargs="+",
        default=["framework", "backbone", "transport", "framework+backbone+transport"],
        help=(
            "Grouping keys for aggregate plots. "
            "Use '+' for composite keys (e.g. framework+transport)."
        ),
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=15,
        help="Moving-average window for per-frame line plots.",
    )
    parser.add_argument(
        "--max-runs-line-plot",
        type=int,
        default=12,
        help="Skip dense line plot if run count exceeds this number.",
    )
    parser.add_argument(
        "--write-manifest-template",
        action="store_true",
        help="Write inferred metadata template CSV to outdir.",
    )
    parser.add_argument(
        "--error-bar",
        choices=["none", "std", "sem"],
        default="std",
        help="Error bar type for bar plots: std or sem.",
    )
    return parser.parse_args()


def to_float(value: str) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    low = text.lower()
    if low.startswith("n/a"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def to_int(value: str) -> Optional[int]:
    fval = to_float(value)
    if fval is None:
        return None
    return int(round(fval))


def infer_metadata(path: Path) -> Dict[str, str]:
    name = path.stem
    low = name.lower()

    if "sat_hmr" in low or "sathmr" in low:
        framework = "SAT-HMR"
    elif "romp" in low:
        framework = "ROMP"
    else:
        framework = "Unknown"

    if "hrnet" in low:
        backbone = "HRNet"
    elif "resnet" in low:
        backbone = "ResNet"
    elif framework == "SAT-HMR":
        backbone = "SAT"
    else:
        backbone = "Unknown"

    if "tcp" in low:
        transport = "TCP"
    elif "udp" in low:
        transport = "UDP"
    elif low.startswith("inference_stats_"):
        # Legacy naming in some ROMP/SAT-HMR scripts (non-TCP benchmark outputs)
        transport = "UDP"
    else:
        transport = "Unknown"

    return {
        "run_name": name,
        "framework": framework,
        "backbone": backbone,
        "transport": transport,
        "model": framework if framework != "Unknown" else name,
        "tags": "",
    }


def load_manifest(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            normalized = {
                str(k).strip().lower(): str(v).strip()
                for k, v in row.items()
                if k is not None
            }
            if not any(normalized.values()):
                continue
            rows.append(normalized)
    return rows


def find_manifest_row(path: Path, manifest_rows: Sequence[Dict[str, str]]) -> Optional[Dict[str, str]]:
    abs_path = str(path.resolve())
    candidates = {abs_path, str(path), path.name, path.stem}
    for row in manifest_rows:
        key = row.get("path", "").strip()
        if not key:
            continue
        if key in candidates:
            return row
        if abs_path.endswith(key):
            return row
    return None


def apply_manifest(meta: Dict[str, str], row: Optional[Dict[str, str]]) -> Dict[str, str]:
    if row is None:
        return meta
    out = dict(meta)
    for key in VALID_META_KEYS:
        val = row.get(key, "").strip()
        if val:
            out[key] = val
    return out


def expand_input_paths(inputs: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(x for x in p.rglob("*.csv") if x.is_file()))
            continue
        if p.is_file():
            files.append(p)
            continue
        matched = [Path(m) for m in glob.glob(item, recursive=True)]
        if matched:
            files.extend(sorted(x for x in matched if x.is_file()))

    dedup: Dict[str, Path] = {}
    for fpath in files:
        dedup[str(fpath.resolve())] = fpath
    return sorted(dedup.values(), key=lambda x: str(x))


def parse_frame_section(frame_header: Sequence[str], frame_rows: Sequence[Sequence[str]]) -> Tuple[
    List[Tuple[int, Optional[float], Optional[float]]], List[float], List[float]
]:
    header = [h.strip() for h in frame_header]
    lower = [h.lower() for h in header]

    frame_idx = lower.index("frame_id") if "frame_id" in lower else 0

    infer_idx = None
    total_idx = None
    for i, name in enumerate(lower):
        if infer_idx is None and "inference" in name:
            infer_idx = i
        if total_idx is None and "total" in name:
            total_idx = i

    series: List[Tuple[int, Optional[float], Optional[float]]] = []
    infer_vals: List[float] = []
    total_vals: List[float] = []

    for row_id, row in enumerate(frame_rows):
        if not row:
            continue
        fid_raw = row[frame_idx] if frame_idx < len(row) else str(row_id)
        fid = to_int(fid_raw)
        if fid is None:
            fid = row_id

        inf_val = to_float(row[infer_idx]) if infer_idx is not None and infer_idx < len(row) else None
        tot_val = to_float(row[total_idx]) if total_idx is not None and total_idx < len(row) else None

        series.append((fid, inf_val, tot_val))
        if inf_val is not None:
            infer_vals.append(inf_val)
        if tot_val is not None:
            total_vals.append(tot_val)

    return series, infer_vals, total_vals


def parse_benchmark_csv(path: Path) -> Tuple[List[Tuple[int, Optional[float], Optional[float]]], Dict[str, float], List[float], List[float]]:
    section = ""
    summary_header: List[str] = []
    summary_rows: Dict[str, List[str]] = {}
    frame_header: List[str] = []
    frame_rows: List[List[str]] = []

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for raw in reader:
            row = [cell.strip() for cell in raw]
            if not row or not any(row):
                continue
            first = row[0]
            lower_first = first.lower()

            if first.startswith("#"):
                if "統計サマリー" in first or "summary" in lower_first:
                    section = "summary"
                    summary_header = []
                elif "全フレームデータ" in first or "frame" in lower_first:
                    section = "frames"
                    frame_header = []
                continue

            if section == "summary":
                if not summary_header:
                    summary_header = row
                    continue
                summary_rows[row[0]] = row[1:]
                continue

            if section == "frames":
                if not frame_header:
                    frame_header = row
                    continue
                frame_rows.append(row)
                continue

            if lower_first == "frame_id":
                section = "frames"
                frame_header = row

    if not frame_header:
        raise ValueError("Frame section not found")

    series, inf_vals, total_vals = parse_frame_section(frame_header, frame_rows)
    metrics = compute_metrics(inf_vals, total_vals, summary_rows)
    return series, metrics, inf_vals, total_vals


def calc_std_sem(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return math.nan, math.nan
    std = float(np.std(arr))
    sem = float(std / math.sqrt(arr.size))
    return std, sem


def compute_metrics(
    inference_times: Sequence[float], total_times: Sequence[float], summary_rows: Dict[str, Sequence[str]]
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    inf = np.array(inference_times, dtype=np.float64) if inference_times else np.array([], dtype=np.float64)
    tot = np.array(total_times, dtype=np.float64) if total_times else np.array([], dtype=np.float64)

    if inf.size > 0:
        metrics["mean_inference_ms"] = float(np.mean(inf))
        std_inf, sem_inf = calc_std_sem(inf)
        metrics["std_inference_ms"] = std_inf
        metrics["sem_inference_ms"] = sem_inf
        metrics["median_inference_ms"] = float(np.median(inf))
        metrics["min_inference_ms"] = float(np.min(inf))
        metrics["max_inference_ms"] = float(np.max(inf))
        metrics["fps_inference"] = float(1000.0 / metrics["mean_inference_ms"])
        fps_inf = 1000.0 / inf
        std_fps_inf, sem_fps_inf = calc_std_sem(fps_inf)
        metrics["std_fps_inference"] = std_fps_inf
        metrics["sem_fps_inference"] = sem_fps_inf
        metrics["n_inference_frames"] = float(inf.size)
    else:
        metrics["mean_inference_ms"] = math.nan
        metrics["std_inference_ms"] = math.nan
        metrics["sem_inference_ms"] = math.nan
        metrics["median_inference_ms"] = math.nan
        metrics["min_inference_ms"] = math.nan
        metrics["max_inference_ms"] = math.nan
        metrics["fps_inference"] = math.nan
        metrics["std_fps_inference"] = math.nan
        metrics["sem_fps_inference"] = math.nan
        metrics["n_inference_frames"] = 0.0

    if tot.size > 0:
        metrics["mean_total_ms"] = float(np.mean(tot))
        std_tot, sem_tot = calc_std_sem(tot)
        metrics["std_total_ms"] = std_tot
        metrics["sem_total_ms"] = sem_tot
        metrics["median_total_ms"] = float(np.median(tot))
        metrics["min_total_ms"] = float(np.min(tot))
        metrics["max_total_ms"] = float(np.max(tot))
        metrics["fps_total"] = float(1000.0 / metrics["mean_total_ms"])
        fps_tot = 1000.0 / tot
        std_fps_tot, sem_fps_tot = calc_std_sem(fps_tot)
        metrics["std_fps_total"] = std_fps_tot
        metrics["sem_fps_total"] = sem_fps_tot
        metrics["n_total_frames"] = float(tot.size)
    else:
        metrics["mean_total_ms"] = math.nan
        metrics["std_total_ms"] = math.nan
        metrics["sem_total_ms"] = math.nan
        metrics["median_total_ms"] = math.nan
        metrics["min_total_ms"] = math.nan
        metrics["max_total_ms"] = math.nan
        metrics["fps_total"] = math.nan
        metrics["std_fps_total"] = math.nan
        metrics["sem_fps_total"] = math.nan
        metrics["n_total_frames"] = 0.0

    # fallback values from summary rows when needed
    if math.isnan(metrics["fps_total"]):
        fps_total_row = summary_rows.get("FPS (全体)", [])
        for value in fps_total_row:
            parsed = to_float(value)
            if parsed is not None:
                metrics["fps_total"] = parsed
                break

    return metrics


def weighted_average(values: Sequence[float], weights: Sequence[float]) -> float:
    filtered = [(v, w) for v, w in zip(values, weights) if not math.isnan(v) and w > 0]
    if not filtered:
        return math.nan
    vals, wts = zip(*filtered)
    return float(np.average(vals, weights=wts))


def concat_or_empty(arrays: Sequence[Sequence[float]]) -> np.ndarray:
    parts = [np.asarray(a, dtype=np.float64) for a in arrays if len(a) > 0]
    if not parts:
        return np.array([], dtype=np.float64)
    return np.concatenate(parts, axis=0)


def aggregate_by_group(runs: Sequence[RunResult], group_spec: str) -> List[Dict[str, object]]:
    keys = [k.strip() for k in group_spec.split("+") if k.strip()]
    buckets: Dict[Tuple[str, ...], List[RunResult]] = {}
    for run in runs:
        key = tuple(str(getattr(run, k, "Unknown")) for k in keys)
        buckets.setdefault(key, []).append(run)

    rows: List[Dict[str, object]] = []
    for key, members in sorted(buckets.items(), key=lambda kv: kv[0]):
        frameworks = sorted({m.framework for m in members})
        frameworks_text = ",".join(frameworks)
        label_base = " | ".join(key)
        if "framework" in keys:
            label = label_base
        else:
            label = f"[{frameworks_text}] {label_base}"

        inf_vals = [m.summary_metrics["mean_inference_ms"] for m in members]
        inf_wts = [max(m.frame_count, 1) for m in members]
        tot_vals = [m.summary_metrics["mean_total_ms"] for m in members]
        tot_wts = [max(m.detected_count, 1) for m in members]

        mean_inf = weighted_average(inf_vals, inf_wts)
        mean_tot = weighted_average(tot_vals, tot_wts)
        inf_concat = concat_or_empty([m.inference_times for m in members])
        tot_concat = concat_or_empty([m.total_times for m in members])
        std_inf, sem_inf = calc_std_sem(inf_concat)
        std_tot, sem_tot = calc_std_sem(tot_concat)

        fps_inf_concat = (1000.0 / inf_concat) if inf_concat.size > 0 else np.array([], dtype=np.float64)
        fps_tot_concat = (1000.0 / tot_concat) if tot_concat.size > 0 else np.array([], dtype=np.float64)
        std_fps_inf, sem_fps_inf = calc_std_sem(fps_inf_concat)
        std_fps_tot, sem_fps_tot = calc_std_sem(fps_tot_concat)

        row: Dict[str, object] = {
            "group_spec": group_spec,
            "group_label": label,
            "frameworks": frameworks_text,
            "run_count": len(members),
            "sum_frames": sum(m.frame_count for m in members),
            "sum_detected_frames": sum(m.detected_count for m in members),
            "mean_inference_ms": mean_inf,
            "std_inference_ms": std_inf,
            "sem_inference_ms": sem_inf,
            "mean_total_ms": mean_tot,
            "std_total_ms": std_tot,
            "sem_total_ms": sem_tot,
            "fps_inference": (1000.0 / mean_inf) if not math.isnan(mean_inf) and mean_inf > 0 else math.nan,
            "std_fps_inference": std_fps_inf,
            "sem_fps_inference": sem_fps_inf,
            "fps_total": (1000.0 / mean_tot) if not math.isnan(mean_tot) and mean_tot > 0 else math.nan,
            "std_fps_total": std_fps_tot,
            "sem_fps_total": sem_fps_tot,
            "n_inference_frames": int(inf_concat.size),
            "n_total_frames": int(tot_concat.size),
        }
        for idx, key_name in enumerate(keys):
            row[key_name] = key[idx]
        rows.append(row)
    return rows


def moving_average(values: Sequence[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    if window <= 1 or arr.size < window:
        return arr
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(arr, kernel, mode="same")


def save_bar_plot(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    ylabel: str,
    outpath: Path,
    errors: Optional[Sequence[float]] = None,
    error_label: str = "",
    descending: bool = False,
) -> None:
    if errors is None:
        errors = [math.nan] * len(values)
    data = [
        (l, v, e)
        for l, v, e in zip(labels, values, errors)
        if not math.isnan(v)
    ]
    if not data:
        return
    if len(data) < 2:
        print(f"[SKIP] single-series bar plot: {outpath.name}")
        return
    if descending:
        data.sort(key=lambda x: x[1], reverse=True)
    else:
        data.sort(key=lambda x: x[0])

    xlabels = [d[0] for d in data]
    yvals = [d[1] for d in data]
    yerrs = [0.0 if math.isnan(d[2]) else max(d[2], 0.0) for d in data]
    has_error_bar = any(err > 0.0 for err in yerrs)

    width = max(10, 0.8 * len(xlabels))
    fig, ax = plt.subplots(figsize=(width, 5.5))
    ax.bar(
        np.arange(len(xlabels)),
        yvals,
        yerr=yerrs if has_error_bar else None,
        capsize=4 if has_error_bar else 0,
        ecolor="black",
        error_kw={"elinewidth": 1.0},
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    if has_error_bar and error_label:
        ax.text(
            0.01,
            0.99,
            f"Error bar: {error_label}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="dimgray",
        )
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def save_boxplot(runs: Sequence[RunResult], outpath: Path) -> None:
    labels: List[str] = []
    data: List[List[float]] = []
    for run in runs:
        if run.inference_times:
            labels.append(run.run_label)
            data.append(run.inference_times)
    if not data:
        return

    width = max(10, 0.8 * len(labels))
    fig, ax = plt.subplots(figsize=(width, 5.5))
    ax.boxplot(data, showfliers=False)
    ax.set_title("Inference Time Distribution by Run")
    ax.set_ylabel("Inference Time (ms)")
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def save_line_plot(runs: Sequence[RunResult], outpath: Path, smooth_window: int) -> None:
    usable = [r for r in runs if len(r.frame_series) >= 2]
    if not usable:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for run in usable:
        xs = [f[0] for f in run.frame_series]
        ys_raw = [f[1] if f[1] is not None else math.nan for f in run.frame_series]
        ys = np.asarray(ys_raw, dtype=np.float64)
        valid = ~np.isnan(ys)
        if not np.any(valid):
            continue
        xs_valid = np.asarray(xs, dtype=np.float64)[valid]
        ys_valid = ys[valid]
        ys_smoothed = moving_average(ys_valid, smooth_window)
        ax.plot(xs_valid, ys_smoothed, linewidth=1.6, label=run.run_label)

    ax.set_title(f"Inference Time per Frame (MA window={smooth_window})")
    ax.set_xlabel("Frame ID")
    ax.set_ylabel("Inference Time (ms)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def write_summary_csv(runs: Sequence[RunResult], outpath: Path) -> None:
    columns = [
        "run_name",
        "run_label",
        "path",
        "framework",
        "backbone",
        "transport",
        "model",
        "tags",
        "frame_count",
        "detected_count",
        "mean_inference_ms",
        "std_inference_ms",
        "sem_inference_ms",
        "median_inference_ms",
        "min_inference_ms",
        "max_inference_ms",
        "fps_inference",
        "std_fps_inference",
        "sem_fps_inference",
        "mean_total_ms",
        "std_total_ms",
        "sem_total_ms",
        "median_total_ms",
        "min_total_ms",
        "max_total_ms",
        "fps_total",
        "std_fps_total",
        "sem_fps_total",
        "n_inference_frames",
        "n_total_frames",
    ]
    with outpath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for run in runs:
            row = {
                "run_name": run.run_name,
                "run_label": run.run_label,
                "path": str(run.path),
                "framework": run.framework,
                "backbone": run.backbone,
                "transport": run.transport,
                "model": run.model,
                "tags": run.tags,
                "frame_count": run.frame_count,
                "detected_count": run.detected_count,
            }
            row.update(run.summary_metrics)
            writer.writerow(row)


def write_frame_long_csv(runs: Sequence[RunResult], outpath: Path) -> None:
    columns = [
        "run_name",
        "run_label",
        "framework",
        "backbone",
        "transport",
        "frame_id",
        "inference_time_ms",
        "total_time_ms",
    ]
    with outpath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for run in runs:
            for frame_id, inf, tot in run.frame_series:
                writer.writerow(
                    {
                        "run_name": run.run_name,
                        "run_label": run.run_label,
                        "framework": run.framework,
                        "backbone": run.backbone,
                        "transport": run.transport,
                        "frame_id": frame_id,
                        "inference_time_ms": "" if inf is None else f"{inf:.6f}",
                        "total_time_ms": "" if tot is None else f"{tot:.6f}",
                    }
                )


def write_group_csv(group_rows: Sequence[Dict[str, object]], outpath: Path) -> None:
    if not group_rows:
        return
    keys = sorted({k for row in group_rows for k in row.keys()})
    with outpath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in group_rows:
            writer.writerow(row)


def write_manifest_template(runs: Sequence[RunResult], outpath: Path) -> None:
    columns = ["path", "run_name", "framework", "backbone", "transport", "model", "tags"]
    with outpath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "path": str(run.path),
                    "run_name": run.run_name,
                    "framework": run.framework,
                    "backbone": run.backbone,
                    "transport": run.transport,
                    "model": run.model,
                    "tags": run.tags,
                }
            )


def build_runs(csv_paths: Sequence[Path], manifest_rows: Sequence[Dict[str, str]]) -> List[RunResult]:
    runs: List[RunResult] = []
    for path in csv_paths:
        try:
            frame_series, metrics, inf_vals, tot_vals = parse_benchmark_csv(path)
        except Exception as exc:
            print(f"[SKIP] {path}: parse error ({exc})")
            continue

        meta = infer_metadata(path)
        meta = apply_manifest(meta, find_manifest_row(path, manifest_rows))
        run = RunResult(
            path=path,
            run_name=meta["run_name"],
            run_label=make_run_label(meta["run_name"], meta["framework"]),
            framework=meta["framework"],
            backbone=meta["backbone"],
            transport=meta["transport"],
            model=meta["model"],
            tags=meta["tags"],
            frame_count=len(frame_series),
            detected_count=len(tot_vals),
            inference_times=inf_vals,
            total_times=tot_vals,
            frame_series=frame_series,
            summary_metrics=metrics,
        )
        runs.append(run)
    return runs


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, str]] = []
    if args.manifest:
        manifest_path = Path(args.manifest).expanduser()
        if not manifest_path.exists():
            print(f"[WARN] manifest not found: {manifest_path}")
        else:
            manifest_rows = load_manifest(manifest_path)
            print(f"[INFO] Loaded manifest rows: {len(manifest_rows)}")

    csv_paths = expand_input_paths(args.inputs)
    if not csv_paths:
        print("[ERROR] No CSV files found from --inputs")
        return 1

    print(f"[INFO] CSV files found: {len(csv_paths)}")
    runs = build_runs(csv_paths, manifest_rows)
    if not runs:
        print("[ERROR] No valid benchmark CSV parsed.")
        return 1

    print(f"[INFO] Parsed runs: {len(runs)}")
    write_summary_csv(runs, outdir / "benchmark_comparison_summary.csv")
    write_frame_long_csv(runs, outdir / "benchmark_comparison_frames_long.csv")

    if args.write_manifest_template:
        write_manifest_template(runs, outdir / "manifest_template.csv")

    error_key_suffix = {"none": "", "std": "std", "sem": "sem"}[args.error_bar]

    def get_run_error(metric_key: str) -> List[float]:
        if args.error_bar == "none":
            return [math.nan for _ in runs]
        metric_to_error = {
            "mean_inference_ms": f"{error_key_suffix}_inference_ms",
            "mean_total_ms": f"{error_key_suffix}_total_ms",
            "fps_inference": f"{error_key_suffix}_fps_inference",
            "fps_total": f"{error_key_suffix}_fps_total",
        }
        err_key = metric_to_error[metric_key]
        return [float(r.summary_metrics.get(err_key, math.nan)) for r in runs]

    # run-level plots
    run_labels = [r.run_label for r in runs]
    save_bar_plot(
        run_labels,
        [r.summary_metrics["mean_inference_ms"] for r in runs],
        "Mean Inference Time by Run",
        "Inference Time (ms)",
        outdir / "runs_mean_inference_ms.png",
        errors=get_run_error("mean_inference_ms"),
        error_label=args.error_bar.upper() if args.error_bar != "none" else "",
    )
    save_bar_plot(
        run_labels,
        [r.summary_metrics["mean_total_ms"] for r in runs],
        "Mean Total Time by Run",
        "Total Time (ms)",
        outdir / "runs_mean_total_ms.png",
        errors=get_run_error("mean_total_ms"),
        error_label=args.error_bar.upper() if args.error_bar != "none" else "",
    )
    save_bar_plot(
        run_labels,
        [r.summary_metrics["fps_inference"] for r in runs],
        "FPS (Inference) by Run",
        "FPS",
        outdir / "runs_fps_inference.png",
        errors=get_run_error("fps_inference"),
        error_label=args.error_bar.upper() if args.error_bar != "none" else "",
        descending=True,
    )
    save_bar_plot(
        run_labels,
        [r.summary_metrics["fps_total"] for r in runs],
        "FPS (Total) by Run",
        "FPS",
        outdir / "runs_fps_total.png",
        errors=get_run_error("fps_total"),
        error_label=args.error_bar.upper() if args.error_bar != "none" else "",
        descending=True,
    )
    save_boxplot(runs, outdir / "runs_inference_boxplot.png")

    if len(runs) <= args.max_runs_line_plot:
        save_line_plot(runs, outdir / "runs_inference_per_frame_ma.png", args.smooth_window)
    else:
        print(
            f"[INFO] Skip line plot: runs={len(runs)} > max_runs_line_plot={args.max_runs_line_plot}"
        )

    # group-level plots
    all_group_rows: List[Dict[str, object]] = []
    for spec in args.group_by:
        group_rows = aggregate_by_group(runs, spec)
        all_group_rows.extend(group_rows)
        spec_tag = spec.replace("+", "_")
        write_group_csv(group_rows, outdir / f"group_{spec_tag}_summary.csv")

        def get_group_error(metric_key: str) -> List[float]:
            if args.error_bar == "none":
                return [math.nan for _ in group_rows]
            metric_to_error = {
                "mean_inference_ms": f"{error_key_suffix}_inference_ms",
                "mean_total_ms": f"{error_key_suffix}_total_ms",
                "fps_inference": f"{error_key_suffix}_fps_inference",
                "fps_total": f"{error_key_suffix}_fps_total",
            }
            err_key = metric_to_error[metric_key]
            vals: List[float] = []
            for row in group_rows:
                raw = row.get(err_key, math.nan)
                vals.append(float(raw) if raw is not None else math.nan)
            return vals

        save_bar_plot(
            [str(r["group_label"]) for r in group_rows],
            [float(r["mean_inference_ms"]) for r in group_rows],
            f"Mean Inference Time by {spec}",
            "Inference Time (ms)",
            outdir / f"group_{spec_tag}_mean_inference_ms.png",
            errors=get_group_error("mean_inference_ms"),
            error_label=args.error_bar.upper() if args.error_bar != "none" else "",
        )
        save_bar_plot(
            [str(r["group_label"]) for r in group_rows],
            [float(r["mean_total_ms"]) for r in group_rows],
            f"Mean Total Time by {spec}",
            "Total Time (ms)",
            outdir / f"group_{spec_tag}_mean_total_ms.png",
            errors=get_group_error("mean_total_ms"),
            error_label=args.error_bar.upper() if args.error_bar != "none" else "",
        )
        save_bar_plot(
            [str(r["group_label"]) for r in group_rows],
            [float(r["fps_inference"]) for r in group_rows],
            f"FPS (Inference) by {spec}",
            "FPS",
            outdir / f"group_{spec_tag}_fps_inference.png",
            errors=get_group_error("fps_inference"),
            error_label=args.error_bar.upper() if args.error_bar != "none" else "",
            descending=True,
        )
        save_bar_plot(
            [str(r["group_label"]) for r in group_rows],
            [float(r["fps_total"]) for r in group_rows],
            f"FPS (Total) by {spec}",
            "FPS",
            outdir / f"group_{spec_tag}_fps_total.png",
            errors=get_group_error("fps_total"),
            error_label=args.error_bar.upper() if args.error_bar != "none" else "",
            descending=True,
        )

    if all_group_rows:
        write_group_csv(all_group_rows, outdir / "benchmark_comparison_group_summary.csv")

    print(f"[DONE] Output directory: {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

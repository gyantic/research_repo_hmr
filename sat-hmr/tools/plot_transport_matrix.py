#!/usr/bin/env python3
"""Plot transport/precision comparisons from transport_precision_matrix.csv."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def _to_float(text: str) -> float:
    s = (text or "").strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def load_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "transport": (r.get("transport") or "").strip().lower(),
                    "precision": (r.get("precision") or "").strip().lower(),
                    "inference_ms_mean": _to_float(r.get("inference_ms_mean", "")),
                    "inference_ms_std": _to_float(r.get("inference_ms_std", "")),
                    "inference_ms_sem": _to_float(r.get("inference_ms_sem", "")),
                    "total_ms_mean": _to_float(r.get("total_ms_mean", "")),
                    "total_ms_std": _to_float(r.get("total_ms_std", "")),
                    "total_ms_sem": _to_float(r.get("total_ms_sem", "")),
                    "fps_inst_mean": _to_float(r.get("fps_inst_mean", "")),
                    "fps_inst_std": _to_float(r.get("fps_inst_std", "")),
                    "fps_inst_sem": _to_float(r.get("fps_inst_sem", "")),
                }
            )
    return rows


def save_bar(
    labels: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    out: Path,
    errors: Optional[List[float]] = None,
    error_label: str = "",
) -> None:
    if not labels:
        return
    if len(labels) < 2:
        print(f"[SKIP] single-series bar plot: {out.name}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    if errors is None:
        errors = [float("nan")] * len(values)
    yerrs = [0.0 if math.isnan(e) else max(0.0, e) for e in errors]
    has_error = any(e > 0.0 for e in yerrs)
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(
        labels,
        values,
        yerr=yerrs if has_error else None,
        capsize=4 if has_error else 0,
        ecolor="black",
        error_kw={"elinewidth": 1.0},
    )
    for b, v, e in zip(bars, values, yerrs):
        y = v + e if e > 0 else v
        plt.text(
            b.get_x() + b.get_width() / 2.0,
            y,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.25)
    if has_error and error_label:
        plt.gca().text(
            0.01,
            0.99,
            f"Error bar: {error_label}",
            transform=plt.gca().transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="dimgray",
        )
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def _err_field(base: str, error_bar: str) -> str:
    if error_bar == "std":
        return f"{base}_std"
    if error_bar == "sem":
        return f"{base}_sem"
    return ""


def plot_udp_precision(rows: List[Dict[str, object]], outdir: Path, error_bar: str) -> None:
    order = ["fp32", "fp16", "int8"]
    filt = [r for r in rows if r["transport"] == "udp" and r["precision"] in order]
    if not filt:
        return
    by_prec = {str(r["precision"]): r for r in filt}
    labels = [p.upper() for p in order if p in by_prec]
    inf_vals = [float(by_prec[p]["inference_ms_mean"]) for p in order if p in by_prec]
    fps_vals = [float(by_prec[p]["fps_inst_mean"]) for p in order if p in by_prec]
    inf_errs = [float(by_prec[p].get(_err_field("inference_ms", error_bar), float("nan"))) for p in order if p in by_prec]
    fps_errs = [float(by_prec[p].get(_err_field("fps_inst", error_bar), float("nan"))) for p in order if p in by_prec]

    save_bar(
        labels,
        inf_vals,
        "SAT-HMR (UDP): Precision vs Inference Mean (ms)",
        "inference_ms_mean",
        outdir / "sathmr_udp_precision_inference_ms.png",
        errors=inf_errs,
        error_label=error_bar if error_bar != "none" else "",
    )
    save_bar(
        labels,
        fps_vals,
        "SAT-HMR (UDP): Precision vs FPS",
        "fps_inst_mean",
        outdir / "sathmr_udp_precision_fps.png",
        errors=fps_errs,
        error_label=error_bar if error_bar != "none" else "",
    )


def plot_fp32_transport(rows: List[Dict[str, object]], outdir: Path, error_bar: str) -> None:
    # UDP/none row uses precision=fp32, TCP row uses precision=fp32_torch_tcp_benchmark.
    keep = []
    for r in rows:
        t = str(r["transport"])
        p = str(r["precision"])
        if t in {"udp", "tcp", "none"} and p in {"fp32", "fp32_torch_tcp_benchmark"}:
            keep.append(r)
    if not keep:
        return

    # Prefer UDP over none when both exist.
    picked: Dict[str, Dict[str, object]] = {}
    for r in keep:
        t = str(r["transport"])
        if t == "none" and "udp" in picked:
            continue
        if t == "udp" and "none" in picked:
            picked.pop("none", None)
        picked[t] = r

    labels_order = [x for x in ["udp", "tcp", "none"] if x in picked]
    labels = [x.upper() for x in labels_order]
    inf_vals = [float(picked[x]["inference_ms_mean"]) for x in labels_order]
    fps_vals = [float(picked[x]["fps_inst_mean"]) for x in labels_order]
    inf_errs = [float(picked[x].get(_err_field("inference_ms", error_bar), float("nan"))) for x in labels_order]
    fps_errs = [float(picked[x].get(_err_field("fps_inst", error_bar), float("nan"))) for x in labels_order]

    save_bar(
        labels,
        inf_vals,
        "SAT-HMR FP32: Transport vs Inference Mean (ms)",
        "inference_ms_mean",
        outdir / "sathmr_fp32_transport_inference_ms.png",
        errors=inf_errs,
        error_label=error_bar if error_bar != "none" else "",
    )
    save_bar(
        labels,
        fps_vals,
        "SAT-HMR FP32: Transport vs FPS",
        "fps_inst_mean",
        outdir / "sathmr_fp32_transport_fps.png",
        errors=fps_errs,
        error_label=error_bar if error_bar != "none" else "",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot selected charts from transport matrix CSV.")
    p.add_argument("--matrix_csv", required=True, help="transport_precision_matrix.csv path")
    p.add_argument("--outdir", default="", help="Output directory (default: matrix_csv parent)")
    p.add_argument("--error-bar", choices=["none", "std", "sem"], default="sem", help="Error bar type.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    matrix_csv = Path(args.matrix_csv)
    if not matrix_csv.is_file():
        print(f"[ERROR] matrix csv not found: {matrix_csv}")
        return 2
    outdir = Path(args.outdir) if args.outdir else matrix_csv.parent
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(matrix_csv)
    if not rows:
        print(f"[ERROR] empty matrix csv: {matrix_csv}")
        return 2

    plot_udp_precision(rows, outdir, args.error_bar)
    plot_fp32_transport(rows, outdir, args.error_bar)
    print(f"[DONE] plots written to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

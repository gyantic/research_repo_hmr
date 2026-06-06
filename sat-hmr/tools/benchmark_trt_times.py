#!/usr/bin/env python3
"""Run trtexec for multiple engines and summarize timing JSON into CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_engine_item(text: str) -> Tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"Invalid --engine '{text}'. Use LABEL=PATH")
    label, path = text.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError(f"Invalid --engine '{text}'. Use LABEL=PATH")
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark TensorRT engines via trtexec")
    p.add_argument(
        "--engine",
        action="append",
        type=parse_engine_item,
        required=True,
        help="Engine spec in LABEL=PATH format. Repeatable.",
    )
    p.add_argument("--trtexec", default="/usr/src/tensorrt/bin/trtexec", help="trtexec path")
    p.add_argument("--shape", default="image:1x3x728x1288", help="Input shapes for trtexec --shapes")
    p.add_argument("--warmup", type=int, default=500, help="trtexec --warmUp")
    p.add_argument("--duration", type=int, default=5, help="trtexec --duration")
    p.add_argument("--iterations", type=int, default=100, help="trtexec --iterations")
    p.add_argument("--output_csv", default="plots/int8_speed_check.csv", help="Summary CSV output")
    p.add_argument("--times_dir", default="plots", help="Directory to save exportTimes JSON")
    p.add_argument(
        "--extra_arg",
        action="append",
        default=[],
        help="Extra argument passed to trtexec as-is. Repeatable.",
    )
    return p.parse_args()


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _percentile(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def load_times_summary(times_path: Path) -> Dict[str, float]:
    data = json.loads(times_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected exportTimes JSON format: {times_path}")
    if len(data) == 0:
        raise RuntimeError(f"No timing samples in exportTimes JSON: {times_path}")

    lat = np.asarray([float(x.get("latencyMs", np.nan)) for x in data], dtype=np.float64)
    comp = np.asarray([float(x.get("computeMs", np.nan)) for x in data], dtype=np.float64)
    h2d = np.asarray([float(x.get("h2dMs", np.nan)) for x in data], dtype=np.float64)
    d2h = np.asarray([float(x.get("d2hMs", np.nan)) for x in data], dtype=np.float64)

    lat = lat[np.isfinite(lat)]
    comp = comp[np.isfinite(comp)]
    h2d = h2d[np.isfinite(h2d)]
    d2h = d2h[np.isfinite(d2h)]

    lat_mean = float(np.mean(lat)) if lat.size else float("nan")
    comp_mean = float(np.mean(comp)) if comp.size else float("nan")
    h2d_mean = float(np.mean(h2d)) if h2d.size else float("nan")
    d2h_mean = float(np.mean(d2h)) if d2h.size else float("nan")

    return {
        "samples": int(len(data)),
        "lat_mean_ms": lat_mean,
        "lat_p95_ms": _percentile(lat, 95.0),
        "compute_mean_ms": comp_mean,
        "compute_p95_ms": _percentile(comp, 95.0),
        "h2d_mean_ms": h2d_mean,
        "d2h_mean_ms": d2h_mean,
        "qps_from_lat": (1000.0 / lat_mean) if np.isfinite(lat_mean) and lat_mean > 0 else float("nan"),
        "qps_from_compute": (1000.0 / comp_mean) if np.isfinite(comp_mean) and comp_mean > 0 else float("nan"),
    }


def run_trtexec(
    trtexec: Path,
    engine_path: Path,
    times_path: Path,
    shape: str,
    warmup: int,
    duration: int,
    iterations: int,
    extra_args: List[str],
) -> str:
    cmd = [
        str(trtexec),
        f"--loadEngine={engine_path}",
        f"--shapes={shape}",
        f"--warmUp={warmup}",
        f"--duration={duration}",
        f"--iterations={iterations}",
        f"--exportTimes={times_path}",
    ]
    cmd.extend(extra_args)

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "trtexec failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"returncode: {proc.returncode}\n"
            f"output_tail:\n{proc.stdout[-4000:]}"
        )
    return proc.stdout


def main() -> int:
    args = parse_args()
    trtexec = Path(args.trtexec)
    if not trtexec.is_file():
        print(f"[ERROR] trtexec not found: {trtexec}")
        return 2

    times_dir = Path(args.times_dir)
    times_dir.mkdir(parents=True, exist_ok=True)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for label, engine_path in args.engine:
        if not engine_path.is_file():
            print(f"[ERROR] engine not found: {engine_path}")
            return 2
        safe_label = _safe_name(label)
        times_path = times_dir / f"trt_{safe_label}_times.json"

        print(f"[INFO] benchmark {label}: {engine_path}")
        run_trtexec(
            trtexec=trtexec,
            engine_path=engine_path,
            times_path=times_path,
            shape=args.shape,
            warmup=args.warmup,
            duration=args.duration,
            iterations=args.iterations,
            extra_args=args.extra_arg,
        )
        summary = load_times_summary(times_path)
        row: Dict[str, object] = {
            "engine": label,
            "engine_path": str(engine_path),
            "times_json": str(times_path),
            **summary,
        }
        rows.append(row)

    if not rows:
        print("[ERROR] no benchmark rows")
        return 2

    base = rows[0]
    base_lat = float(base["lat_mean_ms"])
    base_comp = float(base["compute_mean_ms"])
    for row in rows:
        lat = float(row["lat_mean_ms"])
        comp = float(row["compute_mean_ms"])
        row["speedup_vs_base_pct_lat"] = (
            ((base_lat / lat) - 1.0) * 100.0 if np.isfinite(base_lat) and np.isfinite(lat) and lat > 0 else float("nan")
        )
        row["speedup_vs_base_pct_compute"] = (
            ((base_comp / comp) - 1.0) * 100.0
            if np.isfinite(base_comp) and np.isfinite(comp) and comp > 0
            else float("nan")
        )

    fieldnames = [
        "engine",
        "engine_path",
        "times_json",
        "samples",
        "lat_mean_ms",
        "lat_p95_ms",
        "compute_mean_ms",
        "compute_p95_ms",
        "h2d_mean_ms",
        "d2h_mean_ms",
        "qps_from_lat",
        "qps_from_compute",
        "speedup_vs_base_pct_lat",
        "speedup_vs_base_pct_compute",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[DONE] Wrote benchmark CSV: {output_csv}")
    print("[INFO] Summary:")
    for row in rows:
        print(
            f"  - {row['engine']}: lat={float(row['lat_mean_ms']):.4f} ms, "
            f"compute={float(row['compute_mean_ms']):.4f} ms, "
            f"speedup_vs_base(lat)={float(row['speedup_vs_base_pct_lat']):.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


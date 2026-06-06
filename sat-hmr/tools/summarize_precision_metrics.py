#!/usr/bin/env python3
"""Summarize fp32/fp16/int8 per-frame metrics CSVs into one compact CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import numpy as np


def _to_float(row: Dict[str, str], key: str) -> float:
    val = (row.get(key, "") or "").strip()
    if not val:
        return float("nan")
    try:
        return float(val)
    except ValueError:
        return float("nan")


def summarize_csv(path: Path) -> Dict[str, float]:
    rows: List[Dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        return {
            "frames": 0.0,
            "preprocess_ms_mean": float("nan"),
            "inference_ms_mean": float("nan"),
            "inference_ms_std": float("nan"),
            "inference_ms_sem": float("nan"),
            "postprocess_ms_mean": float("nan"),
            "render_ms_mean": float("nan"),
            "total_ms_mean": float("nan"),
            "total_ms_std": float("nan"),
            "total_ms_sem": float("nan"),
            "fps_inst_mean": float("nan"),
            "fps_inst_std": float("nan"),
            "fps_inst_sem": float("nan"),
            "fps_avg_30_last": float("nan"),
        }

    keys = [
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
        "render_ms",
        "total_ms",
        "fps_inst",
    ]
    sums = {k: 0.0 for k in keys}
    counts = {k: 0 for k in keys}
    vals = {k: [] for k in keys}
    for r in rows:
        for k in keys:
            v = _to_float(r, k)
            if v == v:  # not nan
                sums[k] += v
                counts[k] += 1
                vals[k].append(v)

    def _std_sem(k: str) -> tuple[float, float]:
        arr = np.asarray(vals[k], dtype=np.float64)
        if arr.size <= 1:
            return float("nan"), float("nan")
        std = float(np.std(arr))
        sem = float(std / math.sqrt(arr.size))
        return std, sem

    inf_std, inf_sem = _std_sem("inference_ms")
    tot_std, tot_sem = _std_sem("total_ms")
    fps_std, fps_sem = _std_sem("fps_inst")

    out = {
        "frames": float(len(rows)),
        "preprocess_ms_mean": (sums["preprocess_ms"] / counts["preprocess_ms"]) if counts["preprocess_ms"] else float("nan"),
        "inference_ms_mean": (sums["inference_ms"] / counts["inference_ms"]) if counts["inference_ms"] else float("nan"),
        "inference_ms_std": inf_std,
        "inference_ms_sem": inf_sem,
        "postprocess_ms_mean": (sums["postprocess_ms"] / counts["postprocess_ms"]) if counts["postprocess_ms"] else float("nan"),
        "render_ms_mean": (sums["render_ms"] / counts["render_ms"]) if counts["render_ms"] else float("nan"),
        "total_ms_mean": (sums["total_ms"] / counts["total_ms"]) if counts["total_ms"] else float("nan"),
        "total_ms_std": tot_std,
        "total_ms_sem": tot_sem,
        "fps_inst_mean": (sums["fps_inst"] / counts["fps_inst"]) if counts["fps_inst"] else float("nan"),
        "fps_inst_std": fps_std,
        "fps_inst_sem": fps_sem,
        "fps_avg_30_last": _to_float(rows[-1], "fps_avg_30"),
    }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize precision benchmark metrics.")
    p.add_argument("--fp32_csv", required=True)
    p.add_argument("--fp16_csv", required=True)
    p.add_argument("--int8_csv", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths = {
        "fp32": Path(args.fp32_csv),
        "fp16": Path(args.fp16_csv),
        "int8": Path(args.int8_csv),
    }
    for name, p in paths.items():
        if not p.is_file():
            print(f"[ERROR] missing {name} csv: {p}")
            return 2

    rows_out: List[Dict[str, object]] = []
    for name, p in paths.items():
        s = summarize_csv(p)
        rows_out.append(
            {
                "precision": name,
                "metrics_csv": str(p),
                **s,
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "precision",
        "metrics_csv",
        "frames",
        "preprocess_ms_mean",
        "inference_ms_mean",
        "inference_ms_std",
        "inference_ms_sem",
        "postprocess_ms_mean",
        "render_ms_mean",
        "total_ms_mean",
        "total_ms_std",
        "total_ms_sem",
        "fps_inst_mean",
        "fps_inst_std",
        "fps_inst_sem",
        "fps_avg_30_last",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    print(f"[DONE] Wrote summary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Merge transport benchmark outputs into one compact matrix CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional


def _to_float(text: str) -> float:
    s = (text or "").strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def load_precision_summary(path: Path, transport: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "transport": transport,
                    "precision": (r.get("precision") or "").strip(),
                    "inference_ms_mean": _to_float(r.get("inference_ms_mean", "")),
                    "inference_ms_std": _to_float(r.get("inference_ms_std", "")),
                    "inference_ms_sem": _to_float(r.get("inference_ms_sem", "")),
                    "total_ms_mean": _to_float(r.get("total_ms_mean", "")),
                    "total_ms_std": _to_float(r.get("total_ms_std", "")),
                    "total_ms_sem": _to_float(r.get("total_ms_sem", "")),
                    "fps_inst_mean": _to_float(r.get("fps_inst_mean", "")),
                    "fps_inst_std": _to_float(r.get("fps_inst_std", "")),
                    "fps_inst_sem": _to_float(r.get("fps_inst_sem", "")),
                    "fps_avg_30_last": _to_float(r.get("fps_avg_30_last", "")),
                    "source": str(path),
                }
            )
    return rows


def load_tcp_stats(path: Path) -> Optional[Dict[str, object]]:
    # Expected section:
    # 指標,推論時間 (ms),送信時間 (ms),全体処理時間 (ms)
    # 平均,99.11,0.05,100.06
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    avg_row: Optional[List[str]] = None
    for row in rows:
        if row and row[0].strip() == "平均":
            avg_row = row
            break
    if avg_row is None:
        return None

    inf_ms = _to_float(avg_row[1] if len(avg_row) > 1 else "")
    total_ms = _to_float(avg_row[-1] if len(avg_row) > 1 else "")
    fps_inst = (1000.0 / total_ms) if total_ms == total_ms and total_ms > 0 else float("nan")
    return {
        "transport": "tcp",
        "precision": "fp32_torch_tcp_benchmark",
        "inference_ms_mean": inf_ms,
        "inference_ms_std": float("nan"),
        "inference_ms_sem": float("nan"),
        "total_ms_mean": total_ms,
        "total_ms_std": float("nan"),
        "total_ms_sem": float("nan"),
        "fps_inst_mean": fps_inst,
        "fps_inst_std": float("nan"),
        "fps_inst_sem": float("nan"),
        "fps_avg_30_last": float("nan"),
        "source": str(path),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect transport benchmark matrix.")
    p.add_argument("--no_udp_summary", required=True, help="precision_summary.csv from no-udp run")
    p.add_argument("--udp_summary", default="", help="precision_summary.csv from udp run")
    p.add_argument("--tcp_stats_csv", default="", help="TCP stats CSV from sathmr_tcp_benchmark.py")
    p.add_argument("--output", required=True, help="Output matrix CSV")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_rows: List[Dict[str, object]] = []

    no_udp_path = Path(args.no_udp_summary)
    if not no_udp_path.is_file():
        print(f"[ERROR] no_udp summary not found: {no_udp_path}")
        return 2
    out_rows.extend(load_precision_summary(no_udp_path, "none"))

    if args.udp_summary:
        udp_path = Path(args.udp_summary)
        if not udp_path.is_file():
            print(f"[ERROR] udp summary not found: {udp_path}")
            return 2
        out_rows.extend(load_precision_summary(udp_path, "udp"))

    if args.tcp_stats_csv:
        tcp_path = Path(args.tcp_stats_csv)
        if not tcp_path.is_file():
            print(f"[ERROR] tcp stats not found: {tcp_path}")
            return 2
        tcp_row = load_tcp_stats(tcp_path)
        if tcp_row is None:
            print(f"[ERROR] could not parse TCP stats CSV: {tcp_path}")
            return 2
        out_rows.append(tcp_row)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "transport",
        "precision",
        "inference_ms_mean",
        "inference_ms_std",
        "inference_ms_sem",
        "total_ms_mean",
        "total_ms_std",
        "total_ms_sem",
        "fps_inst_mean",
        "fps_inst_std",
        "fps_inst_sem",
        "fps_avg_30_last",
        "source",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print(f"[DONE] Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

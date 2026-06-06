#!/usr/bin/env python3
"""Summarize per-frame inference timing CSVs into one comparison CSV."""

import argparse
import csv
import hashlib
import math
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--warmup_frames", type=int, default=10)
    parser.add_argument("--romp_csv", default="")
    parser.add_argument("--sathmr_csv", default="")
    parser.add_argument("--fastsam_csv", default="")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional CSV manifest with one row per model/variant. Extra speed and accuracy metadata columns are preserved.",
    )
    return parser.parse_args()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            frame = row.get("frame", "")
            try:
                int(frame)
            except (TypeError, ValueError):
                break
            rows.append(row)
    return rows


def numeric(rows, column):
    values = []
    for row in rows:
        raw = row.get(column, "")
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            values.append(val)
    return np.asarray(values, dtype=np.float64)


def stats(values):
    if values.size == 0:
        return (math.nan, math.nan, math.nan)
    return (float(np.mean(values)), float(np.median(values)), float(np.var(values)))


def make_summary(
    method,
    dtype,
    env,
    raw_csv,
    inference_column,
    command,
    video_path,
    video_sha,
    warmup_frames,
    skip_rows=0,
    status="ok",
    note="",
    backbone="",
    compile_value="",
    quantization_mode="",
    runtime_variant="",
    batch_size="",
    batch_size_note="",
    weight_size_unit="",
    primary_weight_mb="",
    weight_total_mb="",
    weight_files="",
    weight_files_mb="",
    weight_note="",
    accuracy_dataset="",
    accuracy_protocol="",
    accuracy_status="pending",
    mpjpe_mm="",
    pa_mpjpe_mm="",
    pve_mm="",
    pck="",
    auc="",
    map_score="",
    accuracy_note="",
):
    raw_csv = Path(raw_csv)
    rows = read_rows(raw_csv) if status == "ok" and raw_csv.exists() else []
    if skip_rows > 0:
        rows = rows[skip_rows:]
    inf_ms = numeric(rows, inference_column)
    total_ms = numeric(rows, "total_ms")
    inf_fps = 1000.0 / inf_ms[inf_ms > 0]
    total_fps = 1000.0 / total_ms[total_ms > 0]

    inf_ms_mean, inf_ms_median, inf_ms_var = stats(inf_ms)
    inf_fps_mean, inf_fps_median, inf_fps_var = stats(inf_fps)
    total_ms_mean, total_ms_median, total_ms_var = stats(total_ms)
    total_fps_mean, total_fps_median, total_fps_var = stats(total_fps)

    return {
        "method": method,
        "backbone": backbone,
        "dtype": dtype,
        "quantization_mode": quantization_mode,
        "compile": compile_value,
        "runtime_variant": runtime_variant,
        "batch_size": batch_size,
        "batch_size_note": batch_size_note,
        "weight_size_unit": weight_size_unit,
        "primary_weight_mb": primary_weight_mb,
        "weight_total_mb": weight_total_mb,
        "weight_files": weight_files,
        "weight_files_mb": weight_files_mb,
        "weight_note": weight_note,
        "conda_env": env,
        "input_video": str(video_path),
        "warmup_frames": warmup_frames,
        "skipped_csv_rows": skip_rows,
        "measured_frames": len(rows),
        "inference_metric_source": inference_column,
        "inference_ms_mean": inf_ms_mean,
        "inference_ms_median": inf_ms_median,
        "inference_ms_variance": inf_ms_var,
        "inference_fps_mean": inf_fps_mean,
        "inference_fps_median": inf_fps_median,
        "inference_fps_variance": inf_fps_var,
        "total_ms_mean": total_ms_mean,
        "total_ms_median": total_ms_median,
        "total_ms_variance": total_ms_var,
        "total_fps_mean": total_fps_mean,
        "total_fps_median": total_fps_median,
        "total_fps_variance": total_fps_var,
        "accuracy_dataset": accuracy_dataset,
        "accuracy_protocol": accuracy_protocol,
        "accuracy_status": accuracy_status,
        "mpjpe_mm": mpjpe_mm,
        "pa_mpjpe_mm": pa_mpjpe_mm,
        "pve_mm": pve_mm,
        "pck": pck,
        "auc": auc,
        "map": map_score,
        "accuracy_note": accuracy_note,
        "video_sha256": video_sha,
        "raw_csv": str(raw_csv),
        "status": status,
        "note": note,
        "command": command,
    }


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    video_path = Path(args.video)
    video_sha = sha256_file(video_path)

    if args.manifest:
        summaries = []
        for row in read_manifest(args.manifest):
            row_video = Path(row.get("input_video") or video_path)
            row_video_sha = sha256_file(row_video) if row_video.exists() else video_sha
            summaries.append(
                make_summary(
                    row.get("method", ""),
                    row.get("dtype", ""),
                    row.get("conda_env", ""),
                    row.get("raw_csv", ""),
                    row.get("inference_column", "inference_ms"),
                    row.get("command", ""),
                    row_video,
                    row_video_sha,
                    int(row.get("warmup_frames") or args.warmup_frames),
                    int(row.get("skip_rows") or 0),
                    row.get("status", "ok") or "ok",
                    row.get("note", ""),
                    row.get("backbone", ""),
                    row.get("compile", ""),
                    row.get("quantization_mode", ""),
                    row.get("runtime_variant", ""),
                    row.get("batch_size", ""),
                    row.get("batch_size_note", ""),
                    row.get("weight_size_unit", ""),
                    row.get("primary_weight_mb", ""),
                    row.get("weight_total_mb", ""),
                    row.get("weight_files", ""),
                    row.get("weight_files_mb", ""),
                    row.get("weight_note", ""),
                    row.get("accuracy_dataset", ""),
                    row.get("accuracy_protocol", ""),
                    row.get("accuracy_status", "pending") or "pending",
                    row.get("mpjpe_mm", ""),
                    row.get("pa_mpjpe_mm", ""),
                    row.get("pve_mm", ""),
                    row.get("pck", ""),
                    row.get("auc", ""),
                    row.get("map", ""),
                    row.get("accuracy_note", ""),
                )
            )
    else:
        summaries = [
            make_summary(
                "ROMP",
                "default",
                "romp_3.9",
                Path(args.romp_csv),
                "inference_ms",
                "conda run -n romp_3.9 python -u romp_video_demo.py --no_render --no_save --metrics_csv ...",
                video_path,
                video_sha,
                args.warmup_frames,
            ),
            make_summary(
                "SAT-HMR",
                "default",
                "sathmr-copy",
                Path(args.sathmr_csv),
                "inference_ms",
                "conda run -n sathmr-copy python -u video_demo.py --no_render --no_save --metrics_csv ...",
                video_path,
                video_sha,
                args.warmup_frames,
            ),
            make_summary(
                "Fast-SAM-3D-Body",
                "default",
                "sam_3d_body",
                Path(args.fastsam_csv),
                "inference_ms",
                "conda run -n sam_3d_body python -u benchmark_fastsam3dbody.py --output_csv ...",
                video_path,
                video_sha,
                args.warmup_frames,
            ),
        ]

    fieldnames = list(summaries[0].keys())
    with output.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            out = row.copy()
            for key, value in list(out.items()):
                if isinstance(value, float):
                    out[key] = "" if math.isnan(value) else f"{value:.6f}"
            writer.writerow(out)

    print(f"Saved summary: {output}")


if __name__ == "__main__":
    main()

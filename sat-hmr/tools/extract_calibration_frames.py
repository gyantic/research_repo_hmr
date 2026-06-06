#!/usr/bin/env python3
"""Extract calibration frames for TensorRT INT8 PTQ."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract INT8 calibration frames from a video")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--outdir", default="datas/calibration", help="Output directory")
    parser.add_argument("--num-frames", type=int, default=800, help="Number of frames to extract")
    parser.add_argument("--start-frame", type=int, default=0, help="Inclusive start frame")
    parser.add_argument("--end-frame", type=int, default=-1, help="Exclusive end frame (-1 means video end)")
    parser.add_argument("--prefix", default="calib", help="Output file prefix")
    parser.add_argument("--jpg-quality", type=int, default=95, help="JPEG quality 1-100")
    return parser.parse_args()


def pick_uniform_indices(start: int, end: int, n: int) -> List[int]:
    if n <= 0 or end <= start:
        return []
    total = end - start
    if n >= total:
        return list(range(start, end))
    idx = np.linspace(start, end - 1, n, dtype=np.int64)
    # Ensure uniqueness after rounding
    idx = np.unique(idx)
    return [int(x) for x in idx.tolist()]


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}")
        return 2

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, args.start_frame)
    end = total_frames if args.end_frame < 0 else min(args.end_frame, total_frames)
    indices = pick_uniform_indices(start, end, args.num_frames)
    if not indices:
        print("[ERROR] No frame indices selected. Check --start-frame/--end-frame/--num-frames.")
        return 2

    print(
        f"[INFO] video={args.video} total_frames={total_frames} "
        f"extract_range=[{start},{end}) selected={len(indices)}"
    )

    manifest_path = outdir / "calibration_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "image_path"])

        for i, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            image_name = f"{args.prefix}_{i:05d}_f{frame_idx:06d}.jpg"
            image_path = outdir / image_name
            cv2.imwrite(
                str(image_path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpg_quality)],
            )
            writer.writerow([frame_idx, str(image_path)])

    cap.release()
    print(f"[DONE] Extracted frames to: {outdir.resolve()}")
    print(f"[DONE] Manifest: {manifest_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


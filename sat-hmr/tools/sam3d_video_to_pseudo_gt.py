#!/usr/bin/env python3
"""Run SAM-3D-Body on a video and export pseudo-GT sequence data."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAM3D video -> pseudo-GT npz")
    p.add_argument("--video", required=True, help="Input video path")
    p.add_argument("--sam3d_root", default="/home/kyuta/sam-3d-body", help="SAM-3D-Body repo root")
    p.add_argument("--output_dir", default="plots/sam3d_video", help="Output directory")
    p.add_argument("--frames_dir", default="", help="Extracted frames dir (default: <output_dir>/frames)")
    p.add_argument("--output_npz", default="", help="Output npz path (default: <output_dir>/sam3d_pseudo_gt.npz)")
    p.add_argument("--manifest_csv", default="", help="Output manifest csv path (default: <output_dir>/manifest.csv)")

    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--end_frame", type=int, default=-1, help="Inclusive end frame. -1 means until video end.")
    p.add_argument("--stride", type=int, default=1, help="Frame sampling stride")
    p.add_argument("--max_frames", type=int, default=-1, help="Maximum number of extracted frames. -1 means no limit.")
    p.add_argument("--jpg_quality", type=int, default=95)
    p.add_argument("--reuse_frames", action="store_true", help="Reuse existing extracted frames if present")
    p.add_argument("--extract_only", action="store_true", help="Only extract frames and skip SAM3D inference")

    p.add_argument("--checkpoint_path", default="", help="SAM3D checkpoint path")
    p.add_argument("--mhr_path", default="", help="MHR asset path")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")

    p.add_argument("--detector_name", default="vitdet", help="Set empty string to disable detector")
    p.add_argument("--segmentor_name", default="", help="Set empty string to disable segmentor")
    p.add_argument("--fov_name", default="", help="Set empty string to disable fov estimator")
    p.add_argument("--detector_path", default="")
    p.add_argument("--segmentor_path", default="")
    p.add_argument("--fov_path", default="")
    p.add_argument("--bbox_thresh", type=float, default=0.8)
    p.add_argument("--use_mask", action="store_true")
    p.add_argument("--inference_type", choices=["full", "body", "hand"], default="full")
    return p.parse_args()


def ensure_parent(path: str) -> None:
    parent = Path(path).parent
    if str(parent) and str(parent) != ".":
        parent.mkdir(parents=True, exist_ok=True)


def default_checkpoint_paths(sam3d_root: Path) -> Tuple[str, str]:
    ckpt = sam3d_root / "checkpoints" / "sam-3d-body-dinov3" / "model.ckpt"
    mhr = sam3d_root / "checkpoints" / "sam-3d-body-dinov3" / "assets" / "mhr_model.pt"
    return str(ckpt), str(mhr)


def extract_frames(
    video_path: str,
    frames_dir: Path,
    start_frame: int,
    end_frame: int,
    stride: int,
    max_frames: int,
    jpg_quality: int,
) -> List[Tuple[int, Path]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame < 0:
        end = total - 1
    else:
        end = min(end_frame, total - 1)
    start = max(0, start_frame)
    if end < start:
        cap.release()
        return []

    frames_dir.mkdir(parents=True, exist_ok=True)

    selected: List[Tuple[int, Path]] = []
    idx = 0
    written = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx < start:
            idx += 1
            continue
        if idx > end:
            break
        if (idx - start) % max(1, stride) != 0:
            idx += 1
            continue
        if max_frames > 0 and written >= max_frames:
            break

        out_path = frames_dir / f"frame_{idx:06d}.jpg"
        cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
        selected.append((idx, out_path))
        written += 1
        idx += 1

    cap.release()
    return selected


def list_existing_frames(frames_dir: Path) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    for p in sorted(frames_dir.glob("frame_*.jpg")):
        stem = p.stem
        try:
            fid = int(stem.split("_")[-1])
        except Exception:
            continue
        out.append((fid, p))
    return out


def build_estimator(args: argparse.Namespace):
    sam3d_root = Path(args.sam3d_root).resolve()
    if not sam3d_root.exists():
        raise FileNotFoundError(f"sam3d_root not found: {sam3d_root}")

    if str(sam3d_root) not in sys.path:
        sys.path.insert(0, str(sam3d_root))

    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    checkpoint_path = args.checkpoint_path
    mhr_path = args.mhr_path
    if not checkpoint_path or not mhr_path:
        d_ckpt, d_mhr = default_checkpoint_paths(sam3d_root)
        checkpoint_path = checkpoint_path or d_ckpt
        mhr_path = mhr_path or d_mhr

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available. Fallback to CPU.")
        device = "cpu"

    model, model_cfg = load_sam_3d_body(checkpoint_path=checkpoint_path, mhr_path=mhr_path, device=device)

    human_detector = None
    human_segmentor = None
    fov_estimator = None

    detector_name = args.detector_name.strip()
    segmentor_name = args.segmentor_name.strip()
    fov_name = args.fov_name.strip()

    if detector_name:
        from tools.build_detector import HumanDetector  # type: ignore

        human_detector = HumanDetector(
            name=detector_name,
            device=torch.device(device),
            path=args.detector_path,
        )
    if segmentor_name:
        from tools.build_sam import HumanSegmentor  # type: ignore

        human_segmentor = HumanSegmentor(
            name=segmentor_name,
            device=torch.device(device),
            path=args.segmentor_path,
        )
    if fov_name:
        from tools.build_fov_estimator import FOVEstimator  # type: ignore

        fov_estimator = FOVEstimator(
            name=fov_name,
            device=torch.device(device),
            path=args.fov_path,
        )

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )
    return estimator


def parse_person(person: Dict[str, Any]) -> Dict[str, np.ndarray | float]:
    out: Dict[str, np.ndarray | float] = {}
    out["joints3d"] = np.asarray(person["pred_keypoints_3d"], dtype=np.float32)
    out["joints2d"] = np.asarray(person["pred_keypoints_2d"], dtype=np.float32)
    out["verts"] = np.asarray(person["pred_vertices"], dtype=np.float32)
    out["transl"] = np.asarray(person["pred_cam_t"], dtype=np.float32).reshape(-1)[:3]
    out["bbox"] = np.asarray(person["bbox"], dtype=np.float32).reshape(-1)[:4]
    out["focal_length"] = float(person["focal_length"])
    return out


def run_sam3d(frames: Sequence[Tuple[int, Path]], estimator, args: argparse.Namespace) -> Dict[str, Any]:
    frame_ids: List[int] = []
    image_paths: List[str] = []
    person_counts: List[int] = []
    joints2d_all: List[np.ndarray] = []
    joints3d_all: List[np.ndarray] = []
    verts_all: List[np.ndarray] = []
    transl_all: List[np.ndarray] = []
    bboxes_all: List[np.ndarray] = []
    focal_all: List[np.ndarray] = []

    failed = 0

    for fid, img_path in tqdm(frames, desc="sam3d_infer"):
        frame_ids.append(fid)
        image_paths.append(str(img_path))
        try:
            outputs = estimator.process_one_image(
                str(img_path),
                bbox_thr=args.bbox_thresh,
                use_mask=args.use_mask,
                inference_type=args.inference_type,
            )
            if outputs is None:
                outputs = []
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]

            people: List[Dict[str, np.ndarray | float]] = []
            for person in outputs:
                try:
                    people.append(parse_person(person))
                except Exception:
                    continue

            person_counts.append(len(people))
            if len(people) == 0:
                joints2d_all.append(np.zeros((0, 24, 2), dtype=np.float32))
                joints3d_all.append(np.zeros((0, 24, 3), dtype=np.float32))
                verts_all.append(np.zeros((0, 0, 3), dtype=np.float32))
                transl_all.append(np.zeros((0, 3), dtype=np.float32))
                bboxes_all.append(np.zeros((0, 4), dtype=np.float32))
                focal_all.append(np.zeros((0,), dtype=np.float32))
                continue

            joints2d_all.append(np.stack([p["joints2d"] for p in people], axis=0))
            joints3d_all.append(np.stack([p["joints3d"] for p in people], axis=0))
            verts_all.append(np.stack([p["verts"] for p in people], axis=0))
            transl_all.append(np.stack([p["transl"] for p in people], axis=0))
            bboxes_all.append(np.stack([p["bbox"] for p in people], axis=0))
            focal_all.append(np.asarray([p["focal_length"] for p in people], dtype=np.float32))
        except Exception:
            failed += 1
            person_counts.append(0)
            joints2d_all.append(np.zeros((0, 24, 2), dtype=np.float32))
            joints3d_all.append(np.zeros((0, 24, 3), dtype=np.float32))
            verts_all.append(np.zeros((0, 0, 3), dtype=np.float32))
            transl_all.append(np.zeros((0, 3), dtype=np.float32))
            bboxes_all.append(np.zeros((0, 4), dtype=np.float32))
            focal_all.append(np.zeros((0,), dtype=np.float32))

    return {
        "frame_ids": np.asarray(frame_ids, dtype=np.int64),
        "image_paths": np.asarray(image_paths, dtype=object),
        "person_counts": np.asarray(person_counts, dtype=np.int32),
        "joints2d": np.asarray(joints2d_all, dtype=object),
        "joints3d": np.asarray(joints3d_all, dtype=object),
        "verts": np.asarray(verts_all, dtype=object),
        "transl": np.asarray(transl_all, dtype=object),
        "bboxes": np.asarray(bboxes_all, dtype=object),
        "focal_length": np.asarray(focal_all, dtype=object),
        "failed_frames": int(failed),
    }


def write_manifest(path: str, frames: Sequence[Tuple[int, Path]], person_counts: np.ndarray) -> None:
    ensure_parent(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "image_path", "persons"])
        for i, (fid, img_path) in enumerate(frames):
            persons = int(person_counts[i]) if i < len(person_counts) else -1
            w.writerow([fid, str(img_path), persons])


def main() -> int:
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = Path(args.frames_dir) if args.frames_dir else (out_dir / "frames")
    output_npz = args.output_npz or str(out_dir / "sam3d_pseudo_gt.npz")
    manifest_csv = args.manifest_csv or str(out_dir / "manifest.csv")

    if args.reuse_frames and frames_dir.exists():
        frames = list_existing_frames(frames_dir)
        print(f"[INFO] Reusing existing frames: {len(frames)} from {frames_dir}")
    else:
        print("[INFO] Extracting frames...")
        frames = extract_frames(
            video_path=args.video,
            frames_dir=frames_dir,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            stride=max(1, args.stride),
            max_frames=args.max_frames,
            jpg_quality=args.jpg_quality,
        )
        print(f"[INFO] Extracted frames: {len(frames)} -> {frames_dir}")

    if len(frames) == 0:
        raise RuntimeError("No frames selected. Check --start_frame/--end_frame/--stride/--max_frames.")

    if args.extract_only:
        write_manifest(manifest_csv, frames, np.full((len(frames),), -1, dtype=np.int32))
        print(f"[DONE] Frame extraction only. Manifest: {manifest_csv}")
        return 0

    estimator = build_estimator(args)
    result = run_sam3d(frames, estimator, args)

    ensure_parent(output_npz)
    np.savez_compressed(output_npz, **result)
    write_manifest(manifest_csv, frames, result["person_counts"])

    print(f"[DONE] Pseudo-GT npz: {output_npz}")
    print(f"[DONE] Manifest: {manifest_csv}")
    print(
        "[INFO] Summary: "
        f"frames={len(frames)} "
        f"detected_frames={(result['person_counts'] > 0).sum()} "
        f"failed_frames={result['failed_frames']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


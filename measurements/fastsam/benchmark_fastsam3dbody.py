#!/usr/bin/env python3
"""Run Fast-SAM-3D-Body video inference timing and write per-frame CSV."""

import argparse
import csv
import os
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[2] / "fast-sam-3d-body"))
    parser.add_argument("--video", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--warmup_frames", type=int, default=10)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--bbox_thresh", type=float, default=0.5)
    parser.add_argument(
        "--dtype_variant",
        choices=["fp32", "bf16", "int8", "int8_bf16"],
        default="int8",
        help="Precision/engine variant to benchmark.",
    )
    parser.add_argument(
        "--backbone_engine",
        default="./checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_int8.engine",
    )
    parser.add_argument(
        "--fov_engine",
        default="./checkpoints/moge_trt/moge_dinov2_encoder_int8.engine",
    )
    parser.add_argument(
        "--yolo_model",
        default="./checkpoints/yolo/yolo11m-pose.engine",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="./checkpoints/sam-3d-body-dinov3/model.ckpt",
        help="SAM 3D Body model.ckpt to load for timing.",
    )
    parser.add_argument(
        "--mhr_path",
        default="./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt",
        help="MHR model path to load for timing.",
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile and MHR CUDA graph for this timing run.",
    )
    return parser.parse_args()


COMPILE_DISABLED_ENV = {
    "USE_COMPILE": "0",
    "USE_COMPILE_BACKBONE": "0",
    "DECODER_COMPILE": "0",
    "INTERM_COMPILE": "0",
    "MHR_USE_CUDA_GRAPH": "0",
}

# Overrides on top of the base env, which already is the TensorRT INT8 backbone + FOV setup
# (so "int8" needs no override).
DTYPE_VARIANT_ENV = {
    "fp32": {"LAYER_DTYPE": "fp32", "USE_TRT_BACKBONE": "0", "FOV_TRT": "0"},
    # bf16 runs eager: compile / CUDA graph caused a mixed-dtype addmm failure.
    "bf16": {"LAYER_DTYPE": "bf16", "USE_TRT_BACKBONE": "0", "FOV_TRT": "0", **COMPILE_DISABLED_ENV},
    "int8": {},
    "int8_bf16": {"LAYER_DTYPE": "bf16", "MHR_USE_CUDA_GRAPH": "0"},
}


def set_perf_env(args):
    """Export the Fast-SAM-3D-Body runtime switches.

    Must run before `mocap.core.setup_estimator` is imported: tools/build_fov_estimator.py
    and the DINOv3 backbone read FOV_* / USE_TRT_BACKBONE / TRT_BACKBONE_PATH at import time,
    so the FOV_* values that build_default_estimator() sets later do not reach them.
    """
    env = {
        "SAM3D_QUIET": "1",
        "TORCHINDUCTOR_CUDAGRAPH_TREES": "0",
        "GPU_HAND_PREP": "1",
        "LAYER_DTYPE": "fp32",
        "SKIP_KEYPOINT_PROMPT": "1",
        "KEYPOINT_PROMPT_INTERM_INTERVAL": "999",
        "IMG_SIZE": str(args.image_size),
        "MHR_NO_CORRECTIVES": "1",
        "FOV_TRT": "1",
        "FOV_TRT_PATH": args.fov_engine,
        "FOV_FAST": "1",
        "FOV_MODEL": "s",
        "FOV_LEVEL": "0",
        "BODY_INTERM_PRED_LAYERS": "0,1,2",
        "HAND_INTERM_PRED_LAYERS": "0,1",
        "MHR_USE_CUDA_GRAPH": "1",
        "USE_COMPILE": "1",
        "USE_COMPILE_BACKBONE": "1",
        "DECODER_COMPILE": "1",
        "COMPILE_MODE": "reduce-overhead",
        "COMPILE_WARMUP_BATCH_SIZES": "1",
        "USE_TRT_BACKBONE": "1",
        "TRT_BACKBONE_PATH": args.backbone_engine,
        "TORCHINDUCTOR_CACHE_DIR": os.path.expanduser("~/.cache/torchinductor"),
        "TRITON_CACHE_DIR": os.path.expanduser("~/.cache/triton"),
    }
    env.update(DTYPE_VARIANT_ENV[args.dtype_variant])
    if args.no_compile:
        env.update(COMPILE_DISABLED_ENV)
    os.environ.update(env)


def main():
    args = parse_args()
    set_perf_env(args)

    repo = Path(args.repo).resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))

    # Deferred imports: the Fast-SAM-3D-Body modules read the env set above at import time.
    import cv2
    import numpy as np
    import torch

    from mocap.core.setup_estimator import build_default_estimator

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading Fast-SAM-3D-Body estimator")
    print(f"[INFO] repo={repo}")
    print(f"[INFO] video={args.video}")
    print(f"[INFO] dtype_variant={args.dtype_variant}")
    print(f"[INFO] LAYER_DTYPE={os.environ.get('LAYER_DTYPE')}")
    print(f"[INFO] USE_COMPILE={os.environ.get('USE_COMPILE')}")
    print(f"[INFO] MHR_USE_CUDA_GRAPH={os.environ.get('MHR_USE_CUDA_GRAPH')}")
    print(f"[INFO] USE_TRT_BACKBONE={os.environ.get('USE_TRT_BACKBONE')}")
    print(f"[INFO] FOV_TRT={os.environ.get('FOV_TRT')}")
    print(f"[INFO] backbone_engine={os.environ.get('TRT_BACKBONE_PATH')}")
    print(f"[INFO] fov_engine={os.environ.get('FOV_TRT_PATH')}")
    print(f"[INFO] checkpoint_path={args.checkpoint_path}")
    print(f"[INFO] mhr_path={args.mhr_path}")

    estimator = build_default_estimator(
        image_size=args.image_size,
        yolo_model_path=args.yolo_model,
        fov_model_size="s",
        fov_resolution_level=0,
        fov_fixed_size=args.image_size,
        fov_fast_mode=True,
        checkpoint_path=args.checkpoint_path,
        mhr_path=args.mhr_path,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video: {width}x{height}, FPS={fps:.2f}, frames={total_frames}")

    print(f"[WARMUP] {args.warmup_frames} frames")
    for idx in range(args.warmup_frames):
        ok, frame_bgr = cap.read()
        if not ok:
            raise SystemExit("Video ended during warmup")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        estimator.last_timings = {}
        _ = estimator.process_one_image(
            frame_rgb,
            bbox_thr=args.bbox_thresh,
            use_mask=False,
            hand_box_source="yolo_pose",
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if (idx + 1) % 5 == 0:
            print(f"[WARMUP] {idx + 1}/{args.warmup_frames}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print("[INFO] Measurement started")

    with output_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "frame",
                "dtype_variant",
                "persons",
                "detection_ms",
                "fov_ms",
                "inference_ms",
                "backbone_ms",
                "decoders_ms",
                "total_ms",
                "fps_inst",
                "inference_fps_inst",
            ]
        )

        frame_idx = 0
        while True:
            if args.max_frames and frame_idx >= args.max_frames:
                break
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            estimator.last_timings = {}

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = estimator.process_one_image(
                frame_rgb,
                bbox_thr=args.bbox_thresh,
                use_mask=False,
                hand_box_source="yolo_pose",
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_ms = (time.perf_counter() - t0) * 1000.0

            timings = dict(getattr(estimator, "last_timings", {}) or {})
            inference_ms = float(timings.get("model_ms", np.nan))
            fps_inst = 1000.0 / total_ms if total_ms > 0 else np.nan
            inference_fps = 1000.0 / inference_ms if inference_ms > 0 else np.nan
            writer.writerow(
                [
                    frame_idx,
                    args.dtype_variant,
                    len(outputs),
                    f"{float(timings.get('detection_ms', np.nan)):.4f}",
                    f"{float(timings.get('fov_ms', np.nan)):.4f}",
                    f"{inference_ms:.4f}",
                    f"{float(timings.get('backbone_ms', np.nan)):.4f}",
                    f"{float(timings.get('decoders_ms', np.nan)):.4f}",
                    f"{total_ms:.4f}",
                    f"{fps_inst:.6f}",
                    f"{inference_fps:.6f}",
                ]
            )
            if frame_idx % 30 == 0:
                fp.flush()
                print(
                    f"frame={frame_idx}/{total_frames} persons={len(outputs)} "
                    f"inference_ms={inference_ms:.1f} total_ms={total_ms:.1f}"
                )
            frame_idx += 1

    cap.release()
    print(f"[INFO] Saved {output_csv}")


if __name__ == "__main__":
    main()

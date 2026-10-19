#!/usr/bin/env python3
"""Calibration-image helpers shared by build_trt_int8_calibrated.py and quantize_onnx_qdq_ptq.py.

Depends only on numpy and OpenCV so that it can be used in a TensorRT / ONNX
Runtime environment without PyTorch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np


# ImageNet statistics used by SAT-HMR's DINOv2 encoder.
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def process_img_like_base(img: np.ndarray, input_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Resize so that the longer side equals input_size. Returns (img, (h, w))."""
    ori_h, ori_w = img.shape[:2]
    if ori_w >= ori_h:
        resize_rate = input_size / float(ori_w)
        new_w = input_size
        new_h = int(resize_rate * ori_h)
    else:
        resize_rate = input_size / float(ori_h)
        new_h = input_size
        new_w = int(resize_rate * ori_w)
    img_resized = cv2.resize(img, (new_w, new_h))
    return img_resized, (new_h, new_w)


def preprocess_calibration_image(
    image_path: Path,
    input_size: int,
    patch_size: int,
    expected_hw: Tuple[int, int],
) -> np.ndarray:
    """Read an image and apply SAT-HMR preprocessing; returns a float32 CHW array.

    BGR -> RGB, [0, 1] scaling, ImageNet standardization, zero padding to a
    multiple of patch_size. Raises if the padded size differs from expected_hw
    (the engine / ONNX input shape).
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    img_resized, img_size = process_img_like_base(img, input_size)
    pad_h = ((img_size[0] + patch_size - 1) // patch_size) * patch_size
    pad_w = ((img_size[1] + patch_size - 1) // patch_size) * patch_size

    if (pad_h, pad_w) != expected_hw:
        raise RuntimeError(
            f"Preprocess shape mismatch for {image_path}: got {(pad_h, pad_w)} expected {expected_hw}"
        )

    pad_img = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
    pad_img[: img_size[0], : img_size[1]] = img_resized

    rgb = pad_img[:, :, ::-1].astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
    return np.ascontiguousarray(chw)


def resolve_manifest_path(raw_path: str, manifest_path: Path) -> Path:
    """Resolve a manifest entry: absolute path, else relative to the manifest, else to cwd."""
    p = Path(raw_path)
    if p.is_absolute() and p.exists():
        return p
    cands = [manifest_path.parent / raw_path, Path(raw_path)]
    for c in cands:
        if c.exists():
            return c
    return cands[0]


def unique_paths(paths: Iterable[Path]) -> List[Path]:
    """Drop duplicates (by resolved path) while keeping the first-seen order."""
    out: List[Path] = []
    seen = set()
    for p in paths:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out

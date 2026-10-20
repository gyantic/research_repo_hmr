#!/usr/bin/env python3
"""Check numeric parity between PyTorch SAT-HMR and exported ONNX model."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.sat_hmr_export_utils import (
    SatHmrOnnxWrapper,
    build_dummy_inputs,
    build_infer_model,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SAT-HMR PyTorch vs ONNX outputs")
    parser.add_argument("--onnx", required=True, help="Path to exported ONNX file")
    parser.add_argument("--weights", default="./weights/sat_hmr/sat_644.pth")
    parser.add_argument("--input-size", type=int, default=1288)
    parser.add_argument("--img-h", type=int, default=728)
    parser.add_argument("--img-w", type=int, default=1288)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--ort-provider", default="", help="Force ORT provider name")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--strict", action="store_true", help="Return non-zero if any output exceeds tolerances")
    xg = parser.add_mutually_exclusive_group()
    xg.add_argument(
        "--disable-xformers",
        action="store_true",
        help="Disable xFormers and use fallback path for parity check.",
    )
    xg.add_argument(
        "--enable-xformers",
        action="store_true",
        help="Keep xFormers enabled.",
    )
    return parser.parse_args()


def summarize_diff(name: str, a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    abs_diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), 1e-12)
    rel_diff = abs_diff / denom
    return {
        "name": name,
        "shape_match": float(a.shape == b.shape),
        "max_abs": float(np.max(abs_diff)) if abs_diff.size > 0 else 0.0,
        "mean_abs": float(np.mean(abs_diff)) if abs_diff.size > 0 else 0.0,
        "max_rel": float(np.max(rel_diff)) if rel_diff.size > 0 else 0.0,
        "mean_rel": float(np.mean(rel_diff)) if rel_diff.size > 0 else 0.0,
    }


def main() -> int:
    args = parse_args()
    if not args.enable_xformers:
        os.environ.setdefault("XFORMERS_DISABLED", "1")

    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - runtime dependency check
        print(f"[ERROR] onnxruntime import failed: {exc}")
        print("Install with: pip install onnxruntime or onnxruntime-gpu")
        return 2

    device = resolve_device(args.device)
    if str(device) != args.device:
        print(f"[WARN] Requested device={args.device}, fallback to device={device}.")

    print("[INFO] Building PyTorch model...")
    model, key_stats = build_infer_model(args.input_size, args.weights, str(device))
    print(
        "[INFO] weights loaded: "
        f"missing={key_stats['missing_keys']} unexpected={key_stats['unexpected_keys']}"
    )
    wrapper = SatHmrOnnxWrapper(model).to(device).eval()
    image, img_size = build_dummy_inputs(args.img_h, args.img_w, device)

    with torch.no_grad():
        torch_outs = wrapper(image, img_size)
    torch_np = [x.detach().cpu().numpy() for x in torch_outs]

    available = ort.get_available_providers()
    providers = available
    if args.ort_provider:
        if args.ort_provider in available:
            providers = [args.ort_provider]
        else:
            print(
                f"[WARN] Requested provider {args.ort_provider} not in available providers {available}. "
                "Falling back to default provider list."
            )

    print(f"[INFO] ONNX Runtime providers: {providers}")
    sess = ort.InferenceSession(args.onnx, providers=providers)
    input_names = [x.name for x in sess.get_inputs()]
    output_names = [x.name for x in sess.get_outputs()]
    if len(input_names) < 2:
        print(f"[ERROR] Unexpected ONNX inputs: {input_names}")
        return 2

    ort_inputs = {
        input_names[0]: image.detach().cpu().numpy().astype(np.float32),
        input_names[1]: img_size.detach().cpu().numpy().astype(np.int64),
    }
    ort_outs = sess.run(output_names, ort_inputs)

    print("[INFO] Parity summary")
    print("name,shape_ok,max_abs,mean_abs,max_rel,mean_rel,pass")
    has_fail = False
    for idx, name in enumerate(output_names):
        a = torch_np[idx]
        b = ort_outs[idx]
        if a.shape != b.shape:
            print(f"{name},0,NaN,NaN,NaN,NaN,0")
            has_fail = True
            continue
        stats = summarize_diff(name, a, b)
        ok = (stats["max_abs"] <= args.atol) or (stats["max_rel"] <= args.rtol)
        print(
            f"{name},1,{stats['max_abs']:.6g},{stats['mean_abs']:.6g},"
            f"{stats['max_rel']:.6g},{stats['mean_rel']:.6g},{int(ok)}"
        )
        if not ok:
            has_fail = True

    if has_fail and args.strict:
        print("[ERROR] Parity check failed under strict mode.")
        return 1

    print("[DONE] Parity check completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

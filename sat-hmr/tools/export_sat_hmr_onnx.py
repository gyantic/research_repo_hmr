#!/usr/bin/env python3
"""Export SAT-HMR inference graph to ONNX."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import onnx

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.sat_hmr_export_utils import (
    SatHmrOnnxWrapper,
    build_dummy_inputs,
    build_infer_model,
    resolve_device,
)
from tools.fix_onnx_where_broadcast import apply_fix_passes, check_and_save, format_fix_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SAT-HMR to ONNX")
    parser.add_argument("--weights", default="./weights/sat_hmr/sat_644.pth")
    parser.add_argument("--input-size", type=int, default=1288)
    parser.add_argument("--img-h", type=int, default=728)
    parser.add_argument("--img-w", type=int, default=1288)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--out", default="datas/onnx/sat_hmr.onnx")
    parser.add_argument(
        "--sat-num-lvls",
        type=int,
        default=-1,
        help="Override sat_cfg.num_lvls for export only (SAT remains enabled).",
    )
    parser.add_argument(
        "--drop-img-size-input",
        action="store_true",
        help="Allow ONNX optimization to drop img_size from graph inputs (legacy behavior).",
    )
    parser.add_argument(
        "--apply-runtime-fix",
        action="store_true",
        help="Apply SAT-HMR ONNX runtime-fix passes after export.",
    )
    parser.add_argument(
        "--fixed-out",
        default="",
        help="Output path for fixed ONNX. Default: <out_stem>_runtimefix.onnx",
    )
    parser.add_argument(
        "--dynamic-shape",
        action="store_true",
        help="Enable dynamic H/W axes for image input.",
    )
    xg = parser.add_mutually_exclusive_group()
    xg.add_argument(
        "--disable-xformers",
        action="store_true",
        help="Disable xFormers and use export-safe fallback path.",
    )
    xg.add_argument(
        "--enable-xformers",
        action="store_true",
        help="Keep xFormers enabled.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.enable_xformers:
        os.environ.setdefault("XFORMERS_DISABLED", "1")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    if str(device) != args.device:
        print(f"[WARN] Requested device={args.device}, fallback to device={device}.")

    print("[INFO] Building SAT-HMR model...")
    model, key_stats = build_infer_model(
        args.input_size,
        args.weights,
        str(device),
        sat_num_lvls_override=args.sat_num_lvls,
    )
    print(
        "[INFO] weights loaded: "
        f"missing={key_stats['missing_keys']} unexpected={key_stats['unexpected_keys']}"
    )

    keep_img_size_input = not args.drop_img_size_input
    wrapper = SatHmrOnnxWrapper(model, keep_img_size_input=keep_img_size_input).to(device).eval()
    image, img_size = build_dummy_inputs(args.img_h, args.img_w, device)

    input_names = ["image", "img_size"]
    output_names = [
        "pred_poses",
        "pred_betas",
        "pred_transl",
        "pred_confs",
        "pred_boxes",
    ]
    if keep_img_size_input:
        output_names.append("pred_intrinsics")
    dynamic_axes = None
    if args.dynamic_shape:
        dynamic_axes = {
            "image": {2: "height", 3: "width"},
        }

    do_constant_folding = device.type == "cpu"
    if not do_constant_folding:
        print("[INFO] CUDA export detected: disable ONNX constant folding to avoid device-mix errors")

    print(f"[INFO] Exporting ONNX: {out_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (image, img_size),
            str(out_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=args.opset,
            do_constant_folding=do_constant_folding,
            dynamic_axes=dynamic_axes,
        )

    print("[DONE] ONNX export complete")
    print(f"[DONE] Output: {out_path.resolve()}")

    if args.apply_runtime_fix:
        fixed_out = Path(args.fixed_out) if args.fixed_out else out_path.with_name(f"{out_path.stem}_runtimefix.onnx")
        fixed_out.parent.mkdir(parents=True, exist_ok=True)
        print("[INFO] Applying runtime-fix passes...")
        model_onnx, counts = apply_fix_passes(onnx.load(str(out_path)))
        check_and_save(model_onnx, fixed_out)
        print(f"[DONE] Runtime-fix summary: {format_fix_counts(counts)}")
        print(f"[DONE] Fixed ONNX: {fixed_out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

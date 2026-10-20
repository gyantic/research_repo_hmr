#!/usr/bin/env python3
"""Create QDQ INT8 ONNX via ONNX Runtime static PTQ.

ONNX(FP32)
    ↓ ONNX Runtimeでキャリブレーション
    ↓ スケールを計算してQDQノードとしてONNXに埋め込む
ONNX(INT8 QDQ済み)
    ↓ trtexec / TensorRT API
TensorRTエンジン

参考 (既定設定での実測):
    int8_related_layers = 500 / 1329  ≒ 37%
    int8_outputs        = 465 / 801   ≒ 58%
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnx
from onnx import numpy_helper
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

from calib_image_utils import collect_calibration_images, preprocess_calibration_image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Static PTQ (QDQ) for SAT-HMR ONNX")
    p.add_argument("--onnx_in", required=True, help="Input ONNX path")
    p.add_argument("--onnx_out", required=True, help="Output quantized ONNX path")
    p.add_argument("--calib_manifest", default="", help="Calibration manifest CSV")
    p.add_argument("--calib_dir", default="", help="Calibration image directory")
    p.add_argument("--input_name", default="image", help="ONNX input tensor name")
    p.add_argument("--input_size", type=int, default=1288, help="SAT-HMR preprocess input_size")
    p.add_argument("--patch_size", type=int, default=56, help="Patch alignment used in preprocess")
    p.add_argument("--expected_h", type=int, default=728, help="Expected H after preprocess")
    p.add_argument("--expected_w", type=int, default=1288, help="Expected W after preprocess")
    p.add_argument("--max_images", type=int, default=80, help="Use at most N calibration images") #80枚で統計的に十分(きゃりぶ)
    p.add_argument("--per_channel", action="store_true", help="Enable per-channel weight quantization")
    p.add_argument(
        "--activation_type",
        choices=["qint8", "quint8"],
        default="qint8",
        help="Activation quant type",
    )
    p.add_argument(
        "--weight_type",
        choices=["qint8", "quint8"],
        default="qint8",
        help="Weight quant type",
    )
    p.add_argument(
        "--no_symmetric",
        action="store_true",
        help="Disable symmetric quantization options (TensorRT usually needs symmetric QDQ)",
    )
    p.add_argument(
        "--op_types",
        default="Conv,MatMul",
        help="Comma-separated ONNX op types to quantize. Empty uses ONNX Runtime default.",
    )
    p.add_argument(
        "--nodes_to_exclude",
        default="",
        help="Comma-separated ONNX node names to exclude from quantization.",
    )
    p.add_argument(
        "--nodes_to_quantize",
        default="",
        help="Comma-separated ONNX node names to force-quantize.",
    )
    return p.parse_args()


def collect_images(args: argparse.Namespace) -> List[Path]:
    images = collect_calibration_images(args.calib_manifest, args.calib_dir, args.max_images)
    if not images:
        raise RuntimeError("No calibration images found. Use --calib_manifest and/or --calib_dir")
    return images


def parse_quant_type(text: str) -> QuantType:
    if text == "qint8":
        return QuantType.QInt8
    if text == "quint8":
        return QuantType.QUInt8
    raise ValueError(f"Unsupported quant type: {text}")


def parse_csv_list(text: str) -> List[str]:
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def _reshape_for_axis(vec: np.ndarray, ndim: int, axis: int) -> np.ndarray:
    shape = [1] * ndim
    shape[axis] = vec.shape[0]
    return vec.reshape(shape)


def _dq_axis(node: onnx.NodeProto, ndim: int) -> int:
    """DequantizeLinear axis attribute (default 1) with negative values wrapped; not range-checked."""
    axis = 1
    for attr in node.attribute:
        if attr.name == "axis":
            axis = int(attr.i)
            break
    if axis < 0:
        axis += ndim
    return axis


def _dequantize(q_arr: np.ndarray, s_arr: np.ndarray, zp_arr, axis: int):
    """(q - zero_point) * scale, or None if the per-axis scale / zero point does not match q."""
    qf = q_arr.astype(np.float32)
    if s_arr.ndim == 0:
        zpf = float(zp_arr) if zp_arr is not None else 0.0
        return (qf - zpf) * float(s_arr)

    if s_arr.shape[0] != q_arr.shape[axis]:
        return None
    s_b = _reshape_for_axis(s_arr, q_arr.ndim, axis)
    if zp_arr is None:
        z_b = 0.0
    elif zp_arr.ndim == 0:
        z_b = float(zp_arr)
    elif zp_arr.shape[0] != q_arr.shape[axis]:
        return None
    else:
        z_b = _reshape_for_axis(zp_arr, q_arr.ndim, axis)
    return (qf - z_b) * s_b


def apply_trt_int32_dq_bias_fix(onnx_path: Path) -> int:
    """Replace int32 DequantizeLinear constant paths with pre-dequantized FP32 initializers."""
    model = onnx.load(str(onnx_path))
    graph = model.graph
    init_map = {x.name: x for x in graph.initializer}

    kept_nodes = []
    replaced = 0

    for node in graph.node:
        if node.op_type != "DequantizeLinear":
            kept_nodes.append(node)
            continue
        if len(node.input) < 2 or len(node.output) < 1:
            kept_nodes.append(node)
            continue

        q_init = init_map.get(node.input[0])
        s_init = init_map.get(node.input[1])
        zp_init = init_map.get(node.input[2]) if len(node.input) >= 3 else None
        if q_init is None or s_init is None:
            kept_nodes.append(node)
            continue

        q_arr = numpy_helper.to_array(q_init)
        # TRT 10.13 cannot parse DQ with int32/int64 data tensors.
        if q_arr.dtype not in (np.int32, np.int64):
            kept_nodes.append(node)
            continue

        s_arr = numpy_helper.to_array(s_init).astype(np.float32)
        zp_arr = numpy_helper.to_array(zp_init).astype(np.float32) if zp_init is not None else None

        axis = _dq_axis(node, q_arr.ndim)
        if axis < 0 or axis >= q_arr.ndim:
            kept_nodes.append(node)
            continue

        deq = _dequantize(q_arr, s_arr, zp_arr, axis)
        if deq is None:
            kept_nodes.append(node)
            continue

        # Drop this DQ node: its consumers already refer to node.output[0], which the
        # new initializer now provides.
        graph.initializer.append(numpy_helper.from_array(deq.astype(np.float32), name=node.output[0]))
        replaced += 1

    if replaced == 0:
        return 0

    del graph.node[:]
    graph.node.extend(kept_nodes)
    onnx.save(model, str(onnx_path))
    return replaced


class ImageCalibrationReader(CalibrationDataReader):
    def __init__(
        self,
        image_paths: List[Path],
        input_name: str,
        input_size: int,
        patch_size: int,
        expected_hw: Tuple[int, int],
    ):
        self.image_paths = image_paths
        self.input_name = input_name
        self.input_size = input_size
        self.patch_size = patch_size
        self.expected_hw = expected_hw
        self.index = 0

    def get_next(self):
        if self.index >= len(self.image_paths):
            return None
        path = self.image_paths[self.index]
        self.index += 1
        if self.index % 10 == 0 or self.index == len(self.image_paths):
            print(f"[INFO] calibration sample {self.index}/{len(self.image_paths)}")
        # ONNX Runtime calibration feeds one image at a time: add the batch dim.
        arr = preprocess_calibration_image(path, self.input_size, self.patch_size, self.expected_hw)[None, ...]
        return {self.input_name: arr}


def main() -> int:
    args = parse_args()
    onnx_in = Path(args.onnx_in)
    onnx_out = Path(args.onnx_out)
    if not onnx_in.is_file():
        print(f"[ERROR] Input ONNX not found: {onnx_in}")
        return 2
    onnx_out.parent.mkdir(parents=True, exist_ok=True)

    try:
        image_paths = collect_images(args)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 2

    expected_hw = (int(args.expected_h), int(args.expected_w))
    op_types = parse_csv_list(args.op_types)
    nodes_to_exclude = parse_csv_list(args.nodes_to_exclude)
    nodes_to_quantize = parse_csv_list(args.nodes_to_quantize)
    print(f"[INFO] ONNX in: {onnx_in}")
    print(f"[INFO] ONNX out: {onnx_out}")
    print(f"[INFO] calib images: {len(image_paths)}")
    print(f"[INFO] expected HW: {expected_hw}")
    print(f"[INFO] op_types_to_quantize: {op_types if op_types else 'ORT default'}")
    if nodes_to_exclude:
        print(f"[INFO] nodes_to_exclude: {len(nodes_to_exclude)}")
    if nodes_to_quantize:
        print(f"[INFO] nodes_to_quantize: {len(nodes_to_quantize)}")

    reader = ImageCalibrationReader(
        image_paths=image_paths,
        input_name=args.input_name,
        input_size=args.input_size,
        patch_size=args.patch_size,
        expected_hw=expected_hw,
    )

    extra_options = {}
    if not args.no_symmetric:
        # TensorRT ONNX parser generally expects zero-point=0 for INT8 QDQ.
        extra_options["ActivationSymmetric"] = True
        extra_options["WeightSymmetric"] = True

    try:
        quantize_static(
            model_input=str(onnx_in),
            model_output=str(onnx_out),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=parse_quant_type(args.activation_type),
            weight_type=parse_quant_type(args.weight_type),
            per_channel=bool(args.per_channel),
            op_types_to_quantize=op_types,
            nodes_to_exclude=nodes_to_exclude,
            nodes_to_quantize=nodes_to_quantize,
            extra_options=extra_options,
        )
    except Exception as exc:
        print(f"[ERROR] quantize_static failed: {exc}")
        return 2

    try:
        replaced = apply_trt_int32_dq_bias_fix(onnx_out)
        if replaced > 0:
            print(f"[INFO] TRT int32-DQ bias fix applied: replaced_nodes={replaced}")
    except Exception as exc:
        print(f"[ERROR] TRT int32-DQ bias fix failed: {exc}")
        return 2

    size_mib = onnx_out.stat().st_size / (1024.0 * 1024.0) if onnx_out.is_file() else 0.0
    print(f"[DONE] Quantized ONNX: {onnx_out} ({size_mib:.2f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

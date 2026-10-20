#!/usr/bin/env python3
"""Build a TensorRT INT8 engine with image calibration (PTQ)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from calib_image_utils import collect_calibration_images, preprocess_calibration_image


def cuda_call(ret, name: str):
    if isinstance(ret, tuple):
        status = int(ret[0])
        values = ret[1:]
    else:
        status = int(ret)
        values = ()
    if status != 0:
        raise RuntimeError(f"{name} failed (cudaError={status})")
    if len(values) == 0:
        return None
    if len(values) == 1:
        return values[0]
    return values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TensorRT INT8 engine with calibrator")
    p.add_argument("--onnx", required=True, help="ONNX model path")
    p.add_argument("--engine_out", required=True, help="Output TensorRT engine path")
    p.add_argument("--calib_manifest", default="", help="Calibration manifest CSV (from extract_calibration_frames.py)")
    p.add_argument("--calib_dir", default="", help="Calibration image directory (*.jpg/*.png)")
    p.add_argument("--cache", default="datas/trt/sat_hmr_sat_int8.cache", help="Calibration cache output path")
    p.add_argument("--input_size", type=int, default=1288, help="SAT-HMR preprocess input_size")
    p.add_argument("--patch_size", type=int, default=56, help="Patch alignment used in preprocess")
    p.add_argument("--expected_h", type=int, default=-1, help="Expected input H after preprocess (-1: from ONNX input)")
    p.add_argument("--expected_w", type=int, default=-1, help="Expected input W after preprocess (-1: from ONNX input)")
    p.add_argument("--batch_size", type=int, default=1, help="Calibration batch size")
    p.add_argument("--max_images", type=int, default=-1, help="Use at most N calibration images (-1: all)")
    p.add_argument("--workspace_gb", type=float, default=8.0, help="TensorRT workspace size in GiB")
    p.add_argument("--no_fp16_fallback", action="store_true", help="Disable FP16 fallback for unsupported INT8 layers")
    p.add_argument("--verbose", action="store_true", help="Verbose TensorRT logger")
    return p.parse_args()


def load_calibration_images(args: argparse.Namespace) -> List[Path]:
    # Unlike quantize_onnx_qdq_ptq.py, an empty manifest or directory is an error here.
    images = collect_calibration_images(
        args.calib_manifest, args.calib_dir, args.max_images, require_images_per_source=True
    )
    if not images:
        raise RuntimeError("No calibration images provided. Use --calib_manifest and/or --calib_dir")
    return images


def parse_onnx_and_get_input_shape(onnx_path: Path, trt, logger) -> Tuple[object, object, Tuple[int, int, int, int], str]:
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    model_data = onnx_path.read_bytes()
    ok = parser.parse(model_data)
    if not ok:
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        joined = "\n".join(errs[:20])
        raise RuntimeError(f"Failed to parse ONNX:\n{joined}")

    if network.num_inputs < 1:
        raise RuntimeError("ONNX network has no inputs")

    inp = network.get_input(0)
    shape = tuple(int(v) for v in inp.shape)
    if len(shape) != 4:
        raise RuntimeError(f"Only 4D input is supported for calibration. got={shape}")
    if any(v <= 0 for v in shape):
        raise RuntimeError(f"Dynamic input shape is not supported in this script. got={shape}")

    return builder, network, shape, inp.name


def to_bytes(obj) -> bytes:
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    return bytes(memoryview(obj))


def build_engine(args: argparse.Namespace) -> int:
    try:
        import tensorrt as trt  # type: ignore
    except Exception as exc:
        print(f"[ERROR] TensorRT import failed: {exc}")
        return 2

    try:
        from cuda import cudart  # type: ignore
    except Exception:
        try:
            from cuda.bindings import runtime as cudart  # type: ignore
        except Exception as exc:
            print(f"[ERROR] CUDA runtime bindings import failed: {exc}")
            return 2

    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        print(f"[ERROR] ONNX not found: {onnx_path}")
        return 2

    engine_out = Path(args.engine_out)
    cache_path = Path(args.cache)
    engine_out.parent.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    images = load_calibration_images(args)
    if args.batch_size < 1:
        print("[ERROR] --batch_size must be >= 1")
        return 2
    if len(images) < args.batch_size:
        print(f"[ERROR] Not enough images for batch_size={args.batch_size}. images={len(images)}")
        return 2

    logger = trt.Logger(trt.Logger.INFO if args.verbose else trt.Logger.WARNING)
    builder, network, input_shape, input_name = parse_onnx_and_get_input_shape(onnx_path, trt, logger)
    n, c, h, w = input_shape
    if n != args.batch_size:
        print(
            f"[WARN] ONNX batch={n} but --batch_size={args.batch_size}. "
            f"Calibrator batch will use {args.batch_size}, engine input remains fixed."
        )

    expected_h = args.expected_h if args.expected_h > 0 else h
    expected_w = args.expected_w if args.expected_w > 0 else w
    expected_hw = (expected_h, expected_w)

    # Validate one sample early to catch preprocessing mismatch.
    try:
        _ = preprocess_calibration_image(images[0], args.input_size, args.patch_size, expected_hw)
    except Exception as exc:
        print(f"[ERROR] Calibration preprocessing check failed: {exc}")
        return 2

    print(f"[INFO] ONNX: {onnx_path}")
    print(f"[INFO] input_name={input_name} input_shape={input_shape}")
    print(f"[INFO] expected_preprocess_hw={expected_hw} input_size={args.input_size} patch_size={args.patch_size}")
    print(f"[INFO] calibration_images={len(images)}")

    class ImageEntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.image_paths = images
            self.batch_size = int(args.batch_size)
            self.input_size = int(args.input_size)
            self.patch_size = int(args.patch_size)
            self.expected_hw = expected_hw
            self.cache_path = cache_path
            self.index = 0

            elt_count = self.batch_size * c * self.expected_hw[0] * self.expected_hw[1]
            self.nbytes = int(elt_count) * np.dtype(np.float32).itemsize
            self.device_input = int(cuda_call(cudart.cudaMalloc(self.nbytes), "cudaMalloc(calib_input)"))

        def get_batch_size(self):
            return self.batch_size

        def get_batch(self, names: Sequence[str]):
            if self.index >= len(self.image_paths):
                return None

            end = min(self.index + self.batch_size, len(self.image_paths))
            batch_paths = self.image_paths[self.index : end]
            if len(batch_paths) < self.batch_size:
                return None

            batch = np.empty((self.batch_size, c, self.expected_hw[0], self.expected_hw[1]), dtype=np.float32)
            for i, p in enumerate(batch_paths):
                batch[i] = preprocess_calibration_image(
                    p,
                    input_size=self.input_size,
                    patch_size=self.patch_size,
                    expected_hw=self.expected_hw,
                )

            cuda_call(
                cudart.cudaMemcpy(
                    self.device_input,
                    batch.ctypes.data,
                    batch.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                ),
                "cudaMemcpy(calib_input)",
            )
            self.index = end
            if self.index % max(self.batch_size * 50, 1) == 0 or self.index == len(self.image_paths):
                print(f"[INFO] calibration progress: {self.index}/{len(self.image_paths)}")
            return [self.device_input]

        def read_calibration_cache(self):
            if self.cache_path.is_file():
                data = self.cache_path.read_bytes()
                print(f"[INFO] Using calibration cache: {self.cache_path} ({len(data)} bytes)")
                return data
            return None

        def write_calibration_cache(self, cache):
            data = bytes(cache)
            self.cache_path.write_bytes(data)
            print(f"[DONE] Wrote calibration cache: {self.cache_path} ({len(data)} bytes)")

        def __del__(self):
            ptr = getattr(self, "device_input", 0)
            if ptr:
                try:
                    cuda_call(cudart.cudaFree(ptr), "cudaFree(calib_input)")
                except Exception:
                    pass

    calibrator = ImageEntropyCalibrator()
    config = builder.create_builder_config()
    workspace_bytes = int(args.workspace_gb * (1 << 30))
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    config.set_flag(trt.BuilderFlag.INT8)

    if not args.no_fp16_fallback and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    config.int8_calibrator = calibrator

    print("[INFO] Building INT8 engine. This may take several minutes...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("[ERROR] build_serialized_network returned None")
        return 2

    plan_bytes = to_bytes(serialized)
    engine_out.write_bytes(plan_bytes)
    print(f"[DONE] INT8 engine: {engine_out} ({len(plan_bytes)} bytes)")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return build_engine(args)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


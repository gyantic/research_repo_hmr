#!/usr/bin/env python3
"""Build a TensorRT engine with the active Python TensorRT runtime.

Use this when trtexec binary version and Python TensorRT version do not match.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple


def parse_shape_spec(text: str) -> Dict[str, Tuple[int, ...]]:
    out: Dict[str, Tuple[int, ...]] = {}
    if not text:
        return out
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"invalid shape spec: {item}")
        name, dims = item.split(":", 1)
        name = name.strip()
        dims = dims.strip()
        if not name:
            raise ValueError(f"invalid shape spec: {item}")
        if dims.lower() == "scalar" or dims == "":
            out[name] = ()
        else:
            out[name] = tuple(int(x) for x in dims.split("x"))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TensorRT engine via Python API")
    p.add_argument("--onnx", required=True, help="ONNX path")
    p.add_argument("--engine_out", required=True, help="Output engine path")
    p.add_argument("--int8", action="store_true", help="Enable INT8")
    p.add_argument("--fp16", action="store_true", help="Enable FP16")
    p.add_argument("--version_compatible", action="store_true", help="Enable version-compatible engine")
    p.add_argument("--exclude_lean_runtime", action="store_true", help="Exclude embedded lean runtime")
    p.add_argument("--workspace_gb", type=float, default=8.0, help="Workspace size (GiB)")
    p.add_argument("--opt_level", type=int, default=3, help="Builder optimization level (0-5)")
    p.add_argument(
        "--shapes",
        default="",
        help="Static shape overrides for dynamic inputs. Example: image:1x3x728x1288,img_size:1x2",
    )
    p.add_argument("--verbose", action="store_true", help="Verbose TensorRT log")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import tensorrt as trt  # type: ignore

    log_level = trt.Logger.VERBOSE if args.verbose else trt.Logger.WARNING
    logger = trt.Logger(log_level)

    onnx_path = Path(args.onnx)
    out_path = Path(args.engine_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    onnx_bytes = onnx_path.read_bytes()
    if not parser.parse(onnx_bytes):
        print("[ERROR] ONNX parse failed")
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        return 1

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * (1024**3)))
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = int(args.opt_level)

    if args.int8:
        config.set_flag(trt.BuilderFlag.INT8)
    if args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if args.version_compatible:
        config.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)
    if args.exclude_lean_runtime:
        config.set_flag(trt.BuilderFlag.EXCLUDE_LEAN_RUNTIME)

    shapes = parse_shape_spec(args.shapes)
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    if any(int(d) < 0 for t in inputs for d in t.shape):
        # Every input gets a fixed (min = opt = max) shape: --shapes for the dynamic ones.
        profile = builder.create_optimization_profile()
        for t in inputs:
            shape = tuple(int(d) for d in t.shape)
            if any(d < 0 for d in shape):
                if t.name not in shapes:
                    raise RuntimeError(
                        f"Dynamic input {t.name} shape {shape} requires --shapes override"
                    )
                shape = shapes[t.name]
            profile.set_shape(t.name, shape, shape, shape)
        config.add_optimization_profile(profile)

    print(f"[INFO] TensorRT Python: {trt.__version__}")
    print(f"[INFO] ONNX: {onnx_path}")
    print(f"[INFO] OUT:  {out_path}")
    print(
        "[INFO] Flags:"
        f" int8={args.int8} fp16={args.fp16}"
        f" vc={args.version_compatible} exclude_lean={args.exclude_lean_runtime}"
    )

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("[ERROR] build_serialized_network returned None")
        return 1

    out_path.write_bytes(bytes(serialized))
    print(f"[DONE] wrote engine: {out_path} ({out_path.stat().st_size / (1024*1024):.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

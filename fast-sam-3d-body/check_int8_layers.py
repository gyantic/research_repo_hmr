#!/usr/bin/env python3
"""Check tensor dtypes in TRT engine via Python API."""
import os, sys

ENGINES = {
    "INT8": "checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_int8.engine",
    "FP16": "checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_fp16.engine",
}

try:
    import tensorrt as trt
except ImportError:
    sys.exit("tensorrt not installed")

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)

for label, path in ENGINES.items():
    if not os.path.exists(path):
        print(f"{label}: not found")
        continue

    with open(path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    n = engine.num_io_tensors
    counts = {str(trt.DataType.INT8): 0, str(trt.DataType.HALF): 0, str(trt.DataType.FLOAT): 0}
    details = []
    for i in range(n):
        name = engine.get_tensor_name(i)
        dtype = engine.get_tensor_dtype(name)
        key = str(dtype)
        counts[key] = counts.get(key, 0) + 1
        details.append(f"  [{i}] {name}: {dtype}")

    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"\n=== {label} Engine ({size_mb:.0f} MB) ===")
    print(f"  IO tensors: {n}")
    for k, v in counts.items():
        if v > 0:
            print(f"  {k}: {v}")
    print("  Tensor details:")
    for d in details:
        print(d)

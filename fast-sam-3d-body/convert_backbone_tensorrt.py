#!/usr/bin/env python3
"""
Convert DINOv3 Backbone to TensorRT.

Usage:
    python convert_backbone_tensorrt.py --export_onnx
    python convert_backbone_tensorrt.py --convert_trt
    python convert_backbone_tensorrt.py --benchmark
    python convert_backbone_tensorrt.py --all

The backbone accepts:
    Input: [B, 3, 512, 512] RGB image (normalized)
    Output: [B, 1280, 32, 32] feature map



    Instructions:
Step 1: Export and convert to TensorRT

# All-in-one: export ONNX + convert TensorRT + benchmark
python convert_backbone_tensorrt.py --all

# Or run steps individually:
python convert_backbone_tensorrt.py --export_onnx    # Export ONNX
python convert_backbone_tensorrt.py --convert_trt    # Convert to TensorRT
python convert_backbone_tensorrt.py --benchmark      # Performance comparison

Step 2: Run inference with TensorRT

# Set environment variable to enable TensorRT backbone
USE_TRT_BACKBONE=1 python profile_nsight.py --image_path ./notebook/images/dancing.jpg --detector yolo --detector_model ./checkpoints/yolo/yolo11m.engine

# Or specify a custom engine path
USE_TRT_BACKBONE=1 TRT_BACKBONE_PATH=/path/to/backbone.engine python demo_human.py ...


"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

# Add parent directory to path
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)


# Default paths
CHECKPOINT_DIR = os.path.join(parent_dir, "checkpoints", "sam-3d-body-dinov3")
TRT_OUTPUT_DIR = os.path.join(CHECKPOINT_DIR, "backbone_trt")
ONNX_PATH = os.path.join(TRT_OUTPUT_DIR, "backbone_dinov3.onnx")
TRT_PATH_BF16 = os.path.join(TRT_OUTPUT_DIR, "backbone_dinov3_bf16.engine")
TRT_PATH_FP16 = os.path.join(TRT_OUTPUT_DIR, "backbone_dinov3_fp16.engine")
TRT_PATH_INT8 = os.path.join(TRT_OUTPUT_DIR, "backbone_dinov3_int8.engine")
TRT_PATH = TRT_PATH_FP16  # Default to FP16 for better TensorRT optimization
ONNX_QDQ_PATH = os.path.join(TRT_OUTPUT_DIR, "backbone_dinov3_qdq.onnx")

# Model config
IMAGE_SIZE = (512, 512)  # H, W
EMBED_DIM = 1280  # dinov3_vith16plus
PATCH_SIZE = 16
OUTPUT_SIZE = (32, 32)  # 512 / 16 = 32


class BackboneWrapper(nn.Module):
    """
    Wrapper for DINOv3 backbone that exposes a simple forward interface.
    This wraps the get_intermediate_layers call for ONNX export.
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] normalized RGB image
        Returns:
            y: [B, C, H', W'] feature map
        """
        # get_intermediate_layers returns a list of features
        # We take the last one with reshape=True, norm=True
        y = self.encoder.get_intermediate_layers(x, n=1, reshape=True, norm=True)[-1]
        return y


def load_backbone():
    """Load the DINOv3 backbone from checkpoint."""
    from sam_3d_body.build_models import load_sam_3d_body

    # Load model using checkpoint path
    ckpt_path = os.path.join(CHECKPOINT_DIR, "model.ckpt")
    mhr_path = os.path.join(CHECKPOINT_DIR, "assets", "mhr_model.pt")

    model, cfg = load_sam_3d_body(checkpoint_path=ckpt_path, mhr_path=mhr_path)

    # Extract backbone
    backbone = model.backbone
    backbone.eval()

    print(f"Backbone type: {cfg.MODEL.BACKBONE.TYPE}")
    print(f"Embed dim: {backbone.embed_dim}")
    print(f"Patch size: {backbone.patch_size}")

    return backbone


def step1_export_onnx(backbone, batch_sizes=[1, 2, 4]):
    """Export backbone to ONNX with dynamic batch size."""
    print("=" * 60)
    print("Step 1: Export Backbone to ONNX")
    print("=" * 60)

    # Ensure output directory exists
    os.makedirs(TRT_OUTPUT_DIR, exist_ok=True)
    print(f"  Output directory: {TRT_OUTPUT_DIR}")

    # Create wrapper and convert to FP32 for ONNX export
    # (TensorRT will handle FP16 conversion during engine build)
    wrapper = BackboneWrapper(backbone.encoder)
    wrapper.eval()
    wrapper.float()  # Convert to FP32
    wrapper.cuda()

    # Test input (FP32)
    dummy_input = torch.randn(1, 3, *IMAGE_SIZE, device="cuda", dtype=torch.float32)

    # Verify output
    with torch.no_grad():
        output = wrapper(dummy_input)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")

    # Export to ONNX with dynamic batch
    print("  Exporting to ONNX...")

    # Dynamic axes for batch dimension
    dynamic_axes = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }

    torch.onnx.export(
        wrapper,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )

    print(f"  [SUCCESS] Saved to: {ONNX_PATH}")
    print(f"  File size: {os.path.getsize(ONNX_PATH) / 1024 / 1024:.1f} MB")

    # Verify ONNX
    try:
        import onnx
        model = onnx.load(ONNX_PATH)
        onnx.checker.check_model(model)
        print("  ONNX model verified!")
    except Exception as e:
        print(f"  Warning: ONNX verification failed: {e}")

    return True


import tensorrt as _trt_for_calib


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# Default calibration video path (relative to project root)
_DEFAULT_CALIB_VIDEO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos", "sample_video.mp4")


class _VideoCalibrator(_trt_for_calib.IInt8EntropyCalibrator2):
    """INT8 calibrator using real video frames (ImageNet-normalized FP32)."""

    def __init__(self, batch_size=1, video_path=None, num_frames=200):
        super().__init__()
        self._batch_size = batch_size
        self._cache_file = os.path.join(TRT_OUTPUT_DIR, "int8_calibration.cache")

        import cv2
        if video_path is None:
            video_path = _DEFAULT_CALIB_VIDEO

        mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1).cuda()
        std  = torch.tensor(_IMAGENET_STD,  dtype=torch.float32).view(3, 1, 1).cuda()

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // num_frames)
        frames = []
        for i in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
            t = torch.from_numpy(frame).float().cuda().permute(2, 0, 1) / 255.0
            t = (t - mean) / std
            frames.append(t)
            if len(frames) >= num_frames:
                break
        cap.release()
        print(f"  [Calibrator] Loaded {len(frames)} frames from {video_path}")

        all_frames = torch.stack(frames)  # [N, 3, H, W]
        pad = (batch_size - len(all_frames) % batch_size) % batch_size
        if pad > 0:
            all_frames = torch.cat([all_frames, all_frames[:pad]], dim=0)
        self._batches = [
            all_frames[i:i + batch_size].contiguous()
            for i in range(0, len(all_frames), batch_size)
        ]
        self._idx = 0

    def get_batch_size(self):
        return self._batch_size

    def get_batch(self, names):
        if self._idx >= len(self._batches):
            return None
        batch = self._batches[self._idx]
        self._idx += 1
        return [batch.data_ptr()]

    def read_calibration_cache(self):
        if os.path.exists(self._cache_file):
            with open(self._cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self._cache_file, "wb") as f:
            f.write(cache)


def _load_calib_frames(video_path=None, num_frames=200):
    """Load calibration frames from video, ImageNet-normalized, on CUDA."""
    import cv2
    if video_path is None:
        video_path = _DEFAULT_CALIB_VIDEO

    mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1).cuda()
    std  = torch.tensor(_IMAGENET_STD,  dtype=torch.float32).view(1, 3, 1, 1).cuda()

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // num_frames)
    frames = []
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
        t = torch.from_numpy(frame).float().cuda().permute(2, 0, 1).unsqueeze(0) / 255.0
        t = (t - mean) / std
        frames.append(t)
        if len(frames) >= num_frames:
            break
    cap.release()
    print(f"  [Calib] Loaded {len(frames)} frames from {video_path}")
    return frames


def step2b_quantize_onnx(video_path=None, num_frames=200):
    """
    Explicit INT8 PTQ using nvidia-modelopt on PyTorch (CUDA).

    Flow:
      1. Load PyTorch backbone
      2. mtq.quantize(model, INT8_DEFAULT_CFG, forward_loop)  ← inserts fake-quant on CUDA
      3. torch.onnx.export → Q/DQ nodes in ONNX
      4. TRT reads QDQ ONNX → real INT8 kernel selection (no calibrator needed)
    """
    print("\n" + "=" * 60)
    print("Step 2b: Explicit INT8 PTQ via modelopt (PyTorch CUDA)")
    print("=" * 60)

    try:
        import modelopt.torch.quantization as mtq
    except ImportError:
        print("  [ERROR] nvidia-modelopt not installed.")
        print("  Install: pip install nvidia-modelopt")
        return False

    # Load backbone
    print("  Loading backbone...")
    backbone = load_backbone()
    wrapper = BackboneWrapper(backbone.encoder)
    wrapper.eval().cuda().float()

    # Load calibration frames
    calib_frames = _load_calib_frames(video_path=video_path, num_frames=num_frames)

    def forward_loop(model):
        with torch.no_grad():
            for i, frame in enumerate(calib_frames):
                model(frame)
                if (i + 1) % 20 == 0:
                    print(f"    calibrated {i+1}/{len(calib_frames)} frames")

    # Apply INT8 PTQ: inserts TensorQuantizer fake-quant layers, then calibrates
    print("  Applying INT8 quantization (this runs on CUDA)...")
    mtq.quantize(wrapper, mtq.INT8_DEFAULT_CFG, forward_loop)
    mtq.print_quant_summary(wrapper)

    # Export quantized model to ONNX (TensorQuantizer → QuantizeLinear/DequantizeLinear)
    print(f"  Exporting QDQ ONNX to: {ONNX_QDQ_PATH}")
    os.makedirs(TRT_OUTPUT_DIR, exist_ok=True)

    dummy_input = torch.randn(1, 3, *IMAGE_SIZE, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            ONNX_QDQ_PATH,
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
            do_constant_folding=False,   # keep Q/DQ nodes intact
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

    size_mb = os.path.getsize(ONNX_QDQ_PATH) / 1024 / 1024
    print(f"  [SUCCESS] QDQ ONNX saved: {ONNX_QDQ_PATH} ({size_mb:.0f} MB)")

    # Quick sanity: count Q/DQ nodes
    import onnx
    m = onnx.load(ONNX_QDQ_PATH)
    qdq_count = sum(1 for n in m.graph.node if n.op_type in ("QuantizeLinear", "DequantizeLinear"))
    print(f"  Q/DQ nodes in ONNX: {qdq_count}")
    return True


def step2_convert_tensorrt(batch_sizes=[1, 2, 4], use_int8=False):
    """Convert ONNX to TensorRT with FP16."""
    out_path = TRT_PATH_INT8 if use_int8 else TRT_PATH_FP16
    precision_label = "INT8+FP16" if use_int8 else "FP16"
    print("\n" + "=" * 60)
    print(f"Step 2: Convert to TensorRT ({precision_label})")
    print("=" * 60)

    try:
        import tensorrt as trt
    except ImportError:
        print("  [ERROR] TensorRT not installed")
        print("  Install with: pip install tensorrt")
        return False

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # For INT8: use QDQ ONNX (explicit quantization, no calibrator needed)
    # For FP16: use original ONNX
    if use_int8:
        if not os.path.exists(ONNX_QDQ_PATH):
            print(f"  [ERROR] QDQ ONNX not found: {ONNX_QDQ_PATH}")
            print("  Run --quantize_onnx first to generate it.")
            return False
        onnx_to_parse = ONNX_QDQ_PATH
    else:
        onnx_to_parse = ONNX_PATH

    print("  Parsing ONNX...")
    print(f"  ONNX file: {onnx_to_parse}")

    if not parser.parse_from_file(onnx_to_parse):
        for i in range(parser.num_errors):
            print(f"  [ERROR] {parser.get_error(i)}")
        return False

    # Print network info
    print(f"  Network inputs: {network.num_inputs}")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"    {inp.name}: {inp.shape}")
    print(f"  Network outputs: {network.num_outputs}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"    {out.name}: {out.shape}")

    # Set min/opt/max shapes for batch dimension
    min_batch = min(batch_sizes)
    opt_batch = batch_sizes[len(batch_sizes) // 2] if len(batch_sizes) > 1 else batch_sizes[0]
    max_batch = max(batch_sizes)

    # Build config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB

    # Always enable FP16 for layers that cannot run in INT8 (Softmax, LayerNorm, etc.)
    config.set_flag(trt.BuilderFlag.FP16)

    if use_int8:
        # Explicit quantization via Q/DQ nodes in ONNX - no calibrator needed
        # TRT reads QuantizeLinear/DequantizeLinear nodes and uses INT8 for MatMul/Conv
        config.set_flag(trt.BuilderFlag.INT8)
        # Set I/O to FP16 (Q/DQ ONNX has FP32 I/O by ORT convention, but we want FP16)
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            inp.dtype = trt.float16
        for i in range(network.num_outputs):
            out = network.get_output(i)
            out.dtype = trt.float16
        print("  Using explicit INT8 (Q/DQ ONNX) + FP16 fallback, FP16 I/O")
    else:
        # Set input/output layers to use FP16
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            inp.dtype = trt.float16
        for i in range(network.num_outputs):
            out = network.get_output(i)
            out.dtype = trt.float16
        print("  Using FP16 precision (compute + I/O)")

    # Optimization profile for dynamic batch size
    profile = builder.create_optimization_profile()

    profile.set_shape(
        "input",
        (min_batch, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]),  # min
        (opt_batch, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]),  # opt
        (max_batch, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]),  # max
    )
    config.add_optimization_profile(profile)

    print(f"  Batch size range: [{min_batch}, {opt_batch}, {max_batch}]")

    # Build engine
    print("  Building TensorRT engine (this may take several minutes)...")
    engine = builder.build_serialized_network(network, config)

    if engine is None:
        print("  [ERROR] Engine build failed")
        return False

    with open(out_path, "wb") as f:
        f.write(engine)

    print(f"  [SUCCESS] Saved to: {out_path}")
    print(f"  File size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")
    return True


class TRTBackbone:
    """TensorRT inference wrapper for backbone."""

    def __init__(self, engine_path):
        import tensorrt as trt

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Get binding info
        self.input_name = "input"
        self.output_name = "output"

    def __call__(self, x):
        """
        Run TensorRT inference.
        Args:
            x: [B, 3, H, W] input tensor (BF16 or FP32)
        Returns:
            output: [B, C, H', W'] feature map (BF16)
        """
        batch_size = x.shape[0]

        # Set input shape for dynamic batch
        self.context.set_input_shape(self.input_name, x.shape)

        # Allocate output buffer (FP16)
        output = torch.empty(
            batch_size, EMBED_DIM, OUTPUT_SIZE[0], OUTPUT_SIZE[1],
            device=x.device, dtype=torch.float16
        )

        # Set tensor addresses
        x_fp16 = x.half() if x.dtype != torch.float16 else x
        self.context.set_tensor_address(self.input_name, x_fp16.data_ptr())
        self.context.set_tensor_address(self.output_name, output.data_ptr())

        # Execute
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)

        return output


def step3_benchmark(backbone, save_results=True):
    """Benchmark PyTorch vs TensorRT."""
    import datetime, json
    print("\n" + "=" * 60)
    print("Step 3: Benchmark")
    print("=" * 60)

    results = {"timestamp": datetime.datetime.now().isoformat(), "gpu": torch.cuda.get_device_name(0), "batch_sizes": {}}

    # Test different batch sizes
    for batch_size in [1]:
        print(f"\n  Batch size: {batch_size}")

        # Test input
        x = torch.randn(batch_size, 3, *IMAGE_SIZE, device="cuda", dtype=torch.float32)
        batch_results = {}

        # PyTorch (BF16)
        print("  [PyTorch BF16]")
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    _ = backbone(x)
            torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(50):
                with torch.no_grad():
                    _ = backbone(x)
            torch.cuda.synchronize()
            pt_time = (time.perf_counter() - start) * 1000 / 50
            print(f"    Time: {pt_time:.3f} ms/call")
            batch_results["pytorch_bf16_ms"] = round(pt_time, 3)

        # TensorRT FP16
        if os.path.exists(TRT_PATH_FP16):
            print("  [TensorRT FP16]")
            trt_backbone = TRTBackbone(TRT_PATH_FP16)

            # Warmup
            for _ in range(5):
                _ = trt_backbone(x)
            torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(50):
                _ = trt_backbone(x)
            torch.cuda.synchronize()
            trt_time = (time.perf_counter() - start) * 1000 / 50
            print(f"    Time: {trt_time:.3f} ms/call")
            print(f"    Speedup: {pt_time / trt_time:.2f}x")
            batch_results["trt_fp16_ms"] = round(trt_time, 3)
            batch_results["trt_fp16_speedup"] = round(pt_time / trt_time, 2)

            # Verify output
            with torch.cuda.amp.autocast(dtype=torch.float16):
                with torch.no_grad():
                    pt_out = backbone(x)
            trt_out = trt_backbone(x)

            diff = (pt_out.float() - trt_out.float()).abs()
            print(f"    Max diff: {diff.max().item():.6f}")
            print(f"    Mean diff: {diff.mean().item():.6f}")
        else:
            print("  [TensorRT FP16] Engine not found, skipping...")

        # TensorRT INT8
        if os.path.exists(TRT_PATH_INT8):
            print("  [TensorRT INT8+FP16]")
            trt_backbone_int8 = TRTBackbone(TRT_PATH_INT8)

            for _ in range(5):
                _ = trt_backbone_int8(x)
            torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(50):
                _ = trt_backbone_int8(x)
            torch.cuda.synchronize()
            int8_time = (time.perf_counter() - start) * 1000 / 50
            print(f"    Time: {int8_time:.3f} ms/call")
            print(f"    Speedup vs PyTorch: {pt_time / int8_time:.2f}x")
            batch_results["trt_int8_ms"] = round(int8_time, 3)
            batch_results["trt_int8_speedup_vs_pytorch"] = round(pt_time / int8_time, 2)
            if os.path.exists(TRT_PATH_FP16):
                sp = trt_time / int8_time
                print(f"    Speedup vs FP16: {sp:.2f}x")
                batch_results["trt_int8_speedup_vs_fp16"] = round(sp, 2)

        results["batch_sizes"][str(batch_size)] = batch_results

    if save_results:
        import csv
        out_path = os.path.join(TRT_OUTPUT_DIR, "benchmark_results.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "gpu", "batch_size", "pytorch_bf16_ms", "trt_fp16_ms", "trt_fp16_speedup", "trt_int8_ms", "trt_int8_speedup_vs_pytorch", "trt_int8_speedup_vs_fp16"])
            for bs, br in results["batch_sizes"].items():
                writer.writerow([
                    results["timestamp"], results["gpu"], bs,
                    br.get("pytorch_bf16_ms", ""),
                    br.get("trt_fp16_ms", ""),
                    br.get("trt_fp16_speedup", ""),
                    br.get("trt_int8_ms", ""),
                    br.get("trt_int8_speedup_vs_pytorch", ""),
                    br.get("trt_int8_speedup_vs_fp16", ""),
                ])
        print(f"\n  Results saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert DINOv3 Backbone to TensorRT")
    parser.add_argument("--export_onnx", action="store_true", help="Export to ONNX")
    parser.add_argument("--quantize_onnx", action="store_true",
                        help="Explicit INT8 PTQ: add Q/DQ nodes to ONNX (ORT static quantization)")
    parser.add_argument("--convert_trt", action="store_true", help="Convert ONNX to TensorRT")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark PyTorch vs TensorRT")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--int8", action="store_true",
                        help="Build INT8 engine from QDQ ONNX (run --quantize_onnx first)")
    parser.add_argument("--batch_sizes", type=str, default="1,2,4", help="Batch sizes for optimization")
    parser.add_argument("--num_frames", type=int, default=200,
                        help="Number of calibration frames from video")

    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    if args.all:
        args.export_onnx = True
        args.quantize_onnx = args.int8  # only if --int8 is also passed
        args.convert_trt = True
        args.benchmark = True

    if not any([args.export_onnx, args.quantize_onnx, args.convert_trt, args.benchmark]):
        parser.print_help()
        return

    backbone = None

    if args.export_onnx or args.benchmark:
        print("Loading backbone...")
        backbone = load_backbone()

    if args.export_onnx:
        step1_export_onnx(backbone, batch_sizes)

    if args.quantize_onnx:
        step2b_quantize_onnx(num_frames=args.num_frames)

    if args.convert_trt:
        step2_convert_tensorrt(batch_sizes, use_int8=args.int8)

    if args.benchmark:
        step3_benchmark(backbone)


if __name__ == "__main__":
    main()

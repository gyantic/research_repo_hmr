#!/usr/bin/env python3
"""
Convert MoGe2 DINOv2 Encoder to TensorRT.

Usage:
    python convert_moge_encoder_trt.py --export_onnx
    python convert_moge_encoder_trt.py --convert_trt
    python convert_moge_encoder_trt.py --benchmark
    python convert_moge_encoder_trt.py --all

The MoGe2 encoder (DINOv2 ViT-S) accepts:
    Input: [B, 3, 512, 512] RGB image (normalized, FP16)
    Output:
        - features: [B, 384, 35, 35] patch features
        - cls_token: [B, 384] class token
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

from trt_calibration_frames import (
    DEFAULT_CALIB_VIDEO,
    DEFAULT_NUM_CALIB_FRAMES,
    sample_video_frames_rgb,
)

# Default paths
TRT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "moge_trt")
ONNX_PATH = os.path.join(TRT_OUTPUT_DIR, "moge_dinov2_encoder.onnx")
TRT_PATH = os.path.join(TRT_OUTPUT_DIR, "moge_dinov2_encoder_fp16.engine")
INT8_CALIB_CACHE_PATH = os.path.join(TRT_OUTPUT_DIR, "moge_int8_calib.cache")

# Model config for MoGe2 ViT-S with resolution_level=0
IMAGE_SIZE = 512
EMBED_DIM = 384  # ViT-S
TOKEN_ROWS = 35
TOKEN_COLS = 35
BATCH_SIZE = 1  # Fixed batch size for TRT


class MoGeEncoderWrapper(nn.Module):
    """
    Wrapper for MoGe2 DINOv2 encoder that exposes a simple forward interface.
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] normalized RGB image (FP16)
        Returns:
            features: [B, C, H', W'] patch features
            cls_token: [B, C] class token
        """
        # MoGe2 encoder returns (features, cls_token)
        # features: [B, embed_dim, token_rows, token_cols]
        # cls_token: [B, embed_dim]
        features, cls_token = self.encoder(
            x,
            token_rows=TOKEN_ROWS,
            token_cols=TOKEN_COLS,
            return_class_token=True
        )
        return features, cls_token


def load_moge_encoder():
    """Load the MoGe2 DINOv2 encoder."""
    from moge.model.v2 import MoGeModel

    # Load MoGe2 ViT-S model
    model_path = "Ruicheng/moge-2-vits-normal"
    print(f"Loading MoGe2 model from: {model_path}")

    moge_model = MoGeModel.from_pretrained(model_path).cuda()
    moge_model.half()
    moge_model.eval()

    # Extract encoder
    encoder = moge_model.encoder
    print(f"Encoder type: {type(encoder)}")

    # Enable ONNX compatible mode to avoid unsupported ops (antialias interpolation)
    if hasattr(encoder, 'backbone'):
        encoder.backbone.onnx_compatible_mode = True
        print("  Enabled ONNX compatible mode on backbone")

    return encoder


def export_onnx():
    """Export MoGe2 encoder to ONNX."""
    print("\n" + "=" * 60)
    print("Step 1: Export to ONNX")
    print("=" * 60)

    # Create output directory
    os.makedirs(TRT_OUTPUT_DIR, exist_ok=True)

    # Load encoder
    encoder = load_moge_encoder()
    wrapper = MoGeEncoderWrapper(encoder)
    wrapper.eval()

    # Monkey-patch F.interpolate to disable antialias (not supported in ONNX)
    import torch.nn.functional as F
    _original_interpolate = F.interpolate

    def _interpolate_no_antialias(*args, **kwargs):
        kwargs.pop('antialias', None)  # Remove antialias if present
        return _original_interpolate(*args, **kwargs)

    F.interpolate = _interpolate_no_antialias
    print("  Patched F.interpolate to disable antialias for ONNX export")

    # Create dummy input
    dummy_input = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, device="cuda", dtype=torch.float16)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Input dtype: {dummy_input.dtype}")

    # Test forward pass
    with torch.no_grad():
        features, cls_token = wrapper(dummy_input)
        print(f"Output features shape: {features.shape}")
        print(f"Output cls_token shape: {cls_token.shape}")

    # Export to ONNX
    print(f"\nExporting to ONNX: {ONNX_PATH}")

    torch.onnx.export(
        wrapper,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["features", "cls_token"],
        dynamic_axes=None,  # Fixed batch size
    )

    print(f"ONNX exported successfully: {ONNX_PATH}")
    print(f"File size: {os.path.getsize(ONNX_PATH) / 1024 / 1024:.2f} MB")


def _make_video_calibrator(trt, video_path=DEFAULT_CALIB_VIDEO, num_frames=DEFAULT_NUM_CALIB_FRAMES):
    """Build an IInt8EntropyCalibrator2 over video frames (FP16 RGB scaled to [0, 1]).

    Takes the ``tensorrt`` module as an argument because TensorRT is imported lazily.
    """

    class _VideoCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            frames = [
                torch.from_numpy(frame).cuda().permute(2, 0, 1).half() / 255.0
                for frame in sample_video_frames_rgb(
                    video_path, num_frames, width=IMAGE_SIZE, height=IMAGE_SIZE
                )
            ]
            print(f"  [MoGe Calibrator] Loaded {len(frames)} frames from {video_path}")

            all_frames = torch.stack(frames)  # [N, 3, H, W]
            self._batches = [
                all_frames[i:i + BATCH_SIZE].contiguous()
                for i in range(0, len(all_frames), BATCH_SIZE)
            ]
            self._next_batch_idx = 0

        def get_batch_size(self):
            return BATCH_SIZE

        def get_batch(self, names):
            if self._next_batch_idx >= len(self._batches):
                return None
            batch = self._batches[self._next_batch_idx]
            self._next_batch_idx += 1
            return [batch.data_ptr()]

        def read_calibration_cache(self):
            # Never reuse a cache: always calibrate from the current video frames.
            return None

        def write_calibration_cache(self, cache):
            with open(INT8_CALIB_CACHE_PATH, "wb") as f:
                f.write(cache)

    return _VideoCalibrator()


def convert_trt(use_int8=False):
    """Convert ONNX to TensorRT engine."""
    print("\n" + "=" * 60)
    print(f"Step 2: Convert to TensorRT ({'INT8+FP16' if use_int8 else 'FP16'})")
    print("=" * 60)

    import tensorrt as trt

    # Check ONNX exists
    if not os.path.exists(ONNX_PATH):
        print(f"Error: ONNX file not found: {ONNX_PATH}")
        print("Run --export_onnx first")
        return

    # Create builder
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    print(f"Parsing ONNX: {ONNX_PATH}")
    with open(ONNX_PATH, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"ONNX parsing error: {parser.get_error(error)}")
            return

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
    config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16

    out_path = TRT_PATH
    if use_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = _make_video_calibrator(trt)
        out_path = TRT_PATH.replace("_fp16.engine", "_int8.engine")

    # Build engine
    print("Building TensorRT engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("Error: Failed to build TensorRT engine")
        return

    # Save engine
    print(f"Saving engine to: {out_path}")
    with open(out_path, "wb") as f:
        f.write(serialized_engine)

    print(f"TensorRT engine saved successfully")
    print(f"File size: {os.path.getsize(out_path) / 1024 / 1024:.2f} MB")


def benchmark():
    """Benchmark PyTorch vs TensorRT."""
    print("\n" + "=" * 60)
    print("Step 3: Benchmark")
    print("=" * 60)

    import tensorrt as trt

    # Load PyTorch encoder
    encoder = load_moge_encoder()
    wrapper = MoGeEncoderWrapper(encoder)
    wrapper.eval()

    # Create input
    dummy_input = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, device="cuda", dtype=torch.float16)

    # Warmup PyTorch
    print("\nWarming up PyTorch...")
    for _ in range(10):
        with torch.no_grad():
            _ = wrapper(dummy_input)
    torch.cuda.synchronize()

    # Benchmark PyTorch
    print("Benchmarking PyTorch...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            _ = wrapper(dummy_input)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - t0) / 100 * 1000
    print(f"PyTorch: {pytorch_time:.2f} ms")

    # Check TRT engine exists
    if not os.path.exists(TRT_PATH):
        print(f"\nTensorRT engine not found: {TRT_PATH}")
        print("Run --convert_trt first")
        return

    # Load TRT engine
    print("\nLoading TensorRT engine...")
    logger = trt.Logger(trt.Logger.WARNING)
    with open(TRT_PATH, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # Allocate output buffers
    features_buf = torch.empty((BATCH_SIZE, EMBED_DIM, TOKEN_ROWS, TOKEN_COLS),
                               device="cuda", dtype=torch.float16)
    cls_buf = torch.empty((BATCH_SIZE, EMBED_DIM), device="cuda", dtype=torch.float16)

    # Set tensor addresses
    context.set_input_shape("image", tuple(dummy_input.shape))
    context.set_tensor_address("image", dummy_input.data_ptr())
    context.set_tensor_address("features", features_buf.data_ptr())
    context.set_tensor_address("cls_token", cls_buf.data_ptr())

    # Warmup TRT
    print("Warming up TensorRT...")
    for _ in range(10):
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    # Benchmark TRT
    print("Benchmarking TensorRT...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    trt_time = (time.perf_counter() - t0) / 100 * 1000
    print(f"TensorRT: {trt_time:.2f} ms")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"PyTorch:   {pytorch_time:.2f} ms")
    print(f"TensorRT:  {trt_time:.2f} ms")
    print(f"Speedup:   {pytorch_time / trt_time:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Convert MoGe2 encoder to TensorRT")
    parser.add_argument("--export_onnx", action="store_true", help="Export to ONNX")
    parser.add_argument("--convert_trt", action="store_true", help="Convert ONNX to TensorRT")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark PyTorch vs TensorRT")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--int8", action="store_true", help="Enable INT8+FP16 (entropy calibration on 200 frames of datasets/videos/sample_video.mp4)")

    args = parser.parse_args()

    if args.all:
        export_onnx()
        convert_trt(use_int8=args.int8)
        benchmark()
    elif args.export_onnx:
        export_onnx()
    elif args.convert_trt:
        convert_trt(use_int8=args.int8)
    elif args.benchmark:
        benchmark()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

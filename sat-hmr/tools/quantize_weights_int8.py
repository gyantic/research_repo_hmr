#!/usr/bin/env python3
"""
Weight-only INT8 quantization for SAT-HMR .pth files.

Quantizes Linear weights to INT8 (per-channel) to reduce file size (~half).
The quantized .pth can be loaded with load_quantized_state_dict() below,
which dequantizes weights before loading into the model.

Usage:
    # Quantize
    python tools/quantize_weights_int8.py \
        --input weights/sat_hmr/sat_644.pth \
        --output weights/sat_hmr/sat_644_int8w.pth

    # Verify
    python tools/quantize_weights_int8.py \
        --input weights/sat_hmr/sat_644.pth \
        --output weights/sat_hmr/sat_644_int8w.pth \
        --verify
"""

from __future__ import annotations

import argparse
import pathlib

import torch


def quantize_state_dict(state_dict: dict) -> tuple[dict, dict]:
    """Quantize all 2D Linear weights in state_dict to INT8 (per output channel).

    Returns:
        quant_dict: state dict with *.weight replaced by *.weight_int8 + *.weight_scale
        stats: summary counts
    """
    quant_dict = {}
    n_quantized = 0
    n_skipped = 0

    for key, tensor in state_dict.items():
        if key.endswith(".weight") and tensor.ndim == 2:
            w = tensor.float()
            # per output-channel scale
            scale = w.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8) / 127.0
            w_int8 = (w / scale).round().clamp(-128, 127).to(torch.int8)
            quant_dict[key.replace(".weight", ".weight_int8")] = w_int8
            quant_dict[key.replace(".weight", ".weight_scale")] = scale.squeeze(1).half()
            n_quantized += 1
        else:
            quant_dict[key] = tensor
            n_skipped += 1

    stats = {"quantized": n_quantized, "skipped": n_skipped}
    return quant_dict, stats


def dequantize_state_dict(quant_dict: dict) -> dict:
    """Dequantize INT8 weights back to float32 for model.load_state_dict().

    Same expansion as sathmr_common.dequantize_int8_weight_state_dict_if_needed(); kept here
    so this tool does not pull in sathmr_common's demo dependencies.
    """
    state_dict = {}
    for key, tensor in quant_dict.items():
        if key.endswith(".weight_int8"):
            base = key[: -len("_int8")]  # e.g. "foo.weight"
            scale = quant_dict[base + "_scale"].float().unsqueeze(1)
            state_dict[base] = (tensor.float() * scale).to(torch.float32)
        elif not key.endswith(".weight_scale"):
            state_dict[key] = tensor
    return state_dict


def load_quantized_state_dict(path: str | pathlib.Path, map_location="cpu") -> dict:
    """Load a quantized .pth and return a dequantized state dict ready for model.load_state_dict()."""
    quant_dict = torch.load(path, map_location=map_location, weights_only=True)
    return dequantize_state_dict(quant_dict)


def file_size_mb(path: pathlib.Path) -> float:
    return path.stat().st_size / 1024 / 1024


def main() -> None:
    p = argparse.ArgumentParser("Quantize SAT-HMR weights to INT8")
    p.add_argument("--input", required=True, help="Input .pth path")
    p.add_argument("--output", required=True, help="Output quantized .pth path")
    p.add_argument("--verify", action="store_true", help="Verify max abs error after quantization")
    args = p.parse_args()

    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)

    print(f"Loading {input_path} ({file_size_mb(input_path):.1f} MB)...")
    state_dict = torch.load(input_path, map_location="cpu", weights_only=True)

    print("Quantizing weights...")
    quant_dict, stats = quantize_state_dict(state_dict)
    print(f"  Quantized: {stats['quantized']} tensors")
    print(f"  Skipped:   {stats['skipped']} tensors")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(quant_dict, output_path)
    print(f"Saved {output_path} ({file_size_mb(output_path):.1f} MB)")
    print(f"Compression: {file_size_mb(input_path):.1f} MB -> {file_size_mb(output_path):.1f} MB "
          f"({file_size_mb(output_path)/file_size_mb(input_path)*100:.1f}%)")

    if args.verify:
        print("\nVerifying dequantization error...")
        deq_dict = dequantize_state_dict(quant_dict)
        max_errs = []
        for key, orig in state_dict.items():
            if key in deq_dict and key.endswith(".weight"):
                err = (deq_dict[key] - orig.float()).abs().max().item()
                max_errs.append((key, err))
        max_errs.sort(key=lambda x: -x[1])
        print("  Top-5 max abs errors:")
        for key, err in max_errs[:5]:
            print(f"    {key}: {err:.6f}")
        print(f"  Overall max abs error: {max_errs[0][1]:.6f}" if max_errs else "  No quantized tensors to verify")


if __name__ == "__main__":
    main()

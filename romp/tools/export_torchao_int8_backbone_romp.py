#!/usr/bin/env python3
"""
Pre-convert the ROMP HRNet32 backbone weights to torchao int8 weight-only.

The script builds only the HRNet32 backbone, loads the floating-point
checkpoint, applies torchao IntxWeightOnlyConfig(int8) to the backbone, and
saves a checkpoint whose `backbone.*` weights are quantized.

Usage:
    python tools/export_torchao_int8_backbone_romp.py \
        --input trained_models/ROMP_HRNet32_V1.pkl \
        --output trained_models/

Output:
    trained_models/ROMP_HRNet32_V1_hrnet32_backbone_int8wo.pth

"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from checkpoint_io import file_size_mb, load_checkpoint, strip_module_prefix


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if not hasattr(np, "bool"):
    np.bool = np.bool_  # type: ignore[attr-defined]

import romp  # noqa: F401  # Adds ROMP lib paths.


def extract_backbone_state_dict(state_dict: dict) -> dict:
    normalized = strip_module_prefix(state_dict)
    backbone_state_dict = {
        key.removeprefix("backbone."): value
        for key, value in normalized.items()
        if key.startswith("backbone.")
    }
    if not backbone_state_dict:
        raise RuntimeError("No `backbone.*` weights were found in the input checkpoint.")
    return backbone_state_dict


def detect_backbone_prefix(state_dict: dict) -> str:
    if any(key.startswith("module.backbone.") for key in state_dict):
        return "module.backbone."
    if any(key.startswith("backbone.") for key in state_dict):
        return "backbone."
    raise RuntimeError("Could not detect a `backbone.` prefix in the checkpoint keys.")


def merge_backbone_state_dict(state_dict: dict, backbone_state_dict: dict) -> dict:
    merged = dict(state_dict)
    backbone_prefix = detect_backbone_prefix(state_dict)
    for key, value in backbone_state_dict.items():
        merged[f"{backbone_prefix}{key}"] = value
    return merged


def build_output_payload(raw_checkpoint, merged_state_dict: dict, source_checkpoint: pathlib.Path) -> dict:
    meta = {
        "quantization": "torchao_intx_weight_only_int8",
        "scope": "backbone",
        "backbone": "hrnet32",
        "source_checkpoint": str(source_checkpoint),
    }
    if isinstance(raw_checkpoint, dict) and "model_state_dict" in raw_checkpoint:
        payload = dict(raw_checkpoint)
        payload["model_state_dict"] = merged_state_dict
        payload_meta = dict(payload.get("meta", {}))
        payload_meta.update(meta)
        payload["meta"] = payload_meta
        return payload
    return {
        "model_state_dict": merged_state_dict,
        "meta": meta,
    }


def is_int8_quant_target(module: nn.Module) -> bool:
    # Same layer selection as romp/base.py::_is_torchao_int8_weight_only_target.
    return isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, "weight")


def quant_target_modules(modules: Iterable[tuple[str, nn.Module]]) -> list[tuple[str, nn.Module]]:
    return [(name, module) for name, module in modules if is_int8_quant_target(module)]


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ROMP HRNet32 backbone weights to torchao int8 weight-only."
    )
    parser.add_argument("--input", required=True, help="Input checkpoint (.pkl / .pth)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--configs_yml", default="configs/video.yml", help="ROMP config yaml")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device used while building and quantizing the model")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_torchao_conv_int8_api():
    try:
        from torchao.quantization import IntxWeightOnlyConfig, quantize_
    except Exception as exc:
        raise RuntimeError(
            "This exporter requires a torchao build that provides "
            "IntxWeightOnlyConfig for Conv2d/Linear int8 weight-only export."
        ) from exc
    return quantize_, IntxWeightOnlyConfig


def main() -> None:
    args = parse_cli()
    inp = pathlib.Path(args.input)
    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    romp_args = [
        "--configs_yml", args.configs_yml,
        "--backbone", "hrnet",
        "--model_version", "1",
        "--model_precision", "fp32",
        "--gpu", "0" if device.type == "cuda" else "-1",
    ]
    sys.argv = [sys.argv[0]] + romp_args

    from config import ConfigContext, parse_args  # type: ignore
    from models.hrnet_32 import HigherResolutionNet  # type: ignore
    quantize_, IntxWeightOnlyConfig = resolve_torchao_conv_int8_api()

    print(f"[INFO] Loading float checkpoint: {inp} ({file_size_mb(inp):.1f} MB)")
    raw_checkpoint, state_dict = load_checkpoint(inp)
    backbone_state_dict = extract_backbone_state_dict(state_dict)

    with ConfigContext(parse_args(romp_args)):
        model = HigherResolutionNet().eval().to(device)

    incompatible = model.load_state_dict(backbone_state_dict, strict=False)
    if incompatible.missing_keys:
        print(f"[WARN] Missing keys: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"[WARN] Unexpected keys: {len(incompatible.unexpected_keys)}")

    backbone_targets = quant_target_modules(model.named_modules())
    print(f"[INFO] Quantizing {len(backbone_targets)} HRNet32 Conv2d/Linear modules on {device} ...")

    quantize_(
        model,
        IntxWeightOnlyConfig(weight_dtype=torch.int8),
        filter_fn=lambda module, _fqn: is_int8_quant_target(module),
        device=device,
    )

    out_path = out_dir / f"{inp.stem}_hrnet32_backbone_int8wo.pth"
    merged_state_dict = merge_backbone_state_dict(state_dict, model.state_dict())
    payload = build_output_payload(raw_checkpoint, merged_state_dict, inp)
    torch.save(payload, out_path)
    print(f"[INFO] Saved: {out_path} ({file_size_mb(out_path):.1f} MB)")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()

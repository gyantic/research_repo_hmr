#!/usr/bin/env python3
"""
Export SAT-HMR weights quantized with torchao (int4, fp8).

Usage:
    python tools/export_torchao_weights.py \
        --input  weights/sat_hmr/sat_644.pth \
        --output weights/sat_hmr/ \
        --modes  int4,fp8
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# reuse load_config from measure_torchao_scopes
sys.path.insert(0, str(ROOT / "tools"))
from measure_torchao_scopes import load_config


def _int4_filter(module: nn.Module, fqn: str) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    w = module.weight
    if type(w) not in (torch.Tensor, nn.Parameter):
        return False
    return w.shape[0] % 16 == 0 and w.shape[1] % 16 == 0


def quantize_int4(model: nn.Module) -> nn.Module:
    from torchao.quantization import quantize_, int4_weight_only
    model.to(torch.bfloat16)
    quantize_(model, int4_weight_only(), filter_fn=_int4_filter)
    return model


def quantize_fp8(model: nn.Module) -> nn.Module:
    from torchao.quantization import quantize_, float8_weight_only
    model.to(torch.bfloat16)
    quantize_(model, float8_weight_only())
    return model


def load_base_model(ckpt_path: pathlib.Path) -> nn.Module:
    from models import build_sat_model
    cfg = load_config(input_size=1288)
    model, _ = build_sat_model(cfg, set_criterion=False)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, help="Input .pth (fp32 checkpoint)")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--modes",  default="int4,fp8", help="Comma-separated: int4,fp8")
    args = ap.parse_args()

    inp = pathlib.Path(args.input)
    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = [m.strip() for m in args.modes.split(",")]
    stem = inp.stem

    for mode in modes:
        print(f"\n[INFO] === mode={mode} ===")
        print("[INFO] Loading base model ...")
        model = load_base_model(inp)

        if mode == "int4":
            print("[INFO] Applying int4_weight_only (bfloat16 base) ...")
            model = quantize_int4(model)
            suffix = "_torchao_int4w"
        elif mode == "fp8":
            print("[INFO] Applying float8_weight_only (bfloat16 base) ...")
            model = quantize_fp8(model)
            suffix = "_torchao_fp8w"
        else:
            raise ValueError(f"Unknown mode: {mode}")

        out_path = out_dir / f"{stem}{suffix}.pth"
        print(f"[INFO] Saving → {out_path} ...")
        torch.save(model.state_dict(), out_path)
        size_mb = out_path.stat().st_size / 1024**2
        print(f"[INFO] Done. Size: {size_mb:.1f} MB")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

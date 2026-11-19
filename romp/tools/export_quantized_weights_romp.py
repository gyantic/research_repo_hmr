#!/usr/bin/env python3
"""
Pre-convert ROMP/BEV checkpoint weights to bf16.

モデル構造を構築せず state dict を直接変換するため、学習用依存ライブラリ不要。

Usage:
    python tools/export_quantized_weights_romp.py \
        --input  trained_models/ROMP_HRNet32_V1.pkl \
        --output trained_models/

    # ResNet-50 の場合も同様
    python tools/export_quantized_weights_romp.py \
        --input  trained_models/ROMP_ResNet50_V1.pkl \
        --output trained_models/

Output:
    trained_models/ROMP_HRNet32_V1_bf16.pth  (サイズ約50%削減)

    --format int8_raw を指定すると trained_models/<name>_int8_raw_weights.pth を出力する。

Loading:
    configs/video.yml で model_precision: 'bf16' を設定し、
    model_path を生成した _bf16.pth ファイルに指定する。
    _build_model_ が model.to(bfloat16) を呼ぶため、
    copy_state_dict が自動でキャストする。
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import torch

from checkpoint_io import file_size_mb, load_checkpoint, strip_module_prefix


def convert_state_dict_bf16(state_dict: dict) -> dict:
    """float32 / float16 テンソルをすべて bfloat16 に変換する。
    int / bool テンソルはそのまま保持する。"""
    out = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            out[k] = v.to(torch.bfloat16)
        else:
            out[k] = v
    return out


def convert_state_dict_int8_raw(state_dict: dict) -> collections.OrderedDict:
    """浮動小数テンソルを int8 に変換する（キー・順序はそのまま）。"""
    return collections.OrderedDict(
        (k, v.to(torch.int8) if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
        for k, v in state_dict.items()
    )


def load_state_dict(path: pathlib.Path) -> dict:
    """DataParallel prefix ('module.') を除去して state dict を返す。"""
    _raw, state_dict = load_checkpoint(path)
    return strip_module_prefix(state_dict)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pre-convert ROMP/BEV checkpoint weights to bfloat16."
    )
    ap.add_argument("--input",  required=True, help="Input checkpoint (.pkl / .pth)")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--format", default="bf16", choices=["bf16", "int8_raw"],
                    help="bf16 (default) or int8_raw")
    args = ap.parse_args()

    inp = pathlib.Path(args.input)
    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.format == "int8_raw":
        print(f"[INFO] Loading: {inp}  ({file_size_mb(inp):.1f} MB)")
        _raw, sd = load_checkpoint(inp)
        out_path = out_dir / f"{inp.stem}_int8_raw_weights.pth"
        torch.save(convert_state_dict_int8_raw(sd), out_path)
        print(f"[INFO] Saved: {out_path}  ({file_size_mb(out_path):.1f} MB)")
        return

    print(f"[INFO] Loading: {inp}  ({file_size_mb(inp):.1f} MB)")
    sd = load_state_dict(inp)

    n_total = len(sd)
    n_float = sum(1 for v in sd.values()
                  if isinstance(v, torch.Tensor) and v.is_floating_point())
    print(f"[INFO] Tensors: {n_total} total, {n_float} floating-point")

    print("[INFO] Converting to bfloat16 ...")
    sd_bf16 = convert_state_dict_bf16(sd)

    out_path = out_dir / f"{inp.stem}_bf16.pth"
    torch.save(sd_bf16, out_path)
    ratio = file_size_mb(out_path) / file_size_mb(inp) * 100
    print(f"[INFO] Saved: {out_path}  ({file_size_mb(out_path):.1f} MB, {ratio:.1f}% of original)")
    print("[INFO] Done.")
    print()
    print("使い方:")
    print(f"  configs/video.yml で以下を設定してください:")
    print(f"    model_path: {out_path}")
    print(f"    model_precision: 'bf16'")


if __name__ == "__main__":
    main()

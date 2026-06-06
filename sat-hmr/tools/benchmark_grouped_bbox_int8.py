#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.grouped_linear import (
    GroupedInt8LinearReference,
    GroupedLinearReference,
    loop_grouped_linear_forward,
)
from models.triton_grouped_int8 import GroupedInt8LinearTriton


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Benchmark grouped bbox_embed hidden-path FP16 vs INT8 references")
    p.add_argument("--weights", default="./weights/sat_hmr/sat_644.pth")
    p.add_argument("--input_size", type=int, default=1288)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype_cfg", default="float16", help="fp32, float16, or bfloat16")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_queries", type=int, default=None)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_csv", default="plots/grouped_bbox_hidden_int8_bench.csv")
    p.add_argument("--out_json", default="plots/grouped_bbox_hidden_int8_bench.json")
    return p.parse_args()


def resolve_runtime_dtype(dtype_cfg: str) -> torch.dtype:
    value = str(dtype_cfg).lower()
    if value in {"bfloat16", "bf16", "bfp16"}:
        return torch.bfloat16
    if value in {"float16", "fp16", "half"}:
        return torch.float16
    if value in {"float32", "fp32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype_cfg: {dtype_cfg}")


def load_config(input_size: int):
    demo_cfg_path = ROOT / "configs" / "run" / "demo.yaml"
    with demo_cfg_path.open("r", encoding="utf-8") as f:
        demo_config = yaml.safe_load(f)

    model_name = demo_config.get("model", "default")
    model_cfg_path = ROOT / "configs" / "models" / f"{model_name}.yaml"
    with model_cfg_path.open("r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)

    class Config:
        pass

    args = Config()
    for k, v in model_config.items():
        setattr(args, k, v)
    for k, v in demo_config.items():
        setattr(args, k, v)
    args.mode = "infer"
    args.pretrain = False
    args.pretrain_path = None
    args.input_size = input_size
    return args


def time_fn(fn, warmup: int, iters: int) -> Dict[str, float]:
    times: List[float] = []
    with torch.no_grad():
        for _ in range(warmup):
            _ = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }


def flatten_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1)


def hidden_loop(stage0: Sequence[torch.nn.Linear], stage1: Sequence[torch.nn.Linear], x: torch.Tensor) -> torch.Tensor:
    y = F.relu(loop_grouped_linear_forward(stage0, x))
    return F.relu(loop_grouped_linear_forward(stage1, y))


def hidden_grouped_fp16(stage0: GroupedLinearReference, stage1: GroupedLinearReference, x: torch.Tensor) -> torch.Tensor:
    y = F.relu(stage0(x))
    return F.relu(stage1(y))


def hidden_grouped_int8(stage0: GroupedInt8LinearReference, stage1: GroupedInt8LinearReference, x: torch.Tensor) -> torch.Tensor:
    y = F.relu(stage0(x))
    return F.relu(stage1(y))


def hidden_grouped_int8_triton(stage0: GroupedInt8LinearTriton, stage1: GroupedInt8LinearTriton, x: torch.Tensor) -> torch.Tensor:
    y = F.relu(stage0(x))
    return F.relu(stage1(y))


def hidden_grouped_int8_triton_fused_stage1_quant(
    stage0: GroupedInt8LinearTriton,
    stage1: GroupedInt8LinearTriton,
    x: torch.Tensor,
) -> torch.Tensor:
    batch_shape = tuple(int(v) for v in x.shape[1:-1])
    x_q, x_scale = stage0.quantize_activation(x)
    y_q, y_scale = stage0.forward_prequantized_relu_requant(x_q, x_scale, batch_shape=batch_shape)
    y_q = y_q.reshape(y_q.shape[0], -1, y_q.shape[-1])
    y_scale = y_scale.reshape(y_scale.shape[0], -1)
    y = stage1.forward_prequantized(y_q, y_scale, batch_shape=batch_shape, out_dtype=x.dtype)
    return F.relu(y)


def hidden_grouped_int8_triton_fused_stage0_input_and_stage1_quant(
    stage0: GroupedInt8LinearTriton,
    stage1: GroupedInt8LinearTriton,
    x: torch.Tensor,
) -> torch.Tensor:
    batch_shape = tuple(int(v) for v in x.shape[1:-1])
    y_q, y_scale = stage0.forward_fused_input_relu_requant(x, batch_shape=batch_shape)
    y_q = y_q.reshape(y_q.shape[0], -1, y_q.shape[-1])
    y_scale = y_scale.reshape(y_scale.shape[0], -1)
    y = stage1.forward_prequantized(y_q, y_scale, batch_shape=batch_shape, out_dtype=x.dtype)
    return F.relu(y)


def main() -> int:
    from models import build_sat_model

    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    runtime_dtype = resolve_runtime_dtype(args.dtype_cfg)

    out_csv = pathlib.Path(args.out_csv)
    out_json = pathlib.Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.input_size)
    model, _ = build_sat_model(cfg, set_criterion=False)
    state_dict = torch.load(args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    if runtime_dtype != torch.float32:
        model = model.to(runtime_dtype)
    model.eval()

    bbox_embed = list(model.bbox_embed)
    group_count = len(bbox_embed)
    hidden_dim = int(bbox_embed[0].layers[0].in_features)
    num_queries = int(args.num_queries if args.num_queries is not None else model.num_queries)

    stage0_linears = [bbox.layers[0] for bbox in bbox_embed]
    stage1_linears = [bbox.layers[1] for bbox in bbox_embed]

    grouped_stage0_fp16 = GroupedLinearReference.from_linears(stage0_linears, impl="bmm").to(
        device=device, dtype=runtime_dtype
    )
    grouped_stage1_fp16 = GroupedLinearReference.from_linears(stage1_linears, impl="bmm").to(
        device=device, dtype=runtime_dtype
    )
    grouped_stage0_int8 = GroupedInt8LinearReference.from_linears(stage0_linears).to(device=device)
    grouped_stage1_int8 = GroupedInt8LinearReference.from_linears(stage1_linears).to(device=device)
    grouped_stage0_int8_triton = None
    grouped_stage1_int8_triton = None
    use_triton = device.type == "cuda" and GroupedInt8LinearTriton.is_supported()
    if use_triton:
        grouped_stage0_int8_triton = GroupedInt8LinearTriton.from_linears(stage0_linears).to(device=device)
        grouped_stage1_int8_triton = GroupedInt8LinearTriton.from_linears(stage1_linears).to(device=device)

    x = torch.randn(group_count, args.batch_size, num_queries, hidden_dim, device=device, dtype=runtime_dtype)

    with torch.no_grad():
        y_loop = hidden_loop(stage0_linears, stage1_linears, x)
        y_grouped_fp16 = hidden_grouped_fp16(grouped_stage0_fp16, grouped_stage1_fp16, x)
        y_grouped_int8 = hidden_grouped_int8(grouped_stage0_int8, grouped_stage1_int8, x)
        y_grouped_int8_triton = (
            hidden_grouped_int8_triton(grouped_stage0_int8_triton, grouped_stage1_int8_triton, x)
            if use_triton
            else None
        )
        y_grouped_int8_triton_fused_stage1q = (
            hidden_grouped_int8_triton_fused_stage1_quant(
                grouped_stage0_int8_triton,
                grouped_stage1_int8_triton,
                x,
            )
            if use_triton
            else None
        )

    loop_flat = flatten_tensor(y_loop)
    grouped_fp16_flat = flatten_tensor(y_grouped_fp16)
    grouped_int8_flat = flatten_tensor(y_grouped_int8)
    denom = torch.clamp(loop_flat.abs(), min=1e-8)
    max_abs_grouped_fp16 = float((loop_flat - grouped_fp16_flat).abs().max().item())
    max_abs_grouped_int8 = float((loop_flat - grouped_int8_flat).abs().max().item())
    max_rel_grouped_fp16 = float(((loop_flat - grouped_fp16_flat).abs() / denom).max().item())
    max_rel_grouped_int8 = float(((loop_flat - grouped_int8_flat).abs() / denom).max().item())
    max_abs_grouped_int8_triton = None
    max_rel_grouped_int8_triton = None
    max_abs_grouped_int8_triton_fused_stage1q = None
    max_rel_grouped_int8_triton_fused_stage1q = None
    if y_grouped_int8_triton is not None:
        grouped_int8_triton_flat = flatten_tensor(y_grouped_int8_triton)
        max_abs_grouped_int8_triton = float((loop_flat - grouped_int8_triton_flat).abs().max().item())
        max_rel_grouped_int8_triton = float(((loop_flat - grouped_int8_triton_flat).abs() / denom).max().item())
    if y_grouped_int8_triton_fused_stage1q is not None:
        grouped_int8_triton_fused_stage1q_flat = flatten_tensor(y_grouped_int8_triton_fused_stage1q)
        max_abs_grouped_int8_triton_fused_stage1q = float(
            (loop_flat - grouped_int8_triton_fused_stage1q_flat).abs().max().item()
        )
        max_rel_grouped_int8_triton_fused_stage1q = float(
            ((loop_flat - grouped_int8_triton_fused_stage1q_flat).abs() / denom).max().item()
        )

    loop_stats = time_fn(lambda: hidden_loop(stage0_linears, stage1_linears, x), args.warmup, args.iters)
    grouped_fp16_stats = time_fn(
        lambda: hidden_grouped_fp16(grouped_stage0_fp16, grouped_stage1_fp16, x),
        args.warmup,
        args.iters,
    )
    grouped_int8_stats = time_fn(
        lambda: hidden_grouped_int8(grouped_stage0_int8, grouped_stage1_int8, x),
        args.warmup,
        args.iters,
    )
    grouped_int8_triton_stats = (
        time_fn(
            lambda: hidden_grouped_int8_triton(grouped_stage0_int8_triton, grouped_stage1_int8_triton, x),
            args.warmup,
            args.iters,
        )
        if use_triton
        else None
    )
    grouped_int8_triton_fused_stage1q_stats = (
        time_fn(
            lambda: hidden_grouped_int8_triton_fused_stage1_quant(
                grouped_stage0_int8_triton,
                grouped_stage1_int8_triton,
                x,
            ),
            args.warmup,
            args.iters,
        )
        if use_triton
        else None
    )
    grouped_int8_triton_fused_stage0in_stage1q_stats = (
        time_fn(
            lambda: hidden_grouped_int8_triton_fused_stage0_input_and_stage1_quant(
                grouped_stage0_int8_triton,
                grouped_stage1_int8_triton,
                x,
            ),
            args.warmup,
            args.iters,
        )
        if use_triton
        else None
    )
    triton_breakdown_rows = []
    if use_triton:
        with torch.no_grad():
            stage0_q_ref, stage0_scale_ref = grouped_stage0_int8_triton.quantize_activation(x)
            stage0_out_ref = grouped_stage0_int8_triton.forward_prequantized(
                stage0_q_ref,
                stage0_scale_ref,
                batch_shape=(args.batch_size, num_queries),
                out_dtype=runtime_dtype,
            )
            stage1_input_ref = F.relu(stage0_out_ref)
            stage1_q_ref, stage1_scale_ref = grouped_stage1_int8_triton.quantize_activation(stage1_input_ref)
            stage0_fused_q_ref, stage0_fused_scale_ref = grouped_stage0_int8_triton.forward_prequantized_relu_requant(
                stage0_q_ref,
                stage0_scale_ref,
                batch_shape=(args.batch_size, num_queries),
            )
            stage0_fused_q_ref = stage0_fused_q_ref.reshape(group_count, -1, stage0_fused_q_ref.shape[-1])
            stage0_fused_scale_ref = stage0_fused_scale_ref.reshape(group_count, -1)
            stage0_fused_input_q_ref, stage0_fused_input_scale_ref = (
                grouped_stage0_int8_triton.forward_fused_input_relu_requant(
                    x,
                    batch_shape=(args.batch_size, num_queries),
                )
            )
            stage0_fused_input_q_ref = stage0_fused_input_q_ref.reshape(
                group_count, -1, stage0_fused_input_q_ref.shape[-1]
            )
            stage0_fused_input_scale_ref = stage0_fused_input_scale_ref.reshape(group_count, -1)

        stage0_quant_stats = time_fn(
            lambda: grouped_stage0_int8_triton.quantize_activation(x),
            args.warmup,
            args.iters,
        )
        stage0_kernel_stats = time_fn(
            lambda: grouped_stage0_int8_triton.forward_prequantized(
                stage0_q_ref,
                stage0_scale_ref,
                batch_shape=(args.batch_size, num_queries),
                out_dtype=runtime_dtype,
            ),
            args.warmup,
            args.iters,
        )
        stage1_quant_stats = time_fn(
            lambda: grouped_stage1_int8_triton.quantize_activation(stage1_input_ref),
            args.warmup,
            args.iters,
        )
        stage1_kernel_stats = time_fn(
            lambda: grouped_stage1_int8_triton.forward_prequantized(
                stage1_q_ref,
                stage1_scale_ref,
                batch_shape=(args.batch_size, num_queries),
                out_dtype=runtime_dtype,
            ),
            args.warmup,
            args.iters,
        )
        stage0_fused_relu_requant_stats = time_fn(
            lambda: grouped_stage0_int8_triton.forward_prequantized_relu_requant(
                stage0_q_ref,
                stage0_scale_ref,
                batch_shape=(args.batch_size, num_queries),
            ),
            args.warmup,
            args.iters,
        )
        stage1_kernel_from_fused_stats = time_fn(
            lambda: grouped_stage1_int8_triton.forward_prequantized(
                stage0_fused_q_ref,
                stage0_fused_scale_ref,
                batch_shape=(args.batch_size, num_queries),
                out_dtype=runtime_dtype,
            ),
            args.warmup,
            args.iters,
        )
        stage0_fused_input_relu_requant_stats = time_fn(
            lambda: grouped_stage0_int8_triton.forward_fused_input_relu_requant(
                x,
                batch_shape=(args.batch_size, num_queries),
            ),
            args.warmup,
            args.iters,
        )
        stage1_kernel_from_fused_input_stats = time_fn(
            lambda: grouped_stage1_int8_triton.forward_prequantized(
                stage0_fused_input_q_ref,
                stage0_fused_input_scale_ref,
                batch_shape=(args.batch_size, num_queries),
                out_dtype=runtime_dtype,
            ),
            args.warmup,
            args.iters,
        )

        for impl_name, stats in [
            ("triton_stage0_quant_only", stage0_quant_stats),
            ("triton_stage0_kernel_only", stage0_kernel_stats),
            ("triton_stage1_quant_only", stage1_quant_stats),
            ("triton_stage1_kernel_only", stage1_kernel_stats),
            ("triton_stage0_fused_relu_requant", stage0_fused_relu_requant_stats),
            ("triton_stage1_kernel_from_fused", stage1_kernel_from_fused_stats),
            ("triton_stage0_fused_input_relu_requant", stage0_fused_input_relu_requant_stats),
            ("triton_stage1_kernel_from_fused_input", stage1_kernel_from_fused_input_stats),
        ]:
            triton_breakdown_rows.append(
                {
                    "impl": impl_name,
                    "group_count_per_stage": group_count,
                    "num_hidden_linears": group_count * 2,
                    "batch_size": args.batch_size,
                    "num_queries": num_queries,
                    "hidden_dim": hidden_dim,
                    "median_ms": stats["median_ms"],
                    "min_ms": stats["min_ms"],
                    "p95_ms": stats["p95_ms"],
                    "speedup_vs_fp16_loop": loop_stats["median_ms"] / stats["median_ms"],
                    "max_abs_diff": "",
                    "max_rel_diff": "",
                }
            )

    rows = [
        {
            "impl": "fp16_loop",
            "group_count_per_stage": group_count,
            "num_hidden_linears": group_count * 2,
            "batch_size": args.batch_size,
            "num_queries": num_queries,
            "hidden_dim": hidden_dim,
            "median_ms": loop_stats["median_ms"],
            "min_ms": loop_stats["min_ms"],
            "p95_ms": loop_stats["p95_ms"],
            "speedup_vs_fp16_loop": 1.0,
            "max_abs_diff": 0.0,
            "max_rel_diff": 0.0,
        },
        {
            "impl": "fp16_grouped_bmm",
            "group_count_per_stage": group_count,
            "num_hidden_linears": group_count * 2,
            "batch_size": args.batch_size,
            "num_queries": num_queries,
            "hidden_dim": hidden_dim,
            "median_ms": grouped_fp16_stats["median_ms"],
            "min_ms": grouped_fp16_stats["min_ms"],
            "p95_ms": grouped_fp16_stats["p95_ms"],
            "speedup_vs_fp16_loop": loop_stats["median_ms"] / grouped_fp16_stats["median_ms"],
            "max_abs_diff": max_abs_grouped_fp16,
            "max_rel_diff": max_rel_grouped_fp16,
        },
        {
            "impl": "grouped_int8_vmap",
            "group_count_per_stage": group_count,
            "num_hidden_linears": group_count * 2,
            "batch_size": args.batch_size,
            "num_queries": num_queries,
            "hidden_dim": hidden_dim,
            "median_ms": grouped_int8_stats["median_ms"],
            "min_ms": grouped_int8_stats["min_ms"],
            "p95_ms": grouped_int8_stats["p95_ms"],
            "speedup_vs_fp16_loop": loop_stats["median_ms"] / grouped_int8_stats["median_ms"],
            "max_abs_diff": max_abs_grouped_int8,
            "max_rel_diff": max_rel_grouped_int8,
        },
    ]
    if grouped_int8_triton_stats is not None:
        rows.append(
            {
                "impl": "grouped_int8_triton",
                "group_count_per_stage": group_count,
                "num_hidden_linears": group_count * 2,
                "batch_size": args.batch_size,
                "num_queries": num_queries,
                "hidden_dim": hidden_dim,
                "median_ms": grouped_int8_triton_stats["median_ms"],
                "min_ms": grouped_int8_triton_stats["min_ms"],
                "p95_ms": grouped_int8_triton_stats["p95_ms"],
                "speedup_vs_fp16_loop": loop_stats["median_ms"] / grouped_int8_triton_stats["median_ms"],
                "max_abs_diff": max_abs_grouped_int8_triton,
                "max_rel_diff": max_rel_grouped_int8_triton,
            }
        )
    if grouped_int8_triton_fused_stage1q_stats is not None:
        rows.append(
            {
                "impl": "grouped_int8_triton_fused_stage1q",
                "group_count_per_stage": group_count,
                "num_hidden_linears": group_count * 2,
                "batch_size": args.batch_size,
                "num_queries": num_queries,
                "hidden_dim": hidden_dim,
                "median_ms": grouped_int8_triton_fused_stage1q_stats["median_ms"],
                "min_ms": grouped_int8_triton_fused_stage1q_stats["min_ms"],
                "p95_ms": grouped_int8_triton_fused_stage1q_stats["p95_ms"],
                "speedup_vs_fp16_loop": loop_stats["median_ms"] / grouped_int8_triton_fused_stage1q_stats["median_ms"],
                "max_abs_diff": max_abs_grouped_int8_triton_fused_stage1q,
                "max_rel_diff": max_rel_grouped_int8_triton_fused_stage1q,
            }
        )
    if use_triton:
        with torch.no_grad():
            y_grouped_int8_triton_fused_stage0in_stage1q = hidden_grouped_int8_triton_fused_stage0_input_and_stage1_quant(
                grouped_stage0_int8_triton,
                grouped_stage1_int8_triton,
                x,
            )
        grouped_int8_triton_fused_stage0in_stage1q_flat = flatten_tensor(y_grouped_int8_triton_fused_stage0in_stage1q)
        max_abs_grouped_int8_triton_fused_stage0in_stage1q = float(
            (loop_flat - grouped_int8_triton_fused_stage0in_stage1q_flat).abs().max().item()
        )
        max_rel_grouped_int8_triton_fused_stage0in_stage1q = float(
            ((loop_flat - grouped_int8_triton_fused_stage0in_stage1q_flat).abs() / denom).max().item()
        )
        rows.append(
            {
                "impl": "grouped_int8_triton_fused_stage0in_stage1q",
                "group_count_per_stage": group_count,
                "num_hidden_linears": group_count * 2,
                "batch_size": args.batch_size,
                "num_queries": num_queries,
                "hidden_dim": hidden_dim,
                "median_ms": grouped_int8_triton_fused_stage0in_stage1q_stats["median_ms"],
                "min_ms": grouped_int8_triton_fused_stage0in_stage1q_stats["min_ms"],
                "p95_ms": grouped_int8_triton_fused_stage0in_stage1q_stats["p95_ms"],
                "speedup_vs_fp16_loop": loop_stats["median_ms"] / grouped_int8_triton_fused_stage0in_stage1q_stats["median_ms"],
                "max_abs_diff": max_abs_grouped_int8_triton_fused_stage0in_stage1q,
                "max_rel_diff": max_rel_grouped_int8_triton_fused_stage0in_stage1q,
            }
        )
    if grouped_int8_triton_stats is not None:
        rows.extend(triton_breakdown_rows)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "impl",
                "group_count_per_stage",
                "num_hidden_linears",
                "batch_size",
                "num_queries",
                "hidden_dim",
                "median_ms",
                "min_ms",
                "p95_ms",
                "speedup_vs_fp16_loop",
                "max_abs_diff",
                "max_rel_diff",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    out_json.write_text(
        json.dumps(
            {
                "device": str(device),
                "dtype": str(runtime_dtype),
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    for row in rows:
        max_abs_diff = row["max_abs_diff"]
        if isinstance(max_abs_diff, (int, float)):
            max_abs_str = f"{max_abs_diff:.3e}"
        else:
            max_abs_str = str(max_abs_diff)
        print(
            f"[RESULT] impl={row['impl']} median={row['median_ms']:.4f}ms "
            f"speedup_vs_fp16_loop={row['speedup_vs_fp16_loop']:.3f} "
            f"max_abs_diff={max_abs_str}"
        )
    print(f"[DONE] csv: {out_csv}")
    print(f"[DONE] json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
from typing import Dict, Iterable, List

import numpy as np
import torch
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Check grouped decoder query-side projections against the original loop")
    p.add_argument("--weights", default="./weights/sat_hmr/sat_644.pth")
    p.add_argument("--input_size", type=int, default=1288)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype_cfg", default="float16", help="fp32, float16, or bfloat16")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_queries", type=int, default=None)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_csv", default="plots/grouped_decoder_query_proj_check.csv")
    p.add_argument("--out_json", default="plots/grouped_decoder_query_proj_check.json")
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


def flatten_outputs(outputs: Iterable[object]) -> torch.Tensor:
    flat: List[torch.Tensor] = []
    for item in outputs:
        if item is None:
            continue
        if isinstance(item, torch.Tensor):
            flat.append(item.reshape(-1))
        elif isinstance(item, (list, tuple)):
            flat.append(flatten_outputs(item))
        else:
            raise TypeError(f"Unsupported output type: {type(item)}")
    return torch.cat(flat, dim=0)


def set_grouped_query_proj(layers, enabled: bool, target: str) -> None:
    for layer in layers:
        if target == "self":
            layer.use_grouped_self_query_proj = enabled
            layer.use_grouped_cross_query_proj = False
        elif target == "cross":
            layer.use_grouped_self_query_proj = False
            layer.use_grouped_cross_query_proj = enabled
        else:
            raise ValueError(f"Unsupported target: {target}")
        layer.grouped_query_proj_impl = "bmm"
        layer.rebuild_grouped_query_proj_refs()


def set_fused_int8_query_proj(layers, enabled: bool, target: str) -> None:
    for layer in layers:
        if target == "self":
            layer.use_fused_int8_self_query_proj = enabled
            layer.use_fused_input_int8_self_query_proj = False
        else:
            raise ValueError(f"Unsupported target: {target}")
        layer.rebuild_grouped_query_proj_refs()


def set_fused_input_int8_query_proj(layers, enabled: bool, target: str) -> None:
    for layer in layers:
        if target == "self":
            layer.use_fused_input_int8_self_query_proj = enabled
            layer.use_fused_int8_self_query_proj = False
        else:
            raise ValueError(f"Unsupported target: {target}")
        layer.rebuild_grouped_query_proj_refs()


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

    layers = list(model.decoder.decoder.layers)
    hidden_dim = int(model.hidden_dim)
    num_queries = int(args.num_queries if args.num_queries is not None else model.num_queries)
    seq_len = args.batch_size * num_queries

    tgt = torch.randn(seq_len, hidden_dim, device=device, dtype=runtime_dtype)
    query_pos = torch.randn_like(tgt)
    query_sine_embed = torch.randn_like(tgt)

    def run_self_all():
        return [
            torch.stack(layer.forward_self_query_proj(tgt, query_pos), dim=0)
            for layer in layers
        ]

    def run_cross_all():
        return [
            flatten_outputs(layer.forward_cross_query_proj(tgt, query_pos, query_sine_embed))
            for layer in layers
        ]

    results = []

    for case_name, fn in [
        ("decoder.self_query_pre", run_self_all),
        ("decoder.cross_query_pre", run_cross_all),
    ]:
        target = "self" if "self" in case_name else "cross"
        set_fused_input_int8_query_proj(layers, enabled=False, target="self")
        set_fused_int8_query_proj(layers, enabled=False, target="self")
        set_grouped_query_proj(layers, enabled=False, target=target)
        with torch.no_grad():
            y_loop = fn()
        set_grouped_query_proj(layers, enabled=True, target=target)
        with torch.no_grad():
            y_grouped = fn()

        loop_flat = flatten_outputs(y_loop)
        grouped_flat = flatten_outputs(y_grouped)
        max_abs = float((loop_flat - grouped_flat).abs().max().item())
        denom = torch.clamp(loop_flat.abs(), min=1e-8)
        max_rel = float(((loop_flat - grouped_flat).abs() / denom).max().item())

        set_grouped_query_proj(layers, enabled=False, target=target)
        loop_stats = time_fn(fn, args.warmup, args.iters)
        set_grouped_query_proj(layers, enabled=True, target=target)
        grouped_stats = time_fn(fn, args.warmup, args.iters)

        results.extend(
            [
                {
                    "case": case_name,
                    "impl": "loop",
                    "num_layers": len(layers),
                    "batch_size": args.batch_size,
                    "num_queries": num_queries,
                    "hidden_dim": hidden_dim,
                    "median_ms": loop_stats["median_ms"],
                    "min_ms": loop_stats["min_ms"],
                    "p95_ms": loop_stats["p95_ms"],
                    "speedup_vs_loop": 1.0,
                    "max_abs_diff": 0.0,
                    "max_rel_diff": 0.0,
                },
                {
                    "case": case_name,
                    "impl": "grouped_bmm",
                    "num_layers": len(layers),
                    "batch_size": args.batch_size,
                    "num_queries": num_queries,
                    "hidden_dim": hidden_dim,
                    "median_ms": grouped_stats["median_ms"],
                    "min_ms": grouped_stats["min_ms"],
                    "p95_ms": grouped_stats["p95_ms"],
                    "speedup_vs_loop": loop_stats["median_ms"] / grouped_stats["median_ms"],
                    "max_abs_diff": max_abs,
                    "max_rel_diff": max_rel,
                },
            ]
        )

        if target == "self" and device.type == "cuda":
            set_grouped_query_proj(layers, enabled=False, target="self")
            set_fused_int8_query_proj(layers, enabled=True, target="self")
            with torch.no_grad():
                y_fused_i8 = fn()
            fused_i8_flat = flatten_outputs(y_fused_i8)
            max_abs_i8 = float((loop_flat - fused_i8_flat).abs().max().item())
            max_rel_i8 = float(((loop_flat - fused_i8_flat).abs() / denom).max().item())
            fused_i8_stats = time_fn(fn, args.warmup, args.iters)
            results.append(
                {
                    "case": case_name,
                    "impl": "fused_int8_triton",
                    "num_layers": len(layers),
                    "batch_size": args.batch_size,
                    "num_queries": num_queries,
                    "hidden_dim": hidden_dim,
                    "median_ms": fused_i8_stats["median_ms"],
                    "min_ms": fused_i8_stats["min_ms"],
                    "p95_ms": fused_i8_stats["p95_ms"],
                    "speedup_vs_loop": loop_stats["median_ms"] / fused_i8_stats["median_ms"],
                    "max_abs_diff": max_abs_i8,
                    "max_rel_diff": max_rel_i8,
                }
            )
            set_fused_int8_query_proj(layers, enabled=False, target="self")

            set_fused_input_int8_query_proj(layers, enabled=True, target="self")
            with torch.no_grad():
                y_fused_input_i8 = fn()
            fused_input_i8_flat = flatten_outputs(y_fused_input_i8)
            max_abs_input_i8 = float((loop_flat - fused_input_i8_flat).abs().max().item())
            max_rel_input_i8 = float(((loop_flat - fused_input_i8_flat).abs() / denom).max().item())
            fused_input_i8_stats = time_fn(fn, args.warmup, args.iters)
            results.append(
                {
                    "case": case_name,
                    "impl": "fused_input_int8_triton",
                    "num_layers": len(layers),
                    "batch_size": args.batch_size,
                    "num_queries": num_queries,
                    "hidden_dim": hidden_dim,
                    "median_ms": fused_input_i8_stats["median_ms"],
                    "min_ms": fused_input_i8_stats["min_ms"],
                    "p95_ms": fused_input_i8_stats["p95_ms"],
                    "speedup_vs_loop": loop_stats["median_ms"] / fused_input_i8_stats["median_ms"],
                    "max_abs_diff": max_abs_input_i8,
                    "max_rel_diff": max_rel_input_i8,
                }
            )
            set_fused_input_int8_query_proj(layers, enabled=False, target="self")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "impl",
                "num_layers",
                "batch_size",
                "num_queries",
                "hidden_dim",
                "median_ms",
                "min_ms",
                "p95_ms",
                "speedup_vs_loop",
                "max_abs_diff",
                "max_rel_diff",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    out_json.write_text(
        json.dumps(
            {
                "device": str(device),
                "dtype": str(runtime_dtype),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    for row in results:
        print(
            f"[RESULT] case={row['case']} impl={row['impl']} median={row['median_ms']:.4f}ms "
            f"speedup_vs_loop={row['speedup_vs_loop']:.3f} max_abs_diff={row['max_abs_diff']:.3e}"
        )
    print(f"[DONE] csv: {out_csv}")
    print(f"[DONE] json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

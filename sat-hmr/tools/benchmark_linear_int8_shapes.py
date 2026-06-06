#!/usr/bin/env python3
"""
Microbenchmark representative SAT-HMR Linear shapes with fp16 nn.Linear
vs bitsandbytes Linear8bitLt.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import time
from typing import Dict, List

import numpy as np
import torch


DEFAULT_CASES = [
    {"name": "encoder.attn.qkv", "m": 582, "k": 768, "n": 2304},
    {"name": "encoder.attn.proj", "m": 582, "k": 768, "n": 768},
    {"name": "encoder.mlp.fc1", "m": 582, "k": 768, "n": 3072},
    {"name": "encoder.mlp.fc2", "m": 582, "k": 3072, "n": 768},
    {"name": "decoder.attn.proj", "m": 225, "k": 768, "n": 768},
    {"name": "decoder.ffn.fc1", "m": 225, "k": 768, "n": 2048},
    {"name": "decoder.ffn.fc2", "m": 225, "k": 2048, "n": 768},
    {"name": "decoder.bbox.hidden", "m": 225, "k": 768, "n": 768},
    {"name": "decoder.bbox.out", "m": 225, "k": 768, "n": 4},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Benchmark SAT-HMR Linear shapes for fp16 vs bnb int8")
    p.add_argument("--device", default="cuda")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--bnb_threshold", type=float, default=0.0)
    p.add_argument("--profile_case", default=None)
    p.add_argument("--profile_impl", choices=["fp16", "bnb", "both"], default="both")
    p.add_argument("--profile_iters", type=int, default=20)
    p.add_argument("--profile_topk", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_csv", default="plots/linear_int8_shape_bench.csv")
    p.add_argument("--out_json", default="plots/linear_int8_shape_bench.json")
    return p.parse_args()


def make_fp16_linear(k: int, n: int, device: torch.device) -> torch.nn.Linear:
    linear = torch.nn.Linear(k, n, bias=True)
    return linear.to(device=device, dtype=torch.float16)


def make_bnb_linear_from_fp16(
    fp16_linear: torch.nn.Linear,
    device: torch.device,
    threshold: float,
):
    import bitsandbytes.nn as bnb_nn

    linear = bnb_nn.Linear8bitLt(
        fp16_linear.in_features,
        fp16_linear.out_features,
        bias=fp16_linear.bias is not None,
        has_fp16_weights=False,
        threshold=threshold,
        device=None,
    )
    cpu_state_dict = {k: v.detach().cpu() for k, v in fp16_linear.state_dict().items()}
    linear.load_state_dict(cpu_state_dict)
    return linear.to(device=device)


def time_module(module: torch.nn.Module, x: torch.Tensor, warmup: int, iters: int) -> Dict[str, float]:
    times: List[float] = []
    module.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = module(x)
        torch.cuda.synchronize()
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = module(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

    return {
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }


def bnb_state_summary(module) -> Dict[str, object]:
    weight = module.weight
    state = module.state
    return {
        "weight_dtype": str(weight.data.dtype),
        "has_fp16_weights": bool(state.has_fp16_weights),
        "threshold": float(state.threshold),
        "weight_cb": getattr(weight, "CB", None) is not None,
        "state_cb": getattr(state, "CB", None) is not None,
        "state_scb": getattr(state, "SCB", None) is not None,
    }


def profile_module(
    module: torch.nn.Module,
    x: torch.Tensor,
    iters: int,
    topk: int,
) -> str:
    activities = [torch.profiler.ProfilerActivity.CPU]
    sort_key = "self_cpu_time_total"
    if x.is_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        sort_key = "self_cuda_time_total"

    module.eval()
    with torch.no_grad():
        with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
            for _ in range(iters):
                _ = module(x)
                if x.is_cuda:
                    torch.cuda.synchronize()
    return prof.key_averages().table(sort_by=sort_key, row_limit=topk)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    out_csv = pathlib.Path(args.out_csv)
    out_json = pathlib.Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for case in DEFAULT_CASES:
        m = case["m"]
        k = case["k"]
        n = case["n"]
        name = case["name"]

        x = torch.randn(m, k, device=device, dtype=torch.float16)
        fp16_linear = make_fp16_linear(k, n, device)
        bnb_linear = make_bnb_linear_from_fp16(fp16_linear, device, args.bnb_threshold)

        fp16_stats = time_module(fp16_linear, x, warmup=args.warmup, iters=args.iters)
        bnb_stats = time_module(bnb_linear, x, warmup=args.warmup, iters=args.iters)
        bnb_state = bnb_state_summary(bnb_linear)

        row = {
            "case": name,
            "m": m,
            "k": k,
            "n": n,
            "fp16_median_ms": fp16_stats["median_ms"],
            "fp16_min_ms": fp16_stats["min_ms"],
            "fp16_p95_ms": fp16_stats["p95_ms"],
            "bnb_median_ms": bnb_stats["median_ms"],
            "bnb_min_ms": bnb_stats["min_ms"],
            "bnb_p95_ms": bnb_stats["p95_ms"],
            "bnb_vs_fp16_ratio": (
                bnb_stats["median_ms"] / fp16_stats["median_ms"]
                if fp16_stats["median_ms"] > 0
                else 0.0
            ),
            "bnb_weight_dtype": bnb_state["weight_dtype"],
            "bnb_has_fp16_weights": bnb_state["has_fp16_weights"],
            "bnb_weight_cb": bnb_state["weight_cb"],
            "bnb_state_cb": bnb_state["state_cb"],
            "bnb_state_scb": bnb_state["state_scb"],
            "bnb_threshold": args.bnb_threshold,
        }
        results.append(row)
        print(
            f"[CASE] {name} m={m} k={k} n={n} "
            f"fp16_med={row['fp16_median_ms']:.4f}ms "
            f"bnb_med={row['bnb_median_ms']:.4f}ms "
            f"ratio={row['bnb_vs_fp16_ratio']:.3f} "
            f"bnb_state_cb={row['bnb_state_cb']}"
        )

        if args.profile_case == name:
            profile_targets = []
            if args.profile_impl in ("fp16", "both"):
                profile_targets.append(("fp16", fp16_linear))
            if args.profile_impl in ("bnb", "both"):
                profile_targets.append(("bnb", bnb_linear))

            for impl_name, module in profile_targets:
                table = profile_module(module, x, args.profile_iters, args.profile_topk)
                profile_path = out_csv.parent / (
                    f"{name.replace('.', '_')}_{impl_name}_profile.txt"
                )
                profile_path.write_text(table + "\n", encoding="utf-8")
                print(f"[PROFILE] {impl_name} -> {profile_path}")

    fields = [
        "case",
        "m",
        "k",
        "n",
        "fp16_median_ms",
        "fp16_min_ms",
        "fp16_p95_ms",
        "bnb_median_ms",
        "bnb_min_ms",
        "bnb_p95_ms",
        "bnb_vs_fp16_ratio",
        "bnb_weight_dtype",
        "bnb_has_fp16_weights",
        "bnb_weight_cb",
        "bnb_state_cb",
        "bnb_state_scb",
        "bnb_threshold",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "device": str(device),
                "warmup": args.warmup,
                "iters": args.iters,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[DONE] csv: {out_csv}")
    print(f"[DONE] json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

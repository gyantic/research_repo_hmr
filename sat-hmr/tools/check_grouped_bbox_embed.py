#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.grouped_bbox_embed import GroupedBBoxEmbedReference, loop_bbox_embed_forward


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Check grouped decoder.bbox_embed against the original loop")
    p.add_argument("--weights", default="./weights/sat_hmr/sat_644.pth")
    p.add_argument("--input_size", type=int, default=1288)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype_cfg", default="float16", help="fp32, float16, or bfloat16")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_queries", type=int, default=None)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_csv", default="plots/grouped_bbox_embed_check.csv")
    p.add_argument("--out_json", default="plots/grouped_bbox_embed_check.json")
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

    x = torch.randn(group_count, args.batch_size, num_queries, hidden_dim, device=device, dtype=runtime_dtype)

    grouped_bmm = GroupedBBoxEmbedReference.from_bbox_embed(bbox_embed, impl="bmm").to(device=device, dtype=runtime_dtype)
    grouped_einsum = GroupedBBoxEmbedReference.from_bbox_embed(bbox_embed, impl="einsum").to(device=device, dtype=runtime_dtype)

    with torch.no_grad():
        y_loop = loop_bbox_embed_forward(bbox_embed, x)
        y_bmm = grouped_bmm(x)
        y_einsum = grouped_einsum(x)

    max_abs_bmm = float((y_loop - y_bmm).abs().max().item())
    max_abs_einsum = float((y_loop - y_einsum).abs().max().item())
    denom_loop = torch.clamp(y_loop.abs(), min=1e-8)
    max_rel_bmm = float(((y_loop - y_bmm).abs() / denom_loop).max().item())
    max_rel_einsum = float(((y_loop - y_einsum).abs() / denom_loop).max().item())

    loop_stats = time_fn(lambda: loop_bbox_embed_forward(bbox_embed, x), args.warmup, args.iters)
    bmm_stats = time_fn(lambda: grouped_bmm(x), args.warmup, args.iters)
    einsum_stats = time_fn(lambda: grouped_einsum(x), args.warmup, args.iters)

    rows = [
        {
            "impl": "loop",
            "group_count": group_count,
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
            "impl": "grouped_bmm",
            "group_count": group_count,
            "batch_size": args.batch_size,
            "num_queries": num_queries,
            "hidden_dim": hidden_dim,
            "median_ms": bmm_stats["median_ms"],
            "min_ms": bmm_stats["min_ms"],
            "p95_ms": bmm_stats["p95_ms"],
            "speedup_vs_loop": loop_stats["median_ms"] / bmm_stats["median_ms"],
            "max_abs_diff": max_abs_bmm,
            "max_rel_diff": max_rel_bmm,
        },
        {
            "impl": "grouped_einsum",
            "group_count": group_count,
            "batch_size": args.batch_size,
            "num_queries": num_queries,
            "hidden_dim": hidden_dim,
            "median_ms": einsum_stats["median_ms"],
            "min_ms": einsum_stats["min_ms"],
            "p95_ms": einsum_stats["p95_ms"],
            "speedup_vs_loop": loop_stats["median_ms"] / einsum_stats["median_ms"],
            "max_abs_diff": max_abs_einsum,
            "max_rel_diff": max_rel_einsum,
        },
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "impl",
                "group_count",
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
        print(
            f"[RESULT] impl={row['impl']} median={row['median_ms']:.4f}ms "
            f"speedup_vs_loop={row['speedup_vs_loop']:.3f} "
            f"max_abs_diff={row['max_abs_diff']:.3e}"
        )
    print(f"[DONE] csv: {out_csv}")
    print(f"[DONE] json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""計測・評価の出力（各 run.sh の出力先）から、論文の図の入力 paper_results.csv を作る。

速度は生データ（raw_*.csv）を summarize_speed.py と同じ計算で集計し、精度は各モデルの評価出力
（accuracy_*）から読む。固定値は原論文の値（paper_*）だけ。

    python measurements/make_paper_results.py results/measurements/rerun_<日時> [--output paper_results.csv]

複数の出力先を渡すと、同じ名前のファイルは後に渡した方を使う。見つからないファイルは警告を出し、
速度の生データが無い行は書かない。
"""

import argparse
import csv
import json
from pathlib import Path

from summarize_speed import summarize

REPO_ROOT = Path(__file__).resolve().parents[1]

# 原論文の値 [mm]（図2の★）
ROMP_HRNET_PAPER = {"paper_MPJPE": 85.5, "paper_PA-MPJPE": 53.3, "paper_PVE": 103.1}    # ROMP Table 2, HRNet-32
ROMP_RESNET_PAPER = {"paper_MPJPE": 89.3, "paper_PA-MPJPE": 53.5, "paper_PVE": 105.6}   # ROMP Table 2, ResNet-50
SATHMR_PAPER = {"paper_PA-MPJPE": 41.6}                                                 # SAT-HMR Table 2, 644*
FASTSAM_PAPER = {"paper_MPJPE": 59.5, "paper_PA-MPJPE": 30.4, "paper_PVE": 68.9}       # Fast SAM 3D Body Table 1

# (method, backbone, dtype, quantization_mode, compile, runtime_variant, UDP, 速度の生データ, 精度の出力, 原論文の値)
CONFIGS = [
    ("ROMP", "HRNet32", "fp32", "none", True, "torch_compile", False, "raw_romp_fp32.csv", "accuracy_romp_hrnet_fp32.json", ROMP_HRNET_PAPER),
    ("ROMP", "HRNet32", "fp32", "none", False, "eager", False, "raw_romp_hrnet_fp32_no_compile.csv", "accuracy_romp_hrnet_fp32.json", ROMP_HRNET_PAPER),
    ("ROMP", "HRNet32", "bf16", "none", True, "torch_compile", False, "raw_romp_hrnet_bf16.csv", "accuracy_romp_hrnet_bf16.json", {}),
    ("ROMP", "HRNet32", "bf16", "none", False, "eager", False, "raw_romp_hrnet_bf16_no_compile.csv", "accuracy_romp_hrnet_bf16.json", {}),
    ("ROMP", "ResNet50", "fp32", "none", True, "torch_compile", False, "raw_romp_resnet_fp32.csv", "accuracy_romp_resnet_fp32.json", ROMP_RESNET_PAPER),
    ("ROMP", "ResNet50", "fp32", "none", False, "eager", False, "raw_romp_resnet_fp32_no_compile.csv", "accuracy_romp_resnet_fp32.json", ROMP_RESNET_PAPER),
    ("ROMP", "ResNet50", "bf16", "none", True, "torch_compile", False, "raw_romp_resnet_bf16.csv", "accuracy_romp_resnet_bf16.json", {}),
    ("ROMP", "ResNet50", "bf16", "none", False, "eager", False, "raw_romp_resnet_bf16_no_compile.csv", "accuracy_romp_resnet_bf16.json", {}),
    ("FastSAM3DBody", "", "fp32", "none", True, "eager_true_fp32", False, "raw_fastsam_fp32_compile.csv", "accuracy_fastsam_fp32", {}),
    ("FastSAM3DBody", "", "fp32", "none", False, "eager_true_fp32", False, "raw_fastsam_fp32_no_compile.csv", "accuracy_fastsam_fp32", FASTSAM_PAPER),
    ("FastSAM3DBody", "", "bf16", "none", True, "eager_autocast", False, "raw_fastsam_bf16_compile.csv", "accuracy_fastsam_bf16", {}),
    ("FastSAM3DBody", "", "bf16", "none", False, "eager_autocast", False, "raw_fastsam_bf16_no_compile.csv", "accuracy_fastsam_bf16", {}),
    ("FastSAM3DBody", "", "fp32", "int8_trt", True, "tensorrt_int8_backbone_fov", False, "raw_fastsam_int8_compile.csv", "accuracy_fastsam_int8", {}),
    ("FastSAM3DBody", "", "fp32", "int8_trt", False, "tensorrt_int8_backbone_fov", False, "raw_fastsam_int8_no_compile.csv", "accuracy_fastsam_int8", {}),
    ("SAT-HMR", "", "fp32", "none", True, "torch_compile_encoder_decoder_head", False, "raw_sathmr_fp32_compile_encoder_decoder_heads.csv", "accuracy_sathmr_fp32.txt", SATHMR_PAPER),
    ("SAT-HMR", "", "fp32", "none", False, "eager", False, "raw_sathmr_fp32.csv", "accuracy_sathmr_fp32.txt", SATHMR_PAPER),
    ("SAT-HMR", "", "bf16", "none", False, "eager", False, "raw_sathmr_bf16.csv", "accuracy_sathmr_bf16.txt", {}),
    ("SAT-HMR", "", "bf16", "none", True, "torch_compile_encoder_decoder_heads", False, "raw_sathmr_bf16_compile_encoder_decoder_heads.csv", "accuracy_sathmr_bf16.txt", {}),
    ("SAT-HMR", "", "bf16", "int8_dynamic_no_int8_attn", False, "eager_no_int8_attn", False, "raw_sathmr_int8w_dynamic_eager_no_int8_attn.csv", "accuracy_sathmr_int8.txt", {}),
    ("SAT-HMR", "", "bf16", "int8_dynamic_no_int8_attn", True, "torch_compile_encoder_only", False, "raw_sathmr_int8w_dynamic_compile_encoder_only_no_int8_attn.csv", "accuracy_sathmr_int8.txt", {}),
    ("SAT-HMR", "", "bf16", "int8_dynamic_no_int8_attn", True, "torch_compile_encoder_decoder", False, "raw_sathmr_int8w_dynamic_compile_encoder_decoder_no_int8_attn.csv", "accuracy_sathmr_int8.txt", {}),
    ("SAT-HMR", "", "bf16", "int8_dynamic_no_int8_attn", True, "torch_compile_quantized_linear", False, "raw_sathmr_int8w_dynamic_compile_no_int8_attn.csv", "accuracy_sathmr_int8.txt", {}),
    ("SAT-HMR", "", "bf16", "int8_dynamic_no_int8_attn", True, "torch_compile_quantized_linear", True, "raw_sathmr_udp_int8w_dynamic_compile_no_int8_attn.csv", "accuracy_sathmr_int8.txt", {}),
    ("ROMP", "HRNet32", "bf16", "none", True, "torch_compile", True, "raw_romp_hrnet_bf16_udp_ue.csv", "accuracy_romp_hrnet_bf16.json", {}),
    ("FastSAM3DBody", "", "fp32", "int8_trt", True, "tensorrt_int8_backbone_fov", True, "raw_fastsam3dbody_int8_trt_udp_ue_frames.csv", "accuracy_fastsam_int8", {}),
]

# run_publisher_ue.py の計測CSVは計測開始後のフレームだけを含むため、先頭を除かない
NO_SKIP_RAW = {"raw_fastsam3dbody_int8_trt_udp_ue_frames.csv"}
SKIP_ROWS = 10

COLUMNS = ["method", "backbone", "dtype", "quantization_mode", "compile", "runtime_variant",
           "robust_infer_ms_median", "robust_infer_ms_mad", "MPJPE", "PA-MPJPE", "PVE", "PA-PVE",
           "status", "paper_MPJPE", "paper_PA-MPJPE", "paper_PVE", "communication_protocol"]


def read_romp_accuracy(path):
    """measurements/romp/evaluate_3dpw.py の JSON（キーは '<データセット>-<指標>'）。"""
    results = json.loads(path.read_text())
    by_metric = {name.split("-", 1)[1]: value for name, value in results.items()}
    return {"MPJPE": by_metric["H36M14_MPJPE"], "PA-MPJPE": by_metric["H36M14_PA"],
            "PVE": by_metric["PVE"], "PA-PVE": by_metric["PA_PVE"]}


def read_sathmr_accuracy(path):
    """SAT-HMR の main.py --mode eval が書く results.txt（'名前: 値' の行）。"""
    results = dict(line.split(": ", 1) for line in path.read_text().splitlines() if ": " in line)
    return {"MPJPE": float(results["MPJPE"]), "PA-MPJPE": float(results["PA-MPJPE"]),
            "PVE": float(results["MVE"]), "PA-PVE": float(results["PA-MVE"])}


def read_fastsam_accuracy(path):
    """measurements/fastsam/evaluate_3dpw.py の JSON（本論文の評価 h36m14_global_orient）。"""
    results = json.loads((path / "metrics_h36m14_global_orient.json").read_text())
    return {"MPJPE": results["MPJPE"]["mean"], "PA-MPJPE": results["PA-MPJPE"]["mean"],
            "PVE": results["PVE"]["mean"], "PA-PVE": results["MPVAE"]["mean"]}


def read_accuracy(path):
    if path.suffix == ".json":
        return read_romp_accuracy(path)
    if path.suffix == ".txt":
        return read_sathmr_accuracy(path)
    return read_fastsam_accuracy(path)


def find(run_dirs, name):
    """後に渡した出力先を優先して探す。"""
    for run_dir in reversed(run_dirs):
        if (run_dir / name).exists():
            return run_dir / name
    print(f"[WARN] not found: {name}")
    return None


def make_row(run_dirs, config):
    method, backbone, dtype, quantization, compiled, variant, udp, raw_name, accuracy_name, paper = config
    raw_path = find(run_dirs, raw_name)
    if raw_path is None:
        return None
    speed = summarize(raw_path, 0 if raw_name in NO_SKIP_RAW else SKIP_ROWS)
    row = {
        "method": method, "backbone": backbone, "dtype": dtype, "quantization_mode": quantization,
        "compile": "TRUE" if compiled else "FALSE", "runtime_variant": variant,
        "robust_infer_ms_median": round(speed["inference_ms_median"], 2),
        "robust_infer_ms_mad": round(speed["inference_ms_mad"], 2),
        "status": "ok", "communication_protocol": "UDP" if udp else "",
        **paper,
    }
    accuracy_path = find(run_dirs, accuracy_name)
    if accuracy_path is not None:
        row.update({metric: round(value, 2) for metric, value in read_accuracy(accuracy_path).items()})
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", nargs="+", type=Path, help="run.sh の出力先")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "paper_results.csv")
    args = parser.parse_args()

    rows = [row for row in (make_row(args.run_dirs, config) for config in CONFIGS) if row is not None]
    with open(args.output, "w", newline="", encoding="cp932") as fp:
        writer = csv.DictWriter(fp, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

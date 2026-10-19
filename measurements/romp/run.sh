#!/bin/bash
# ROMP の論文用計測を実行する。romp_3.9 環境を有効にして、リポジトリのルートから実行する。
#
#   bash measurements/romp/run.sh speed     # Table 1・図2: HRNet32/ResNet50 × fp32/bf16 × compile あり/なし
#   bash measurements/romp/run.sh accuracy  # 図2: 3DPW の精度評価（HRNet32/ResNet50 × fp32/bf16）
#   bash measurements/romp/run.sh udp       # 図5: HRNet32 bf16 compile で UE へ UDP 送信しながら計測（UE を先に起動）
#
# 生データは results/measurements/rerun_<日時>/ に書く（既存の生データは上書きしない）。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
VIDEO="${VIDEO:-$REPO_ROOT/datasets/videos/sample_video.mp4}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/results/measurements/rerun_$(date +%Y%m%d_%H%M%S)}"
CONFIG_DIR="$REPO_ROOT/measurements/romp/configs"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
cd "$REPO_ROOT/romp"
export PYTHONPATH="$PWD:$PWD/romp"

run_demo() {  # <設定YAML> <dtype_cfg> <出力CSV名>
    "$PYTHON" -u romp_video_demo.py -i "$VIDEO" --configs_yml "$CONFIG_DIR/$1" --dtype_cfg "$2" \
        --no_render --no_save --metrics_csv "$OUT_DIR/$3" --warmup_frames 10
}

case "${1:-}" in
    speed)
        run_demo romp_video_fp32.yml                  fp32 raw_romp_fp32.csv
        run_demo romp_video_fp32_no_compile.yml       fp32 raw_romp_hrnet_fp32_no_compile.csv
        run_demo romp_video_bf16.yml                  bf16 raw_romp_hrnet_bf16.csv
        run_demo romp_video_bf16_no_compile.yml       bf16 raw_romp_hrnet_bf16_no_compile.csv
        run_demo romp_resnet_video_fp32.yml           fp32 raw_romp_resnet_fp32.csv
        run_demo romp_resnet_video_fp32_no_compile.yml fp32 raw_romp_resnet_fp32_no_compile.csv
        run_demo romp_resnet_video_bf16.yml           bf16 raw_romp_resnet_bf16.csv
        run_demo romp_resnet_video_bf16_no_compile.yml bf16 raw_romp_resnet_bf16_no_compile.csv
        ;;
    accuracy)
        for config in hrnet_fp32 hrnet_bf16 resnet_fp32 resnet_bf16; do
            "$PYTHON" -u "$REPO_ROOT/measurements/romp/evaluate_3dpw.py" --configs_yml "configs/h36m14_$config.yml" \
                --output_json "$OUT_DIR/accuracy_romp_$config.json"
        done
        ;;
    udp)
        "$PYTHON" -u "$REPO_ROOT/measurements/romp/benchmark_romp_udp_ue.py" --video "$VIDEO" \
            --configs_yml "$REPO_ROOT/romp/configs/video.yml" --stats_csv "$OUT_DIR/raw_romp_hrnet_bf16_udp_ue.csv"
        ;;
    *)
        echo "usage: bash measurements/romp/run.sh {speed|accuracy|udp}" >&2
        exit 2
        ;;
esac
echo "raw data: $OUT_DIR"

#!/bin/bash
# SAT-HMR の論文用計測を実行する。SAT-HMR の環境（sathmr-copy）を有効にして、リポジトリのルートから実行する。
#
#   bash measurements/sathmr/run.sh speed          # Table 1・図2: fp32 / bf16 / INT8 の compile あり/なし
#   bash measurements/sathmr/run.sh compile_scope  # 図3: INT8 で compile 範囲を encoder のみ・encoder+decoder にする（残りの2段階は speed）
#   bash measurements/sathmr/run.sh accuracy       # 図2: 3DPW の精度評価（fp32 / bf16 / INT8）
#   bash measurements/sathmr/run.sh udp            # 図5: INT8 compile で UE へ UDP 送信しながら計測（UE を先に起動）
#
# 生データは results/measurements/rerun_<日時>/ に書く（既存の生データは上書きしない）。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
VIDEO="${VIDEO:-$REPO_ROOT/datasets/videos/sample_video.mp4}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/results/measurements/rerun_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
cd "$REPO_ROOT/sat-hmr"

INT8=(--weights ./weights/sat_hmr/sat_644_int8w.pth --dtype_cfg bfloat16 --ptq_mode int8_dynamic --no_int8_attn)
COMPILE_ALL=(--compile --compile_encoder_only --compile_decoder_only --compile_heads --compile_dynamic)

run_demo() {  # <出力CSV名> <video_demo.py の引数...>
    local csv="$1"; shift
    "$PYTHON" -u video_demo.py -i "$VIDEO" --no_render --no_save --metrics_csv "$OUT_DIR/$csv" --warmup_frames 10 "$@"
}

evaluate() {  # <configs/run の設定名> <出力名>
    local results="results/$1/evaluation/3dpw_test/thresh_0.3/results.txt"
    rm -f "$results"
    "$PYTHON" -u main.py --mode eval --cfg "$1"
    cp "$results" "$OUT_DIR/accuracy_sathmr_$2.txt"
}

case "${1:-}" in
    speed)
        run_demo raw_sathmr_fp32.csv --weights ./weights/sat_hmr/sat_644.pth --dtype_cfg fp32
        run_demo raw_sathmr_fp32_compile_encoder_decoder_heads.csv \
            --weights ./weights/sat_hmr/sat_644.pth --dtype_cfg fp32 "${COMPILE_ALL[@]}"
        run_demo raw_sathmr_bf16.csv --weights ./weights/sat_hmr/sat_644_bf16.pth --dtype_cfg bf16
        run_demo raw_sathmr_bf16_compile_encoder_decoder_heads.csv \
            --weights ./weights/sat_hmr/sat_644_bf16.pth --dtype_cfg bfloat16 "${COMPILE_ALL[@]}"
        run_demo raw_sathmr_int8w_dynamic_eager_no_int8_attn.csv "${INT8[@]}"
        run_demo raw_sathmr_int8w_dynamic_compile_no_int8_attn.csv "${INT8[@]}" "${COMPILE_ALL[@]}" --compile_quantized_linear
        ;;
    compile_scope)
        run_demo raw_sathmr_int8w_dynamic_compile_encoder_only_no_int8_attn.csv "${INT8[@]}" \
            --compile --compile_encoder_only --compile_dynamic --compile_quantized_linear
        run_demo raw_sathmr_int8w_dynamic_compile_encoder_decoder_no_int8_attn.csv "${INT8[@]}" \
            --compile --compile_encoder_only --compile_decoder_only --compile_dynamic --compile_quantized_linear
        ;;
    accuracy)
        evaluate eval_3dpw fp32
        evaluate eval_3dpw_bf16 bf16
        evaluate eval_3dpw_int8w_dynamic int8
        ;;
    udp)
        run_demo raw_sathmr_udp_int8w_dynamic_compile_no_int8_attn.csv "${INT8[@]}" "${COMPILE_ALL[@]}" \
            --compile_quantized_linear --udp --udp_only --local_port 5555 --ue_port 62004
        ;;
    *)
        echo "usage: bash measurements/sathmr/run.sh {speed|compile_scope|accuracy|udp}" >&2
        exit 2
        ;;
esac
echo "raw data: $OUT_DIR"

#!/bin/bash
# Fast-SAM-3D-Body の論文用計測を実行する。sam_3d_body 環境を有効にして、リポジトリのルートから実行する。
#
#   bash measurements/fastsam/run.sh speed     # Table 1・図2: fp32 / bf16 / INT8 × compile あり/なし
#   bash measurements/fastsam/run.sh accuracy  # 図2・脚注1: 3DPW 推論（step0）と精度評価（evaluate_3dpw.py）
#   bash measurements/fastsam/run.sh udp       # 図5: INT8 で UE へ UDP 送信しながら計測（UE を先に起動）
#
# 生データは results/measurements/rerun_<日時>/ に書く（既存の生データは上書きしない）。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
VIDEO="${VIDEO:-$REPO_ROOT/datasets/videos/sample_video.mp4}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/results/measurements/rerun_$(date +%Y%m%d_%H%M%S)}"
MEASUREMENTS="$REPO_ROOT/measurements/fastsam"
CHECKPOINT="$REPO_ROOT/fast-sam-3d-body/checkpoints/sam-3d-body-dinov3"
FP32_CHECKPOINT="$REPO_ROOT/fast-sam-3d-body/checkpoints/fastsam3dbody_fp32"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
cd "$REPO_ROOT/fast-sam-3d-body"

# fp32 構成は、元のチェックポイントの設定だけを TRAIN.USE_FP16=false にしたもので計測した。
make_fp32_checkpoint() {
    [ -f "$FP32_CHECKPOINT/model_config.yaml" ] && return
    mkdir -p "$FP32_CHECKPOINT"
    ln -sfn "$CHECKPOINT/model.ckpt" "$FP32_CHECKPOINT/model.ckpt"
    ln -sfn "$CHECKPOINT/assets" "$FP32_CHECKPOINT/assets"
    sed 's/^  USE_FP16: true$/  USE_FP16: false/' "$CHECKPOINT/model_config.yaml" > "$FP32_CHECKPOINT/model_config.yaml"
}

# step0 推論の共通環境変数（3DPW 推論・UDP 計測）
COMMON_ENV=(SAM3D_QUIET=1 TORCHINDUCTOR_CUDAGRAPH_TREES=0 GPU_HAND_PREP=1 SKIP_KEYPOINT_PROMPT=1
    KEYPOINT_PROMPT_INTERM_INTERVAL=999 MHR_NO_CORRECTIVES=1 FOV_FAST=1 FOV_MODEL=s FOV_LEVEL=0
    BODY_INTERM_PRED_LAYERS=0,1,2 HAND_INTERM_PRED_LAYERS=0,1)
EAGER_ENV=(MHR_USE_CUDA_GRAPH=0 USE_COMPILE=0 USE_COMPILE_BACKBONE=0 DECODER_COMPILE=0 INTERM_COMPILE=0 USE_TRT_BACKBONE=0 FOV_TRT=0)
INT8_ENV=(IMG_SIZE=512 LAYER_DTYPE=fp32 MHR_USE_CUDA_GRAPH=1 USE_COMPILE=1 USE_COMPILE_BACKBONE=1 DECODER_COMPILE=1
    COMPILE_MODE=reduce-overhead FOV_TRT=1 FOV_TRT_PATH="$REPO_ROOT/fast-sam-3d-body/checkpoints/moge_trt/moge_dinov2_encoder_int8.engine"
    USE_TRT_BACKBONE=1 TRT_BACKBONE_PATH="$CHECKPOINT/backbone_trt/backbone_dinov3_int8.engine")

benchmark() {  # <出力CSV名> <チェックポイント> <dtype_variant> [--no_compile]
    local csv="$1" checkpoint="$2" variant="$3"; shift 3
    "$PYTHON" -u "$MEASUREMENTS/benchmark_fastsam3dbody.py" --video "$VIDEO" --dtype_variant "$variant" "$@" \
        --warmup_frames 10 --image_size 512 --checkpoint_path "$checkpoint/model.ckpt" --mhr_path "$checkpoint/assets/mhr_model.pt" \
        --output_csv "$OUT_DIR/$csv"
}

infer_and_evaluate() {  # <名前> <チェックポイント> <環境変数...>
    local name="$1" checkpoint="$2"; shift 2
    local work="$OUT_DIR/accuracy_fastsam_$name"
    env "${COMMON_ENV[@]}" "$@" "$PYTHON" -u mhr2smpl/multi_view/step0_infer_3dpw.py \
        --data_dir "$REPO_ROOT/datasets/3dpw" --output_dir "$work" --split test --frame_step 10 --max_samples 2000 \
        --local_checkpoint_path "$checkpoint" --local_mhr_path "$checkpoint/assets/mhr_model.pt" \
        --detector yolo_pose --detector_model checkpoints/yolo/yolo11m-pose.engine --hand_box_source yolo_pose --context_factor 1.5
    for protocol in h36m14_global_orient paper_3dpw14; do
        "$PYTHON" -u "$MEASUREMENTS/evaluate_3dpw.py" --samples_dir "$work/samples" --pa_protocol "$protocol" \
            --output_json "$work/metrics_$protocol.json" --output_csv "$work/metrics_$protocol.csv"
    done
}

case "${1:-}" in
    speed)
        make_fp32_checkpoint
        benchmark raw_fastsam_fp32_compile.csv    "$FP32_CHECKPOINT" fp32
        benchmark raw_fastsam_fp32_no_compile.csv "$FP32_CHECKPOINT" fp32 --no_compile
        benchmark raw_fastsam_bf16_compile.csv    "$CHECKPOINT" bf16
        benchmark raw_fastsam_bf16_no_compile.csv "$CHECKPOINT" bf16 --no_compile
        benchmark raw_fastsam_int8_compile.csv    "$CHECKPOINT" int8
        benchmark raw_fastsam_int8_no_compile.csv "$CHECKPOINT" int8 --no_compile
        ;;
    accuracy)
        make_fp32_checkpoint
        infer_and_evaluate fp32 "$FP32_CHECKPOINT" IMG_SIZE=384 LAYER_DTYPE=fp32 "${EAGER_ENV[@]}"
        infer_and_evaluate bf16 "$CHECKPOINT" IMG_SIZE=384 LAYER_DTYPE=bf16 "${EAGER_ENV[@]}"
        infer_and_evaluate int8 "$CHECKPOINT" "${INT8_ENV[@]}"
        ;;
    udp)
        env "${COMMON_ENV[@]}" "${INT8_ENV[@]}" "$PYTHON" -u run_publisher_ue.py --source video --video "$VIDEO" \
            --intrinsics "$MEASUREMENTS/camera.json" --no-loop --image-size 512 --yolo-model checkpoints/yolo/yolo11m-pose.engine \
            --nn-model-dir mhr2smpl/experiments/multiview_n30000_e500 --mhr2smpl-mapping-path mhr2smpl/data/mhr2smpl_mapping.npz \
            --metrics-csv "$OUT_DIR/raw_fastsam3dbody_int8_trt_udp_ue_frames.csv" \
            --summary-csv "$OUT_DIR/raw_fastsam3dbody_int8_trt_udp_ue_summary.csv" \
            --ue udp --ue-port 62004 --local-port 5555
        ;;
    *)
        echo "usage: bash measurements/fastsam/run.sh {speed|accuracy|udp}" >&2
        exit 2
        ;;
esac
echo "raw data: $OUT_DIR"

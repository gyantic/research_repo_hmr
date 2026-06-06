#!/usr/bin/env bash
set -euo pipefail

# Run fp32 (Torch), fp16 (TRT), int8 (TRT) in one shot and store outputs in a single run dir.

VIDEO=""
FP16_ENGINE="datas/trt/sat_hmr_sat_sameonnx_fp16.engine"
INT8_ENGINE="datas/trt/sat_hmr_sat_sameonnx_int8_qdq.engine"
RUN_NAME=""
INPUT_SIZE="1288"
CONF_THRESH="0.1"
ENABLE_UDP="0"
UE_PORT="62004"
LOCAL_PORT="5555"
CLEAN_LEGACY="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video)
      VIDEO="$2"; shift 2;;
    --fp16-engine)
      FP16_ENGINE="$2"; shift 2;;
    --int8-engine)
      INT8_ENGINE="$2"; shift 2;;
    --run-name)
      RUN_NAME="$2"; shift 2;;
    --input-size)
      INPUT_SIZE="$2"; shift 2;;
    --conf-thresh)
      CONF_THRESH="$2"; shift 2;;
    --udp)
      ENABLE_UDP="1"; shift;;
    --ue-port)
      UE_PORT="$2"; shift 2;;
    --local-port)
      LOCAL_PORT="$2"; shift 2;;
    --clean-legacy)
      CLEAN_LEGACY="1"; shift;;
    *)
      echo "[ERROR] Unknown arg: $1"
      exit 2;;
  esac
done

if [[ -z "${VIDEO}" ]]; then
  echo "Usage:"
  echo "  bash tools/run_precision_suite.sh --video videos/sample_video.mp4 [--udp] [--clean-legacy]"
  exit 2
fi

if [[ ! -f "${VIDEO}" ]]; then
  echo "[ERROR] video not found: ${VIDEO}"
  exit 2
fi
if [[ ! -f "${FP16_ENGINE}" ]]; then
  echo "[ERROR] fp16 engine not found: ${FP16_ENGINE}"
  exit 2
fi
if [[ ! -f "${INT8_ENGINE}" ]]; then
  echo "[ERROR] int8 engine not found: ${INT8_ENGINE}"
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${RUN_NAME}" ]]; then
  RUN_NAME="${TS}"
fi
RUN_DIR="plots/precision_runs/${RUN_NAME}"
mkdir -p "${RUN_DIR}/logs"

if [[ "${CLEAN_LEGACY}" == "1" ]]; then
  ARCHIVE_DIR="plots/legacy_archive/${TS}"
  mkdir -p "${ARCHIVE_DIR}"
  echo "[INFO] archiving root-level legacy plot files -> ${ARCHIVE_DIR}"
  find plots -maxdepth 1 -type f \( -name "*.csv" -o -name "*.json" -o -name "*.npz" -o -name "*.log" \) -print0 | \
    while IFS= read -r -d '' f; do
      mv -v "$f" "${ARCHIVE_DIR}/"
    done
fi

COMMON_ARGS=(
  -i "${VIDEO}"
  --input_size "${INPUT_SIZE}"
  --conf_thresh "${CONF_THRESH}"
  --no_display
  --no_render
  --no_save
)

if [[ "${ENABLE_UDP}" == "1" ]]; then
  COMMON_ARGS+=(--udp --udp_only --ue_port "${UE_PORT}" --local_port "${LOCAL_PORT}")
fi

echo "[STEP] fp32 Torch"
python -u video_demo.py \
  "${COMMON_ARGS[@]}" \
  --metrics_csv "${RUN_DIR}/fp32_metrics.csv" \
  --pred_npz "${RUN_DIR}/fp32_pred.npz" \
  2>&1 | tee "${RUN_DIR}/logs/fp32.log"

echo "[STEP] fp16 TensorRT"
python -u video_demo_trt.py \
  "${COMMON_ARGS[@]}" \
  --engine "${FP16_ENGINE}" \
  --metrics_csv "${RUN_DIR}/fp16_metrics.csv" \
  --pred_npz "${RUN_DIR}/fp16_pred.npz" \
  2>&1 | tee "${RUN_DIR}/logs/fp16.log"

echo "[STEP] int8 TensorRT"
python -u video_demo_trt.py \
  "${COMMON_ARGS[@]}" \
  --engine "${INT8_ENGINE}" \
  --metrics_csv "${RUN_DIR}/int8_metrics.csv" \
  --pred_npz "${RUN_DIR}/int8_pred.npz" \
  2>&1 | tee "${RUN_DIR}/logs/int8.log"

echo "[STEP] summarize"
python tools/summarize_precision_metrics.py \
  --fp32_csv "${RUN_DIR}/fp32_metrics.csv" \
  --fp16_csv "${RUN_DIR}/fp16_metrics.csv" \
  --int8_csv "${RUN_DIR}/int8_metrics.csv" \
  --output "${RUN_DIR}/precision_summary.csv"

ln -sfn "${RUN_NAME}" "plots/precision_runs/latest"

echo "[DONE] run_dir: ${RUN_DIR}"
echo "[DONE] summary: ${RUN_DIR}/precision_summary.csv"
echo "[DONE] latest symlink: plots/precision_runs/latest"

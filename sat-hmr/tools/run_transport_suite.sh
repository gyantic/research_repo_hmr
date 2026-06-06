#!/usr/bin/env bash
set -euo pipefail

# Run transport-level benchmark suite:
# 1) none (no UDP/TCP)
# 2) udp  (optional)
# 3) tcp  (optional, torch benchmark script)

VIDEO=""
FP16_ENGINE="datas/trt/sat_hmr_sat_sameonnx_fp16.engine"
INT8_ENGINE="datas/trt/sat_hmr_sat_sameonnx_int8_qdq.engine"
RUN_NAME=""
INPUT_SIZE="1288"
CONF_THRESH="0.1"

WITH_UDP="0"
WITH_TCP="0"
UE_PORT="62004"
LOCAL_PORT="5555"
TCP_HOST="127.0.0.1"
TCP_PORT="5555"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video) VIDEO="$2"; shift 2;;
    --fp16-engine) FP16_ENGINE="$2"; shift 2;;
    --int8-engine) INT8_ENGINE="$2"; shift 2;;
    --run-name) RUN_NAME="$2"; shift 2;;
    --input-size) INPUT_SIZE="$2"; shift 2;;
    --conf-thresh) CONF_THRESH="$2"; shift 2;;
    --with-udp) WITH_UDP="1"; shift;;
    --with-tcp) WITH_TCP="1"; shift;;
    --ue-port) UE_PORT="$2"; shift 2;;
    --local-port) LOCAL_PORT="$2"; shift 2;;
    --tcp-host) TCP_HOST="$2"; shift 2;;
    --tcp-port) TCP_PORT="$2"; shift 2;;
    *) echo "[ERROR] Unknown arg: $1"; exit 2;;
  esac
done

if [[ -z "${VIDEO}" ]]; then
  echo "Usage:"
  echo "  bash tools/run_transport_suite.sh --video videos/sample_video.mp4 [--with-udp] [--with-tcp]"
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

RUN_DIR="plots/transport_runs/${RUN_NAME}"
mkdir -p "${RUN_DIR}/logs"

NO_UDP_RUN="${RUN_NAME}_no_udp"
NO_UDP_SUMMARY="plots/precision_runs/${NO_UDP_RUN}/precision_summary.csv"

echo "[STEP] transport=none"
bash tools/run_precision_suite.sh \
  --video "${VIDEO}" \
  --fp16-engine "${FP16_ENGINE}" \
  --int8-engine "${INT8_ENGINE}" \
  --run-name "${NO_UDP_RUN}" \
  --input-size "${INPUT_SIZE}" \
  --conf-thresh "${CONF_THRESH}"

UDP_SUMMARY=""
if [[ "${WITH_UDP}" == "1" ]]; then
  UDP_RUN="${RUN_NAME}_udp"
  UDP_SUMMARY="plots/precision_runs/${UDP_RUN}/precision_summary.csv"
  echo "[STEP] transport=udp (UE receiver must be ready)"
  bash tools/run_precision_suite.sh \
    --video "${VIDEO}" \
    --fp16-engine "${FP16_ENGINE}" \
    --int8-engine "${INT8_ENGINE}" \
    --run-name "${UDP_RUN}" \
    --input-size "${INPUT_SIZE}" \
    --conf-thresh "${CONF_THRESH}" \
    --udp \
    --ue-port "${UE_PORT}" \
    --local-port "${LOCAL_PORT}"
fi

TCP_CSV=""
if [[ "${WITH_TCP}" == "1" ]]; then
  TCP_CSV="${RUN_DIR}/tcp_stats.csv"
  echo "[STEP] transport=tcp (requires TCP receiver/client connection)"
  SATHMR_VIDEO_PATH="${VIDEO}" \
  SATHMR_STATS_CSV="${TCP_CSV}" \
  SATHMR_TCP_HOST="${TCP_HOST}" \
  SATHMR_TCP_PORT="${TCP_PORT}" \
  SATHMR_INPUT_SIZE="${INPUT_SIZE}" \
  SATHMR_CONF_THRESH="${CONF_THRESH}" \
  python sathmr_tcp_benchmark.py 2>&1 | tee "${RUN_DIR}/logs/tcp.log"
fi

MATRIX_CSV="${RUN_DIR}/transport_precision_matrix.csv"
CMD=(python tools/collect_transport_matrix.py --no_udp_summary "${NO_UDP_SUMMARY}" --output "${MATRIX_CSV}")
if [[ -n "${UDP_SUMMARY}" ]]; then
  CMD+=(--udp_summary "${UDP_SUMMARY}")
fi
if [[ -n "${TCP_CSV}" ]]; then
  CMD+=(--tcp_stats_csv "${TCP_CSV}")
fi
"${CMD[@]}"

python tools/plot_transport_matrix.py \
  --matrix_csv "${MATRIX_CSV}" \
  --outdir "${RUN_DIR}"

ln -sfn "${RUN_NAME}" "plots/transport_runs/latest"

echo "[DONE] matrix: ${MATRIX_CSV}"
echo "[DONE] latest symlink: plots/transport_runs/latest"

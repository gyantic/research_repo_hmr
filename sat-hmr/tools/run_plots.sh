#!/usr/bin/env bash
set -euo pipefail

# One-shot plot runner for transport + accuracy charts.
#
# Usage:
#   bash tools/run_plots.sh
#   bash tools/run_plots.sh --error-bar std
#   bash tools/run_plots.sh --transport-run-dir plots/transport_runs/latest --accuracy-outdir plots/accuracy_plots

ERROR_BAR="sem"
TRANSPORT_RUN_DIR="plots/transport_runs/latest"
ACCURACY_OUTDIR="plots/accuracy_plots"
INCLUDE_COMPARE="1"
BENCH_OUTDIR="plots/benchmark_plots_mixed"
ROMP_GLOB="/home/kyuta/ROMP/*inference_stats*.csv"
SATHMR_GLOB="datas/*inference_stats*.csv"
SKIP_BENCH="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --error-bar)
      ERROR_BAR="$2"; shift 2;;
    --transport-run-dir)
      TRANSPORT_RUN_DIR="$2"; shift 2;;
    --accuracy-outdir)
      ACCURACY_OUTDIR="$2"; shift 2;;
    --bench-outdir)
      BENCH_OUTDIR="$2"; shift 2;;
    --romp-glob)
      ROMP_GLOB="$2"; shift 2;;
    --sathmr-glob)
      SATHMR_GLOB="$2"; shift 2;;
    --no-compare)
      INCLUDE_COMPARE="0"; shift;;
    --skip-bench)
      SKIP_BENCH="1"; shift;;
    -h|--help)
      cat <<'EOF'
Usage:
  bash tools/run_plots.sh [options]

Options:
  --error-bar {none,std,sem}      Error bar type (default: sem)
  --transport-run-dir <dir>       Transport run directory (default: plots/transport_runs/latest)
  --accuracy-outdir <dir>         Accuracy plot output directory (default: plots/accuracy_plots)
  --bench-outdir <dir>            ROMP vs SAT-HMR benchmark outdir (default: plots/benchmark_plots_mixed)
  --romp-glob <glob>              ROMP inference_stats glob (default: /home/kyuta/ROMP/*inference_stats*.csv)
  --sathmr-glob <glob>            SAT-HMR inference_stats glob (default: datas/*inference_stats*.csv)
  --no-compare                    Do not include *compare*.csv for accuracy plot
  --skip-bench                    Skip ROMP vs SAT-HMR benchmark plots
  -h, --help                      Show this help
EOF
      exit 0;;
    *)
      echo "[ERROR] Unknown arg: $1"
      exit 2;;
  esac
done

case "${ERROR_BAR}" in
  none|std|sem) ;;
  *)
    echo "[ERROR] --error-bar must be one of: none/std/sem"
    exit 2;;
esac

MATRIX_CSV="${TRANSPORT_RUN_DIR}/transport_precision_matrix.csv"
if [[ ! -f "${MATRIX_CSV}" ]]; then
  echo "[ERROR] transport matrix not found: ${MATRIX_CSV}"
  exit 2
fi

echo "[STEP] transport plots"
python tools/plot_transport_matrix.py \
  --matrix_csv "${MATRIX_CSV}" \
  --outdir "${TRANSPORT_RUN_DIR}" \
  --error-bar "${ERROR_BAR}"

echo "[STEP] accuracy plots"
if [[ "${INCLUDE_COMPARE}" == "1" ]]; then
  python tools/plot_accuracy_comparison.py \
    --inputs "plots/*summary*.csv" "plots/*compare*.csv" \
    --outdir "${ACCURACY_OUTDIR}" \
    --error-bar "${ERROR_BAR}"
else
  python tools/plot_accuracy_comparison.py \
    --inputs "plots/*summary*.csv" \
    --outdir "${ACCURACY_OUTDIR}" \
    --error-bar "${ERROR_BAR}"
fi

if [[ "${SKIP_BENCH}" != "1" ]]; then
  shopt -s nullglob
  ROMP_FILES=( ${ROMP_GLOB} )
  SATHMR_FILES=( ${SATHMR_GLOB} )
  shopt -u nullglob

  if [[ ${#ROMP_FILES[@]} -gt 0 && ${#SATHMR_FILES[@]} -gt 0 ]]; then
    echo "[STEP] ROMP vs SAT-HMR benchmark plots"
    python tools/plot_benchmark_comparison.py \
      --inputs "${ROMP_GLOB}" "${SATHMR_GLOB}" \
      --outdir "${BENCH_OUTDIR}" \
      --error-bar "${ERROR_BAR}" \
      --group-by framework backbone transport framework+backbone+transport
  else
    echo "[SKIP] benchmark plots: missing inference_stats CSVs"
    echo "       ROMP files: ${#ROMP_FILES[@]} (glob=${ROMP_GLOB})"
    echo "       SAT-HMR files: ${#SATHMR_FILES[@]} (glob=${SATHMR_GLOB})"
  fi
fi

echo "[DONE] transport out: ${TRANSPORT_RUN_DIR}"
echo "[DONE] accuracy out: ${ACCURACY_OUTDIR}"
if [[ "${SKIP_BENCH}" != "1" ]]; then
  echo "[DONE] benchmark out: ${BENCH_OUTDIR}"
fi

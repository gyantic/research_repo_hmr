#!/usr/bin/env bash
set -euo pipefail

# Cleanup helper for benchmark/result data.
# Default behavior: move old top-level entries in plots/ to archive.

APPLY=0
DELETE_MODE=0
TARGET_PLOTS="plots"
TARGET_DATAS="datas"
KEEP_PLOTS=("precision_runs" "transport_runs" "legacy_archive")
INCLUDE_ACTIVE_RUNS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply) APPLY=1; shift;;
    --delete) DELETE_MODE=1; shift;;
    --include-active-runs) INCLUDE_ACTIVE_RUNS=1; shift;;
    --plots-dir) TARGET_PLOTS="$2"; shift 2;;
    --datas-dir) TARGET_DATAS="$2"; shift 2;;
    *) echo "[ERROR] Unknown arg: $1"; exit 2;;
  esac
done

if [[ ! -d "${TARGET_PLOTS}" ]]; then
  echo "[ERROR] plots dir not found: ${TARGET_PLOTS}"
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_DIR="${TARGET_PLOTS}/legacy_archive/${TS}"

is_kept() {
  local name="$1"
  if [[ "${INCLUDE_ACTIVE_RUNS}" == "1" ]]; then
    if [[ "${name}" == "legacy_archive" ]]; then
      return 0
    fi
    return 1
  fi
  for k in "${KEEP_PLOTS[@]}"; do
    if [[ "${name}" == "${k}" ]]; then
      return 0
    fi
  done
  return 1
}

PLOT_ITEMS=()
while IFS= read -r item; do
  base="$(basename "$item")"
  if is_kept "${base}"; then
    continue
  fi
  PLOT_ITEMS+=("${item}")
done < <(find "${TARGET_PLOTS}" -mindepth 1 -maxdepth 1 -print | sort)

DATA_ITEMS=()
if [[ -d "${TARGET_DATAS}" ]]; then
  while IFS= read -r item; do
    DATA_ITEMS+=("${item}")
  done < <(find "${TARGET_DATAS}" -mindepth 1 -maxdepth 1 -type f -name '*inference_stats*.csv' -print | sort)
fi

echo "[PLAN] plots entries to cleanup: ${#PLOT_ITEMS[@]}"
for p in "${PLOT_ITEMS[@]}"; do
  echo "  - ${p}"
done
echo "[PLAN] datas stats csv to cleanup: ${#DATA_ITEMS[@]}"
for d in "${DATA_ITEMS[@]}"; do
  echo "  - ${d}"
done

if [[ "${APPLY}" != "1" ]]; then
  echo "[DRY-RUN] no changes applied. add --apply to execute."
  exit 0
fi

if [[ "${DELETE_MODE}" == "1" ]]; then
  echo "[APPLY] delete mode"
  for p in "${PLOT_ITEMS[@]}"; do
    rm -rf "${p}"
  done
  for d in "${DATA_ITEMS[@]}"; do
    rm -f "${d}"
  done
else
  echo "[APPLY] archive mode -> ${ARCHIVE_DIR}"
  mkdir -p "${ARCHIVE_DIR}"
  for p in "${PLOT_ITEMS[@]}"; do
    mv "${p}" "${ARCHIVE_DIR}/"
  done
  for d in "${DATA_ITEMS[@]}"; do
    mv "${d}" "${ARCHIVE_DIR}/"
  done
fi

echo "[DONE] cleanup completed."

#!/bin/bash
# 論文の計測・評価をすべて実行し、paper_results.csv を作る。リポジトリのルートから実行する。
#
#   bash measurements/run_all.sh
#
# 各モデルを conda 環境で実行する（環境名は ROMP_ENV・SATHMR_ENV・FASTSAM_ENV で変更できる）。
# UDP の計測は UE の受信 Blueprint を先に起動しておく必要がある（ue/README.md）。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export OUT_DIR="${OUT_DIR:-$REPO_ROOT/results/measurements/rerun_$(date +%Y%m%d_%H%M%S)}"
ROMP_ENV="${ROMP_ENV:-romp_3.9}"
SATHMR_ENV="${SATHMR_ENV:-sathmr-copy}"
FASTSAM_ENV="${FASTSAM_ENV:-sam_3d_body}"

run() {  # <conda 環境> <run.sh> <モード>
    conda run --no-capture-output -n "$1" bash "$REPO_ROOT/measurements/$2" "$3"
}

for mode in speed accuracy udp; do
    run "$ROMP_ENV" romp/run.sh "$mode"
done
for mode in speed compile_scope accuracy udp; do
    run "$SATHMR_ENV" sathmr/run.sh "$mode"
done
for mode in speed accuracy udp; do
    run "$FASTSAM_ENV" fastsam/run.sh "$mode"
done

python3 "$REPO_ROOT/measurements/make_paper_results.py" "$OUT_DIR"

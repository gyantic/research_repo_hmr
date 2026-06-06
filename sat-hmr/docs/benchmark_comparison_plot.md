# Benchmark Comparison Plot Guide

## 現在のプロット方針（固定）
やる比較は次の3系統に固定する。

1. SAT-HMR: `fp32 / fp16 / int8`（すべて UDP）
2. SAT-HMR: `UDP vs TCP`（`fp32` 固定）
3. `ROMP vs SAT-HMR`（framework比較）

### 1,2 の実行（SAT-HMR transport/precision）
`tools/run_transport_suite.sh` を使う。  
このスクリプトは `transport_precision_matrix.csv` と対応PNGを自動生成する。

```bash
bash tools/run_transport_suite.sh \
  --video videos/sample_video.mp4 \
  --run-name transport_suite_01 \
  --with-udp \
  --with-tcp
```

主な出力:
- `plots/transport_runs/<run>/transport_precision_matrix.csv`
- `plots/transport_runs/<run>/sathmr_udp_precision_inference_ms.png`
- `plots/transport_runs/<run>/sathmr_udp_precision_fps.png`
- `plots/transport_runs/<run>/sathmr_fp32_transport_inference_ms.png`
- `plots/transport_runs/<run>/sathmr_fp32_transport_fps.png`

`run_transport_suite.sh` 経由の棒グラフは、取得できる場合 `sem` エラーバー付きで保存される。

### 3 の実行（ROMP vs SAT-HMR）
既存の `tools/plot_benchmark_comparison.py` を使う。

```bash
python tools/plot_benchmark_comparison.py \
  --inputs "/home/kyuta/ROMP/*inference_stats*.csv" "/home/kyuta/SAT-HMR/datas/*inference_stats*.csv" \
  --outdir plots/benchmark_plots_mixed \
  --error-bar std \
  --group-by framework backbone transport framework+backbone+transport
```

### 精度専用プロット（別ファイル）
精度比較は速度比較と分離し、`tools/plot_accuracy_comparison.py` を使う。
入力は `eval_pseudo_gt_sequence.py` や `eval_3dpw_trt.py` の summary CSV（比較CSVも可）。

```bash
python tools/plot_accuracy_comparison.py \
  --inputs "plots/*summary*.csv" "plots/*compare*.csv" \
  --outdir plots/accuracy_plots \
  --error-bar sem
```

`--error-bar` は `std` / `sem` / `none` を指定可能。  
8本の棒グラフすべてに同じ方式のエラーバーを適用する。
`MPJPE/PVE` 系は対応する `*details*.csv` から、`precision/recall/f1/detection_rate` は `tp/fp/fn` から誤差を算出する。

主な出力:
- `plots/accuracy_plots/accuracy_comparison_summary.csv`
- `plots/accuracy_plots/runs_mpjpe_mm.png`
- `plots/accuracy_plots/runs_pa_mpjpe_mm.png`
- `plots/accuracy_plots/runs_pve_mm.png`
- `plots/accuracy_plots/runs_pa_pve_mm.png`
- `plots/accuracy_plots/runs_precision.png`
- `plots/accuracy_plots/runs_recall.png`
- `plots/accuracy_plots/runs_f1.png`
- `plots/accuracy_plots/runs_detection_rate.png`

## 目的
ROMP / SAT-HMR の推論計測CSVをまとめて比較し、`framework` / `backbone` / `transport` などの軸で可視化する。

対応CSV例:
- `inference_stats_*.csv`
- `TCP_inference_stats_*.csv`

## 追加スクリプト
- `tools/plot_benchmark_comparison.py`

量子化方針と実行手順は `docs/quantization_plan.md` を参照。

## SAT-HMRベンチCSVの保存先
- `sathmr_udp_benchmark.py` / `sathmr_tcp_benchmark.py` は `datas/` を自動作成し、その配下にCSVを保存する。

## 基本実行例
```bash
python tools/plot_benchmark_comparison.py \
  --inputs "/home/kyuta/ROMP/*inference_stats*.csv" "/home/kyuta/SAT-HMR/datas/*inference_stats*.csv" \
  --outdir plots/benchmark_plots \
  --error-bar std \
  --group-by framework backbone transport framework+backbone+transport \
  --write-manifest-template
```

## 実行パターン例
### SAT-HMRのみ
```bash
python tools/plot_benchmark_comparison.py \
  --inputs "/home/kyuta/SAT-HMR/datas/*inference_stats*.csv" \
  --outdir plots/benchmark_plots_sathmr \
  --error-bar std \
  --group-by backbone transport framework+backbone+transport
```

### ROMPのみ (ROMP同士比較)
```bash
python tools/plot_benchmark_comparison.py \
  --inputs "/home/kyuta/ROMP/*inference_stats*.csv" \
  --outdir plots/benchmark_plots_romp \
  --error-bar std \
  --group-by backbone transport framework+backbone+transport
```

### ROMP + SAT-HMR 混在比較
```bash
python tools/plot_benchmark_comparison.py \
  --inputs "/home/kyuta/ROMP/*inference_stats*.csv" "/home/kyuta/SAT-HMR/datas/*inference_stats*.csv" \
  --outdir plots/benchmark_plots_mixed \
  --error-bar std \
  --group-by framework backbone transport framework+backbone+transport
```

## 出力
`--outdir` 配下に以下を出力:
- `benchmark_comparison_summary.csv`
  - 実験(run)ごとの要約
- `benchmark_comparison_frames_long.csv`
  - フレーム単位のロング形式
- `runs_*.png`
  - run単位の比較 (mean/fps/boxplot/line)
- `group_*_summary.csv`, `group_*.png`
  - 指定軸ごとの集計とプロット
- `manifest_template.csv` (`--write-manifest-template` 時)
  - 推定メタデータ雛形

`run_name` とは別に `run_label` を出力し、常に `ROMP:...` / `SAT-HMR:...` のように framework を明記する。
ROMP同士比較でも `run_label` で ROMP 実験であることが明示される。

`group-by` に `framework` を含めない場合でも、`group_label` は `[ROMP] UDP` や `[ROMP,SAT-HMR] UDP` のように framework 情報を自動で付与する。

`--error-bar` で棒グラフのエラーバーを指定できる:
- `std` (標準偏差)
- `sem` (標準誤差)
- `none` (エラーバー無し)

指定した方式は画像左上に `Error bar: ...` として表示される。

## 比較軸の設計
`--group-by` で柔軟に指定可能。

例:
- `framework` (ROMP / SAT-HMR)
- `backbone` (HRNet / ResNet / SAT)
- `transport` (TCP / UDP)
- `framework+backbone+transport` (複合条件)

## メタデータ上書き (任意)
ファイル名からの自動推定が不十分な場合は `--manifest` で上書きする。

### manifest CSV 形式
```csv
path,run_name,framework,backbone,transport,model,tags
/home/kyuta/ROMP/inference_stats_ROMP_HRNet32_V1.csv,ROMP_HRNet32_UDP,ROMP,HRNet,UDP,ROMP_HRNet32,v1
/home/kyuta/SAT-HMR/datas/inference_stats_sat_hmr_UDP.csv,SATHMR_UDP,SAT-HMR,SAT,UDP,SAT-HMR,baseline
```

`path` は以下のいずれかで一致可能:
- 絶対パス
- 相対パス
- ファイル名
- 拡張子なしファイル名

## 注意点
- `mean_total_ms` と `fps_total` は、CSV内の有効な `total_time_ms` のみで計算される。
- 検出なしフレームが多い実験では、`fps_total` と `fps_inference` の差が大きくなる。
- run数が多い場合、可読性確保のためフレーム時系列プロットは自動スキップされる。

# TensorRT INT8 仕様メモ（SAT-HMR）

## 1. 目的
- SAT-HMR の INT8 最適化を、再現可能な手順で実行する。
- FP16 と INT8 を公平比較できるように、同一ONNX起点のフローに統一する。
- 失敗しやすいポイント（QDQ / VC / layer info）を事前に回避する。

## 2. 用語
- `FP16` は TensorRT engine の精度指定を指す。
- `INT8` は TensorRT engine の精度指定を指す。
- `ONNX` は変換元モデル。engine精度とは別概念。
- `compute_mean_ms` は `trtexec --exportTimes` の `computeMs` 平均（GPU計算時間）を指す。
- `qps_from_compute` は `1000 / compute_mean_ms` で計算する推論レート（query per second）。
- `lat_mean_ms` は `latencyMs` 平均で、H2D/D2Hや実行オーバーヘッドを含む。
- `qps_from_lat` は `1000 / lat_mean_ms`。実運用の体感速度に近い。
- 注意: `qps_from_compute` は「推論コア性能」の指標で、`video_demo_trt.py` の表示FPSとは一致しない。

## 3. 現在の正規フロー
- 入口ONNXは 1つに固定する。  
  例: `datas/onnx/sat_hmr_sat_runtimefix.onnx`
- 同一ONNXから次を作る。
  - FP16 engine
  - QDQ INT8 ONNX
  - QDQ INT8 ONNXからINT8 engine
- その2engineを同条件でベンチし、速度差を算出する。

実行スクリプト:
- `tools/resolve_int8_slowdown.py`

補助スクリプト:
- `tools/quantize_onnx_qdq_ptq.py`
- `tools/benchmark_trt_times.py`

現在採用中のengine:
- FP16: `datas/trt/sat_hmr_sat_sameonnx_fp16.engine`
- INT8: `datas/trt/sat_hmr_sat_sameonnx_int8_qdq.engine`


## 4. INT8をどう作っているか（理論）
### 4.1 QDQ PTQ
- SAT-HMRは学習済み重みをそのまま使い、再学習なしでINT8化する。
- そのためにONNXへ `QuantizeLinear/DequantizeLinear`（QDQ）を挿入し、どの演算を量子化対象にするかを明示する。
- 校正用画像（calibration）で活性値の分布を観測し、スケール/ゼロ点を推定する。
- 重みは静的量子化、活性は推論時の範囲を想定した静的スケールで量子化する。

### 4.2 `Conv/MatMul` 中心
- INT8化の主な速度寄与は、計算量が大きい線形演算（`Conv`, `MatMul`）に集中する。
- 一方で `Gemm` や一部の小演算は、biasの型やブロードキャスト仕様でパーサ失敗やフォールバックを起こしやすい。
- そのため、まずは主要演算をINT8化し、壊れやすいノードは除外して安定性を優先する。

### 4.3 TensorRT側で何が起きるか
- TensorRTはQDQ付きONNXを受け取り、INT8実行可能な層はINT8カーネルへ割り当てる。
- ただし未対応演算や精度保護が必要な箇所はFP16/FP32へ自動フォールバックする。
- 実際のengineは「純INT8」ではなく「INT8 + FP16/FP32混在」になることが多い。
- そのため、速度は「INT8化率」と「フォールバック層の重さ」で決まる。

### 4.4 Calibration方式との違い
- QDQ方式: ONNXに量子化情報を埋め込んでからengine化する。量子化意図をONNX側で保持しやすい。
- Calibration方式: ONNXはFPのままにして、TensorRTビルド時に校正してINT8スケールを決める。
- 本プロジェクトの主比較は「同一ONNX起点のQDQ方式」を基準にしている。

### 4.5 同一ONNX比較を行う理由
- FP16とINT8の差を正しく見るには、モデル構造差を消して精度設定差だけを比較する必要がある。
- そのため、同一ONNXからFP16 engineとINT8 engineを作り、同条件で速度比較する。

### 4.6 Calibration画像の枚数と選定方針（再現性）
- 抽出元は動画で、`tools/extract_calibration_frames.py` がフレーム範囲内を一様間隔でサンプリングする。
- 既定では抽出枚数は `800`（`--num-frames`）。
- QDQ化時（`tools/quantize_onnx_qdq_ptq.py`）は `max_images=80` を既定とし、manifest順の先頭 `80` 枚を使用する。
- 乱択は使っていないため、同一manifestなら選定結果は決定的（seed依存なし）。
- したがって再現手順は「同一動画・同一抽出範囲・同一manifest」を固定する。

## 5. 出力ファイル
- 速度比較CSV: `plots/fp16_vs_int8_same_onnx.csv`
- INT8 layer info: `plots/int8_sameonnx_layerinfo_detailed.txt`
- INT8 coverage CSV: `plots/int8_coverage_summary.csv`
- 実行ログ: `plots/logs/*.log`

## 6. 主要な実装仕様
- QDQ対象opの既定は `Conv,MatMul`。
- `Gemm` は既定から外す（bias DQ問題を回避するため）。
- QDQ後に `int32/int64` の `DequantizeLinear`（特にbias）を FP32 initializer に展開する修正を適用する。
- 実運用は TensorRT 10.13 に統一する（`trtexec` と `python tensorrt` を同一バージョンに合わせる）。
- 現在採用中の2つのengine（sameonnx FP16/INT8）は非VC系として運用する。
- engine build時に `--profilingVerbosity=detailed` を指定して詳細layer情報を取りやすくする。
- layer infoが `LAYER_NAMES_ONLY` しか取れない場合は `names_only` モードで近似coverageを出す。

## 7. 既知エラーと対処
- `...bias_DequantizeLinear` で parse失敗  
  原因: TRT 10.13 が int32 bias DQ を受け付けない。  
  対処: `tools/quantize_onnx_qdq_ptq.py` の自動修正でbias DQをFP32定数へ展開。

- `The engine plan file is not compatible with this version of TensorRT`  
  原因: engine作成時と実行時のTensorRTバージョン不一致（例: 10.13 vs 10.15）。  
  対処: 作る側と読む側のTensorRTを同一バージョンへ統一する。

- `benchmark_trt_times.py: argument --extra_arg: expected one argument`  
  原因: `--extra_arg` の渡し方不正。  
  対処: `--extra_arg=...` 形式で渡す。

- coverageが `0/0`  
  原因: log形式が `LAYER_NAMES_ONLY` で詳細dtypeが無い。  
  対処: `names_only` 解析へフォールバックし、`total_layers` と `int8_related_layers` を出す。

## 8. 判定基準
- 速度判定は `plots/fp16_vs_int8_same_onnx.csv` の `compute_mean_ms` を優先する。
- 判定式は `delta_ms = INT8_compute_mean_ms - FP16_compute_mean_ms` とする。
- `delta_ms < 0` なら成功（INT8が高速）。
- `delta_ms > 0` なら失敗（INT8が遅い）。  
  同義に `INT8_compute_mean_ms > FP16_compute_mean_ms` なら失敗。
- `qps_from_compute` で見る場合は逆向きで、`INT8_qps > FP16_qps` が成功。
- 可能なら coverage も確認する。
  - `coverage_mode=detailed`: `int8_output_ratio_pct` を利用。
  - `coverage_mode=names_only`: `int8_related_layers / total_layers` を補助的に利用。

## 9. 直近結果
- run: `rerun_20260226_002802`
- 参照:
  - `plots/precision_runs/rerun_20260226_002802_no_udp/precision_summary.csv`
  - `plots/precision_runs/rerun_20260226_002802_udp/precision_summary.csv`
  - `plots/transport_runs/rerun_20260226_002802/transport_precision_matrix.csv`

### 9.1 no_udp（実運用: 前後処理込み）
| precision | `inference_ms_mean` | `fps_inst_mean` |
| --- | ---: | ---: |
| `fp32` | 52.3225 | 14.1389 |
| `fp16` | 13.1064 | 41.9104 |
| `int8` | 14.5347 | 37.5989 |

### 9.2 udp（実運用: 前後処理 + UDP送信込み）
| precision | `inference_ms_mean` | `fps_inst_mean` |
| --- | ---: | ---: |
| `fp32` | 96.7141 | 9.0449 |
| `fp16` | 21.7198 | 30.5305 |
| `int8` | 25.3509 | 28.6494 |

### 9.3 TCP（fp32固定）
| transport | precision | `inference_ms_mean` | `fps_inst_mean` |
| --- | --- | ---: | ---: |
| `tcp` | `fp32_torch_tcp_benchmark` | 99.9800 | 10.1802 |

### 9.4 比較差分（INT8 - FP16）
| 条件 | `delta_inference_ms` | `delta_fps_inst` | `FP16/INT8` |
| --- | ---: | ---: | ---: |
| `no_udp` | +1.4283 | -4.3116 | 0.9017x |
| `udp` | +3.6311 | -1.8811 | 0.8568x |

判定:
- 最新runでは `no_udp` / `udp` の両方で `INT8_inference_ms_mean > FP16_inference_ms_mean`。
- したがって、**このrunの実運用条件では INT8 は FP16 より遅い**。

補足:
- ここは `video_demo_trt.py` の実運用値で、前後処理・通信を含む。
- `trtexec --noDataTransfers` の純推論コア値は別測定として扱う。

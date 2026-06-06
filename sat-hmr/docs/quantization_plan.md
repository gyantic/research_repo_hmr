# SAT-HMR Quantization Plan

## 1. 目的
- UE連携リアルタイム推論のFPSを改善する。
- ただし「精度悪化で挙動が壊れる」状態は避ける。

優先順位:
1. 実運用FPS
2. 挙動安定性（ジャンプ/ガタつき/ロスト増加なし）
3. 推論精度（検出率・conf分布）

## 2. 前提と結論
- 現行のSAT-HMRは量子化未導入（INT8/INT4なし）。
- `torch.quantization.quantize_dynamic` は主にCPU向けで、GPU運用では優先度が低い。
- 本命は `TensorRT FP16 -> TensorRT INT8(PTQ)` の順。

## 3. 実施フェーズ

### Phase 0: Baseline固定（必須）
- 実施内容:
  - 現行PyTorch推論（FP32/AMP）で基準CSVを保存。
  - 保存先: `datas/`
- 成果物:
  - `datas/inference_stats_sat_hmr_UDP.csv`
  - `datas/TCP_inference_stats_sat_hmr.csv`
  - 比較プロット出力（`docs/benchmark_plots_*`）
- 完了条件:
  - 以後の比較で同じ入力動画/同じ条件を再現できること。

### Phase 1: TensorRT FP16（量子化前の本命最適化）
- 実施内容:
  - SAT-HMRをONNXへ書き出し。
  - ONNXからTensorRT FP16エンジン作成。
  - PyTorch版と同一入力で速度・出力差分を比較。
- 成果物:
  - `datas/onnx/*.onnx`
  - `datas/trt/*.engine`（FP16）
  - 比較CSV（PyTorch vs TRT-FP16）
- 完了条件:
  - FPS改善が確認できる（目標: +20%以上）。
  - 目視品質劣化が許容範囲。

### Phase 2: TensorRT INT8 PTQ（本命量子化）
- 実施内容:
  - 校正データ（webcam条件）500〜1000フレーム準備。
  - INT8キャリブレーション付きでエンジン作成。
  - FP16/FP32と比較評価。
- 成果物:
  - `datas/calibration/*.npy` or `*.jpg`
  - `datas/trt/*_int8.engine`
  - 比較CSV・比較プロット
- 完了条件:
  - FPS改善（目標: FP16比 +10〜20%、FP32比 +30%以上）
  - 品質劣化が閾値内（下記評価基準）

### Phase 3: QAT/蒸留（必要時のみ）
- 実施内容:
  - INT8 PTQで劣化が大きい場合のみ検討。
  - QATまたは蒸留（例: SAM3D蒸留）を導入。
- 成果物:
  - 再学習済み重み
  - 再評価レポート

## 4. 評価基準（Go / No-Go）

### 4.1 速度
- `fps_inference`, `fps_total`
- `mean_inference_ms`, `P95_inference_ms`（可能なら追加）

### 4.2 安定性
- no detection率
- outlier skip率（`udp_outlier_mode=skip`時）
- 連続skip最大長

### 4.3 品質
- conf分布（中央値/P10）
- 目視での破綻（急な骨格崩壊、ジャンプ時フリーズ）

### 4.4 合否ライン（初期案）
- 速度:
  - FP16: Baseline比 +20%以上
  - INT8: FP16比 +10%以上（またはFP32比 +30%以上）
- 品質:
  - no detection率の悪化 +2%以内
  - UE動作の破綻が増えない

## 5. すぐやるタスク（実装順）
1. ONNX export用のSAT-HMRラッパー作成（1入力/1出力を固定）。
2. ONNX出力とPyTorch出力の誤差検証スクリプト作成。
3. TensorRT FP16推論経路を `webcam_demo.py` / ベンチに追加（backend切替）。
4. 校正データ生成スクリプト作成（動画/カメラからNフレーム抽出）。
5. TensorRT INT8エンジン生成と比較計測。

## 6. コマンド運用の目安

### 6.0 まずFP16でやること（最優先）
1. ONNX書き出し
2. PyTorch vs ONNX差分確認
3. TensorRT FP16エンジン作成
4. FPS/品質比較（Baseline vs FP16）

### 6.1 ONNX書き出し（追加済み）
```bash
python tools/export_sat_hmr_onnx.py \
  --weights ./weights/sat_hmr/sat_644.pth \
  --input-size 1288 \
  --img-h 728 --img-w 1288 \
  --device cpu \
  --out datas/onnx/sat_hmr.onnx
```

### 6.2 PyTorch vs ONNX 出力差分確認（追加済み）
```bash
python tools/check_sat_hmr_onnx_parity.py \
  --onnx datas/onnx/sat_hmr.onnx \
  --weights ./weights/sat_hmr/sat_644.pth \
  --input-size 1288 \
  --img-h 728 --img-w 1288 \
  --device cpu \
  --atol 1e-3 --rtol 1e-2
```

### 6.3 TensorRT FP16エンジン作成（次に実行）
```bash
trtexec \
  --onnx=datas/onnx/sat_hmr.onnx \
  --saveEngine=datas/trt/sat_hmr_fp16.engine \
  --fp16
```

### 6.4 INT8校正用フレーム抽出（追加済み）
```bash
python tools/extract_calibration_frames.py \
  --video videos/sample_video.mp4 \
  --outdir datas/calibration \
  --num-frames 800
```

### 6.5 前提パッケージ
- ONNX関連スクリプトは `onnx` / `onnxruntime` が必要。
- 未導入時は以下を利用:
  - `pip install onnx onnxruntime`

### 6.6 比較プロット（既存）
```bash
python tools/plot_benchmark_comparison.py \
  --inputs "/home/kyuta/ROMP/*inference_stats*.csv" "/home/kyuta/SAT-HMR/datas/*inference_stats*.csv" \
  --outdir docs/benchmark_plots_mixed \
  --error-bar std \
  --group-by framework backbone transport framework+backbone+transport
```

### 6.7 ベンチCSV保存先
- SAT-HMRベンチは自動で `datas/` にCSVを保存する。

### 6.8 CPU動的量子化（補助策・実装済み）
- これは `torch.ao.quantization.quantize_dynamic` のCPU専用最適化。
- 本命はFP16/TensorRTで、CPU動的量子化は「GPUなし環境」向け。

`webcam_demo.py`:
```bash
python webcam_demo.py \
  --camera 0 \
  --udp --udp_only \
  --device cpu \
  --cpu_int8_dynamic \
  --conf_thresh 0.1
```

`sathmr_udp_benchmark.py` / `sathmr_tcp_benchmark.py`:
```bash
export SATHMR_DEVICE_MODE=cpu
export SATHMR_CPU_INT8_DYNAMIC=1
export SATHMR_CPU_THREADS=4
export SATHMR_CPU_INTEROP_THREADS=1
python sathmr_udp_benchmark.py
python sathmr_tcp_benchmark.py
```

## 7. リスクと回避策
- リスク: ONNX変換で非対応OPが出る
  - 回避: まずFP16最小経路で変換可能範囲を確定
- リスク: INT8で精度劣化
  - 回避: 校正データを実運用ドメインに寄せる
- リスク: UE挙動が破綻
  - 回避: outlier設定・jump例外の既存保護を維持して評価

## 8. 進捗管理
- ステータス: `[未着手] / [進行中] / [完了]`

- `[進行中]` Phase 0 Baseline固定
  - `datas/` へのCSV保存・比較プロット基盤は整備済み。
- `[完了]` CPU動的量子化の即時運用経路
  - `webcam_demo.py`: `--device cpu --cpu_int8_dynamic` に対応。
  - `sathmr_udp_benchmark.py` / `sathmr_tcp_benchmark.py`:
    - `SATHMR_DEVICE_MODE` / `SATHMR_CPU_INT8_DYNAMIC` でCPU INT8に切替可能。
    - CPU時の計測分岐（`perf_counter`）を実装済み。
- `[進行中]` Phase 1 TensorRT FP16
  - 追加済み:
    - `tools/export_sat_hmr_onnx.py`
    - `tools/check_sat_hmr_onnx_parity.py`
    - `tools/extract_calibration_frames.py`
  - モデル側のCPU/exportフォールバック修正:
    - `models/decoder.py`
    - `models/encoders/dinov2/layers/block.py`
    - `utils/transforms.py`
  - 現在のブロッカー:
    - `onnx` / `onnxruntime` 未導入
- `[未着手]` Phase 2 TensorRT INT8 PTQ
- `[未着手]` Phase 3 QAT/蒸留（必要時）

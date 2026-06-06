# ONNX-TensorRT 変換仕様書 (SAT-HMR / SAT有効)

## 0. この文書の対象
この文書は、SAT-HMRを**SAT有効のまま** ONNXへ書き出し、TensorRTで推論を実行するために追加・変更したコードを整理したものです。

前提:
- SATは無効化しない (`sat_cfg.use_sat=True`)
- 主対象はリアルタイム用途の `B=1` (batch size 1)
- 変換後は `trtexec` でエンジン化して計測する

## 1. 変更したファイル一覧
1. `tools/export_sat_hmr_onnx.py`
2. `tools/fix_onnx_where_broadcast.py`
3. `tools/sat_hmr_export_utils.py`
4. `models/decoder.py`

## 2. 各ファイルの実装内容

### 2.1 `tools/export_sat_hmr_onnx.py`
目的:
- SAT有効モデルをONNXへ書き出す
- 書き出し直後に runtime-fix を自動適用できるようにする

実装ポイント:
- `--opset` のデフォルトを `13` に設定 (`tools/export_sat_hmr_onnx.py:38`)
- `--apply-runtime-fix` 追加 (`tools/export_sat_hmr_onnx.py:46`)
- `--fixed-out` 追加 (`tools/export_sat_hmr_onnx.py:50`)
- `--drop-img-size-input` 追加 (`tools/export_sat_hmr_onnx.py:40`)
- CUDA export時は constant folding を無効化 (`tools/export_sat_hmr_onnx.py:119`)
  - 目的: CPU/CUDA混在テンソルの shape inference 失敗を回避
- `--apply-runtime-fix` 指定時、以下を順に適用 (`tools/export_sat_hmr_onnx.py:143`)
  - `patch_where_broadcast`
  - `rewrite_problematic_bool_where`
  - `rewrite_single_output_split`
- 保存前に `onnx.checker` と `shape_inference` を実行 (`tools/export_sat_hmr_onnx.py:147`)

出力:
- 通常ONNX: `--out`
- 修正済みONNX: `--fixed-out` 未指定時 `<out_stem>_runtimefix.onnx`

### 2.2 `tools/fix_onnx_where_broadcast.py`
目的:
- SAT-HMR ONNXの実行時エラーを起こしやすいノード構造を、後処理で修正する

実装パス:
1. `patch_where_broadcast` (`tools/fix_onnx_where_broadcast.py:55`)
- 問題: `Where` で `cond:[N]`, `y:[N,...]` のbroadcast不一致
- 対処: `Unsqueeze(cond, axes=[1..])` を挿入して rank を合わせる

2. `rewrite_problematic_bool_where` (`tools/fix_onnx_where_broadcast.py:119`)
- 問題: 一部実行系で `bool Where` が未実装/不安定
- 対処: `Where(cond, x, y)` を算術合成へ置換
  - `cond_i32 * x_i32 + (1-cond_i32) * y_i32`
  - 最後に `Cast(..., BOOL)`

3. `rewrite_single_output_split` (`tools/fix_onnx_where_broadcast.py:201`)
- 問題: 出力1個の `Split` に固定長が焼き付き、実行時shapeがずれて失敗
- 対処: `Split(1出力)` を `Identity` に置換

CLI:
- `--skip-bool-where-rewrite`
- `--skip-single-split-rewrite`

### 2.3 `tools/sat_hmr_export_utils.py`
目的:
- ONNX exportの共通化
- `img_size` を入力として保持できるラッパー提供

実装ポイント:
- `build_infer_model` (`tools/sat_hmr_export_utils.py:44`)
  - `build_sat_model(cfg, set_criterion=False)` でSAT有効モデルを構築
  - weightsをロードして `eval()` へ
- `SatHmrOnnxWrapper` (`tools/sat_hmr_export_utils.py:67`)
  - 入力: `image`, `img_size`
  - 出力: `pred_poses`, `pred_betas`, `pred_transl`, `pred_confs`, `pred_boxes`
  - `keep_img_size_input=True` 時は `pred_intrinsics` も出力 (`tools/sat_hmr_export_utils.py:89`)
    - 目的: `img_size` 依存をグラフに残し、最適化で入力から落ちるのを防ぐ

### 2.4 `models/decoder.py`
目的:
- ONNX export時に可変長情報が固定化される影響を減らし、実行安定性を上げる

実装ポイント:
- xFormers利用可否の明示分岐 (`models/decoder.py:18`)
- xFormers未使用時の `torch_attention` フォールバック (`models/decoder.py:343`, `models/decoder.py:346`)
- `block_diag_bias_from_lens` 実装 (`models/decoder.py:97`)
- Cross-Attentionで `single_batch` 判定を追加 (`models/decoder.py:386`)
  - `len(tgt_lens)==1 and len(memory_lens)==1` のとき `attn_bias=None`
  - B=1推論ではblock-diagonal biasを作らない (`models/decoder.py:395`)

## 3. 変換～実行手順

### 3.1 ONNX export + runtime fix
```bash
python tools/export_sat_hmr_onnx.py \
  --weights ./weights/sat_hmr/sat_644.pth \
  --input-size 1288 \
  --img-h 728 --img-w 1288 \
  --device cuda \
  --out datas/onnx/sat_hmr_sat.onnx \
  --apply-runtime-fix
```

生成物:
- `datas/onnx/sat_hmr_sat.onnx`
- `datas/onnx/sat_hmr_sat_runtimefix.onnx`

### 3.2 TensorRT engine build
```bash
mkdir -p datas/trt
/usr/src/tensorrt/bin/trtexec \
  --onnx=datas/onnx/sat_hmr_sat_runtimefix.onnx \
  --saveEngine=/home/kyuta/SAT-HMR/datas/trt/sat_hmr_sat_fp16.engine \
  --fp16 --skipInference
```

### 3.3 TensorRT benchmark
```bash
mkdir -p plots
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=/home/kyuta/SAT-HMR/datas/trt/sat_hmr_sat_fp16.engine \
  --shapes=image:1x3x728x1288 \
  --warmUp=500 --duration=5 --iterations=100 \
  --exportTimes=/home/kyuta/SAT-HMR/plots/trt_sat_fp16_times.json
```

## 4. トラブルと対処履歴

### 4.1 `onnx` 未インストール
症状:
- `No module named 'onnx'`

対処:
- `pip install onnx`

### 4.2 `trtexec` 実行不可 / CUDAドライバ不整合
症状例:
- `CUDA driver version is insufficient for CUDA runtime version`
- `/usr/bin/trtexec` が無い

対処:
- `libnvinfer-bin` 導入
- 実体パスは `/usr/src/tensorrt/bin/trtexec`
- driver / CUDA / TensorRT の整合を取る

### 4.3 TensorRT parse失敗 (`Where` shape error)
症状:
- `broadcast dimensions must be conformable`

対処:
- `patch_where_broadcast` を適用

### 4.4 実行時失敗 (`Where` 実装非対応)
症状:
- `Could not find an implementation for Where...`

対処:
- `rewrite_problematic_bool_where` で算術演算に置換

### 4.5 実行時失敗 (`Split` 固定長不一致)
症状:
- `Cannot split using values in 'split' attribute...`

対処:
- `rewrite_single_output_split` で `Identity` へ置換

### 4.6 エンジン保存失敗
症状:
- `Cannot write to FileStreamWriter`

対処:
- 保存先ディレクトリを先に作成 (`datas/trt`)

### 4.7 `img_size` 入力が消える
症状:
- ONNX入力が `image` のみになる

対処:
- `pred_intrinsics` を出力し、`img_size`依存を維持
- 明示的に消したい場合のみ `--drop-img-size-input`

## 5. 入出力仕様 (現行)

通常 (`--drop-img-size-input` なし):
- 入力:
  - `image: [1, 3, H, W]`
  - `img_size: [2]` (H, W)
- 出力:
  - `pred_poses`, `pred_betas`, `pred_transl`, `pred_confs`, `pred_boxes`, `pred_intrinsics`

`--drop-img-size-input` あり:
- `img_size` が最適化で落ちる可能性あり
- 固定解像度運用向け

## 6. 既知の制約
- 主対象は `B=1`。
- `runtimefix` を前提にしている。
- 量子化(INT8/INT4)はこの実装範囲外（本書はONNX→TensorRT FP16経路）。

## 7. 計測結果の見方 (trtexec)
- `H2D Latency`: Host(CPU) -> Device(GPU) 転送時間
- `GPU Compute Time`: GPU上の純推論時間
- `D2H Latency`: Device(GPU) -> Host(CPU) 転送時間

補足:
- `Throughput (qps)` は batch=1 なら実質FPSに相当
- `Enqueue Time` は外れ値が混ざることがあるため、比較の主軸は `GPU Compute Time` と `Throughput` を推奨




tensorRTとは？：推論エンジンの生成と推論の実行
推論エンジン　trtファイルに書かれる. weightsファイルやネットワークの定義の入ったconfigファイルをまとめてネットワークを様々な工夫により高速化したもの
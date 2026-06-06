# Calibration とは（SAT-HMR / TensorRT INT8）

## 1. 定義
- Calibration は、**INT8量子化で使う scale を実データから決める処理**です。
- 出力は `*.cache`（TensorRTの校正キャッシュ）です。

## 2. 何をしているか
- INT8の表現範囲は `-128..127`。
- FP32値 `x` を INT8値 `q` に写すため、scale `s` が必要。

\[
q=\mathrm{clip}\left(\mathrm{round}\left(\frac{x}{s}\right),-128,127\right),\quad \hat{x}=s\,q
\]

- Calibration は、各テンソルの分布を見て `s` を決める工程です。
- `s` が悪いと、飽和（clamp）や丸め誤差が増えて精度が崩れます。

## 3. 入力と出力
- 入力:
  - ONNXモデル
  - 校正用画像（実運用に近いフレーム）
- 出力:
  - `datas/trt/*.cache`
  - その cache を使って作る INT8 engine

## 4. このリポジトリでの実行

### 4.1 校正フレーム抽出
```bash
python tools/extract_calibration_frames.py \
  --video videos/sample_video.mp4 \
  --outdir datas/calibration \
  --num-frames 500
```

### 4.2 cache + INT8 engine 作成
```bash
python tools/build_trt_int8_calibrated.py \
  --onnx datas/onnx/sat_hmr_sat_l2_softsat_dropis_runtimefix.onnx \
  --engine_out datas/trt/sat_hmr_sat_l2_softsat_int8_calib.engine \
  --calib_manifest datas/calibration/calibration_manifest.csv \
  --cache datas/trt/sat_hmr_sat_l2_int8.cache \
  --batch_size 1 \
  --max_images 500
```

## 5. 重要ポイント
- `--int8` だけでは不十分。CalibrationなしINT8は崩れやすい。
- SAT-HMRは実運用では `INT8 + FP16 fallback`（`--int8 --calib --fp16`）が現実的。
- strict all-INT8（`*:int8`強制）は `Gather/Shape/Cast` 系で失敗しやすい。

## 6. 今回の環境でのチェック項目
- cache が存在するか:
  - `find datas/trt -name '*.cache'`
- engine が 0 byte でないか:
  - `ls -lh datas/trt/*.engine`
- build と run の TensorRT バージョンが一致しているか:
  - build: `trtexec`
  - run: `python -c "import tensorrt as trt; print(trt.__version__)"`

## 7. 典型的な失敗パターン
- `cache` なしでINT8作成: 動くが姿勢が不安定になりやすい。
- `engine` が 0 byte: ビルド失敗。
- TensorRT version mismatch: `deserializeCudaEngine` 失敗。

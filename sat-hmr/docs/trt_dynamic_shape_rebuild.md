# TRT Dynamic Shape Engine Rebuild

## 背景

現在の TRT engine は static shape `(1, 3, 728, 1288)` で固定されている。
3DPW には portrait 画像が含まれるため、eval_3dpw_trt.py での精度評価が正確にできない。

## 手順

### Step 1: ONNX 再エクスポート（dynamic shape 有効）

```bash
conda run -n sathmr python tools/export_sat_hmr_onnx.py \
  --weights weights/sat_hmr/sat_644_3dpw.pth \
  --output datas/onnx/sat_hmr_dynamic.onnx \
  --dynamic-shape
```

### Step 2: TRT engine ビルド

```bash
conda run -n sathmr python tools/build_trt_engine_py.py \
  --onnx datas/onnx/sat_hmr_dynamic.onnx \
  --output datas/trt/sat_hmr_dynamic_fp16.engine \
  --fp16 \
  --shapes "image:1x3x728x1288,img_size:1x2"
```

dynamic shape の場合は min/opt/max profile を指定する必要がある。
3DPW の代表的な image shape 範囲に合わせて設定すること。

## 注意

- engine の再ビルドには数十分かかる
- dynamic shape engine は static より若干遅くなる場合がある

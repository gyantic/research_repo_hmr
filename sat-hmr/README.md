# sat-hmr（SAT-HMR 改変版）

元のリポジトリ: [ChiSu001/SAT-HMR](https://github.com/ChiSu001/SAT-HMR) @ `7b157fa` のコードに本研究の改変を加えたもの。
元のリポジトリの README は [README_UPSTREAM.md](README_UPSTREAM.md) に保存。

## 主な独自追加
- **デモ／UE 連携**: `video_demo.py`（Torch 基準・SMPL 姿勢を UE へ UDP 送信＋描画）、`video_demo_onnx.py`（ONNX Runtime）、`video_demo_trt.py`（TensorRT）、`webcam_demo.py` / `webcam_demo_trt.py`（Web カメラ実時間）。
- **速度ベンチ**: `sathmr_tcp_benchmark.py` / `sathmr_udp_benchmark.py`。
- **int8 量子化中核**: `models/int8_attention.py` `static_int8_fused.py`（`grouped_linear.py` `triton_grouped_int8.py` は `tools/benchmark_grouped_bbox_int8.py` 用）。
- **共通ヘルパー**: `sathmr_common.py`（前処理・SMPL→UE 変換・UDP 追跡などデモ／ベンチマーク共通）、`trt_runner.py`（TensorRT 実行）。
- **ツール群** `tools/`（量子化・TensorRT/ONNX エクスポート・キャリブレーション・3DPW 評価・transport ベンチ 等）と一括実行 `run_*.sh`。
- **設定**: `configs/run/eval_3dpw_*.yaml`・`train_qat.yaml`。

## 環境
conda 環境ファイル [environment.yml](environment.yml)（env 名 `sathmr-copy`）を同梱。
作成: `conda env create -f environment.yml` → `conda activate sathmr-copy`。

## 実行の準備
- SAT-HMR 重み（例 `weights/sat_hmr/sat_644*.pth`）・DINOv2 系バックボーン・データセット（3DPW 等）。元のリポジトリの手順で取得し配置。
- 論文の SAT-HMR 速度/精度は `video_demo.py` で計測。

## 動画推論

```bash
conda activate sathmr-copy
cd sat-hmr
python video_demo.py \
  --input /path/to/input.mp4 \
  --weights weights/sat_hmr/sat_644.pth \
  --no_render --no_save
```

`--no_render --no_save` は描画・動画保存を省略します。量子化、コンパイル、UDP配信などの条件は `python video_demo.py --help` で確認してください。

評価コードは [measurements](../measurements/README.md)、研究全体の構成は [ルートREADME](../README.md) を参照してください。

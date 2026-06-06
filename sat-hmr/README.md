# sat-hmr（SAT-HMR 改変版）

上流: [ChiSu001/SAT-HMR](https://github.com/ChiSu001/SAT-HMR) @ `7b157fa` のコードに本研究の改変を加えたもの。
上流の README は [README_UPSTREAM.md](README_UPSTREAM.md) に保存。

## 主な独自追加
- **デモ／UE 連携**: `video_demo.py`（Torch 基準・SMPL 姿勢を UE へ UDP 送信＋描画）、`video_demo_onnx.py`（ONNX Runtime）、`video_demo_trt.py`（TensorRT）、`webcam_demo.py` / `webcam_demo_trt.py`（Web カメラ実時間）。
- **速度ベンチ**: `sathmr_tcp_benchmark.py` / `sathmr_udp_benchmark.py`。
- **int8 量子化中核**: `models/int8_modules.py` `int8_attention.py` `grouped_linear.py` `grouped_bbox_embed.py` `triton_grouped_int8.py` `static_int8_fused.py`。
- **ツール群** `tools/`（量子化・TensorRT/ONNX エクスポート・キャリブレーション・3DPW 評価・プロット・transport ベンチ 等）と一括実行 `run_*.sh`。
- **設定 / ドキュメント**: `configs/run/eval_3dpw_*.yaml`・`train_qat.yaml`、`docs/`（量子化計画・int8 仕様・TensorRT 等）。

## 環境
conda 環境ファイル [environment.yml](environment.yml)（env 名 `sathmr-copy`）を同梱。
作成: `conda env create -f environment.yml` → `conda activate sathmr-copy`。

## 実行に必要（リポジトリ非収録）
- SAT-HMR 重み（例 `weights/sat_hmr/sat_644*.pth`）・DINOv2 系バックボーン・データセット（3DPW 等）。上流の手順で取得し配置。
- 論文の SAT-HMR 速度/精度は `video_demo.py` で計測。

## 除外したもの
重み (`weights/`)・データセット (`datasets/` `datas/`)・venv・各種出力 (`plots/` `outputs/` `*.nsys-rep` `*.sqlite` `*_kern.txt`)・動画。
demo に内蔵された UE 送信機能と重複する `udp_communication_UE*.py`、作者が「不要」と明記した `inspect_torchao_runtime.py`、
一時/調査用ツール（`resolve_int8_slowdown.py` `record_token_lens.py` `compare_model_sizes.py`）は整理時に除外。

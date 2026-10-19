# fast-sam-3d-body（Fast-SAM-3D-Body 改変版）

元のリポジトリ: [yangtiming/Fast-SAM-3D-Body](https://github.com/yangtiming/Fast-SAM-3D-Body) @ `49b0fe0` のコードに本研究の改変を加えたもの。
元のリポジトリの README は [README_UPSTREAM.md](README_UPSTREAM.md) に保存。

## 主な独自追加・改変
- `run_publisher_ue.py` — リアルタイム SMPL パブリッシャ。ZMQ 配信に加え Unreal Engine へ UDP/TCP 送信。int8/TensorRT バックボーン・CUDA Graph・compile を環境変数で制御。論文の FastSAM int8_trt/UE 速度で使用。
- `mocap/realtime/ue_sender.py` — `run_publisher_ue.py` が使う UE 送信モジュール。
- `check_int8_layers.py` — TRT エンジンの各テンソル dtype（INT8/FP16/FP32）を検査。
- `profile_components.py` — forward の各コンポーネント別タイミング計測。
- TensorRT 変換まわりの改変: `build_tensorrt.sh`、`convert_backbone_tensorrt.py`、`convert_moge_encoder_trt.py`、`convert_yolo_pose_trt.py`、`sam_3d_body/models/backbones/dinov3_tensorrt.py` 等。
- `mhr2smpl/multi_view/step0_infer_3dpw.py` — 3DPW 推論サンプル (sample_*.npz) 生成（FastSAM 精度評価の入力）。

## 環境
conda 環境ファイル [environment.yml](environment.yml)（env 名 `sam_3d_body`）を同梱。
作成: `conda env create -f environment.yml` → `conda activate sam_3d_body`。

## 実行の準備
- `checkpoints/`（SAM-3D-Body / DINOv3 / MoGe / YOLO 重み・TensorRT エンジン）、MHR モデル、3DPW データ。公式配布元から取得し配置。
- 精度評価は [../measurements/fastsam/evaluate_3dpw.py](../measurements/fastsam/evaluate_3dpw.py) で実施。

## 推論と評価

UEへの配信には `run_publisher_ue.py` を使用します。SMPLモデルは既定でリポジトリ直下の `body_models/smpl/SMPL_NEUTRAL.pkl` を使います。MHR→SMPL変換モデル、対応表などの指定は `python run_publisher_ue.py --help` で確認してください。

速度計測には [../measurements/fastsam/benchmark_fastsam3dbody.py](../measurements/fastsam/benchmark_fastsam3dbody.py) を使用します。論文の計測は [../measurements/fastsam/run.sh](../measurements/fastsam/run.sh) でまとめて実行できます。`--repo` にこのモデルのディレクトリ、`--video` に入力動画、`--output_csv` に出力先を指定します。出力先は作業ディレクトリ変更の影響を受けない絶対パスを指定してください。

TensorRTエンジンは使用するGPU・実行環境に合わせて用意します。変換には、このディレクトリのエクスポート・エンジン生成スクリプトを使用します。

精度・速度評価は [measurements](../measurements/README.md)、研究全体の構成は [ルートREADME](../README.md) を参照してください。

# fast-sam-3d-body（Fast-SAM-3D-Body 改変版）

本ディレクトリは、[yangtiming/Fast-SAM-3D-Body](https://github.com/yangtiming/Fast-SAM-3D-Body)（コミット `49b0fe0`）をベースに、本研究向けの機能追加・改変を加えたものです。元のリポジトリのREADMEは [README_UPSTREAM.md](README_UPSTREAM.md) に保存しています。

研究全体の構成については[ルートREADME](../README.md)、論文の精度・速度評価については [measurements/README.md](../measurements/README.md) を参照してください。

---

## 主な独自追加・改変

| ファイル | 内容 |
|---|---|
| [run_publisher_ue.py](run_publisher_ue.py) | リアルタイムSMPLパブリッシャ。ZMQ配信に加え、Unreal EngineへのUDP/TCP送信に対応。INT8/TensorRTバックボーン・CUDA Graph・`torch.compile` を環境変数で制御（論文のFast-SAM-3D-Body INT8 TensorRTおよびUE連携時の速度計測で使用） |
| `mocap/realtime/ue_sender.py` | `run_publisher_ue.py` が使用するUE通信モジュール |
| `build_tensorrt.sh`<br>`convert_backbone_tensorrt.py`<br>`convert_moge_encoder_trt.py`<br>`convert_yolo_pose_trt.py`<br>`sam_3d_body/models/backbones/dinov3_tensorrt.py` | TensorRT変換スクリプトおよびランタイム実装まわりの改変 |
| [mhr2smpl/multi_view/step0_infer_3dpw.py](mhr2smpl/multi_view/step0_infer_3dpw.py) | 3DPW推論サンプル（`sample_*.npz`）の生成スクリプト（Fast-SAM-3D-Bodyの精度評価の入力データを作成） |

---

## セットアップ

詳細な手順は、ルートREADMEの [1. 環境の準備](../README.md#環境の準備) および [2. モデルの取得と配置](../README.md#モデルの取得と配置) を参照してください。

### 1. Conda環境の作成

同梱の [environment.yml](environment.yml) から Conda 環境（環境名：`sam_3d_body`）を作成し、`chumpy` と `detectron2` を追加インストールしたうえで、`chumpy` の Python 3.11 互換修正（[../sat-hmr/docs/fix_chumpy.md](../sat-hmr/docs/fix_chumpy.md)）を適用します。

```bash
conda env create -f environment.yml
conda activate sam_3d_body
python -m pip install --no-build-isolation chumpy==0.70
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git@a1ce2f956a1d2212ad672e3c47d53405c2fe4312"
```

### 2. モデル重み・データセットの準備

- **公式の重みと変換用データ**：`checkpoints/` 配下（SAM 3D Body / DINOv3 / MoGe / YOLO11 Pose の重み）、MHRモデル、MHR→SMPL変換モデル（`mhr2smpl/`）、SMPLモデル（`body_models/smpl/`）、および3DPWデータセットを各公式配布元から取得して配置します（配置先は[ルートREADMEの手順2.2](../README.md#22-各モデルの公式重みの配置)を参照）。
- **量子化重み・TensorRTエンジン**：論文で使用したTensorRTエンジンとbf16の重みは、リポジトリのルートで以下を実行すると所定のパスへ取得・配置されます。
  ```bash
  python download_weights.py fastsam
  ```
  > **Note:** TensorRTエンジン（`.engine`）は作成したGPUおよびTensorRTのバージョンに依存します。実行環境が異なる場合は、後述の[TensorRTエンジンの作成](#tensorrtエンジンの作成)の手順で作成し直してください。

---

## 推論と評価の実行

### Unreal Engineへのリアルタイム配信

UEへの姿勢配信には `run_publisher_ue.py` を使用します。SMPLモデルは既定でリポジトリ直下の `body_models/smpl/SMPL_NEUTRAL.pkl` を参照します。MHR→SMPL変換モデルや対応表（マッピングデータ）などの引数指定については、以下で確認してください。

```bash
conda activate sam_3d_body
cd fast-sam-3d-body
python run_publisher_ue.py --help
```

### 速度計測・精度評価

- **速度計測**：[../measurements/fastsam/benchmark_fastsam3dbody.py](../measurements/fastsam/benchmark_fastsam3dbody.py) を使用します。`--repo` にこのディレクトリ（`fast-sam-3d-body`）、`--video` に入力動画、`--output_csv` に出力先を指定します（`--output_csv` には作業ディレクトリ変更の影響を受けない**絶対パス**を指定してください）。
- **精度評価**：`mhr2smpl/multi_view/step0_infer_3dpw.py` で3DPWの推論サンプル（`sample_*.npz`）を生成したのち、[../measurements/fastsam/evaluate_3dpw.py](../measurements/fastsam/evaluate_3dpw.py) で評価指標を算出します。
- **一括実行**：論文の計測（速度・精度・UE連携時のUDP送信）は、リポジトリのルートから [../measurements/fastsam/run.sh](../measurements/fastsam/run.sh) でまとめて実行できます。

---

<a id="tensorrtエンジンの作成"></a>

## TensorRTエンジンの作成

論文の構成（YOLO11 PoseはFP16、MoGe-2エンコーダーとDINOv3バックボーンはINT8）のTensorRTエンジンを、使用するGPU環境でビルドする手順です。

INT8キャリブレーション（較正）には `datasets/videos/sample_video.mp4` を使用します（[ルートREADMEの手順3.2](../README.md#速度計測の入力動画の配置)に従って配置してください）。`sam_3d_body` 環境を有効化し、このディレクトリ（`fast-sam-3d-body/`）から以下を実行します。

```bash
conda activate sam_3d_body
cd fast-sam-3d-body

python convert_yolo_pose_trt.py --model ./checkpoints/yolo/yolo11m-pose.pt --imgsz 640 --half
python convert_moge_encoder_trt.py --export_onnx
python convert_moge_encoder_trt.py --convert_trt --int8
python convert_backbone_tensorrt.py --quantize_onnx --convert_trt --int8
```

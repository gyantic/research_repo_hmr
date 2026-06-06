# fast-sam-3d-body（Fast-SAM-3D-Body 改変版）

上流: [yangtiming/Fast-SAM-3D-Body](https://github.com/yangtiming/Fast-SAM-3D-Body) @ `49b0fe0` のコードに本研究の改変を加えたもの。
上流の README は [README_UPSTREAM.md](README_UPSTREAM.md) に保存。

## 主な独自追加・改変
- `run_publisher_ue.py` — リアルタイム SMPL パブリッシャ。ZMQ 配信に加え Unreal Engine へ UDP/TCP 送信。int8/TensorRT バックボーン・CUDA Graph・compile を環境変数で制御。論文の FastSAM int8_trt/UE 速度で使用。
- `mocap/realtime/ue_sender.py` — `run_publisher_ue.py` が使う UE 送信モジュール。
- `check_int8_layers.py` — TRT エンジンの各テンソル dtype（INT8/FP16/FP32）を検査。
- `profile_components.py` — forward の各コンポーネント別タイミング計測。
- TensorRT 変換まわりの改変: `build_tensorrt.sh`、`convert_backbone_tensorrt.py`、`convert_moge_encoder_trt.py`、`convert_yolo_pose_trt.py`、`sam_3d_body/models/backbones/dinov3_tensorrt.py` 等。
- `mhr2smpl/multi_view/step0_infer_3dpw.py` — 3DPW 推論サンプル (sample_*.npz) 生成（FastSAM 精度評価の入力）。

## 実行に必要（リポジトリ非収録）
- `checkpoints/`（SAM-3D-Body / DINOv3 / MoGe / YOLO 重み・TensorRT エンジン）、MHR モデル、3DPW データ。上流・公式配布元から取得し配置。
- 本研究では conda 環境 `sam_3d_body`。精度評価は [../measurements/](../measurements/) の評価スクリプトで実施。

## 除外したもの
重み・チェックポイント・TensorRT エンジン（`checkpoints/` 配下。ただしユーザー改変の `model_config.yaml` 等の設定/JSON は保持）、
`mhr2smpl/experiments/*.pth` 等の学習済みモデル・インデックス (`*.npy/*.npz`)、データセット (`datasets/coco8-pose` 等)・`data/`・動画。

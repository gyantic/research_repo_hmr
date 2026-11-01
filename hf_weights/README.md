---
license: other
license_name: mixed
license_link: https://huggingface.co/kimurakmkm123/research_repo_hmr/tree/main/licenses
tags:
- human-mesh-recovery
- quantization
- tensorrt
---

# research_repo_sii2027 の作成済み重み

ROMP・SAT-HMR・Fast-SAM-3D-Body の公式重みを、bf16 化・INT8 量子化・TensorRT エンジン化したものです。論文の計測で使用した重みのみを置いています。

## 使い方

コードリポジトリのルートで次を実行すると、各モデルが読み込む場所に配置されます。

```bash
python download_weights.py            # 全モデル
python download_weights.py sathmr     # モデルを指定（sathmr / romp / fastsam）
```

元の公式重みと SMPL モデルは含みません。それぞれの配布元から取得してください。

## ファイル

| ファイル | 元の重み | 変更内容 | ライセンス |
|---|---|---|---|
| `sat-hmr/weights/sat_hmr/sat_644_bf16.pth` | SAT-HMR `sat_644.pth` | 全テンソルを bf16 に変換 | Apache-2.0 |
| `sat-hmr/weights/sat_hmr/sat_644_int8w.pth` | SAT-HMR `sat_644.pth` | 線形層の重みを INT8 に量子化 | Apache-2.0 |
| `sat-hmr/weights/sat_hmr/sat_644_3dpw_int8w.pth` | SAT-HMR `sat_644_3dpw.pth` | 線形層の重みを INT8 に量子化 | Apache-2.0 |
| `romp/trained_models/ROMP_HRNet32_V1_bf16.pth` | ROMP `ROMP_HRNet32_V1.pkl` | bf16 に変換 | Apache-2.0 |
| `romp/trained_models/ROMP_ResNet50_V1_bf16.pth` | ROMP `ROMP_ResNet50_V1.pkl` | bf16 に変換 | Apache-2.0 |
| `romp/trained_models/ROMP_HRNet32_V1_hrnet32_backbone_int8wo.pth` | ROMP `ROMP_HRNet32_V1.pkl` | バックボーンを INT8 weight-only 量子化 | Apache-2.0 |
| `romp/trained_models/ROMP_ResNet50_V1_int8_raw_weights.pth` | ROMP `ROMP_ResNet50_V1.pkl` | INT8 weight-only 量子化 | Apache-2.0 |
| `fast-sam-3d-body/checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_int8.engine` | SAM 3D Body `model.ckpt`（DINOv3 バックボーン） | TensorRT Model Optimizer で INT8 化した TensorRT エンジン | SAM License |
| `fast-sam-3d-body/checkpoints/bf16/model_bf16.ckpt` | SAM 3D Body `model.ckpt` | fp32 テンソルを bf16 に変換 | SAM License |
| `fast-sam-3d-body/checkpoints/bf16/mhr_model_bf16.pt` | MHR `mhr_model.pt` | fp32 テンソルを bf16 に変換 | Apache-2.0 |
| `fast-sam-3d-body/checkpoints/moge_trt/moge_dinov2_encoder_int8.engine` | MoGe-2 ViT-S normal `model.pt`（エンコーダー） | エントロピー較正で INT8 化した TensorRT エンジン | MIT |
| `fast-sam-3d-body/checkpoints/bf16/model_bf16.pt` | MoGe-2 ViT-S normal `model.pt` | fp32 テンソルを bf16 に変換 | MIT |
| `fast-sam-3d-body/checkpoints/yolo/yolo11m-pose.engine` | Ultralytics `yolo11m-pose.pt` | FP16 の TensorRT エンジンに変換 | AGPL-3.0 |

ROMP の INT8 の2ファイルは、論文ではモデルサイズの報告のみに使用しています。

TensorRT エンジンは NVIDIA RTX 4070・TensorRT 10.13 で生成しました。GPU や TensorRT のバージョンが異なる環境では読み込めないため、コードリポジトリの変換スクリプトで生成し直してください。

## ライセンス

各ファイルは、元の重みのライセンスに従います。ライセンス文は `licenses/` にあります。

| ライセンス | 対象 | ライセンス文 |
|---|---|---|
| Apache-2.0 | SAT-HMR の重み | `licenses/Apache-2.0_SAT-HMR.txt` |
| Apache-2.0 | ROMP の重み | `licenses/Apache-2.0_ROMP.txt` |
| SAM License | SAM 3D Body の重み | `licenses/SAM_License.txt` |
| Apache-2.0 | MHR の重み | `licenses/Apache-2.0_MHR.txt` |
| MIT | MoGe-2 の重み | `licenses/MIT_MoGe.txt` |
| AGPL-3.0 | YOLO11 の重み | `licenses/AGPL-3.0_Ultralytics.txt` |

SAM 3D Body に由来するファイルは SAM License に従って配布します。これらを使った研究結果を発表する場合は、SAM Materials を使用したことを明記してください。

## 元の重みの配布元

- SAT-HMR: [ChiSu001/SAT-HMR](https://huggingface.co/ChiSu001/SAT-HMR)
- ROMP: [Arthur151/ROMP](https://github.com/Arthur151/ROMP)
- SAM 3D Body: [facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3)
- MHR: [facebookresearch/MHR](https://github.com/facebookresearch/MHR)
- MoGe-2: [Ruicheng/moge-2-vits-normal](https://huggingface.co/Ruicheng/moge-2-vits-normal)
- YOLO11: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

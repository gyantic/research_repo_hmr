# romp（ROMP 改変版）

本ディレクトリは、[Arthur151/ROMP](https://github.com/Arthur151/ROMP)（コミット `a8558ae`）をベースに、本研究向けの機能追加・改変を加えたものです。元のリポジトリのREADMEは [README_UPSTREAM.md](README_UPSTREAM.md) に保存しています。

研究全体の構成については[ルートREADME](../README.md)、論文の精度・速度評価については [measurements/README.md](../measurements/README.md) を参照してください。

---

## 主な独自追加・改変

| ファイル | 内容 |
|---|---|
| [romp_video_demo.py](romp_video_demo.py) | 動画入力のベンチマーク用デモスクリプト（`--metrics_csv` でフレームごとの推論速度を出力）。論文のROMP速度評価で使用 |
| [tools/export_quantized_weights_romp.py](tools/export_quantized_weights_romp.py) | 量子化重み（bf16 / INT8）のエクスポート用スクリプト |

---

## セットアップ

詳細な手順は、ルートREADMEの [1. 環境の準備](../README.md#環境の準備) および [2. モデルの取得と配置](../README.md#モデルの取得と配置) を参照してください。

### 1. Conda環境の作成

同梱の [environment.yml](environment.yml) から Conda 環境（環境名：`romp_3.9`）を作成し、`pytorch3d` を追加インストールします。

```bash
conda env create -f environment.yml
conda activate romp_3.9
CUDA_HOME=/usr/local/cuda-11.8 FORCE_CUDA=1 python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@0a59450f0ebbe12d9a8db3de937814932517633b"
```

### 2. モデル重み・SMPLの準備

- **ROMPの学習済みモデル**：公式の学習済みモデル（`ROMP_HRNet32_V1.pkl`、`ROMP_ResNet50_V1.pkl`）を `trained_models/` に配置します。また、論文で使用したbf16・INT8の重みは、リポジトリのルートで `python download_weights.py romp` を実行すると `trained_models/` に取得・配置されます。
- **SMPLモデル**：リポジトリ直下の `body_models/smpl/` を共用します。ROMPが読み込む `SMPL_NEUTRAL.pth`（および3DPW評価用の `SMPL_FEMALE.pth`・`SMPL_MALE.pth`）も、[ルートREADMEの手順2.1](../README.md#21-smplモデルと補助データの配置)に従って `body_models/smpl/` 内に生成してください。
- **実行設定ファイル**：速度評価の実行設定（バックボーン・数値精度・`torch.compile` の有無）は、[../measurements/romp/configs/](../measurements/romp/configs/) 内の `romp_video_*.yml` および `romp_resnet_video_*.yml` にまとまっています。

---

## 動画推論の実行

入力には、使用環境の OpenCV で読み込める動画形式を用意してください。使用するモデルやコンパイルの有無などの条件は `measurements/romp/configs/` のYAMLファイル（`--configs_yml`）で指定します。数値精度だけはコマンドライン引数 `--dtype_cfg`（`fp32` / `bf16`、既定は `fp32`）で指定し、YAMLの `model_precision` より優先されます。

```bash
conda activate romp_3.9
cd romp
PYTHONPATH="$PWD:$PWD/romp" PYOPENGL_PLATFORM=egl python romp_video_demo.py \
  --input /path/to/input.mp4 \
  --configs_yml ../measurements/romp/configs/romp_video_fp32_no_compile.yml \
  --no_render --no_save
```

- `--no_render --no_save` を指定すると、描画および出力動画の保存を省略します。
- `--metrics_csv <出力パス>` を指定すると、フレームごとの推論時間をCSVに出力します。

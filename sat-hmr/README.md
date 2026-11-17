# sat-hmr（SAT-HMR 改変版）

本ディレクトリは、[ChiSu001/SAT-HMR](https://github.com/ChiSu001/SAT-HMR)（コミット `7b157fa`）をベースに、本研究向けの機能追加・改変を加えたものです。元のリポジトリのREADMEは [README_UPSTREAM.md](README_UPSTREAM.md) に保存しています。

研究全体の構成については[ルートREADME](../README.md)、論文の精度・速度評価については [measurements/README.md](../measurements/README.md) を参照してください。

---

## 主な独自追加・改変

| カテゴリ | ファイル・ディレクトリ | 内容 |
|---|---|---|
| **デモ・UE連携** | [video_demo.py](video_demo.py) | PyTorchによる動画推論、および推定したSMPL姿勢のUnreal EngineへのUDP送信・描画（論文のSAT-HMR速度・精度計測でも使用） |
| | `video_demo_trt.py` | TensorRTによる動画推論デモ |
| | `webcam_demo_trt.py` | Webカメラを用いた実時間推論デモ（TensorRT） |
| **INT8量子化中核** | `models/int8_attention.py`<br>`models/static_int8_fused.py` | INT8量子化のコア実装 |
| **共通ヘルパー** | `sathmr_common.py` | 前処理、SMPL→UE座標・姿勢変換、UDP通信・追跡などデモおよびベンチマークの共通処理 |
| | `trt_runner.py` | TensorRTエンジンの実行ラッパー |
| **ツール群** | [tools/](tools/) | 量子化、TensorRT / ONNXエクスポート、キャリブレーション、TensorRTでの3DPW評価スクリプト群 |
| **設定ファイル** | `configs/run/eval_3dpw_*.yaml`<br>`configs/run/train_qat.yaml` | 3DPW精度評価およびQAT（量子化認識学習）用の設定ファイル |

---

## セットアップ

詳細な手順は、ルートREADMEの [1. 環境の準備](../README.md#環境の準備) および [2. モデルの取得と配置](../README.md#モデルの取得と配置) を参照してください。

### 1. Conda環境の作成

同梱の [environment.yml](environment.yml) から Conda 環境（環境名：`sathmr-copy`）を作成し、`chumpy` のインストールおよび Python 3.11 互換修正（[docs/fix_chumpy.md](docs/fix_chumpy.md)）を適用します。

```bash
conda env create -f environment.yml
conda activate sathmr-copy
python -m pip install --no-build-isolation chumpy==0.70
```

### 2. モデル重み・データセットの準備

- **公式の重みとデータセット**：SAT-HMRの学習済み重み（`weights/sat_hmr/sat_644.pth`、`sat_644_3dpw.pth`）、DINOv2系バックボーン、SMPLモデルおよび補助データ、評価用データセット（3DPWなど）を元のリポジトリおよび[ルートREADMEの手順](../README.md#モデルの取得と配置)に従って配置します。
- **量子化重み（bf16・INT8）**：論文で使用した `sat_644_bf16.pth`・`sat_644_int8w.pth`・`sat_644_3dpw_int8w.pth` は、リポジトリのルートで以下を実行すると `weights/sat_hmr/` に取得・配置されます。
  ```bash
  python download_weights.py sathmr
  ```

---

## 動画推論の実行

論文におけるSAT-HMRの推論速度は `video_demo.py` で計測しています。

```bash
conda activate sathmr-copy
cd sat-hmr
python video_demo.py \
  --input /path/to/input.mp4 \
  --weights weights/sat_hmr/sat_644.pth \
  --no_render --no_save
```

- `--no_render --no_save` を指定すると、描画および出力動画の保存を省略します。
- 量子化（bf16 / INT8）、`torch.compile` の適用範囲、Unreal EngineへのUDP配信などのオプションは `python video_demo.py --help` で確認できます。

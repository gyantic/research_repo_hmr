# research_repo_sii2027

3D 人体メッシュ推定 3 手法（ROMP / SAT-HMR / Fast-SAM-3D-Body）の改変コードと、
それらの精度・速度評価コードを 1 リポジトリに集約したもの。リアルタイム量子化推論
（bf16 / int8 / TensorRT）と Unreal Engine への姿勢ストリーミングを中心に拡張している。

> **コードのみのリポジトリです。** データセット・学習済み重み・評価出力 (CSV/動画/ログ) は
> 含みません（`.gitignore` で除外）。各上流リポジトリと公式配布元から別途取得してください。

## 構成

| ディレクトリ | 内容 | 上流 |
|---|---|---|
| [romp/](romp/) | ROMP の改変版（UE 骨格ストリーミング、量子化エクスポート 等） | [Arthur151/ROMP](https://github.com/Arthur151/ROMP) @ `a8558ae` |
| [sat-hmr/](sat-hmr/) | SAT-HMR の改変版（int8 量子化・ONNX/TensorRT・UE 連携・各種ツール） | [ChiSu001/SAT-HMR](https://github.com/ChiSu001/SAT-HMR) @ `7b157fa` |
| [fast-sam-3d-body/](fast-sam-3d-body/) | Fast-SAM-3D-Body の改変版（TensorRT int8・リアルタイム UE 配信） | [yangtiming/Fast-SAM-3D-Body](https://github.com/yangtiming/Fast-SAM-3D-Body) @ `49b0fe0` |
| [measurements/](measurements/) | 論文の評価表 (精度・速度) を出した評価/集計コードと ROMP 実行設定 | 本研究 |

各サブディレクトリの `README.md` に、上流からの主な変更点・実行に必要な
重み/データの入手先・除外したものを記載しています。

## 除外しているもの（リポジトリに含まれない）

- データセット（3DPW, AGORA, COCO, MPII, H36M 等）、`datasets/` `datas/` `data/`
- 学習済み重み・チェックポイント（`*.pth` `*.pt` `*.onnx` `*.engine` 等）、`weights/` `checkpoints/`
- Python 仮想環境（`venv*/`）・ビルド生成物（`*.so` `__pycache__/`）
- 評価の生出力（`raw_*.csv` 等の CSV・`*.log`・`*.bak`・各種出力ディレクトリ）・動画
- `inspect_torchao_runtime.py` 等、作者が「不要」と明記/確認した一時スクリプト

## セットアップ概要

各モデルは元リポジトリと同じ環境で動きます（本研究では conda 環境
`romp_3.9` / `sathmr-copy` / `sam_3d_body` を使用）。

1. 対応する上流リポジトリの手順に従い依存関係をインストール
2. 公式配布元から重み・SMPL/MHR モデル・データセットを取得し、各 README 記載の場所へ配置
3. 各サブディレクトリのスクリプトを実行

## 評価の再現

[measurements/](measurements/) を参照。論文の評価表 (`all_model_results` 相当) の各行は、
そこに置いた評価/集計コードと各モデルの推論スクリプトで生成しています。

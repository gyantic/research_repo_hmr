# romp（ROMP 改変版）

元のリポジトリ: [Arthur151/ROMP](https://github.com/Arthur151/ROMP) @ `a8558ae` のコードに本研究の改変を加えたもの。
元のリポジトリの README は [README_UPSTREAM.md](README_UPSTREAM.md) に保存。

## 主な独自追加
- `romp/generator_bonebasedMesh.py` — SMPL pose を関節クォータニオンへ変換し **TCP** で Unreal Engine へ送信。推論時間の統計 CSV も出力。
- `romp/generator_bonebasedMesh_UDPver.py` — 上記の **UDP** 版（ffmpeg で 12 fps に揃えて入力）。
- `romp_video_demo.py` — 動画入力のベンチマーク用デモ（`--metrics_csv` で per-frame 速度を出力）。論文の ROMP 速度評価で使用。
- `tools/export_torchao_int8_backbone_romp.py` / `export_quantized_weights_romp.py` — int8 量子化重みのエクスポート。

## 環境
conda 環境ファイル [environment.yml](environment.yml)（env 名 `romp_3.9`）を同梱。
作成: `conda env create -f environment.yml` → `conda activate romp_3.9`。

## 実行の準備
- ROMP の学習済みモデル（`trained_models/` へ配置）。SMPL はリポジトリ直下の `body_models/smpl/` を共用し、ROMP 用の `SMPL_NEUTRAL.pth` もそこへ生成する（手順はルート README）。
- 速度評価の実行設定は [../measurements/romp/configs/](../measurements/romp/configs/) の `romp_video_*.yml` / `romp_resnet_video_*.yml`。

## 動画推論

```bash
conda activate romp_3.9
cd romp
PYTHONPATH="$PWD:$PWD/romp" PYOPENGL_PLATFORM=egl python romp_video_demo.py \
  --input /path/to/input.mp4 \
  --configs_yml ../measurements/romp/configs/romp_video_fp32_no_compile.yml \
  --no_render --no_save
```

入力は使用環境のOpenCVで読み込める動画形式を用意してください。モデル、数値精度、コンパイルの条件は `measurements/romp/configs/` のYAMLとコマンド引数で指定します。

評価コードは [measurements](../measurements/README.md)、研究全体の構成は [ルートREADME](../README.md) を参照してください。

# romp（ROMP 改変版）

上流: [Arthur151/ROMP](https://github.com/Arthur151/ROMP) @ `a8558ae` のコードに本研究の改変を加えたもの。
上流の README は [README_UPSTREAM.md](README_UPSTREAM.md) に保存。

## 主な独自追加
- `romp/generator_bonebasedMesh.py` — SMPL pose を関節クォータニオンへ変換し **TCP** で Unreal Engine へ送信。推論時間の統計 CSV も出力。
- `romp/generator_bonebasedMesh_UDPver.py` — 上記の **UDP** 版（ffmpeg 給餌・再現性固定・SCSC2 ヘッダ付き）。
- `romp/make_clothmap.py` / `make_proxy_obj.py` / `make_smpl_face.py` — 衣装スキンウェイト/プロキシ OBJ/SMPL face などの補助アセットを一度だけ生成。
- `romp_video_demo.py` — 動画入力のベンチマーク用デモ（`--metrics_csv` で per-frame 速度を出力）。論文の ROMP 速度評価で使用。
- `tools/export_torchao_int8_backbone_romp.py` / `export_quantized_weights_romp.py` — int8 量子化重みのエクスポート。

## 環境
conda 環境ファイル [environment.yml](environment.yml)（env 名 `romp_3.9`）を同梱。
作成: `conda env create -f environment.yml` → `conda activate romp_3.9`。

## 実行に必要（リポジトリ非収録）
- ROMP の学習済みモデルと SMPL モデルデータ（上流の手順で取得し `~/.romp/` 等へ配置）。
- 速度評価の実行設定は [../measurements/](../measurements/) の `romp_video_*.yml` / `romp_resnet_video_*.yml`。

## 除外したもの
学習済み重み (`trained_models/` 等)・SMPL データ・デモ用動画/画像・各種出力。
頂点のみ送信していた旧ストリーミング試作（generate_mesh / generator_romp / mesh_generate_toUE）や
描画クライアント (renderer*)・動作確認用スクラッチは整理時に除外。

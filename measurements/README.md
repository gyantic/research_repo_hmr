# measurements（評価・計測コード）

論文の評価表（精度・速度、`all_model_results` / `paper_results` 相当）の各行を生成した
評価/集計コードと、ROMP 実行設定のみを収録。**評価結果の CSV 自体は含めません**（データのため除外）。
どのコードがどの行を出したかは、元の `all_model_results.csv` の `command` 列に基づき特定しています。

## 収録物

| ファイル | 役割 | 生成する論文行 |
|---|---|---|
| `benchmark_fastsam3dbody.py` | Fast-SAM-3D-Body の速度計測（`--output_csv` で per-frame） | FastSAM 速度 |
| `evaluate_fastsam3dbody_3dpw_paper_repro.py` | Fast-SAM-3D-Body の 3DPW 精度（MPJPE/PA-MPJPE/PVE） | FastSAM 精度 |
| `summarize_inference_stats.py` | 各モデルの per-frame 速度 CSV を集計しサマリ CSV を生成 | 速度サマリ全体 |
| `romp_video_*.yml`, `romp_resnet_video_*.yml` | ROMP 速度計測の実行設定（`romp_video_demo.py --configs_yml`） | ROMP 速度 |

## 他モデルの計測コードの所在
- ROMP 速度: [../romp/romp_video_demo.py](../romp/romp_video_demo.py)（本フォルダの yml を使用）
- SAT-HMR 速度: [../sat-hmr/video_demo.py](../sat-hmr/video_demo.py)
- FastSAM 速度(int8_trt/UE): [../fast-sam-3d-body/run_publisher_ue.py](../fast-sam-3d-body/run_publisher_ue.py)
- FastSAM 精度の入力 (sample_*.npz): [../fast-sam-3d-body/mhr2smpl/multi_view/step0_infer_3dpw.py](../fast-sam-3d-body/mhr2smpl/multi_view/step0_infer_3dpw.py)

## 除外したもの
評価の生出力・サマリ CSV（`raw_*.csv` / `all_model_results.csv` / `paper_results*.csv` 等）、ログ・`.bak`・
評価出力ディレクトリ。論文表に対応しない探索的な評価/診断スクリプト（`evaluate_*` の他バリアント・
`diagnose_*`・`analyze_*`・`stream_*` 等）も command 列に現れないため非収録。

> 注: `paper_results` の `robust_*`（median/IQR/MAD）列はスクリプト化されておらず（該当コード無し）、
> アドホックに算出されたため再現コードは収録できていません。

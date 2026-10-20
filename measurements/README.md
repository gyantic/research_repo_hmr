# measurements（論文の精度・速度評価）

論文の計測を実行するスクリプト・設定と、集計・図の生成コードを、モデルごとにまとめています。モデル本体の推論コードは各モデルのディレクトリ（`romp/`・`sat-hmr/`・`fast-sam-3d-body/`）にあり、ここから呼び出します。

```text
measurements/
  romp/
    run.sh                     ROMP の計測（速度・精度・UDP）
    benchmark_romp_udp_ue.py   UE へ UDP 送信しながらの計測（図5）
    evaluate_3dpw.py           3DPW の精度評価（結果を JSON に保存）
    configs/                   romp_video_demo.py に渡す設定（バックボーン・数値精度・compile）
  sathmr/
    run.sh                     SAT-HMR の計測（速度・compile 範囲・精度・UDP）
  fastsam/
    run.sh                     Fast-SAM-3D-Body の計測（速度・精度・UDP）
    benchmark_fastsam3dbody.py 速度計測
    evaluate_3dpw.py           3DPW の精度評価
    convert_weights_bf16.py    bf16 重みの作成とサイズの集計（Table 2）
  run_all.sh                   すべての計測・評価を実行して paper_results.csv を作る
  make_paper_results.py        計測・評価の出力から paper_results.csv を作る
  cmu_panoptic/
    make_person_clips.py       図4・UEのデモの CMU Panoptic Studio のクリップを切り出す
  summarize_speed.py           生データから推論時間の中央値・MAD を出す
  paper_figures/               論文の図の生成
```

## 計測の実行

```bash
bash measurements/run_all.sh
```

作成された `paper_results.csv`（リポジトリ直下）を使って、論文の図を作成できます。計測・評価の出力は `results/measurements/rerun_<日時>/` に保存されます。脚注1の値は、Fast-SAM-3D-Body が `accuracy_fastsam_fp32/metrics_paper_3dpw14.json` の `PA-MPJPE` の `mean`、SAT-HMR が `accuracy_sathmr_fp32.txt` の `PAPER-PA-MPJPE` に出ます。

## 論文の図

[paper_figures/](paper_figures/) のスクリプトで論文の図を生成します。pandas・matplotlib が必要です。

| スクリプト | 論文の図 | 出力 |
|---|---|---|
| [plot_speed_vs_accuracy.py](paper_figures/plot_speed_vs_accuracy.py) | 図2（推論時間と3DPW精度。torch.compile ありの構成、エラーバーは2×MAD） | `speed_vs_{PAMPJPE,MPJPE,PVE,PAPVE}__compile.png`、論文値の★付きの `speed_vs_{PAMPJPE,MPJPE,PVE}__compile_with_papervalue.png` |
| [plot_sathmr_int8_compile_scope.py](paper_figures/plot_sathmr_int8_compile_scope.py) | 図3（SAT-HMR INT8 のコンパイル範囲ごとの推論時間） | `sathmr_int8_compile_scope.png` |
| [plot_udp_overhead.py](paper_figures/plot_udp_overhead.py) | 図5（UE同時実行時の推論時間増加） | `udp_overhead.png`、対応付けた表 `udp_overhead.csv` |

```bash
python measurements/paper_figures/plot_speed_vs_accuracy.py
python measurements/paper_figures/plot_sathmr_int8_compile_scope.py
python measurements/paper_figures/plot_udp_overhead.py
```

## 入力データ

| データ | 置き場所 | 用意の方法 |
|---|---|---|
| 速度計測の入力動画（1280×720・約30 fps・528フレーム。3モデル共通。） | `datasets/videos/sample_video.mp4` | [動画ACの素材ページ](https://video-ac.com/video/1201)から HD サイズ（1280×720）をダウンロードし、`sample_video.mp4` として配置します（固定カメラ・単一人物の実写動画）。 |
| 3DPW | `datasets/3dpw/` | ルートREADMEの[3DPWの配置](../README.md#3dpwの配置)を参照してください。 |
| CMU Panoptic Studio の動画（図4・UEのデモ） | `datasets/cmu_panoptic/person_clips/` | ルートREADMEの[CMU Panoptic Studioの動画の配置](../README.md#cmu-panoptic-studioの動画の配置)を参照してください。 |

UEパッケージの起動方法は [ue/README.md](../ue/README.md) を参照してください。

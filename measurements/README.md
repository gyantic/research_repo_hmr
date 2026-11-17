# measurements（論文の精度・速度評価）

本ディレクトリには、論文の計測（推論速度・3DPW精度・UE同時実行時のオーバーヘッド）を実行するスクリプトおよび設定ファイルと、結果の集計・論文図表の生成コードがモデルごとにまとまっています。

モデル本体の推論コードは各モデルのディレクトリ（[romp/](../romp/)・[sat-hmr/](../sat-hmr/)・[fast-sam-3d-body/](../fast-sam-3d-body/)）にあり、本ディレクトリのスクリプトから呼び出します。研究全体の構成については[ルートREADME](../README.md)を参照してください。

---

## ディレクトリ構成

```text
measurements/
├── romp/
│   ├── run.sh                     # ROMP の計測（速度・精度・UDP）
│   ├── benchmark_romp_udp_ue.py   # UE へ UDP 送信しながらの計測（図5）
│   ├── evaluate_3dpw.py           # 3DPW の精度評価（結果を JSON に保存）
│   └── configs/                   # romp_video_demo.py に渡す設定（バックボーン・数値精度・compile）
├── sathmr/
│   └── run.sh                     # SAT-HMR の計測（速度・compile 範囲・精度・UDP）
├── fastsam/
│   ├── run.sh                     # Fast-SAM-3D-Body の計測（速度・精度・UDP）
│   ├── benchmark_fastsam3dbody.py # 速度計測スクリプト
│   ├── evaluate_3dpw.py           # 3DPW の精度評価スクリプト
│   └── convert_weights_bf16.py    # bf16 重みの作成とサイズの集計（Table 2）
├── cmu_panoptic/
│   └── make_person_clips.py       # 図4・UEデモ用の CMU Panoptic Studio クリップを切り出す
├── paper_figures/                 # 論文の図（図2・図3・図5）の生成スクリプト
├── run_all.sh                     # すべての計測・評価を実行して paper_results.csv を生成
├── make_paper_results.py          # 計測・評価の出力から paper_results.csv を生成
└── summarize_speed.py             # 生データから推論時間の中央値・MAD を算出
```

---

## 1. 事前準備・入力データ

計測を実行する前に、ルートREADMEの [1. 環境の準備](../README.md#環境の準備) と [2. モデルの取得と配置](../README.md#モデルの取得と配置) に加え、以下の入力データを配置してください。

| データ | 配置先 | 用意の方法 |
|---|---|---|
| **速度計測の入力動画**<br>（1280×720・約30 fps・528フレーム、3モデル共通） | `datasets/videos/sample_video.mp4` | [動画ACの素材ページ](https://video-ac.com/video/1201)からHDサイズ（1280×720）をダウンロードし、`sample_video.mp4` として配置します（固定カメラ・単一人物の実写動画）。 |
| **3DPWデータセット** | `datasets/3dpw/` | ルートREADMEの [3.1 3DPWの配置](../README.md#3dpwの配置) を参照してください。 |
| **CMU Panoptic Studio の動画**<br>（図4・UEのデモ用） | `datasets/cmu_panoptic/person_clips/` | ルートREADMEの [3.3 CMU Panoptic Studioの動画の配置](../README.md#cmu-panoptic-studioの動画の配置) を参照してください（`python measurements/cmu_panoptic/make_person_clips.py` で生成）。 |

※ UE同時実行時の速度計測（`udp` モード、図5）を行う場合は、あらかじめUEの受信環境を起動してUDPポート `62004` で待機させておく必要があります。起動方法は [ue/README.md](../ue/README.md) およびルートREADMEの [5.3 SAT-HMR（INT8）からUEへの実時間送信](../README.md#sat-hmrint8からueへの実時間送信) を参照してください。

---

## 2. 計測の実行

### すべての計測・評価を一括実行する場合

リポジトリのルートから以下を実行します（内部で各モデルのConda環境 `romp_3.9`・`sathmr-copy`・`sam_3d_body` を順次呼び出します）。

```bash
bash measurements/run_all.sh
```

- 計測・評価の生の出力データは `results/measurements/rerun_<日時>/` に保存されます。
- すべての計測が完了すると、リポジトリ直下に集計結果の `paper_results.csv` が生成されます。
- 論文の脚注1に記載した評価値は、`results/measurements/rerun_<日時>/` 内の以下のファイルに出力されます。
  - Fast-SAM-3D-Body：`accuracy_fastsam_fp32/metrics_paper_3dpw14.json` の `PA-MPJPE`（`mean`）
  - SAT-HMR：`accuracy_sathmr_fp32.txt` の `PAPER-PA-MPJPE`

### モデル・項目ごとに個別実行する場合

各モデルのConda環境を有効化したうえで、リポジトリのルートからモデル別の `run.sh` を実行することもできます。

```bash
# ROMP（romp_3.9 環境）
conda activate romp_3.9
bash measurements/romp/run.sh {speed|accuracy|udp}

# SAT-HMR（sathmr-copy 環境）
conda activate sathmr-copy
bash measurements/sathmr/run.sh {speed|compile_scope|accuracy|udp}

# Fast-SAM-3D-Body（sam_3d_body 環境）
conda activate sam_3d_body
bash measurements/fastsam/run.sh {speed|accuracy|udp}
```

---

## 3. 論文の図の生成

`paper_results.csv` の生成後、[paper_figures/](paper_figures/) 内のスクリプトで論文の図を `results_png/` に出力できます（実行には `pandas` および `matplotlib` が必要です）。

| スクリプト | 対応する論文の図 | 出力ファイル（`results_png/` 配下） |
|---|---|---|
| [plot_speed_vs_accuracy.py](paper_figures/plot_speed_vs_accuracy.py) | **図2**：推論時間と3DPW精度の比較（`torch.compile` ありの構成、エラーバーは 2×MAD） | `speed_vs_{PAMPJPE,MPJPE,PVE,PAPVE}__compile.png`<br>論文値の★付き：`speed_vs_{PAMPJPE,MPJPE,PVE}__compile_with_papervalue.png` |
| [plot_sathmr_int8_compile_scope.py](paper_figures/plot_sathmr_int8_compile_scope.py) | **図3**：SAT-HMR INT8のコンパイル範囲ごとの推論時間 | `sathmr_int8_compile_scope.png` |
| [plot_udp_overhead.py](paper_figures/plot_udp_overhead.py) | **図5**：UE同時実行時の推論時間増加（オーバーヘッド） | `udp_overhead.png`<br>対応付けた集計表：`udp_overhead.csv` |

```bash
python measurements/paper_figures/plot_speed_vs_accuracy.py
python measurements/paper_figures/plot_sathmr_int8_compile_scope.py
python measurements/paper_figures/plot_udp_overhead.py
```

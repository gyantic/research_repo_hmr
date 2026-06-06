# ROMP Video Input vs SAT-HMR 差分・Seed固定仕様

## 1. 目的
- ROMP の動画入力経路について、SAT-HMR の動画入力経路との差分を明確化する。
- ROMP 側の動画入力スクリプトで seed 固定を有効化し、再現性を高める。

## 2. 対象
- ROMP 側
  - `romp/predict/video.py`
  - `romp/generator_bonebasedMesh_UDPver.py`
  - `romp/generator_bonebasedMesh_UDPver_copy.py`
  - `test_student.py`
  - `configs/video.yml`
- SAT-HMR 側（比較用）
  - `sathmr_udp_benchmark.py`
  - `video_demo.py`

## 3. 動画入力の主な相違点
| 観点 | ROMP | SAT-HMR |
|---|---|---|
| 動画フレーム取得（標準） | `romp/predict/video.py` は動画を一度 `video2frame` で画像列に展開して処理 (`romp/predict/video.py:51`, `romp/predict/video.py:67`, `romp/lib/utils/demo_utils.py:97`) | `cv2.VideoCapture` で逐次フレーム処理 (`sathmr_udp_benchmark.py:210`, `video_demo.py:191`) |
| 推論入力サイズ | `configs/video.yml` の `input_size: 512` (`configs/video.yml:18`) | 既定 `input_size: 1288` (`sathmr_udp_benchmark.py:45`, `video_demo.py:61`) |
| 人物選択 | ROMP は `show_largest_person_only: True` を設定可能 (`configs/video.yml:22`) + 任意で tracking/temporal optimization (`romp/predict/video.py:59`, `romp/predict/video.py:63`) | SAT-HMR ベンチは conf 最大の人物を1人選択 (`sathmr_udp_benchmark.py:140`) |
| 時系列処理 | ROMP は OneEuro による temporal optimization を標準機能として持つ (`romp/predict/video.py:113`) | SAT-HMR の動画ベンチには時系列フィルタなし（別実装で追跡/抑制を追加可能） |
| UE送信向け実装 | ROMP は専用スクリプトで動画→推論→UE送信 (`romp/generator_bonebasedMesh_UDPver.py`) | SAT-HMR は `sathmr_udp_benchmark.py` / `webcam_demo.py` で送信 |

## 4. ROMP 側 Seed固定 実装内容
以下の 4 ファイルで seed 固定を統一実装した。

- `romp/predict/video.py:15`
- `romp/generator_bonebasedMesh_UDPver.py:7`
- `romp/generator_bonebasedMesh_UDPver_copy.py:7`
- `test_student.py:11`

実装した固定項目:
- `SEED = 42`
- `random.seed`, `numpy.seed`, `cv2.setRNGSeed`
- `torch.manual_seed`, `torch.cuda.manual_seed`, `torch.cuda.manual_seed_all`
- `torch.backends.cudnn.benchmark = False`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.allow_tf32 = False`
- `torch.backends.cuda.matmul.allow_tf32 = False` (CUDA時)
- `torch.use_deterministic_algorithms(True, warn_only=True)`
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`

## 5. 運用上の注意
- `warn_only=True` のため、非決定演算が残っても実行は継続する。
- 完全決定性を優先して停止させたい場合は `warn_only=False` に変更する。
- ROMP 標準の動画処理はフレーム展開を伴うため、SAT-HMR の逐次読み込みより I/O コストが大きくなる場合がある。

## 6. 確認手順
1. ROMP 動画推論を通常起動する。
   - `CUDA_VISIBLE_DEVICES=0 python -u -m romp.predict.video --configs_yml='configs/video.yml'`
2. 同一動画・同一重みで複数回実行し、出力の一致性（少なくとも統計的な再現性）を確認する。
3. UE送信スクリプト利用時は `romp/generator_bonebasedMesh_UDPver.py` でも同様に比較する。

## 7. 変更履歴
- 2026-02-08: ROMP 動画入力系の seed 固定を統一実装。
- 2026-02-08: SAT-HMR との差分観点を明文化。

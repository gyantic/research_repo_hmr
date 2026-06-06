# SAT-HMR Webcam UDP 連携仕様書

## 1. 目的
- SAT-HMR の `webcam_demo.py` で推定した SMPL 結果を Unreal Engine (UE) へ UDP 送信し、リアルタイムにキャラクタ駆動する。
- 多人数検出時でも UE には「1人のみ」を安定して送信する（1人追跡）。

## 2. 対象
- 実装ファイル: `webcam_demo.py`
- 通信方式: UDP
- 送信先想定: UE 側受信ポート `62004`

## 3. 起動フロー
1. `webcam_demo.py` がカメラを起動し、SAT-HMR モデルをロードする。
2. `--udp` 有効時、Python 側は `--local_port` で HELLO を待受。
3. UE から HELLO を受信した送信元 IP を送信先 IP として採用。
4. 毎フレーム推論し、追跡対象 1 人を選択して UE 形式へ変換し送信。

## 4. CLI 引数（UDP関連）
- `--udp`
  - UE への UDP 送信を有効化。
- `--udp_only`
  - UE送信専用モード。`--no_render --no_display` を自動有効化し、最大FPSを優先。
- `--ue_port` (default: `62004`)
  - UE の受信ポート。
- `--local_port` (default: `5555`)
  - Python 側 HELLO 待受ポート。
- `--udp_track_max_dist_cm` (default: `120.0`)
  - 前フレーム追跡対象からの許容移動距離（cm）。
- `--udp_reinit_conf` (default: `0.55`)
  - 追跡再初期化時に必要な最小信頼度。
- `--udp_hold_frames` (default: `10`)
  - 対象見失い時、追跡状態を保持する最大フレーム数。
- `--udp_trans_jump_cm` (default: `30.0`)
  - 平行移動の外れ値閾値（cm）。閾値を超える急ジャンプを抑制する。`<=0` で無効。
- `--udp_outlier_mode` (default: `skip`, choices: `skip|clamp`)
  - 外れ値処理方式。`skip` は送信しない、`clamp` は移動量を閾値まで制限して送信する。
- `--udp_outlier_max_skip` (default: `12`)
  - `skip` モードで連続スキップ可能な最大フレーム数。超過時は自動で再ベース送信し、長時間フリーズを防ぐ。
- `--udp_jump_conf_thresh` (default: `0.35`)
  - ジャンプ判定の最小 conf。
- `--udp_jump_motion_cm` (default: `20.0`)
  - ジャンプ判定の平行移動量閾値（cm）。
- `--udp_jump_pose_motion` (default: `0.02`)
  - ジャンプ判定の pose 変化量閾値。
- `--udp_jump_debounce` (default: `2`)
  - ジャンプ判定が連続成立する必要フレーム数。
- `--udp_jump_hold_frames` (default: `4`)
  - ジャンプ確定後、外れ値抑制を一時バイパスするフレーム数。
- `--udp_jump_cooldown` (default: `6`)
  - ジャンプ終了後、再判定を抑制するフレーム数。
- `--render_every` (default: `1`)
  - メッシュ描画間引き設定。`N>1` で `N` フレームごとにのみ描画。
- `--amp`
  - CUDA mixed precision 推論を有効化し、推論時間短縮を狙う。

## 5. 推論と対象選択
- 検出候補は `pred_confs > --conf_thresh` を満たす query。
- UDP送信用に候補ごとに以下を保持:
  - `pose` (72,)
  - `transl` (3,)
  - `conf`
  - `query_id`

### 5.1 1人追跡ロジック
- 初期フレーム:
  - `conf` 最大候補を選択（`reason=max_conf`）。
- 追跡中:
  - 前回 `transl` との距離を計算。
  - 最近傍候補が `udp_track_max_dist_cm` 以内なら継続（`reason=nearest:*cm`）。
  - 距離が遠い場合でも `conf >= udp_reinit_conf` なら再初期化（`reason=reinit_conf:*`）。
  - それ以外は送信スキップ（`reason=skip_jump ...`）。
- ジャンプ例外:
  - `conf` と `motion/pose` が連続で閾値を超えた場合、短時間だけ外れ値抑制をバイパスして送信を継続する。
  - 横移動の制限は行わない。
- 見失い時:
  - `missed` を加算し、`udp_hold_frames` 超過で追跡状態をリセット。

## 6. UE送信データ仕様
- 送信ペイロード形式（ASCII）:
  - `"q0,q1,...,q95|t0,t1,t2\n"`
- クォータニオン `q`:
  - 24 関節 x 4 成分 = 96 要素
  - 並び: `(x, y, z, w)` をフラット化
- 平行移動 `t`:
  - 3 成分
  - 単位: cm

### 6.1 座標変換
- `smpl_pose_to_ue_quaternions`:
  - SMPL axis-angle(72) -> quaternion(24x4)
  - UE 座標系へ `x`/`z` 符号反転
  - ルート回転補正あり
- `transl_to_ue`:
  - `y` 符号反転
  - `z=0`
  - m -> cm 変換

## 7. 実行例
### 7.1 デバッグ表示あり
```bash
python webcam_demo.py \
  --camera 0 \
  --conf_thresh 0.1 \
  --width 1280 --height 720 \
  --input_size 1288 \
  --weights ./weights/sat_hmr/sat_644.pth \
  --device cuda \
  --show_fps \
  --udp \
  --ue_port 62004 \
  --local_port 5555 \
  --udp_track_max_dist_cm 80 \
  --udp_reinit_conf 0.6 \
  --udp_hold_frames 15
```

### 7.2 実運用（高FPS優先）
```bash
python webcam_demo.py \
  --camera 0 \
  --conf_thresh 0.1 \
  --width 1280 --height 720 \
  --input_size 1288 \
  --weights ./weights/sat_hmr/sat_644.pth \
  --device cuda \
  --udp \
  --udp_only \
  --amp \
  --ue_port 62004 \
  --local_port 5555
```

## 8. FPS差の調査結果
- 動画入力ベンチマーク（`sathmr_udp_benchmark.py`）は描画処理を行わないため、推論性能に近いFPSが出る。
- `webcam_demo.py` でFPSが3前後になる主因は `render`（メッシュ可視化）で、`show_fps` の `Render: xxx ms` が支配的になる。
- UE送信のみが目的なら `--udp_only`（または `--no_render --no_display`）を推奨。
- プレビューも見たい場合は `--render_every 2` 以上で描画を間引く。

## 9. ログ仕様（UDP）
- HELLO接続:
  - `[WAIT] HELLO on 0.0.0.0:<local_port>`
  - `[HELLO] from (<ip>, <port>) -> dst=(<ip>, <ue_port>)`
- 初回送信:
  - `[INFO] Startup Lag: ... seconds`
- 30フレームごとの送信ログ:
  - `UDP frame=... conf=... persons=... query=... reason=... pose_motion=... trans_motion=... sent=...`
- スキップ時:
  - `UDP frame=... skip (...) persons=...`

## 10. 既知の挙動と注意
- `persons` が急増する環境では候補入れ替わりが発生しうるため、`udp_track_max_dist_cm` を小さめにする。
- `skip_jump` が多すぎる場合は:
  - `udp_track_max_dist_cm` を少し上げる、または `udp_reinit_conf` を下げる。
- UE が動かない場合は以下を優先確認:
  - HELLO が成立しているか
  - `sent` バイト数が継続出力されているか
  - `conf` と `pose_motion` が変化しているか

## 11. 非対象
- 本仕様は `udp_communication_UE_webcam.py` の旧デバッグ実装は対象外。
- 本仕様の正は `webcam_demo.py --udp` の実装。





## 12. 改善施策一覧（再学習なし）
ステータス凡例: `[完了]` / `[一部完了]` / `[未着手]` / `[検討中]`

### 12.1 優先度: 高（すぐ効く）
- `[完了]` `--udp_only --amp` を標準運用化
  - 描画を切り、mixed precision を有効化して推論FPSを最大化する。
- `[完了]` 1人追跡の安定化
  - `udp_track_max_dist_cm` / `udp_reinit_conf` / `udp_hold_frames` を環境に合わせて調整する。
- `[完了]` 外れ値抑制
  - `trans_motion` が閾値超過したフレームをスキップまたはクランプする。
  - 実装パラメータ: `udp_trans_jump_cm`, `udp_outlier_mode`, `udp_outlier_max_skip`
- `[完了]` ジャンプ時の例外通過
  - ジャンプ判定中は短時間だけ外れ値抑制をバイパスして送信継続する。
  - 実装パラメータ: `udp_jump_conf_thresh`, `udp_jump_motion_cm`, `udp_jump_pose_motion`, `udp_jump_debounce`, `udp_jump_hold_frames`, `udp_jump_cooldown`

### 12.2 優先度: 中（品質向上）
- `[未着手]` SAT-HMR出力の時系列平滑化
  - `transl`: EMA または One-Euro Filter
  - `pose`: クォータニオン化後に SLERP で平滑化
- `[未着手]` 信頼度連動平滑
  - `conf` が低いほど平滑化を強くし、誤検出フレームの揺れを抑える。
- `[未着手]` UE側補間（推奨）
  - SAT-HMRは 8-12fps のまま送信し、UEで 30/60fps に補間表示する。

### 12.3 優先度: 低
- `[一部完了]` 入力軽量化
  - `input_size` やカメラ解像度を下げる運用は可能（手動設定）。
- `[未着手]` 最新フレーム優先処理
  - キュー詰まり時に古いフレームを破棄して遅延を抑える。
- `[未着手]` 送信フォーマット最適化
  - ASCII からバイナリ化し、CPU文字列処理コストを削減する。UnrealEngine側の修正が必要なので大変

## 13. 改善施策一覧（再学習あり）
### 13.1 優先度: 高（効果が大きい）
- `[未着手]` 単一人物・Webcamドメイン向け微調整
  - 実運用カメラ条件（室内照明・画角・距離）に寄せた再学習で検出安定性を上げる。
- `[未着手]` 時系列一貫性を意識した学習
  - 連続フレームでの姿勢変化にペナルティを入れ、推論時のガタつきを低減する。

### 13.2 優先度: 中（精度改善）
- `[未着手]` Hard case 追加学習
  - 横向き、手前/奥移動、被写体の部分隠れを追加してロバスト化。
- `[未着手]` Confidence calibration
  - conf と実際の正しさのズレを補正し、追跡切替の誤判定を減らす。

### 13.3 優先度: 中〜高（研究寄り）
- `[検討中]` SAM3D蒸留
  - 目的: SAT-HMR の 3D 一貫性・形状復元を強化し、低conf時の破綻を減らす。
  - 期待効果: 姿勢の連続性向上、オクルージョン時の復元改善。
  - コスト: 教師モデル推論・データ生成・蒸留損失設計で実装工数が大きい。
  - リスク: ドメイン不一致時に教師バイアスを取り込み、実環境で逆効果の可能性。
  - 着手判断の目安:
    - まず「再学習なし施策 + 通常微調整」で目標品質に届かない場合に着手する。
    - 明確な評価指標（MPJPE/PVE、UE再生時ジッタ量、追跡切替回数）を先に固定する。

## 14. 実行順（推奨ロードマップ）
1. 再学習なし施策を先に完了（追跡安定化 + 平滑化 + UE補間）。
2. それでも不足する課題を定量化（ジッタ、ロスト率、再初期化回数）。
3. 単一人物ドメイン微調整を実施。
4. 最後に SAM3D 蒸留のPoCを小規模で評価し、継続判断する。

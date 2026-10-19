"""Stream ROMP SMPL joint rotations to Unreal Engine over TCP (bone-based, no mesh).

Sending joint rotations instead of a mesh lets UE drive its own skinned body/clothing.
Listens on HOST:PORT, waits for UE to connect, then runs ROMP on every frame of
VIDEO_PATH and sends one message per detected frame (format: see
ue_pose_message.py; terminated by "\\x00"). Per-frame inference/total times and a
summary are written to TCP_inference_stats_<model>.csv.

Run from the romp/ directory:
    PYTHONPATH="$PWD:$PWD/romp" python romp/generator_bonebasedMesh.py
"""
import cv2, socket, time, sys, pathlib, csv
import numpy as np
import torch
import yaml

from romp.processor_bootstrap import build_video_processor
from romp.ue_pose_message import format_ue_pose_message

CONFIG_YML = "configs/video.yml"
VIDEO_PATH = "../datasets/videos/sample_video.mp4"
WARMUP_FRAMES = 10


def stats_csv_name(config_yml):
    """Return (model_name, csv_path) derived from model_path in the ROMP config."""
    try:
        with open(config_yml, 'r') as f:
            config = yaml.safe_load(f)
        # model_path は ARGS の下にネストされている場合と直下の場合がある
        if 'ARGS' in config and 'model_path' in config['ARGS']:
            model_path = config['ARGS']['model_path']
        elif 'model_path' in config:
            model_path = config['model_path']
        else:
            model_path = 'unknown_model'
        model_name = pathlib.Path(model_path).stem
        return model_name, f"TCP_inference_stats_{model_name}.csv"
    except Exception as e:
        print(f"[WARNING] Failed to read {config_yml}: {e}")
        return "unknown_model", "TCP_inference_stats_unknown_model.csv"


torch.set_num_threads(2)          # CPUスレッド数制限
torch.set_num_interop_threads(1)  # スレッド間連携も絞る
HOST, PORT = "127.0.0.1", 5555
sock  = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((HOST, PORT))
sock.listen(1)
print("[Listening]", HOST, PORT)

conn, _ = sock.accept()
conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
print("[Connected]")
time_connection_success = time.perf_counter()

model_name, STATS_CSV = stats_csv_name(CONFIG_YML)
print(f"[INFO] Model: {model_name}")
print(f"[INFO] CSV output: {STATS_CSV}")

# Seeding / deterministic settings are applied when this imports romp.predict.video.
romp_proc = build_video_processor(CONFIG_YML, VIDEO_PATH, gpu="-1")

inference_times = []  # 推論時間（ミリ秒）
total_times = []      # 全体処理時間（ミリ秒）: デコード→推論→整形→送信
frame_id = 0

infer_start = torch.cuda.Event(enable_timing=True)
infer_end   = torch.cuda.Event(enable_timing=True)

cap = cv2.VideoCapture(VIDEO_PATH)

print(f"\n[WARMUP] Running {WARMUP_FRAMES} warmup frames...")
for warmup_count in range(1, WARMUP_FRAMES + 1):
    ok, frame = cap.read()
    if not ok:
        print("[ERROR] Video ended during warmup")
        sys.exit(1)
    romp_proc.single_image_forward(frame)
    if warmup_count % 5 == 0:
        print(f"[WARMUP] {warmup_count}/{WARMUP_FRAMES} frames completed")
print("[WARMUP] Complete! Starting measurement...\n")

first_send_done = False
try:
    t_all0 = time.perf_counter()  # 全体の開始時刻

    while cap.isOpened():
        t_loop0 = time.perf_counter()  # ループ開始時刻

        ok, frame = cap.read()
        if not ok:
            break

        # GPU時間で推論を計測
        infer_start.record()
        outs = romp_proc.single_image_forward(frame)
        infer_end.record()
        infer_end.synchronize()
        infer_ms = infer_start.elapsed_time(infer_end)
        inference_times.append(infer_ms)

        if outs is None or not outs["detection_flag"]:
            frame_id += 1
            continue

        pose = outs['params']['poses'][0].cpu().numpy().astype(np.float32)  # (72,)
        cam = outs['params']['cam'][0].cpu().numpy().astype(np.float32)     # (3,)
        msg = format_ue_pose_message(pose, cam) + "\x00"
        t_send0 = time.perf_counter()
        conn.sendall(msg.encode("ascii"))
        if not first_send_done:
            time_first_send = time.perf_counter()
            startup_lag = time_first_send - time_connection_success
            print("-" * 40)
            print(f"[INFO] Startup Lag: {startup_lag:.4f} seconds")
            print("-" * 40)
            first_send_done = True

        send_ms = (time.perf_counter() - t_send0) * 1000.0

        total_ms = (time.perf_counter() - t_loop0) * 1000.0
        total_times.append(total_ms)

        if frame_id % 60 == 0:
            avg_fps = frame_id / (time.perf_counter() - t_all0)
            print(f"frames={frame_id}  avg_fps={avg_fps:.2f}  infer_ms={infer_ms:.1f}  全体_ms={total_ms:.1f}")
            print(send_ms)

        frame_id += 1
    time_last_send = time.perf_counter()

except KeyboardInterrupt:
    print("[Interrupted]")
finally:
    sock.close()
    try:
        cap.release()
    except Exception:
        pass

    # ---- 統計計算と保存 ----
    if len(inference_times) > 0:
        inf_array = np.array(inference_times)
        tot_array = np.array(total_times)

        inf_mean = np.mean(inf_array)
        inf_std = np.std(inf_array)
        inf_median = np.median(inf_array)
        tot_mean = np.mean(tot_array)
        tot_std = np.std(tot_array)
        tot_median = np.median(tot_array)

        print("\n" + "=" * 50)
        print("統計結果")
        print("=" * 50)
        print(f"推論時間 (ms):")
        print(f"  平均: {inf_mean:.2f} ± {inf_std:.2f}")
        print(f"  中央値: {inf_median:.2f}")
        print(f"  最小値: {np.min(inf_array):.2f}")
        print(f"  最大値: {np.max(inf_array):.2f}")
        print(f"\n全体処理時間 (ms):")
        print(f"  平均: {tot_mean:.2f} ± {tot_std:.2f}")
        print(f"  中央値: {tot_median:.2f}")
        print(f"  最小値: {np.min(tot_array):.2f}")
        print(f"  最大値: {np.max(tot_array):.2f}")
        print(f"\n総フレーム数: {len(inference_times)}")
        print("=" * 50)
        total_comm_time = time_last_send - time_connection_success
        print(f"\n[INFO] 通信総時間: {total_comm_time:.4f} 秒")
        print(f"[INFO] 初回送信までの遅延: {startup_lag:.4f} 秒")
        print(f"[INFO] 実データ送信時間: {time_last_send - time_first_send:.4f} 秒")

        # CSVに保存（統計サマリー + 全フレームデータ）
        with open(STATS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(['# 統計サマリー'])
            writer.writerow(['指標', '推論時間 (ms)', '全体処理時間 (ms)'])
            writer.writerow(['平均', f'{inf_mean:.2f}', f'{tot_mean:.2f}'])
            writer.writerow(['標準偏差', f'{inf_std:.2f}', f'{tot_std:.2f}'])
            writer.writerow(['中央値', f'{inf_median:.2f}', f'{tot_median:.2f}'])
            writer.writerow(['最小値', f'{np.min(inf_array):.2f}', f'{np.min(tot_array):.2f}'])
            writer.writerow(['最大値', f'{np.max(inf_array):.2f}', f'{np.max(tot_array):.2f}'])
            writer.writerow(['総フレーム数', len(inference_times), ''])
            writer.writerow([])

            # 全フレームデータ（未検出フレームは total_times に入らないため zip で短い方に揃う）
            writer.writerow(['# 全フレームデータ'])
            writer.writerow(['frame_id', 'inference_time_ms', 'total_time_ms'])
            for i, (inf_t, tot_t) in enumerate(zip(inference_times, total_times)):
                writer.writerow([i, f'{inf_t:.2f}', f'{tot_t:.2f}'])

        print(f"\n統計データを保存: {STATS_CSV}")

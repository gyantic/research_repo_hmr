"""
SAT-HMR Benchmark with UDP (測定用)
ROMPと同様の形式でSMPLパラメータをUEに送信 + 統計計測
"""

import socket, time, sys, pathlib, csv, numpy as np
import cv2, torch
import os
import argparse
from sathmr_common import (
    infer_frame,
    load_config,
    resolve_device,
    setup_reproducibility,
    smpl_pose_to_ue_quaternions,
    transl_to_ue,
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype_cfg", type=float, default="FP32", help="変数の型(fp32, bfp16, int8)")
    return p.parse_args()

# ========= Seed固定 =========
SEED = 42
setup_reproducibility(SEED)

# ========= 実行設定 =========
DEVICE_MODE = os.getenv("SATHMR_DEVICE_MODE", "auto")  # "auto" | "cpu" | "cuda"
CPU_THREADS = int(os.getenv("SATHMR_CPU_THREADS", "2"))
CPU_INTEROP_THREADS = int(os.getenv("SATHMR_CPU_INTEROP_THREADS", "1"))


torch.set_num_threads(CPU_THREADS)
torch.set_num_interop_threads(CPU_INTEROP_THREADS)

VIDEO_PATH = "videos/sample_video.mp4"
UE_HOST = "127.0.0.1"        # HELLOで上書き
UE_PORT = 62004
LOCAL_PORT = 5555
W, H = 1280, 720
INPUT_SIZE = 1288
CONF_THRESH = 0.3
LOG_INTERVAL = 60              # 何フレーム毎に平均FPS表示
WARMUP_FRAMES = 10


# 統計CSVファイル名
MODEL_NAME = "sat_hmr"
STATS_DIR = pathlib.Path("datas")
STATS_DIR.mkdir(parents=True, exist_ok=True)
STATS_CSV = STATS_DIR / f"inference_stats_{MODEL_NAME}_UDP.csv"

print(f"[INFO] Model: {MODEL_NAME}")
print(f"[INFO] CSV output: {STATS_CSV}")

# ========= SAT-HMR インポート =========
from models import build_sat_model


# ---- UDP（HELLO待ち→送信先確定）----
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("0.0.0.0", LOCAL_PORT))
print(f"[WAIT] HELLO on 0.0.0.0:{LOCAL_PORT}")
_, addr = sock.recvfrom(2048)
UE_HOST = addr[0]
print(f"[HELLO] from {addr} -> dst={(UE_HOST, UE_PORT)}")
time_connection_success = time.perf_counter()

# ---- SAT-HMR 初期化 ----
print("\n[INFO] SAT-HMR モデルを読み込み中...")
model_args = load_config(INPUT_SIZE)
model, _ = build_sat_model(model_args, set_criterion=False)

state_dict = torch.load('./weights/sat_hmr/sat_644.pth', map_location='cpu', weights_only=False)
model.load_state_dict(state_dict, strict=False)

device = resolve_device(DEVICE_MODE)

# ---- 引数設定 ----
args = parse_args()
dtype = args.dtype_cfg

model = model.to(device)
if dtype == "bfloat16":
    model = model.to(torch.bfloat16)

model.eval()

print(f"[INFO] モデル読み込み完了 (device: {device})")
print(f"[INFO] CPU threads: {CPU_THREADS}, interop: {CPU_INTEROP_THREADS}")

# ---- 統計記録用リスト ----
inference_times = []  # 推論時間（ミリ秒）
total_times = []      # 全体処理時間（ミリ秒）

# CUDA イベント（GPU時のみ使用）
start_event = None
end_event = None
if device.type == "cuda":
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

# ---- 動画読み込み ----
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
    sys.exit(1)

video_fps = cap.get(cv2.CAP_PROP_FPS)
video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"[INFO] Video: {VIDEO_PATH}")
print(f"[INFO] Resolution: {W}x{H}, FPS: {video_fps:.2f}, Frames: {video_frames}")

# ---- ウォームアップ ----
print(f"\n[WARMUP] Running {WARMUP_FRAMES} warmup frames...")
warmup_count = 0
while warmup_count < WARMUP_FRAMES:
    ok, frame = cap.read()
    if not ok:
        print("[ERROR] Video ended during warmup")
        sys.exit(1)
    if dtype == "bfloat16":
        frame = frame.to(torch.bfloat16)
    _ = infer_frame(model, frame, INPUT_SIZE, CONF_THRESH, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    warmup_count += 1
    if warmup_count % 5 == 0:
        print(f"[WARMUP] {warmup_count}/{WARMUP_FRAMES} frames completed")

# 動画を最初に戻す
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
print("[WARMUP] Complete! Starting measurement...\n")

# ---- メインループ（sleep無し / UE送信込みで計測）----
frame_id = 0
t_all0 = time.perf_counter()
first_send_done = False
time_first_send = None
time_last_send = None

try:
    while True:
        t_loop0 = time.perf_counter()  # ループ総時間の開始

        ok, frame = cap.read()
        if not ok:
            break
        if dtype == "bfloat16":
            frame = frame.to(torch.bfloat16)

        # 推論計測（deviceごとに分岐）
        if device.type == "cuda":
            start_event.record()
            outs = infer_frame(model, frame, INPUT_SIZE, CONF_THRESH, device)
            end_event.record()
            end_event.synchronize()
            infer_ms = start_event.elapsed_time(end_event)
        else:
            t_inf0 = time.perf_counter()
            outs = infer_frame(model, frame, INPUT_SIZE, CONF_THRESH, device)
            infer_ms = (time.perf_counter() - t_inf0) * 1000.0
        inference_times.append(infer_ms)

        if not outs.get("detection_flag", False):
            # 検出なし
            tproc_ms = (time.perf_counter() - t_loop0) * 1000.0
            frame_id += 1
            if frame_id % LOG_INTERVAL == 0:
                avg_fps = frame_id / (time.perf_counter() - t_all0)
                print(f"[no detect] frames={frame_id}  avg_fps={avg_fps:.2f}  infer_ms={infer_ms:.1f}  total_ms={tproc_ms:.1f}")
            continue

        # 座標系変換
        quats = smpl_pose_to_ue_quaternions(outs['pose'])
        trans = transl_to_ue(outs['transl'])

        # 送信フォーマット
        pose_str = ",".join(f"{v:.4f}" for v in quats.flatten())
        trans_str = ",".join(f"{v:.4f}" for v in trans.flatten())
        payload = f"{pose_str}|{trans_str}\n".encode("ascii")

        sock.sendto(payload, (UE_HOST, UE_PORT))
        time_last_send = time.perf_counter()

        if not first_send_done:
            time_first_send = time.perf_counter()
            startup_lag = time_first_send - time_connection_success
            print("-" * 40)
            print(f"[INFO] Startup Lag: {startup_lag:.4f} seconds")
            print("-" * 40)
            first_send_done = True

        # ループ総時間
        tproc_ms = (time.perf_counter() - t_loop0) * 1000.0
        total_times.append(tproc_ms)

        frame_id += 1
        if frame_id % LOG_INTERVAL == 0:
            avg_fps = frame_id / (time.perf_counter() - t_all0)
            print(f"frames={frame_id}  avg_fps={avg_fps:.2f}  infer_ms={infer_ms:.1f}  total_ms={tproc_ms:.1f}  conf={outs['conf']:.2f}")

except KeyboardInterrupt:
    print("\n[Interrupted]")

finally:
    sock.close()
    cap.release()
    
    # ---- 統計計算と保存 ----
    if len(inference_times) > 0:
        inf_array = np.array(inference_times)
        tot_array = np.array(total_times) if len(total_times) > 0 else np.array([0])
        
        # 統計値計算
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
        print(f"  FPS (推論のみ): {1000/inf_mean:.2f}")
        print(f"\n全体処理時間 (ms):")
        print(f"  平均: {tot_mean:.2f} ± {tot_std:.2f}")
        print(f"  中央値: {tot_median:.2f}")
        print(f"  最小値: {np.min(tot_array):.2f}")
        print(f"  最大値: {np.max(tot_array):.2f}")
        print(f"  FPS (全体): {1000/tot_mean:.2f}")
        print(f"\n総フレーム数: {len(inference_times)}")
        print(f"検出フレーム数: {len(total_times)}")
        print("=" * 50)
        
        if time_first_send and time_last_send:
            total_comm_time = time_last_send - time_connection_success
            startup_lag = time_first_send - time_connection_success
            print(f"\n[INFO] 通信総時間: {total_comm_time:.4f} 秒")
            print(f"[INFO] 初回送信までの遅延: {startup_lag:.4f} 秒")
            print(f"[INFO] 実データ送信時間: {time_last_send - time_first_send:.4f} 秒")
        
        # CSVに保存
        with open(STATS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # 統計サマリー
            writer.writerow(['# 統計サマリー'])
            writer.writerow(['設定', '値', ''])
            writer.writerow(['device', device.type, ''])
            writer.writerow([])
            writer.writerow(['指標', '推論時間 (ms)', '全体処理時間 (ms)'])
            writer.writerow(['平均', f'{inf_mean:.2f}', f'{tot_mean:.2f}'])
            writer.writerow(['標準偏差', f'{inf_std:.2f}', f'{tot_std:.2f}'])
            writer.writerow(['中央値', f'{inf_median:.2f}', f'{tot_median:.2f}'])
            writer.writerow(['最小値', f'{np.min(inf_array):.2f}', f'{np.min(tot_array):.2f}'])
            writer.writerow(['最大値', f'{np.max(inf_array):.2f}', f'{np.max(tot_array):.2f}'])
            writer.writerow(['FPS (推論)', f'{1000/inf_mean:.2f}', ''])
            writer.writerow(['FPS (全体)', '', f'{1000/tot_mean:.2f}'])
            writer.writerow(['総フレーム数', len(inference_times), ''])
            writer.writerow(['検出フレーム数', len(total_times), ''])
            writer.writerow([])
            
            # 全フレームデータ
            writer.writerow(['# 全フレームデータ'])
            writer.writerow(['frame_id', 'inference_time_ms', 'total_time_ms'])
            
            tot_idx = 0
            for i, inf_t in enumerate(inference_times):
                if tot_idx < len(total_times):
                    writer.writerow([i, f'{inf_t:.2f}', f'{total_times[tot_idx]:.2f}'])
                    tot_idx += 1
                else:
                    writer.writerow([i, f'{inf_t:.2f}', 'N/A (no detection)'])
        
        print(f"\n統計データを保存: {STATS_CSV}")
    else:
        print("\n[WARNING] No frames processed")

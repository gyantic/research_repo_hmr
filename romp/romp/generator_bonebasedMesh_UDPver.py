"""Stream ROMP SMPL joint rotations to Unreal Engine over UDP.

Waits for a "hello" datagram from UE, then feeds a video through ffmpeg at a fixed
12 fps, runs ROMP on each frame and sends one text datagram per detected frame
(format: see ue_pose_message.py; terminated by "\\n").

Run from the romp/ directory:
    PYTHONPATH="$PWD:$PWD/romp" python romp/generator_bonebasedMesh_UDPver.py
"""
import cv2, socket, time, subprocess, struct
import numpy as np
import torch

from romp.processor_bootstrap import build_video_processor
from romp.ue_pose_message import format_ue_pose_message


def scsc2_pack_len(msg_bytes: bytes, little=True) -> bytes:
    # SCSC2 "By Length" ヘッダ: '<' | endian | uint32 size | '>'
    endian_flag = b'\x00' if little else b'\x01'
    size_bytes  = struct.pack('<I' if little else '>I', len(msg_bytes))
    return b'<' + endian_flag + size_bytes + b'>'

def to_numpy_float32(x):
    if hasattr(x, "detach"):
        return x.detach().float().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)

def read_frame():
    n = W*H*3
    buf = proc.stdout.read(n)           # ★ 12fpsでブロッキング供給される
    if not buf or len(buf) < n:
        return False, None
    frm = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    return True, frm

def sleep_until(deadline):
    """Sleep until perf_counter() reaches ``deadline`` (no-op if already past)."""
    remain = deadline - time.perf_counter()
    if remain > 0:
        time.sleep(remain)


# ===================== ユーザ設定 =====================
UE_PORT = 62004           # UEの受信ポート（Blueprintの受信ノードPortと一致）
LOCAL_PORT = 5555         # Python側のローカルポート
torch.set_num_threads(2)          # CPUスレッド数制限
torch.set_num_interop_threads(1)  # スレッド間連携も絞る
# =====================================================

# ---- UDPソケット（hello受信まで待機 → 以後は sendto(...)）----
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("0.0.0.0", LOCAL_PORT))
print(f"[WAIT] HELLO on 0.0.0.0:{LOCAL_PORT}")

# UE からの hello を受け取るまでブロッキング。送信先は hello の送信元IP。
_hello, addr = sock.recvfrom(2048)
UE_HOST = addr[0]
print(f"[HELLO] from {addr} -> dst={(UE_HOST, UE_PORT)}")

# HELLO受信時刻を記録
time_connection_success = time.perf_counter()

VIDEO_PATH = "demo/videos/sample_video(camera_fixed).mp4"

W, H = 1280, 720
STREAM_FPS = 12
FFMPEG = [
    'ffmpeg',
    '-re',                 # read at the native rate so frames arrive in real time
    '-nostdin', '-loglevel', 'error',
    '-i', VIDEO_PATH,
    '-vf', f'fps={STREAM_FPS},scale={W}:{H}',
    '-pix_fmt', 'bgr24',
    '-f', 'rawvideo', '-'
]
proc = subprocess.Popen(FFMPEG, stdout=subprocess.PIPE, bufsize=W*H*3)

# Seeding / deterministic settings are applied when this imports romp.predict.video.
romp_proc = build_video_processor("configs/video.yml", VIDEO_PATH, gpu="0")

frame_interval_s = 1 / STREAM_FPS

# Frames come from ffmpeg (read_frame); this capture is only used as the loop guard.
cap = cv2.VideoCapture(VIDEO_PATH)

first_send_done = False
prev_time = 0
try:
    frame_id = 0
    t_next = time.perf_counter()

    while cap.isOpened():
        ok, frame = read_frame()
        if not ok:
            break

        t0 = time.perf_counter()
        outs = romp_proc.single_image_forward(frame)
        print(f"infer_ms={(time.perf_counter()-t0)*1000:.1f}")
        if outs is None or not outs["detection_flag"]:
            t_next += frame_interval_s
            sleep_until(t_next)
            frame_id += 1
            continue

        pose = to_numpy_float32(outs['params']['poses'][0])  # (72,)
        cam = to_numpy_float32(outs['params']['cam'][0])     # (3,)
        payload = (format_ue_pose_message(pose, cam) + "\n").encode("ascii")
        sock.sendto(payload, (UE_HOST, UE_PORT))

        if not first_send_done:
            time_first_send = time.perf_counter()
            startup_lag = time_first_send - time_connection_success
            print("-" * 40)
            print(f"[INFO] Startup Lag: {startup_lag:.4f} seconds")
            print("-" * 40)
            first_send_done = True

        current_time = time.perf_counter()
        # 前回送信からの経過時間（1フレームの処理時間）
        delta_time = current_time - prev_time
        prev_time = current_time

        if delta_time > 0:
            fps = 1.0 / delta_time
            print(f"Frame: {frame_id} | Actual FPS: {fps:.2f}")

        t_next += frame_interval_s
        sleep_until(t_next)
        frame_id += 1

except KeyboardInterrupt:
    print("[Interrupted]")
finally:
    cap.release()
    sock.close()

import cv2, socket, time, sys, pathlib, pickle, json, numpy as np
from scipy.spatial.transform import Rotation
import struct
import subprocess, torch, random
import os

# ========= Seed固定 =========
SEED = 42

def setup_reproducibility(seed: int = 42):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


setup_reproducibility(SEED)


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


# ===================== ユーザ設定 =====================
UE_HOST = "172.31.32.1"   # 初期値（hello受信後に送信元IPで上書き）
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

#UE からの hello を受け取るまでブロッキング
data, addr = sock.recvfrom(2048)     
UE_HOST = addr[0]               
print(f"[HELLO] from {addr} -> dst={(UE_HOST, UE_PORT)}")


# HELLO受信時刻を記録
time_connection_success = time.perf_counter()

# VIDEO_PATH = "demo/videos/IMG_9732_reencoded.mp4"
VIDEO_PATH = "demo/videos/sample_video(camera_fixed).mp4"

W, H = 1280, 720       
PIX = 'bgr24'         
FFMPEG = [
    'ffmpeg',
    '-re',                 
    '-nostdin', '-loglevel', 'error',
    '-i', VIDEO_PATH,
    '-vf', f'fps=12,scale={W}:{H}', 
    '-pix_fmt', PIX,
    '-f', 'rawvideo', '-'
]
proc = subprocess.Popen(FFMPEG, stdout=subprocess.PIPE, bufsize=W*H*3)


arg_list = [
    "--configs_yml", "configs/video.yml",
    "--inputs",      VIDEO_PATH,
    "--gpu",         "0"
]
sys.argv = [sys.argv[0]] + arg_list

from romp.lib.config import parse_args, ConfigContext
from romp.predict.video import Video_processor
with ConfigContext(parse_args(arg_list)) as cfg:
    romp_proc = Video_processor(args_set=cfg)

pkl_path = pathlib.Path.home() / ".romp" / "smpl_model_data" / "SMPL_NEUTRAL.pkl"
with open(pkl_path, "rb") as f:
    smpl_data = pickle.load(f, encoding="latin1")

faces = np.asarray(smpl_data["f"], dtype=np.int32)
#dt = 1/28.283786664584916  #入力動画のFPSの逆数で定義している
dt = 1/12
t_next = time.perf_counter()
R_xm90 = np.array([[ 1, 0,  0],
                   [ 0, 0, -1],
                   [ 0, 1,  0]], dtype=np.float32)

cap = cv2.VideoCapture(VIDEO_PATH)

first_send_done = False
prev_time = 0
try:
    prev_t = time.perf_counter()
    frame_id = 0
    t_next = time.perf_counter()

    while cap.isOpened():
        #ok, frame = cap.read()
        ok, frame = read_frame()
        if not ok:
            break
        """
        if frame_id % 2 != 0:  #1フレーム飛ばしで
            t_next += dt
            now = time.perf_counter()
            remain = t_next - now
            if remain > 0:
                time.sleep(remain)
            frame_id += 1
            continue 
        """
        
        t0 = time.perf_counter()
        outs = romp_proc.single_image_forward(frame)
        print(f"infer_ms={(time.perf_counter()-t0)*1000:.1f}")
        if outs is None or not outs["detection_flag"]:
            t_next += dt
            now = time.perf_counter()
            remain = t_next - now
            if remain > 0:
                time.sleep(remain)
            frame_id += 1
            continue
        #print(outs['params'].keys())

        pose  = to_numpy_float32(outs['params']['poses'][0])  # (72,)
        trans = to_numpy_float32(outs['params']['cam'][0])    # (3,)
        """
        print(f"cam:{outs['params']['cam']}")
        print(f"global_orient:{outs['params']['global_orient']}")
        print(f"body_pose:{outs['params']['body_pose']}")
        print(f"poses:{outs['params']['poses']}")
        """

        rotations = Rotation.from_rotvec(pose.reshape(-1, 3))
        quaternions_for_ue = rotations.as_quat()  # (N,4)=(x,y,z,w)

        # UE座標系
        quaternions_for_ue[:,0] *= -1.0
        quaternions_for_ue[:,2] *= -1.0
        flip_root = np.array([1., 0., 0., 0.], dtype=np.float32)
        q0 = Rotation.from_quat(flip_root) * Rotation.from_quat(quaternions_for_ue[0])
        quaternions_for_ue[0] = q0.as_quat().astype(np.float32)

        trans[1] *= -1.0
        trans[2] = 0.0
        trans = trans * 100.0  # m→cm

        pose_str  = ",".join(f"{v:.4f}" for v in quaternions_for_ue.flatten())
        trans_str = ",".join(f"{v:.4f}" for v in trans.flatten())
        msg = f"{pose_str}|{trans_str}\n".encode("ascii")

        payload = f"{pose_str}|{trans_str}\n".encode("ascii")
        sock.sendto(payload, (UE_HOST, UE_PORT))  

        if not first_send_done:
            time_first_send = time.perf_counter()
            startup_lag = time_first_send - time_connection_success
            print("-" * 40)
            print(f"[INFO] Startup Lag: {startup_lag:.4f} seconds")
            print("-" * 40)
            first_send_done = True
        
        current_time = time.perf_counter()
        # 1フレームの処理にかかった時間
        delta_time = current_time - prev_time
        prev_time = current_time # 次のフレームのために現在時刻を保存

        if delta_time > 0:
            fps = 1.0 / delta_time
            print(f"Frame: {frame_id} | Actual FPS: {fps:.2f}")
        
        t_next += dt
        now = time.perf_counter()
        remain = t_next - now
        if remain > 0:
            time.sleep(remain)
        frame_id += 1

except KeyboardInterrupt:
    print("[Interrupted]")
finally:
    cap.release()
    sock.close()


# 動画のFPSを20fps程度に抑えてリアルタイム性を維持する
# 人体の表示をキーボードで表示させる、カメラ位置などもキーボードで動かすように
# 

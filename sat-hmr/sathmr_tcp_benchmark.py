import cv2, socket, time, sys, pathlib, csv, yaml, numpy as np
from scipy.spatial.transform import Rotation
import torch, random
import os
from PIL import Image
from torchvision import transforms

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

# ========= 実行設定 =========
DEVICE_MODE = os.getenv("SATHMR_DEVICE_MODE", "auto")  # "auto" | "cpu" | "cuda"
CPU_THREADS = int(os.getenv("SATHMR_CPU_THREADS", "2"))
CPU_INTEROP_THREADS = int(os.getenv("SATHMR_CPU_INTEROP_THREADS", "1"))


def resolve_device(mode: str) -> torch.device:
    mode = mode.lower()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            print("[WARN] DEVICE_MODE=cuda ですが CUDA が利用できないため CPU にフォールバックします")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


torch.set_num_threads(CPU_THREADS)
torch.set_num_interop_threads(CPU_INTEROP_THREADS)

VIDEO_PATH = os.getenv("SATHMR_VIDEO_PATH", "videos/sample_video.mp4")
HOST = os.getenv("SATHMR_TCP_HOST", "127.0.0.1")
PORT = int(os.getenv("SATHMR_TCP_PORT", "5555"))
INPUT_SIZE = int(os.getenv("SATHMR_INPUT_SIZE", "1288"))
CONF_THRESH = float(os.getenv("SATHMR_CONF_THRESH", "0.3"))
LOG_INTERVAL = int(os.getenv("SATHMR_LOG_INTERVAL", "60"))
WARMUP_FRAMES = int(os.getenv("SATHMR_WARMUP_FRAMES", "10"))
WEIGHTS_PATH = os.getenv("SATHMR_WEIGHTS", "./weights/sat_hmr/sat_644.pth")

# 統計CSVファイル名
MODEL_NAME = os.getenv("SATHMR_MODEL_NAME", "sat_hmr")
STATS_DIR = pathlib.Path(os.getenv("SATHMR_STATS_DIR", "datas"))
STATS_DIR.mkdir(parents=True, exist_ok=True)
STATS_CSV_ENV = os.getenv("SATHMR_STATS_CSV", "").strip()
STATS_CSV = pathlib.Path(STATS_CSV_ENV) if STATS_CSV_ENV else (STATS_DIR / f"TCP_inference_stats_{MODEL_NAME}.csv")

print(f"[INFO] Model: {MODEL_NAME}")
print(f"[INFO] CSV output: {STATS_CSV}")

# ========= SAT-HMR インポート =========
from models import build_sat_model
from utils.transforms import unNormalize


def load_config(input_size):
    """SAT-HMRの設定を読み込み"""
    demo_cfg_path = os.path.join('configs', 'run', 'demo.yaml')
    with open(demo_cfg_path) as f:
        demo_config = yaml.safe_load(f)
    
    model_name = demo_config.get('model', 'default')
    model_cfg_path = os.path.join('configs', 'models', f'{model_name}.yaml')
    with open(model_cfg_path) as f:
        model_config = yaml.safe_load(f)
    
    class Config:
        pass
    
    args = Config()
    for k, v in model_config.items():
        setattr(args, k, v)
    for k, v in demo_config.items():
        setattr(args, k, v)
    
    args.mode = 'infer'
    args.pretrain = False
    args.pretrain_path = None
    args.input_size = input_size
    
    return args


def process_img_like_base(img, input_size):
    """BASEクラスのprocess_imgと同じ処理"""
    ori_h, ori_w = img.shape[:2]
    
    if ori_w >= ori_h:
        resize_rate = input_size / ori_w
        new_w = input_size
        new_h = int(resize_rate * ori_h)
    else:
        resize_rate = input_size / ori_h
        new_h = input_size
        new_w = int(resize_rate * ori_w)
    
    img_resized = cv2.resize(img, (new_w, new_h))
    img_size = (new_h, new_w)
    
    return img_resized, img_size, resize_rate


def preprocess_frame(frame, input_size, use_sat=True):
    """フレームを前処理"""
    img_resized, img_size, resize_rate = process_img_like_base(frame, input_size)
    
    patch_size = 56 if use_sat else 14
    pad_h = (img_size[0] + patch_size - 1) // patch_size * patch_size
    pad_w = (img_size[1] + patch_size - 1) // patch_size * patch_size
    
    pad_img = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
    pad_img[:img_size[0], :img_size[1]] = img_resized
    
    pil_img = Image.fromarray(pad_img[:, :, ::-1].copy())
    
    array2tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    norm_img = array2tensor(pil_img)
    
    return norm_img, img_size


def infer_frame(model, frame, input_size, conf_thresh, device):
    """1フレームを推論してSMPLパラメータを返す"""
    img_tensor, img_size = preprocess_frame(frame, input_size, use_sat=True)
    img_tensor = img_tensor.to(device)
    
    targets = [{
        'img_size': torch.tensor(img_size).to(device),
        'img_path': 'video_frame'
    }]
    
    with torch.no_grad():
        outputs = model([img_tensor], targets)
    
    idx = 0
    confs = outputs['pred_confs'][idx].squeeze(-1)
    select_idx = torch.where(confs > conf_thresh)[0]
    
    detection_flag = len(select_idx) > 0
    
    if not detection_flag:
        return {'detection_flag': False}
    
    # 最も信頼度の高い人物を選択
    best_idx = select_idx[confs[select_idx].argmax()]
    
    result = {
        'detection_flag': True,
        'pose': outputs['pred_poses'][idx][best_idx].detach().cpu().numpy(),
        'betas': outputs['pred_betas'][idx][best_idx].detach().cpu().numpy(),
        'transl': outputs['pred_transl'][idx][best_idx].detach().cpu().numpy(),
        'conf': confs[best_idx].detach().cpu().item(),
    }
    
    return result


def smpl_pose_to_ue_quaternions(pose):
    """SMPLポーズ(72,)をUE用クォータニオン(24, 4)に変換"""
    rotations = Rotation.from_rotvec(pose.reshape(-1, 3))
    quats = rotations.as_quat().astype(np.float32)
    
    quats[:, 0] *= -1.0
    quats[:, 2] *= -1.0
    
    flip_root = np.array([1., 0., 0., 0.], dtype=np.float32)
    quats[0] = (Rotation.from_quat(flip_root) * Rotation.from_quat(quats[0])).as_quat().astype(np.float32)
    
    return quats


def transl_to_ue(transl):
    """SMPL translationをUE座標系に変換"""
    trans = transl.copy().astype(np.float32)
    trans[1] *= -1.0
    trans[2] = 0.0
    trans = trans * 100.0
    return trans


# ---- TCP接続（サーバーとして待機）----
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((HOST, PORT))
sock.listen(1)
print(f"[Listening] {HOST}:{PORT}")

conn, addr = sock.accept()
conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
print(f"[Connected] from {addr}")
time_connection_success = time.perf_counter()

# ---- SAT-HMR 初期化 ----
print("\n[INFO] SAT-HMR モデルを読み込み中...")
model_args = load_config(INPUT_SIZE)
model, _ = build_sat_model(model_args, set_criterion=False)

state_dict = torch.load(WEIGHTS_PATH, map_location='cpu', weights_only=False)
model.load_state_dict(state_dict, strict=False)

device = resolve_device(DEVICE_MODE)

model = model.to(device)
model.eval()

print(f"[INFO] モデル読み込み完了 (device: {device})")
print(f"[INFO] CPU threads: {CPU_THREADS}, interop: {CPU_INTEROP_THREADS}")

# ---- 統計記録用リスト ----
inference_times = []
total_times = []
send_times = []

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
print(f"[INFO] FPS: {video_fps:.2f}, Frames: {video_frames}")

# ---- ウォームアップ ----
print(f"\n[WARMUP] Running {WARMUP_FRAMES} warmup frames...")
warmup_count = 0
while warmup_count < WARMUP_FRAMES:
    ok, frame = cap.read()
    if not ok:
        print("[ERROR] Video ended during warmup")
        sys.exit(1)
    _ = infer_frame(model, frame, INPUT_SIZE, CONF_THRESH, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    warmup_count += 1
    if warmup_count % 5 == 0:
        print(f"[WARMUP] {warmup_count}/{WARMUP_FRAMES} frames completed")

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
print("[WARMUP] Complete! Starting measurement...\n")

# ---- メインループ ----
frame_id = 0
t_all0 = time.perf_counter()
first_send_done = False
time_first_send = None
time_last_send = None

try:
    while cap.isOpened():
        t_loop0 = time.perf_counter()
        
        ok, frame = cap.read()
        if not ok:
            break
        
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
            tproc_ms = (time.perf_counter() - t_loop0) * 1000.0
            frame_id += 1
            continue
        
        # 座標系変換
        quats = smpl_pose_to_ue_quaternions(outs['pose'])
        trans = transl_to_ue(outs['transl'])
        
        # 送信フォーマット
        pose_str = ",".join(f"{v:.4f}" for v in quats.flatten())
        trans_str = ",".join(f"{v:.4f}" for v in trans.flatten())
        msg = f"{pose_str}|{trans_str}\x00"
        
        # TCP送信
        t_send0 = time.perf_counter()
        conn.sendall(msg.encode("ascii"))
        send_ms = (time.perf_counter() - t_send0) * 1000.0
        send_times.append(send_ms)
        
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
            print(f"frames={frame_id}  avg_fps={avg_fps:.2f}  infer_ms={infer_ms:.1f}  send_ms={send_ms:.2f}  total_ms={tproc_ms:.1f}  conf={outs['conf']:.2f}")

except KeyboardInterrupt:
    print("\n[Interrupted]")

finally:
    conn.close()
    sock.close()
    cap.release()
    
    # ---- 統計計算と保存 ----
    if len(inference_times) > 0:
        inf_array = np.array(inference_times)
        tot_array = np.array(total_times) if len(total_times) > 0 else np.array([0])
        send_array = np.array(send_times) if len(send_times) > 0 else np.array([0])
        
        inf_mean = np.mean(inf_array)
        inf_std = np.std(inf_array)
        inf_median = np.median(inf_array)
        tot_mean = np.mean(tot_array)
        tot_std = np.std(tot_array)
        tot_median = np.median(tot_array)
        send_mean = np.mean(send_array)
        send_std = np.std(send_array)
        
        print("\n" + "=" * 50)
        print("統計結果")
        print("=" * 50)
        print(f"推論時間 (ms):")
        print(f"  平均: {inf_mean:.2f} ± {inf_std:.2f}")
        print(f"  中央値: {inf_median:.2f}")
        print(f"  最小値: {np.min(inf_array):.2f}")
        print(f"  最大値: {np.max(inf_array):.2f}")
        print(f"  FPS (推論のみ): {1000/inf_mean:.2f}")
        print(f"\nTCP送信時間 (ms):")
        print(f"  平均: {send_mean:.2f} ± {send_std:.2f}")
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
            
            writer.writerow(['# 統計サマリー'])
            writer.writerow(['設定', '値', '', ''])
            writer.writerow(['device', device.type, '', ''])
            writer.writerow([])
            writer.writerow(['指標', '推論時間 (ms)', '送信時間 (ms)', '全体処理時間 (ms)'])
            writer.writerow(['平均', f'{inf_mean:.2f}', f'{send_mean:.2f}', f'{tot_mean:.2f}'])
            writer.writerow(['標準偏差', f'{inf_std:.2f}', f'{send_std:.2f}', f'{tot_std:.2f}'])
            writer.writerow(['中央値', f'{inf_median:.2f}', f'{np.median(send_array):.2f}', f'{tot_median:.2f}'])
            writer.writerow(['最小値', f'{np.min(inf_array):.2f}', f'{np.min(send_array):.2f}', f'{np.min(tot_array):.2f}'])
            writer.writerow(['最大値', f'{np.max(inf_array):.2f}', f'{np.max(send_array):.2f}', f'{np.max(tot_array):.2f}'])
            writer.writerow(['FPS (推論)', f'{1000/inf_mean:.2f}', '', ''])
            writer.writerow(['FPS (全体)', '', '', f'{1000/tot_mean:.2f}'])
            writer.writerow(['総フレーム数', len(inference_times), '', ''])
            writer.writerow(['検出フレーム数', len(total_times), '', ''])
            writer.writerow([])
            
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

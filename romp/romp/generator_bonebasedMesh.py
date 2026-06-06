import cv2, socket, time, sys, pathlib, pickle, json, numpy as np
from scipy.spatial.transform import Rotation
import yaml
import csv, torch, random
# ========= Seed固定 =========
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    
"""このファイルを実行することで、UEに生成したメッシュの情報を送信することができます。"""
"""骨格ベースで送信することによってUE側で服装に関する処理を効率化できるようにする"""

#TCP通信でUEに送る処理
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

VIDEO_PATH = "demo/videos/sample_video(camera_fixed).mp4"

CONFIG_YML = "configs/video.yml"  
try:
    with open(CONFIG_YML, 'r') as f:
        config = yaml.safe_load(f)
        # ARGS の下にネストされている場合に対応
        if 'ARGS' in config and 'model_path' in config['ARGS']:
            model_path = config['ARGS']['model_path']
        elif 'model_path' in config:
            model_path = config['model_path']
        else:
            model_path = 'unknown_model'
        
        # モデル名を抽出（パスの場合はファイル名のみ、拡張子なし）
        model_name = pathlib.Path(model_path).stem
        STATS_CSV = f"TCP_inference_stats_{model_name}.csv"
except Exception as e:
    print(f"[WARNING] Failed to read {CONFIG_YML}: {e}")
    model_name = "unknown_model"
    STATS_CSV = "TCP_inference_stats_unknown_model.csv"

print(f"[INFO] Model: {model_name}")
print(f"[INFO] CSV output: {STATS_CSV}")

arg_list = [
    "--configs_yml", "configs/video.yml",
    "--inputs",      VIDEO_PATH,
    "--gpu",         "-1"            
]

sys.argv = [sys.argv[0]] + arg_list
from romp.lib.config import parse_args, ConfigContext 
from romp.predict.video import Video_processor 

with ConfigContext(parse_args(arg_list)) as cfg:
    romp_proc = Video_processor(args_set=cfg)
    
pkl_path = pathlib.Path.home() / ".romp" / "smpl_model_data" / "SMPL_NEUTRAL.pkl"
with open(pkl_path, "rb") as f:
    smpl_data = pickle.load(f, encoding="latin1")
    
# ---- 統計記録用リスト ----
inference_times = []  # 推論時間（ミリ秒）
total_times = []      # 全体処理時間（ミリ秒）
    
faces = np.asarray(smpl_data["f"], dtype=np.int32)
faces_str = ",".join(map(str, faces.flatten()))

template_saved = False 
dt = 1/20
t_next = time.perf_counter() 
frame_id = 0

R_xm90 = np.array([[ 1, 0,  0],
                   [ 0, 0, -1],
                   [ 0, 1,  0]], dtype=np.float32)
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)


cap = cv2.VideoCapture(VIDEO_PATH)
    
# ---- ウォームアップ ----
WARMUP_FRAMES = 10
print(f"\n[WARMUP] Running {WARMUP_FRAMES} warmup frames...")
warmup_count = 0
while warmup_count < WARMUP_FRAMES:
    ok, frame = cap.read()
    if not ok:
        print("[ERROR] Video ended during warmup")
        sys.exit(1)
    outs = romp_proc.single_image_forward(frame)
    warmup_count += 1
    if warmup_count % 5 == 0:
        print(f"[WARMUP] {warmup_count}/{WARMUP_FRAMES} frames completed")
print("[WARMUP] Complete! Starting measurement...\n")

first_send_done = False
try:
    prev_t = time.perf_counter()
    t_all0 = time.perf_counter()  # 全体の開始時刻
    
    while cap.isOpened():
        t_loop0 = time.perf_counter()  # ループ開始時刻
        
        ok, frame = cap.read()
        if not ok:
            break
        
        # 推論計測
        t0 = time.perf_counter()
        start.record()
        outs = romp_proc.single_image_forward(frame)
        end.record()
        end.synchronize()
        infer_ms = start.elapsed_time(end)
        #infer_ms = (time.perf_counter() - t0) * 1000.0
        
        # 推論時間を記録
        inference_times.append(infer_ms)
        
        if outs is None or not outs["detection_flag"]:
            tproc_ms = (time.perf_counter() - t_loop0) * 1000.0
            frame_id += 1
            continue
        
        pose  = outs['params']['poses'][0].cpu().numpy().astype(np.float32)
        trans = outs['params']['cam'][0].cpu().numpy().astype(np.float32)
        
        rotations = Rotation.from_rotvec(pose.reshape(-1, 3))
        quaternions_for_ue = rotations.as_quat()

        quaternions_for_ue[:,0] *= -1.0
        quaternions_for_ue[:,2] *= -1.0
        flip_root = np.array([1., 0., 0., 0.], dtype=np.float32)
        q0 = Rotation.from_quat(flip_root) * Rotation.from_quat(quaternions_for_ue[0])
        quaternions_for_ue[0] = q0.as_quat().astype(np.float32)
        
        trans[1] *= -1.0
        trans[2] = 0.0   
        trans = trans * 100
        
        pose_str = ",".join(f"{v:.4f}" for v in quaternions_for_ue.flatten())
        trans_str = ",".join(f"{v:.4f}" for v in trans.flatten())
        msg = f"{pose_str}|{trans_str}\x00"
        t_s=time.perf_counter()
        conn.sendall(msg.encode("ascii"))
        if not first_send_done:
            time_first_send = time.perf_counter()
            startup_lag = time_first_send - time_connection_success
            print("-" * 40)
            print(f"[INFO] Startup Lag: {startup_lag:.4f} seconds")
            print("-" * 40)
            first_send_done = True
        
        
        send_ms=(time.perf_counter()-t_s)*1000.0
        
        # ループ総時間（デコード→推論→整形→送信）
        tproc_ms = (time.perf_counter() - t_loop0) * 1000.0
        total_times.append(tproc_ms)
        
        now_t = time.perf_counter()   
        elapsed = now_t - prev_t    
        prev_t = now_t
        
        if frame_id % 60 == 0:
            avg_fps = frame_id / (time.perf_counter() - t_all0)
            print(f"frames={frame_id}  avg_fps={avg_fps:.2f}  infer_ms={infer_ms:.1f}  全体_ms={tproc_ms:.1f}")
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
            
            # 統計サマリー部分
            writer.writerow(['# 統計サマリー'])
            writer.writerow(['指標', '推論時間 (ms)', '全体処理時間 (ms)'])
            writer.writerow(['平均', f'{inf_mean:.2f}', f'{tot_mean:.2f}'])
            writer.writerow(['標準偏差', f'{inf_std:.2f}', f'{tot_std:.2f}'])
            writer.writerow(['中央値', f'{inf_median:.2f}', f'{tot_median:.2f}'])
            writer.writerow(['最小値', f'{np.min(inf_array):.2f}', f'{np.min(tot_array):.2f}'])
            writer.writerow(['最大値', f'{np.max(inf_array):.2f}', f'{np.max(tot_array):.2f}'])
            writer.writerow(['総フレーム数', len(inference_times), ''])
            writer.writerow([])  # 空行
            
            # 全フレームデータ
            writer.writerow(['# 全フレームデータ'])
            writer.writerow(['frame_id', 'inference_time_ms', 'total_time_ms'])
            for i, (inf_t, tot_t) in enumerate(zip(inference_times, total_times)):
                writer.writerow([i, f'{inf_t:.2f}', f'{tot_t:.2f}'])
        
        print(f"\n統計データを保存: {STATS_CSV}")
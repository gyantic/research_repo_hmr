"""ROMP の推論結果を UDP で Unreal Engine へ送りながら、推論時間と全体処理時間を計測する（図5の ROMP）。

UE の受信 Blueprint が送る HELLO を待ち、その送信元へ 1 フレーム 1 パケットで姿勢を送る。
ウォームアップ 10 フレームの後、動画の全フレームについて推論時間（CUDA イベント）と
全体処理時間（読み込み〜推論〜整形〜送信）を記録し、統計サマリーと全フレームデータを CSV に書く。

romp/ ディレクトリから、ROMP の README と同じ PYTHONPATH で実行する:

    cd romp
    PYTHONPATH="$PWD:$PWD/romp" python ../measurements/romp/benchmark_romp_udp_ue.py \
        --video ../datasets/videos/sample_video.mp4 --stats_csv ../results/measurements/raw_romp_hrnet_bf16_udp_ue.csv
"""
import argparse
import csv
import pathlib
import socket
import sys
import time

import cv2
import numpy as np
import torch
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LOG_INTERVAL_FRAMES = 60  # 平均FPSを表示する間隔


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", default=str(REPO_ROOT / "datasets" / "videos" / "sample_video.mp4"))
    parser.add_argument("--configs_yml", default=str(REPO_ROOT / "romp" / "configs" / "video.yml"),
                        help="ROMP の設定（論文: HRNet32・bf16・torch.compile）")
    parser.add_argument("--stats_csv", default=None,
                        help="出力CSV（既定: inference_stats_<model_path の名前>.csv）")
    parser.add_argument("--ue_port", type=int, default=62004, help="UE の受信ポート")
    parser.add_argument("--local_port", type=int, default=5555, help="HELLO を待つローカルポート")
    parser.add_argument("--warmup_frames", type=int, default=10)
    return parser.parse_args()


def default_stats_csv(configs_yml):
    with open(configs_yml) as f:
        config = yaml.safe_load(f)
    model_path = (config.get("ARGS") or config).get("model_path", "unknown_model")
    return f"inference_stats_{pathlib.Path(model_path).stem}.csv"


def to_numpy_float32(x):
    if hasattr(x, "detach"):
        return x.detach().float().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def wait_for_hello(local_port):
    """UE の HELLO を受け取るまで待ち、(ソケット, 送信先ホスト, HELLO受信時刻) を返す。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", local_port))
    print(f"[WAIT] HELLO on 0.0.0.0:{local_port}")
    _hello, addr = sock.recvfrom(2048)
    print(f"[HELLO] from {addr}")
    return sock, addr[0], time.perf_counter()


def write_stats_csv(path, inference_times_ms, total_times_ms):
    """統計サマリーと全フレームデータを、論文の raw CSV と同じ形式で書く。"""
    inference = np.array(inference_times_ms)
    total = np.array(total_times_ms)
    has_total = len(total) > 0
    total_stats = {
        "mean": np.mean(total) if has_total else np.nan,
        "std": np.std(total) if has_total else np.nan,
        "median": np.median(total) if has_total else np.nan,
    }
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# 統計サマリー"])
        writer.writerow(["指標", "推論時間 (ms)", "全体処理時間 (ms)"])
        writer.writerow(["平均", f"{np.mean(inference):.2f}", f"{total_stats['mean']:.2f}"])
        writer.writerow(["標準偏差", f"{np.std(inference):.2f}", f"{total_stats['std']:.2f}"])
        writer.writerow(["中央値", f"{np.median(inference):.2f}", f"{total_stats['median']:.2f}"])
        writer.writerow(["最小値", f"{np.min(inference):.2f}", f"{np.min(total):.2f}" if has_total else ""])
        writer.writerow(["最大値", f"{np.max(inference):.2f}", f"{np.max(total):.2f}" if has_total else ""])
        writer.writerow(["総フレーム数", len(inference_times_ms), ""])
        writer.writerow([])
        writer.writerow(["# 全フレームデータ"])
        writer.writerow(["frame_id", "inference_time_ms", "total_time_ms"])
        for i, (inference_ms, total_ms) in enumerate(zip(inference_times_ms, total_times_ms)):
            writer.writerow([i, f"{inference_ms:.2f}", f"{total_ms:.2f}"])
    print(f"\n統計データを保存: {path}")
    print(f"推論時間 (ms): 平均 {np.mean(inference):.2f} ± {np.std(inference):.2f}, 中央値 {np.median(inference):.2f}")
    if has_total:
        print(f"全体処理時間 (ms): 平均 {total_stats['mean']:.2f} ± {total_stats['std']:.2f}, 中央値 {total_stats['median']:.2f}")
    print(f"総フレーム数: {len(inference_times_ms)}")


def main():
    args = parse_args()
    stats_csv = args.stats_csv or default_stats_csv(args.configs_yml)
    torch.set_num_threads(2)          # 計測時と同じ CPU スレッド数
    torch.set_num_interop_threads(1)

    sock, ue_host, hello_time = wait_for_hello(args.local_port)

    from romp.processor_bootstrap import build_video_processor
    from romp.ue_pose_message import format_ue_pose_message
    romp_proc = build_video_processor(args.configs_yml, args.video, "0")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {args.video}")
    for _ in range(args.warmup_frames):
        ok, frame = cap.read()
        if not ok:
            sys.exit("[ERROR] Video ended during warmup")
        romp_proc.single_image_forward(frame)
    print(f"[WARMUP] {args.warmup_frames} frames done. Starting measurement...")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    inference_times_ms, total_times_ms = [], []
    first_send_time = last_send_time = None
    measure_start = time.perf_counter()
    try:
        while True:
            loop_start = time.perf_counter()  # 全体処理時間は読み込みから送信まで
            ok, frame = cap.read()
            if not ok:
                break

            start_event.record()
            outs = romp_proc.single_image_forward(frame)
            # 論文の計測時はこの出力表示が推論の計測区間内にあった。計測値を合わせるため残す。
            for key, value in outs.items():
                print(key, value.shape if hasattr(value, "shape") else type(value))
            end_event.record()
            end_event.synchronize()
            inference_ms = start_event.elapsed_time(end_event)
            inference_times_ms.append(inference_ms)

            if outs is not None and outs.get("detection_flag", False):
                pose = to_numpy_float32(outs["params"]["poses"][0])
                cam = to_numpy_float32(outs["params"]["cam"][0])
                payload = (format_ue_pose_message(pose, cam) + "\n").encode("ascii")
                sock.sendto(payload, (ue_host, args.ue_port))
                last_send_time = time.perf_counter()
                if first_send_time is None:
                    first_send_time = last_send_time
                    print(f"[INFO] Startup Lag: {first_send_time - hello_time:.4f} seconds")

            total_ms = (time.perf_counter() - loop_start) * 1000.0
            total_times_ms.append(total_ms)
            frame_count = len(total_times_ms)
            if frame_count % LOG_INTERVAL_FRAMES == 0:
                avg_fps = frame_count / (time.perf_counter() - measure_start)
                print(f"frames={frame_count}  avg_fps={avg_fps:.2f}  infer_ms={inference_ms:.1f}  total_ms={total_ms:.1f}")
    except KeyboardInterrupt:
        print("[Interrupted]")
    finally:
        sock.close()
        cap.release()
        if inference_times_ms:
            write_stats_csv(stats_csv, inference_times_ms, total_times_ms)
        if first_send_time is not None:
            print(f"[INFO] 実データ送信時間: {last_send_time - first_send_time:.4f} 秒")


if __name__ == "__main__":
    main()

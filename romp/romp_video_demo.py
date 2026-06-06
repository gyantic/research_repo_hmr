"""
ROMP Video Demo  (SAT-HMR video_demo.py に準拠)

速度計測だけしたい場合:
    python romp_video_demo.py -i demo/videos/sample_video.mp4 \
        --no_render --no_save \
        --metrics_csv datas/romp_metrics.csv

通常デモ:
    python romp_video_demo.py -i demo/videos/sample_video.mp4 \
        -o demo/output.mp4

bf16 モード:
    python romp_video_demo.py -i demo/videos/sample_video.mp4 \
        --dtype_cfg bf16 --no_render --no_save \
        --metrics_csv datas/romp_metrics_bf16.csv
"""

import argparse
import csv
import os
import pathlib
import random
import sys
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

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

torch.set_num_threads(2)
torch.set_num_interop_threads(1)


def get_args():
    p = argparse.ArgumentParser("ROMP Video Demo")
    p.add_argument("--input",  "-i", required=True,  help="入力動画ファイルパス")
    p.add_argument("--output", "-o", default=None,   help="出力動画ファイルパス")
    p.add_argument("--configs_yml", default="configs/video.yml", help="ROMPコンフィグ")
    p.add_argument("--dtype_cfg",   default="fp32",  help="精度: fp32 / bf16")
    compile_group = p.add_mutually_exclusive_group()
    compile_group.add_argument("--torch_compile", dest="torch_compile", action="store_true",
                               help="torch.compile を有効化")
    compile_group.add_argument("--no_torch_compile", dest="torch_compile", action="store_false",
                               help="torch.compile を無効化")
    torchao_group = p.add_mutually_exclusive_group()
    torchao_group.add_argument("--torchao_int8_weight_only", dest="torchao_int8_weight_only", action="store_true",
                               help="torchao の int8 weight-only を有効化")
    torchao_group.add_argument("--no_torchao_int8_weight_only", dest="torchao_int8_weight_only", action="store_false",
                               help="torchao の int8 weight-only を無効化")
    p.add_argument("--torchao_int8_scope", default=None, choices=["full", "backbone"],
                   help="torchao int8 weight-only の適用範囲")
    p.add_argument("--no_render",   action="store_true", help="メッシュ描画スキップ")
    p.add_argument("--no_save",     action="store_true", help="動画保存スキップ")
    p.add_argument("--metrics_csv", default=None,    help="フレーム毎タイミングCSV出力パス")
    p.add_argument("--warmup_frames", type=int, default=10)
    p.add_argument("--display",     action="store_true", help="画面表示")
    p.set_defaults(torch_compile=None)
    p.set_defaults(torchao_int8_weight_only=None)
    return p.parse_args()


# ========= ROMP 初期化 =========
args_cli = get_args()

arg_list = [
    "--configs_yml", args_cli.configs_yml,
    "--inputs",      args_cli.input,
    "--gpu",         "0",
]
# dtype_cfg を model_precision に変換してconfig側に渡す
_precision = "bf16" if str(args_cli.dtype_cfg).lower() in {"bf16", "bfloat16"} else "fp32"
arg_list += ["--model_precision", _precision]
if args_cli.torch_compile is not None:
    arg_list += ["--torch_compile", "true" if args_cli.torch_compile else "false"]
if args_cli.torchao_int8_weight_only is not None:
    arg_list += ["--torchao_int8_weight_only", "true" if args_cli.torchao_int8_weight_only else "false"]
if args_cli.torchao_int8_scope is not None:
    arg_list += ["--torchao_int8_scope", args_cli.torchao_int8_scope]

sys.argv = [sys.argv[0]] + arg_list

from romp.lib.config import parse_args, ConfigContext
from romp.predict.video import Video_processor
from dataset.internet import img_preprocess as romp_img_preprocess

with ConfigContext(parse_args(arg_list)) as cfg:
    romp_proc = Video_processor(args_set=cfg)

romp_proc.model.eval()


# ========= process_frame =========
def process_frame(romp_proc, frame, no_render=False):
    """1フレームを前処理→推論→後処理し times[str→秒] と num_persons を返す"""
    times = {}

    # --- 前処理 ---
    t0 = time.time()
    meta_data = romp_img_preprocess(
        frame, "0",
        input_size=romp_proc.input_size,
        single_img_input=True,
    )
    if "-1" not in romp_proc.gpu:
        meta_data["image"] = meta_data["image"].cuda()
    times["preprocess"] = time.time() - t0

    # --- 推論 ---
    t0 = time.time()
    outputs = romp_proc.net_forward(meta_data, cfg=romp_proc.demo_cfg)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times["inference"] = time.time() - t0

    # --- 後処理（人物数カウント）---
    t0 = time.time()
    detection_flag = outputs.get("detection_flag", False)
    num_persons = 0
    if detection_flag:
        reorg = outputs.get("reorganize_idx", None)
        if reorg is not None and len(reorg) > 0:
            num_persons = int(reorg.max().item()) + 1
        else:
            num_persons = 1
    times["postprocess"] = time.time() - t0

    # --- 描画（--no_render でスキップ）---
    result_frame = frame
    if not no_render and detection_flag:
        t0 = time.time()
        try:
            from romp.lib.visualization.visualization import Visualizer
            if not hasattr(process_frame, "_vis"):
                process_frame._vis = Visualizer(
                    resolution=(512, 512),
                    result_img_dir=None,
                    with_renderer=True,
                )
            result_frame = process_frame._vis.visulize_result_onorg(
                outputs, meta_data, show_items=["mesh"]
            )
        except Exception:
            pass  # 描画エラーは無視して推論計測に専念
        times["render"] = time.time() - t0

    return result_frame, num_persons, times


# ========= 動画読み込み =========
cap = cv2.VideoCapture(args_cli.input)
if not cap.isOpened():
    print(f"[ERROR] Cannot open: {args_cli.input}")
    sys.exit(1)

video_fps    = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[INFO] Video : {args_cli.input}")
print(f"[INFO] Size  : {frame_w}x{frame_h}, FPS: {video_fps:.2f}, Frames: {total_frames}")
print(f"[INFO] dtype : {args_cli.dtype_cfg}")
print(f"[INFO] torch.compile : {getattr(romp_proc, 'torch_compile', False)}")
print(f"[INFO] torchao int8 weight-only : {getattr(romp_proc, 'torchao_int8_weight_only', False)}")
print(f"[INFO] torchao int8 scope : {getattr(romp_proc, 'torchao_int8_scope', 'full')}")

# ========= ウォームアップ =========
print(f"\n[WARMUP] {args_cli.warmup_frames} frames ...")
for i in range(args_cli.warmup_frames):
    ok, frame = cap.read()
    if not ok:
        print("[ERROR] Video ended during warmup")
        sys.exit(1)
    process_frame(romp_proc, frame, no_render=True)
    if i % 5 == 4:
        print(f"[WARMUP] {i+1}/{args_cli.warmup_frames}")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
print("[WARMUP] Complete!\n")

# ========= 出力動画ライター =========
writer = None
if not args_cli.no_save:
    out_path = args_cli.output or (
        pathlib.Path(args_cli.input).stem + "_romp.mp4"
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, video_fps, (frame_w, frame_h))

# ========= メトリクスCSV =========
metrics_fp     = None
metrics_writer = None
if args_cli.metrics_csv:
    metrics_dir = os.path.dirname(args_cli.metrics_csv)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    metrics_fp     = open(args_cli.metrics_csv, "w", newline="", encoding="utf-8")
    metrics_writer = csv.writer(metrics_fp)
    metrics_writer.writerow([
        "frame", "persons",
        "preprocess_ms", "inference_ms", "postprocess_ms", "render_ms",
        "total_ms", "fps_inst", "fps_avg_30",
    ])

# ========= メインループ =========
all_times    = {"preprocess": [], "inference": [], "postprocess": [], "render": []}
elapsed_times = []
fps_queue    = []
total_persons = 0
pbar = tqdm(total=total_frames, desc="処理中")

try:
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result_frame, num_persons, times = process_frame(
            romp_proc, frame, no_render=args_cli.no_render
        )
        total_persons += num_persons

        for k, v in times.items():
            all_times[k].append(v)

        elapsed = sum(times.values())
        elapsed_times.append(elapsed)
        current_fps = 1.0 / elapsed if elapsed > 0 else 0.0
        fps_queue.append(current_fps)
        if len(fps_queue) > 30:
            fps_queue.pop(0)
        avg_fps_30 = float(np.mean(fps_queue))

        if metrics_writer is not None:
            metrics_writer.writerow([
                frame_idx,
                num_persons,
                f'{times["preprocess"] * 1000:.4f}',
                f'{times["inference"]  * 1000:.4f}',
                f'{times["postprocess"]* 1000:.4f}',
                f'{times["render"]     * 1000:.4f}' if "render" in times else "",
                f'{elapsed             * 1000:.4f}',
                f'{current_fps:.6f}',
                f'{avg_fps_30:.6f}',
            ])
            if frame_idx % 30 == 0:
                metrics_fp.flush()

        if writer is not None:
            writer.write(result_frame)

        if frame_idx % 30 == 0:
            render_ms = f'{times["render"]*1000:.0f}ms' if "render" in times else "skip"
            print(
                f"Frame {frame_idx}/{total_frames} | FPS {current_fps:.1f} | "
                f"persons {num_persons} | "
                f"pre {times['preprocess']*1000:.0f}ms | "
                f"inf {times['inference']*1000:.0f}ms | "
                f"post {times['postprocess']*1000:.0f}ms | "
                f"render {render_ms}"
            )

        if args_cli.display:
            cv2.imshow("ROMP Video Demo", result_frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

        frame_idx += 1
        pbar.update(1)
        pbar.set_postfix({"persons": num_persons, "fps": f"{current_fps:.1f}"})

except KeyboardInterrupt:
    print("\n中断されました")

finally:
    pbar.close()
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # ========= CSVサマリー書き込み =========
    if metrics_fp is not None and len(all_times["inference"]) > 0:
        inf_ms  = np.array(all_times["inference"]) * 1000
        tot_ms  = np.array(elapsed_times) * 1000
        metrics_writer.writerow([])
        metrics_writer.writerow(["# サマリー"])
        metrics_writer.writerow(["指標", "推論時間 (ms)", "全体処理時間 (ms)", "FPS"])
        metrics_writer.writerow(["mean",   f"{np.mean(inf_ms):.2f}",   f"{np.mean(tot_ms):.2f}",   f"{1000/np.mean(tot_ms):.2f}"])
        metrics_writer.writerow(["median", f"{np.median(inf_ms):.2f}", f"{np.median(tot_ms):.2f}", ""])
        metrics_writer.writerow(["std",    f"{np.std(inf_ms):.2f}",    f"{np.std(tot_ms):.2f}",    ""])
        metrics_writer.writerow(["min",    f"{np.min(inf_ms):.2f}",    f"{np.min(tot_ms):.2f}",    ""])
        metrics_writer.writerow(["max",    f"{np.max(inf_ms):.2f}",    f"{np.max(tot_ms):.2f}",    ""])
        metrics_fp.close()
        print(f"保存: {args_cli.metrics_csv}")

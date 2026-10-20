#!/usr/bin/env python3
"""図4・UEのデモで使った CMU Panoptic Studio の8クリップ（各15秒・450フレーム）を、HD 映像（カメラ hd_00_00）から切り出す。

CLIPS の開始・終了フレームは、人物が1人だけ映っている区間（YOLO の人数判定で抽出し、人物の交代で分割して、
両端の人物が映っていないフレームを除いたもの）。各クリップはこの区間の 40% の位置を中心とする連続450フレーム
（区間の先頭から max(60, 区間長×0.4 − 225) フレーム目から）で、ファイル名には区間を付ける。
HD 映像が無ければ CMU の配布サーバ（panoptic-toolbox の scripts/getData.sh と同じ場所）から取得する。

    python measurements/cmu_panoptic/make_person_clips.py [--video_dir datasets/cmu_panoptic/hd_videos] \
        [--output_dir datasets/cmu_panoptic/person_clips]
"""

import argparse
import json
import subprocess
import urllib.request
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VIDEO_URL = "http://domedb.perception.cs.cmu.edu/webdata/dataset/{sequence}/videos/hd_shared_crf20/hd_00_00.mp4"

CLIP_FRAMES = 450  # 15秒（29.97 fps）

# (シーケンス, 区間の人物番号, 区間の開始フレーム, 区間の終了フレーム)
CLIPS = [
    ("171026_pose1", 1, 160, 7453),
    ("171204_pose2", 1, 141, 4395),
    ("171204_pose2", 2, 4654, 8865),  # 概要図（図1）の入力
    ("171204_pose2", 7, 28652, 32879),
    ("171204_pose3", 1, 138, 4485),
    ("171204_pose3", 2, 4886, 9056),
    ("171204_pose4", 1, 128, 4349),
    ("171204_pose4", 3, 9184, 13410),
    ("171204_pose5", 3, 9344, 13540),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video_dir", type=Path, default=REPO_ROOT / "datasets" / "cmu_panoptic" / "hd_videos",
                        help="HD 映像 <シーケンス>_hd_00_00.mp4 の置き場所（無ければ取得する）")
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "datasets" / "cmu_panoptic" / "person_clips")
    return parser.parse_args()


def download_if_missing(sequence, video_dir):
    path = video_dir / f"{sequence}_hd_00_00.mp4"
    if path.exists():
        return path
    video_dir.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".mp4.part")
    print(f"downloading {sequence} ...", flush=True)
    urllib.request.urlretrieve(VIDEO_URL.format(sequence=sequence), partial)
    partial.replace(path)
    return path


def frame_rate(path):
    info = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate", "-of", "json", str(path),
    ]))
    return float(Fraction(info["streams"][0]["r_frame_rate"]))


def clip_start_frame(interval_start, interval_end):
    """区間の 40% の位置を中心とする450フレームの先頭（区間の先頭から最低60フレーム後ろ）。"""
    interval_frames = interval_end - interval_start + 1
    return interval_start + max(60, int(interval_frames * 0.4) - CLIP_FRAMES // 2)


def cut_clip(source, destination, start_frame):
    fps = frame_rate(source)
    partial = destination.with_name(f".{destination.name}.part.mp4")
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start_frame / fps:.9f}", "-i", str(source),
        "-map", "0:v:0", "-an", "-frames:v", str(CLIP_FRAMES),
        "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "20", "-b:v", "0",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial),
    ], check=True)
    partial.replace(destination)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sequence, person, interval_start, interval_end in CLIPS:
        destination = args.output_dir / f"{sequence}_person_{person:02d}_f{interval_start:08d}-{interval_end:08d}.mp4"
        if destination.exists():
            print("exists:", destination.name)
            continue
        source = download_if_missing(sequence, args.video_dir)
        cut_clip(source, destination, clip_start_frame(interval_start, interval_end))
        print("saved:", destination.name, flush=True)


if __name__ == "__main__":
    main()

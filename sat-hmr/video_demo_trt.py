"""
SAT-HMR Video Demo (TensorRT)
- Keep original video_demo.py (Torch) intact.
- Run inference from video file with TensorRT engine (.engine).
"""

import argparse
import csv
import os
import sys
import time
from typing import Dict, Optional

import cv2
import numpy as np
import torch

from configs.paths import smpl_model_path
from models.human_models import SMPL_Layer
from sathmr_common import (
    UdpPoseStreamer,
    add_udp_args,
    build_cam_intrinsics_from_img_size,
    empty_udp_candidates,
    make_udp_candidates,
    setup_render_imports,
)
from trt_runner import (
    TensorRTRunner,
    draw_rec_marker,
    draw_timing_overlay,
    format_timing_status,
    infer_frame,
    normalize_demo_args,
    open_runner_or_exit,
    pick_output,
    print_demo_settings,
    render_mesh_frame,
    select_confident_queries,
    update_fps_window,
)


METRICS_CSV_COLUMNS = [
    'frame',
    'persons',
    'preprocess_ms',
    'inference_ms',
    'postprocess_ms',
    'render_ms',
    'total_ms',
    'fps_inst',
    'fps_avg_30',
    'udp_enabled',
]


def get_args():
    parser = argparse.ArgumentParser('SAT-HMR Video Demo (TensorRT)')
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', default=None, help='Output video path')
    parser.add_argument('--engine', default='datas/trt/sat_hmr_sat_fp16_vc.engine', help='TensorRT engine path')
    parser.add_argument('--conf_thresh', type=float, default=0.3, help='Detection confidence threshold')
    parser.add_argument('--input_size', type=int, default=1288, help='SAT-HMR base input size (for preprocessing)')

    parser.add_argument('--no_render', action='store_true', help='Skip mesh rendering')
    parser.add_argument('--render_every', type=int, default=1, help='Render every N frames')
    parser.add_argument('--show_fps', action='store_true', help='Overlay FPS text')
    parser.add_argument('--show', action='store_true', help='Show preview window')
    parser.add_argument('--no_display', action='store_true', help='Do not display window (also disables rendering)')
    parser.add_argument('--no_save', action='store_true', help='Do not save result video')
    parser.add_argument('--metrics_csv', default=None, help='Per-frame metrics CSV output path')
    parser.add_argument('--pred_npz', default=None, help='Per-frame prediction NPZ output path (for pseudo-GT comparison)')
    parser.add_argument('--udp_only', action='store_true', help='UE send only mode (no render, no display)')

    add_udp_args(parser)

    return parser.parse_args()


def empty_pred_record():
    return {
        'joints2d': np.zeros((0, 24, 2), dtype=np.float32),
        'joints3d': np.zeros((0, 24, 3), dtype=np.float32),
        'verts': np.zeros((0, 0, 3), dtype=np.float32),
        'transl': np.zeros((0, 3), dtype=np.float32),
        'bboxes': np.zeros((0, 4), dtype=np.float32),
        'focal_length': np.zeros((0,), dtype=np.float32),
    }


def reconstruct_smpl_in_camera(
    smpl_layer: SMPL_Layer,
    smpl_device: Optional[torch.device],
    poses: np.ndarray,
    betas: np.ndarray,
    transl: np.ndarray,
    cam_intrinsics: np.ndarray,
):
    """Returns float32 camera-space (verts, 24 joints 3D, their 2D projection)."""
    if smpl_device is None:
        smpl_device = torch.device('cpu')
    with torch.no_grad():
        poses_t = torch.from_numpy(poses).to(device=smpl_device, non_blocking=True)
        betas_t = torch.from_numpy(betas).to(device=smpl_device, non_blocking=True)
        verts, joints = smpl_layer(poses=poses_t, betas=betas_t)
        trans_t = torch.from_numpy(transl).to(device=smpl_device, non_blocking=True).unsqueeze(1)
        verts_cam = verts + trans_t
        joints_cam = joints + trans_t
        verts_np = verts_cam.cpu().numpy().astype(np.float32)
        joints3d = joints_cam[:, :24, :].cpu().numpy().astype(np.float32)

    joints2d = np.zeros((len(poses), 24, 2), dtype=np.float32)
    z = np.clip(joints3d[..., 2:3], 1e-6, None)
    joints2d[..., 0] = joints3d[..., 0] * cam_intrinsics[0, 0] / z[..., 0] + cam_intrinsics[0, 2]
    joints2d[..., 1] = joints3d[..., 1] * cam_intrinsics[1, 1] / z[..., 0] + cam_intrinsics[1, 2]
    return verts_np, joints3d, joints2d


def process_frame_trt(
    runner: TensorRTRunner,
    frame: np.ndarray,
    input_size: int,
    conf_thresh: float,
    no_render: bool,
    smpl_layer: Optional[SMPL_Layer],
    smpl_device: Optional[torch.device],
    render_utils: Optional[Dict[str, object]],
):
    times: Dict[str, float] = {}
    img_tensor, img_size, orig_size, outputs = infer_frame(runner, frame, input_size, times)

    t0 = time.time()
    confs, select_idx = select_confident_queries(outputs, conf_thresh)
    num_persons = int(select_idx.size)

    if 'pred_intrinsics' in outputs:
        cam_intrinsics = outputs['pred_intrinsics'][0].reshape(3, 3).astype(np.float32)
    else:
        cam_intrinsics = build_cam_intrinsics_from_img_size(img_size, input_size)

    udp_candidates = empty_udp_candidates()
    pred_record = empty_pred_record()
    if num_persons > 0:
        pred_poses = pick_output(outputs, 'pred_poses')[0][select_idx].astype(np.float32)
        pred_betas = pick_output(outputs, 'pred_betas')[0][select_idx].astype(np.float32)
        pred_transl = pick_output(outputs, 'pred_transl')[0][select_idx].astype(np.float32)
        pred_boxes = pick_output(outputs, 'pred_boxes')[0][select_idx].astype(np.float32)
        udp_candidates = make_udp_candidates(pred_poses, pred_transl, confs, select_idx)

        # Without an SMPL layer only the network outputs are recorded.
        if smpl_layer is not None:
            pred_record['verts'], pred_record['joints3d'], pred_record['joints2d'] = reconstruct_smpl_in_camera(
                smpl_layer, smpl_device, pred_poses, pred_betas, pred_transl, cam_intrinsics
            )
        pred_record['transl'] = pred_transl
        pred_record['bboxes'] = pred_boxes
        pred_record['focal_length'] = np.full((num_persons,), float(cam_intrinsics[0, 0]), dtype=np.float32)

    times['postprocess'] = time.time() - t0

    if no_render or render_utils is None or smpl_layer is None:
        return frame, num_persons, times, udp_candidates, pred_record

    t0 = time.time()
    result_img = render_mesh_frame(
        render_utils,
        img_tensor,
        img_size,
        orig_size,
        pred_record['verts'] if num_persons > 0 else None,
        smpl_layer.faces,
        torch.from_numpy(cam_intrinsics),
    )
    times['render'] = time.time() - t0

    return result_img, num_persons, times, udp_candidates, pred_record


class PredictionNpzWriter:
    """Collects per-frame predictions and saves them in the pseudo-GT comparison NPZ layout."""

    RECORD_KEYS = ('joints2d', 'joints3d', 'verts', 'transl', 'bboxes', 'focal_length')

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.frame_ids = []
        self.image_paths = []
        self.person_counts = []
        self.records = {key: [] for key in self.RECORD_KEYS}

    def add(self, frame_id: int, num_persons: int, pred_record: Dict[str, np.ndarray]):
        self.frame_ids.append(frame_id)
        self.image_paths.append(f"{self.video_path}#frame_{frame_id:06d}")
        self.person_counts.append(num_persons)
        for key in self.RECORD_KEYS:
            self.records[key].append(pred_record[key])

    def save(self, path: str):
        pred_dir = os.path.dirname(path)
        if pred_dir:
            os.makedirs(pred_dir, exist_ok=True)
        np.savez_compressed(
            path,
            frame_ids=np.asarray(self.frame_ids, dtype=np.int64),
            image_paths=np.asarray(self.image_paths, dtype=object),
            person_counts=np.asarray(self.person_counts, dtype=np.int32),
            **{key: np.asarray(values, dtype=object) for key, values in self.records.items()},
            failed_frames=np.int32(0),
        )


def open_metrics_csv(path: str):
    metrics_dir = os.path.dirname(path)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    metrics_fp = open(path, 'w', newline='', encoding='utf-8')
    metrics_writer = csv.writer(metrics_fp)
    metrics_writer.writerow(METRICS_CSV_COLUMNS)
    return metrics_fp, metrics_writer


def open_video_writer(args, fps: float, width: int, height: int):
    """Returns a cv2.VideoWriter, or None when saving is disabled or the writer fails to open."""
    if args.no_save:
        return None
    if args.output is None:
        base, _ = os.path.splitext(args.input)
        args.output = f'{base}_trt.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not writer.isOpened():
        print('[WARN] VideoWriter の初期化に失敗したため保存を無効化します')
        return None
    print(f'出力動画: {args.output}')
    return writer


def print_time_summary(all_times: Dict[str, list]):
    print('\n--- 処理時間 (平均) ---')
    total_time_ms = 0.0
    for k, v in all_times.items():
        if len(v) > 0:
            avg_ms = float(np.mean(v) * 1000.0)
            total_time_ms += avg_ms
            print(f'  {k}: {avg_ms:.1f} ms')
    if total_time_ms > 0:
        print(f'  合計: {total_time_ms:.1f} ms ({1000.0 / total_time_ms:.1f} FPS)')


def main():
    args = get_args()
    if args.show:
        args.no_display = False
    normalize_demo_args(args)

    # Video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f'エラー: 動画を開けません: {args.input}')
        sys.exit(1)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_fps <= 0:
        video_fps = 30.0

    print(f'入力動画: {args.input}')
    print(f'解像度: {actual_width}x{actual_height}, FPS: {video_fps:.2f}, フレーム数: {total_frames}')

    writer = open_video_writer(args, video_fps, actual_width, actual_height)

    runner = open_runner_or_exit(args.engine)

    render_ok, render_utils = setup_render_imports(args.no_render)
    if not render_ok:
        args.no_render = True

    smpl_layer = None
    smpl_device = None
    if (not args.no_render) or args.pred_npz:
        if args.no_render and args.pred_npz:
            print('[INFO] Initialize SMPL layer for NPZ export...')
        else:
            print('[INFO] Initialize SMPL layer for mesh rendering...')
        smpl_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        smpl_layer = SMPL_Layer(model_path=smpl_model_path, with_genders=False).to(smpl_device).eval()

    print_demo_settings(args)

    print('\n=== 操作方法 ===')
    if not args.no_display:
        print("  'q' または ESC: 終了 (早期終了)")
    else:
        print('  Ctrl+C: 終了')
    print('==================\n')

    frame_count = 0
    fps_queue = []
    total_persons = 0
    all_times = {'preprocess': [], 'inference': [], 'postprocess': [], 'render': []}
    npz_writer = PredictionNpzWriter(args.input) if args.pred_npz else None
    metrics_fp = None
    metrics_writer = None
    if args.metrics_csv:
        metrics_fp, metrics_writer = open_metrics_csv(args.metrics_csv)

    udp = None
    if args.udp:
        udp = UdpPoseStreamer(args)
        udp.wait_for_hello()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('動画の終端に到達しました')
                break

            start_time = time.time()
            render_this_frame = (not args.no_render) and (frame_count % args.render_every == 0)

            result_frame, num_persons, times, udp_candidates, pred_record = process_frame_trt(
                runner=runner,
                frame=frame,
                input_size=args.input_size,
                conf_thresh=args.conf_thresh,
                no_render=not render_this_frame,
                smpl_layer=smpl_layer,
                smpl_device=smpl_device,
                render_utils=render_utils,
            )
            total_persons += num_persons
            if npz_writer is not None:
                npz_writer.add(frame_count, num_persons, pred_record)
            for k, v in times.items():
                if k in all_times:
                    all_times[k].append(v)

            if udp is not None:
                udp.process(frame_count, num_persons, udp_candidates)

            elapsed = time.time() - start_time
            current_fps, avg_fps = update_fps_window(fps_queue, elapsed)
            if metrics_writer is not None:
                metrics_writer.writerow([
                    frame_count,
                    num_persons,
                    f'{times["preprocess"] * 1000.0:.4f}',
                    f'{times["inference"] * 1000.0:.4f}',
                    f'{times["postprocess"] * 1000.0:.4f}',
                    f'{times["render"] * 1000.0:.4f}' if 'render' in times else '',
                    f'{elapsed * 1000.0:.4f}',
                    f'{current_fps:.6f}',
                    f'{avg_fps:.6f}',
                    int(args.udp),
                ])
                if frame_count % 30 == 0:
                    metrics_fp.flush()

            if args.show_fps and not args.no_display:
                draw_timing_overlay(result_frame, current_fps, num_persons, times)

            if writer is not None:
                draw_rec_marker(result_frame, actual_width)
                writer.write(result_frame)

            if args.no_display:
                if frame_count % 30 == 0:
                    print(
                        f"Frame: {frame_count}/{total_frames} | FPS: {current_fps:.1f} | Persons: {num_persons} | "
                        f"{format_timing_status(times)}"
                    )
            else:
                cv2.imshow('SAT-HMR Video Demo (TensorRT)', result_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print('\n終了します')
                    break

            frame_count += 1

    except KeyboardInterrupt:
        print('\n中断されました')
    except Exception as exc:
        print(f'\n[ERROR] {exc}')
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if udp is not None:
            udp.close()
        if metrics_fp is not None:
            metrics_fp.close()
        runner.close()
        cv2.destroyAllWindows()

    print('\n処理完了!')
    print(f'処理フレーム数: {frame_count}')
    print(f'平均検出人数: {total_persons / max(frame_count, 1):.2f}')
    print_time_summary(all_times)
    if writer is not None and args.output:
        print(f'出力動画: {args.output}')
    if args.metrics_csv:
        print(f'測定CSV: {args.metrics_csv}')
    if npz_writer is not None:
        npz_writer.save(args.pred_npz)
        print(f'予測NPZ: {args.pred_npz}')


if __name__ == '__main__':
    main()

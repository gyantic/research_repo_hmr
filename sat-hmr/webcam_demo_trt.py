"""
SAT-HMR Webcam Demo (TensorRT)
- Run inference with TensorRT engine (.engine).
"""

import argparse
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


def get_args():
    parser = argparse.ArgumentParser('SAT-HMR Webcam Demo (TensorRT)')
    parser.add_argument('--engine', default='datas/trt/sat_hmr_sat_fp16_vc.engine', help='TensorRT engine path')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--conf_thresh', type=float, default=0.3, help='Detection confidence threshold')
    parser.add_argument('--input_size', type=int, default=1288, help='SAT-HMR base input size (for preprocessing)')
    parser.add_argument('--width', type=int, default=1280, help='Camera width')
    parser.add_argument('--height', type=int, default=720, help='Camera height')
    parser.add_argument('--save', default=None, help='Save output video path')

    parser.add_argument('--no_render', action='store_true', help='Skip mesh rendering')
    parser.add_argument('--render_every', type=int, default=1, help='Render every N frames')
    parser.add_argument('--show_fps', action='store_true', help='Overlay FPS text')
    parser.add_argument('--no_display', action='store_true', help='Do not display window (also disables rendering)')
    parser.add_argument('--udp_only', action='store_true', help='UE send only mode (no render/no display)')

    add_udp_args(parser)

    return parser.parse_args()


def smpl_verts_in_camera(smpl_layer: SMPL_Layer, outputs: Dict[str, np.ndarray], select_idx: np.ndarray) -> np.ndarray:
    pred_poses = pick_output(outputs, 'pred_poses')[0][select_idx].astype(np.float32)
    pred_betas = pick_output(outputs, 'pred_betas')[0][select_idx].astype(np.float32)
    pred_transl = pick_output(outputs, 'pred_transl')[0][select_idx].astype(np.float32)

    with torch.no_grad():
        verts, _ = smpl_layer(poses=torch.from_numpy(pred_poses), betas=torch.from_numpy(pred_betas))
        verts_cam = verts + torch.from_numpy(pred_transl).unsqueeze(1)
        return verts_cam.cpu().numpy()


def process_frame_trt(
    runner: TensorRTRunner,
    frame: np.ndarray,
    input_size: int,
    conf_thresh: float,
    no_render: bool,
    smpl_layer: Optional[SMPL_Layer],
    render_utils: Optional[Dict[str, object]],
):
    times: Dict[str, float] = {}
    img_tensor, img_size, orig_size, outputs = infer_frame(runner, frame, input_size, times)

    t0 = time.time()
    confs, select_idx = select_confident_queries(outputs, conf_thresh)
    num_persons = int(select_idx.size)

    udp_candidates = empty_udp_candidates()
    if num_persons > 0:
        poses = pick_output(outputs, 'pred_poses')[0][select_idx].astype(np.float32)
        transls = pick_output(outputs, 'pred_transl')[0][select_idx].astype(np.float32)
        udp_candidates = make_udp_candidates(poses, transls, confs, select_idx)

    times['postprocess'] = time.time() - t0

    if no_render or render_utils is None or smpl_layer is None:
        return frame, num_persons, times, udp_candidates

    t0 = time.time()
    verts = None
    cam_intrinsics = None
    if num_persons > 0:
        verts = smpl_verts_in_camera(smpl_layer, outputs, select_idx)
        if 'pred_intrinsics' in outputs:
            cam_intrinsics = torch.from_numpy(outputs['pred_intrinsics'][0].reshape(3, 3).astype(np.float32))
        else:
            cam_intrinsics = torch.from_numpy(build_cam_intrinsics_from_img_size(img_size, input_size))

    result_img = render_mesh_frame(
        render_utils, img_tensor, img_size, orig_size, verts, smpl_layer.faces, cam_intrinsics
    )
    times['render'] = time.time() - t0

    return result_img, num_persons, times, udp_candidates


def main():
    args = get_args()
    normalize_demo_args(args)

    # Camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f'エラー: カメラを開けません (ID: {args.camera})')
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f'カメラID: {args.camera}')
    print(f'解像度: {actual_width}x{actual_height}, FPS: {cam_fps:.2f}')

    # Recorder
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save, fourcc, 30.0, (actual_width, actual_height))
        print(f'録画ファイル: {args.save}')

    runner = open_runner_or_exit(args.engine)

    render_ok, render_utils = setup_render_imports(args.no_render)
    if not render_ok:
        args.no_render = True

    smpl_layer = None
    if not args.no_render:
        print('[INFO] Initialize SMPL layer for mesh rendering...')
        smpl_layer = SMPL_Layer(model_path=smpl_model_path, with_genders=False)

    print_demo_settings(args)

    print('\n=== 操作方法 ===')
    if not args.no_display:
        print("  'q' または ESC: 終了")
        if args.save:
            print("  'r': 録画開始/停止")
    else:
        print('  Ctrl+C: 終了')
    print('==================\n')

    frame_count = 0
    fps_queue = []
    recording = args.save is not None

    udp = None
    if args.udp:
        udp = UdpPoseStreamer(args)
        udp.wait_for_hello()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('カメラからフレームを取得できません')
                break

            start_time = time.time()
            render_this_frame = (not args.no_render) and (frame_count % args.render_every == 0)

            result_frame, num_persons, times, udp_candidates = process_frame_trt(
                runner=runner,
                frame=frame,
                input_size=args.input_size,
                conf_thresh=args.conf_thresh,
                no_render=not render_this_frame,
                smpl_layer=smpl_layer,
                render_utils=render_utils,
            )

            if udp is not None:
                udp.process(frame_count, num_persons, udp_candidates)

            elapsed = time.time() - start_time
            _current_fps, avg_fps = update_fps_window(fps_queue, elapsed)

            if args.show_fps and not args.no_display:
                draw_timing_overlay(result_frame, avg_fps, num_persons, times)

            if recording and writer is not None:
                draw_rec_marker(result_frame, actual_width)
                writer.write(result_frame)

            if args.no_display:
                if frame_count % 30 == 0:
                    print(f'\rFPS: {avg_fps:.1f} | Persons: {num_persons} | {format_timing_status(times)}')
            else:
                cv2.imshow('SAT-HMR Webcam Demo (TensorRT)', result_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print('\n終了します')
                    break
                elif key == ord('r') and args.save:
                    recording = not recording
                    print('録画開始' if recording else '録画停止')

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
        runner.close()
        cv2.destroyAllWindows()

    print('\n処理完了!')
    print(f'処理フレーム数: {frame_count}')
    if fps_queue:
        print(f'平均FPS: {np.mean(fps_queue):.2f}')
    if args.save:
        print(f'録画ファイル: {args.save}')


if __name__ == '__main__':
    main()

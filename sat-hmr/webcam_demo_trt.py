"""
SAT-HMR Webcam Demo (TensorRT)
- Keep original webcam_demo.py intact.
- Run inference with TensorRT engine (.engine).
"""

import argparse
import socket
import sys
import time
from typing import Dict, Optional

import cv2
import numpy as np
import torch

from configs.paths import smpl_model_path
from models.human_models import SMPL_Layer
from utils.transforms import unNormalize
from sathmr_common import (
    build_cam_intrinsics_from_img_size,
    preprocess_frame,
    select_udp_target,
    setup_render_imports,
    smpl_pose_to_ue_quaternions,
    transl_to_ue,
)
from trt_runner import TensorRTRunner


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

    parser.add_argument('--udp', action='store_true', help='Enable UDP send to UE')
    parser.add_argument('--ue_port', type=int, default=62004, help='UE receive port')
    parser.add_argument('--local_port', type=int, default=5555, help='Local bind port for HELLO')

    parser.add_argument('--udp_track_max_dist_cm', type=float, default=120.0)
    parser.add_argument('--udp_reinit_conf', type=float, default=0.55)
    parser.add_argument('--udp_hold_frames', type=int, default=10)
    parser.add_argument('--udp_trans_jump_cm', type=float, default=30.0)
    parser.add_argument('--udp_outlier_mode', choices=['skip', 'clamp'], default='skip')
    parser.add_argument('--udp_outlier_max_skip', type=int, default=12)
    parser.add_argument('--udp_jump_conf_thresh', type=float, default=0.35)
    parser.add_argument('--udp_jump_motion_cm', type=float, default=20.0)
    parser.add_argument('--udp_jump_pose_motion', type=float, default=0.02)
    parser.add_argument('--udp_jump_debounce', type=int, default=2)
    parser.add_argument('--udp_jump_hold_frames', type=int, default=4)
    parser.add_argument('--udp_jump_cooldown', type=int, default=6)

    return parser.parse_args()


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

    t0 = time.time()
    img_tensor, img_size, orig_size = preprocess_frame(frame, input_size, use_sat=True)
    image_np = np.ascontiguousarray(img_tensor.unsqueeze(0).numpy().astype(np.float32))
    times['preprocess'] = time.time() - t0

    expected = runner.get_engine_image_shape()
    if expected is not None and tuple(image_np.shape) != expected:
        raise RuntimeError(
            'Preprocess output shape does not match engine static input shape: '
            f'{tuple(image_np.shape)} vs {expected}. '
            'Use camera resolution/aspect that yields the engine shape.'
        )

    trt_inputs = {runner.image_input_name: image_np}
    if runner.img_size_input_name is not None:
        trt_inputs[runner.img_size_input_name] = np.asarray(img_size, dtype=np.int64)

    t0 = time.time()
    outputs = runner.infer(trt_inputs)
    times['inference'] = time.time() - t0

    def pick(name: str) -> np.ndarray:
        if name not in outputs:
            raise KeyError(f'Output tensor not found: {name}. got={list(outputs.keys())}')
        return outputs[name]

    t0 = time.time()
    pred_confs = pick('pred_confs')
    confs = pred_confs[0].reshape(-1)
    select_idx = np.where(confs > conf_thresh)[0]
    num_persons = int(select_idx.size)

    udp_candidates = {
        'poses': np.empty((0, 72), dtype=np.float32),
        'transls': np.empty((0, 3), dtype=np.float32),
        'confs': np.empty((0,), dtype=np.float32),
        'query_ids': np.empty((0,), dtype=np.int32),
    }

    if num_persons > 0:
        poses = pick('pred_poses')[0][select_idx].astype(np.float32)
        transls = pick('pred_transl')[0][select_idx].astype(np.float32)
        conf_t = confs[select_idx].astype(np.float32)
        qids = select_idx.astype(np.int32)
        udp_candidates = {
            'poses': poses,
            'transls': transls,
            'confs': conf_t,
            'query_ids': qids,
        }

    times['postprocess'] = time.time() - t0

    if no_render or render_utils is None or smpl_layer is None:
        return frame, num_persons, times, udp_candidates

    t0 = time.time()
    tensor_to_BGR = render_utils['tensor_to_BGR']
    pad_img = render_utils['pad_img']
    vis_meshes_img = render_utils['vis_meshes_img']
    get_colors_rgb = render_utils['get_colors_rgb']

    ori_img = tensor_to_BGR(unNormalize(img_tensor).cpu())

    if num_persons > 0:
        pred_poses = pick('pred_poses')[0][select_idx].astype(np.float32)
        pred_betas = pick('pred_betas')[0][select_idx].astype(np.float32)
        pred_transl = pick('pred_transl')[0][select_idx].astype(np.float32)

        with torch.no_grad():
            poses_t = torch.from_numpy(pred_poses)
            betas_t = torch.from_numpy(pred_betas)
            verts, _ = smpl_layer(poses=poses_t, betas=betas_t)
            verts_cam = verts + torch.from_numpy(pred_transl).unsqueeze(1)
            pred_verts = verts_cam.cpu().numpy()

        if 'pred_intrinsics' in outputs:
            pred_intrinsics = torch.from_numpy(outputs['pred_intrinsics'][0].reshape(3, 3).astype(np.float32))
        else:
            pred_intrinsics = torch.from_numpy(build_cam_intrinsics_from_img_size(img_size, input_size))

        colors = get_colors_rgb(num_persons)
        result_img = vis_meshes_img(
            img=ori_img.copy(),
            verts=pred_verts,
            smpl_faces=smpl_layer.faces,
            cam_intrinsics=pred_intrinsics,
            colors=colors,
            padding=True,
        )
    else:
        result_img = pad_img(ori_img, max(ori_img.shape[0], ori_img.shape[1]))

    result_img = result_img[:img_size[0], :img_size[1]]
    result_img = cv2.resize(result_img, (orig_size[1], orig_size[0]))
    times['render'] = time.time() - t0

    return result_img, num_persons, times, udp_candidates


def main():
    args = get_args()

    if args.input_size % 56 != 0:
        print(f'エラー: input_size ({args.input_size}) は56の倍数である必要があります')
        sys.exit(1)
    if args.render_every < 1:
        print('エラー: render_every は1以上で指定してください')
        sys.exit(1)
    if args.no_display:
        args.no_render = True

    if args.udp_only:
        args.no_render = True
        args.no_display = True
        args.show_fps = False

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

    # TensorRT
    print(f'[INFO] TensorRT engine load: {args.engine}')
    try:
        runner = TensorRTRunner(args.engine)
    except Exception as exc:
        print(f'[ERROR] TensorRT runner init failed: {exc}')
        print('ヒント: `pip install tensorrt cuda-python` (環境に合わせて)')
        sys.exit(1)

    runner.describe()

    # Render setup
    render_ok, render_utils = setup_render_imports(args.no_render)
    if not render_ok:
        args.no_render = True

    smpl_layer = None
    if not args.no_render:
        print('[INFO] Initialize SMPL layer for mesh rendering...')
        smpl_layer = SMPL_Layer(model_path=smpl_model_path, with_genders=False)

    print(f'Engine: {args.engine}')
    print(f'信頼度閾値: {args.conf_thresh}')
    print(f'入力サイズ: {args.input_size}')
    if args.udp_only:
        print('UDP専用モード: ON (描画/表示を無効化)')
    if args.no_render:
        print('メッシュ描画: OFF')
    elif args.render_every > 1:
        print(f'メッシュ描画間隔: {args.render_every}フレーム毎')
    if args.udp and not args.no_render:
        print('[WARN] 描画ONだとrender処理が重く、FPSが低下します。実運用は --udp_only または --no_render を推奨')

    if args.udp:
        print(f'UDP送信: ON (local:{args.local_port} -> ue:{args.ue_port})')
        if args.udp_trans_jump_cm > 0:
            print(
                f'外れ値抑制: ON (mode={args.udp_outlier_mode}, '
                f'trans_jump_cm={args.udp_trans_jump_cm:.1f}, max_skip={args.udp_outlier_max_skip})'
            )
        else:
            print('外れ値抑制: OFF')
        print(
            'ジャンプ例外: '
            f'conf>={args.udp_jump_conf_thresh:.2f}, '
            f'motion>={args.udp_jump_motion_cm:.1f}cm or '
            f'pose>={args.udp_jump_pose_motion:.4f}, '
            f'debounce={args.udp_jump_debounce}, '
            f'hold={args.udp_jump_hold_frames}, cooldown={args.udp_jump_cooldown}'
        )

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

    sock = None
    ue_host = None
    first_send_done = False
    time_connection_success = None
    prev_quat = None
    prev_trans = None
    outlier_skip_streak = 0
    jump_candidate_streak = 0
    jump_hold_left = 0
    jump_cooldown_left = 0
    tracker_state = {
        'active': False,
        'transl': None,
        'query_id': None,
        'missed': 0,
    }

    if args.udp:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', args.local_port))
        print(f'[WAIT] HELLO on 0.0.0.0:{args.local_port}')
        _, addr = sock.recvfrom(2048)
        ue_host = addr[0]
        print(f'[HELLO] from {addr} -> dst={(ue_host, args.ue_port)}')
        time_connection_success = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('カメラからフレームを取得できません')
                break

            start_time = time.time()
            if jump_cooldown_left > 0:
                jump_cooldown_left -= 1
            if jump_hold_left > 0:
                jump_hold_left -= 1

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

            if args.udp:
                udp_data, tracker_state, track_reason = select_udp_target(
                    udp_candidates,
                    tracker_state,
                    args.udp_track_max_dist_cm,
                    args.udp_reinit_conf,
                )

                if udp_data is None:
                    outlier_skip_streak = 0
                    jump_candidate_streak = 0
                    tracker_state['missed'] += 1
                    if tracker_state['missed'] > args.udp_hold_frames:
                        tracker_state['active'] = False
                        tracker_state['transl'] = None
                        tracker_state['query_id'] = None
                    if frame_count % 30 == 0:
                        print(f'UDP frame={frame_count} skip ({track_reason}) persons={num_persons}')
                else:
                    quaternions = smpl_pose_to_ue_quaternions(udp_data['pose'])
                    trans = transl_to_ue(udp_data['transl'])

                    if np.isfinite(quaternions).all() and np.isfinite(trans).all():
                        raw_trans_motion = 0.0 if prev_trans is None else float(np.linalg.norm(trans - prev_trans))
                        raw_pose_motion = 0.0 if prev_quat is None else float(np.mean(np.abs(quaternions - prev_quat)))

                        outlier_note = ''
                        send_trans = trans.copy()
                        skip_send = False

                        jump_armed = False
                        if prev_trans is not None and jump_cooldown_left == 0:
                            jump_cond = (
                                udp_data['conf'] >= args.udp_jump_conf_thresh and
                                (
                                    raw_trans_motion >= args.udp_jump_motion_cm or
                                    raw_pose_motion >= args.udp_jump_pose_motion
                                )
                            )
                            if jump_cond:
                                jump_candidate_streak += 1
                                if jump_candidate_streak >= max(1, args.udp_jump_debounce):
                                    jump_hold_left = max(jump_hold_left, args.udp_jump_hold_frames)
                                    jump_candidate_streak = 0
                            else:
                                jump_candidate_streak = 0
                        else:
                            jump_candidate_streak = 0

                        if jump_hold_left > 0:
                            jump_armed = True
                            outlier_skip_streak = 0
                            outlier_note = f' jump_bypass(hold={jump_hold_left})'

                        if (not jump_armed) and prev_trans is not None and args.udp_trans_jump_cm > 0 and raw_trans_motion > args.udp_trans_jump_cm:
                            if args.udp_outlier_mode == 'skip':
                                outlier_skip_streak += 1
                                if outlier_skip_streak <= args.udp_outlier_max_skip:
                                    skip_send = True
                                    outlier_note = (
                                        f'trans_outlier {raw_trans_motion:.2f}>{args.udp_trans_jump_cm:.2f}cm '
                                        f'(skip {outlier_skip_streak}/{args.udp_outlier_max_skip})'
                                    )
                                else:
                                    outlier_note = (
                                        f' outlier=rebase({raw_trans_motion:.2f}cm '
                                        f'after {outlier_skip_streak} skips)'
                                    )
                                    outlier_skip_streak = 0
                            else:
                                delta = trans - prev_trans
                                scale = args.udp_trans_jump_cm / (raw_trans_motion + 1e-9)
                                send_trans = prev_trans + delta * scale
                                outlier_note = (
                                    f' outlier=clamp({raw_trans_motion:.2f}'
                                    f'->{args.udp_trans_jump_cm:.2f}cm)'
                                )
                                outlier_skip_streak = 0
                        else:
                            outlier_skip_streak = 0

                        if skip_send:
                            if frame_count % 30 == 0:
                                print(
                                    f'UDP frame={frame_count} skip ({outlier_note}) persons={num_persons}'
                                )
                        else:
                            pose_str = ','.join(f'{v:.4f}' for v in quaternions.flatten())
                            trans_str = ','.join(f'{v:.4f}' for v in send_trans.flatten())
                            payload = f'{pose_str}|{trans_str}\n'.encode('ascii')
                            sent = sock.sendto(payload, (ue_host, args.ue_port))

                            if not first_send_done:
                                startup_lag = time.perf_counter() - time_connection_success
                                print(f'[INFO] Startup Lag: {startup_lag:.4f} seconds')
                                first_send_done = True

                            if prev_quat is None:
                                pose_motion = 0.0
                                trans_motion = 0.0
                            else:
                                pose_motion = raw_pose_motion
                                trans_motion = float(np.linalg.norm(send_trans - prev_trans))

                            prev_quat = quaternions.copy()
                            prev_trans = send_trans.copy()

                            if jump_armed and jump_hold_left == 1:
                                jump_cooldown_left = max(0, args.udp_jump_cooldown)

                            if frame_count % 30 == 0:
                                print(
                                    f'UDP frame={frame_count} conf={udp_data["conf"]:.3f} '
                                    f'persons={udp_data["num_persons"]} query={udp_data["query_id"]} '
                                    f'reason={track_reason} pose_motion={pose_motion:.5f} '
                                    f'trans_motion={trans_motion:.2f}{outlier_note} sent={sent}'
                                )
                    else:
                        if frame_count % 30 == 0:
                            print(f'UDP frame={frame_count} skip (invalid payload) persons={num_persons}')

            elapsed = time.time() - start_time
            current_fps = 1.0 / elapsed if elapsed > 0 else 0
            fps_queue.append(current_fps)
            if len(fps_queue) > 30:
                fps_queue.pop(0)
            avg_fps = float(np.mean(fps_queue)) if fps_queue else 0.0

            if args.show_fps and not args.no_display:
                cv2.putText(result_frame, f'FPS: {avg_fps:.1f}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(result_frame, f'Persons: {num_persons}', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                y_offset = 110
                cv2.putText(result_frame, f"Preprocess: {times['preprocess']*1000:.1f}ms", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_offset += 30
                cv2.putText(result_frame, f"Inference: {times['inference']*1000:.1f}ms", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_offset += 30
                cv2.putText(result_frame, f"Postprocess: {times['postprocess']*1000:.1f}ms", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                if 'render' in times:
                    y_offset += 30
                    cv2.putText(result_frame, f"Render: {times['render']*1000:.1f}ms", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if recording and writer is not None:
                cv2.circle(result_frame, (actual_width - 30, 30), 10, (0, 0, 255), -1)
                cv2.putText(result_frame, 'REC', (actual_width - 80, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                writer.write(result_frame)

            if args.no_display:
                if frame_count % 30 == 0:
                    render_ms = f"{times['render']*1000:.0f}ms" if 'render' in times else 'skip'
                    print(
                        f"\rFPS: {avg_fps:.1f} | Persons: {num_persons} | "
                        f"Preprocess: {times['preprocess']*1000:.0f}ms | "
                        f"Inference: {times['inference']*1000:.0f}ms | "
                        f"Postprocess: {times['postprocess']*1000:.0f}ms | "
                        f"Render: {render_ms}"
                    )
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
        if sock is not None:
            sock.close()
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

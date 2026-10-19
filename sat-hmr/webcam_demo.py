"""
SAT-HMR Webcam Demo
Webカメラからリアルタイムで3Dメッシュ推定を行います。

主な引数:
  --camera カメラID (デフォルト 0)
  --conf_thresh 検出信頼度閾値 (デフォルト 0.3)
  --save 録画する場合のファイル名
  --width カメラ解像度の幅 (デフォルト 1280)
  --height カメラ解像度の高さ (デフォルト 720)
"""

import argparse
import os
# Windowsでのトラブル回避用のおまじない
os.environ['PYOPENGL_PLATFORM'] = 'pyglet'

import pyrender  # ★ここが重要！他の何よりも先に読み込む
def force_windows_create(self):
    from pyrender.platforms import pyglet_platform
    from pyrender.renderer import Renderer

    # 1. Windows用のウィンドウシステム(Platform)を作成
    self._platform = pyglet_platform.PygletPlatform(self.viewport_width, self.viewport_height)
    self._platform.init_context()

    # 2. 描画を行うレンダラー(Renderer)を作成
    # ★修正点：ここにも width と height を渡す必要がありました
    self._renderer = Renderer(self.viewport_width, self.viewport_height)

# ライブラリの関数を、この関数で上書きする
pyrender.OffscreenRenderer._create = force_windows_create
import sys
import cv2
import numpy as np
import torch
import socket
import time

from models import build_sat_model
from utils.transforms import unNormalize
from utils.visualization import tensor_to_BGR, pad_img, vis_meshes_img, get_colors_rgb
from sathmr_common import (
    load_config,
    preprocess_frame,
    select_udp_target,
    smpl_pose_to_ue_quaternions,
    transl_to_ue,
)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def get_args():
    parser = argparse.ArgumentParser('SAT-HMR Webcam Demo')
    parser.add_argument('--camera', type=int, default=0, help='カメラID')
    parser.add_argument('--conf_thresh', type=float, default=0.3, help='検出信頼度閾値')
    parser.add_argument('--weights', default='./weights/sat_hmr/sat_644.pth', help='重みファイルパス')
    parser.add_argument('--input_size', type=int, default=1288, help='入力画像サイズ')
    parser.add_argument('--device', default='cuda', help='デバイス')
    parser.add_argument('--save', default=None, help='録画ファイル名')
    parser.add_argument('--width', type=int, default=1280, help='カメラ解像度(幅)')
    parser.add_argument('--height', type=int, default=720, help='カメラ解像度(高さ)')
    parser.add_argument('--no_render', action='store_true', help='メッシュ描画スキップ')
    parser.add_argument('--render_every', type=int, default=1, help='メッシュ描画間隔。1=毎フレーム、N>1でNフレーム毎')
    parser.add_argument('--show_fps', action='store_true', help='FPS表示')
    parser.add_argument('--no_display', action='store_true', help='画面表示しない（FPSのみ表示）')
    parser.add_argument('--udp_only', action='store_true', help='UE送信専用モード（no_render/no_displayを有効化）')
    parser.add_argument('--amp', action='store_true', help='CUDA mixed precision で推論を高速化')
    parser.add_argument('--udp', action='store_true', help='UEへUDP送信を有効化')
    parser.add_argument('--ue_port', type=int, default=62004, help='UEの受信ポート')
    parser.add_argument('--local_port', type=int, default=5555, help='Python側待受ポート')
    parser.add_argument('--udp_track_max_dist_cm', type=float, default=120.0, help='追跡継続する最大距離(cm)')
    parser.add_argument('--udp_reinit_conf', type=float, default=0.55, help='追跡再初期化に必要な信頼度')
    parser.add_argument('--udp_hold_frames', type=int, default=10, help='見失い時に追跡を維持するフレーム数')
    parser.add_argument('--udp_trans_jump_cm', type=float, default=30.0, help='平行移動の外れ値閾値(cm)。<=0で無効')
    parser.add_argument('--udp_outlier_mode', choices=['skip', 'clamp'], default='skip', help='外れ値処理: skip=送信しない, clamp=移動量を閾値に制限')
    parser.add_argument('--udp_outlier_max_skip', type=int, default=12, help='skipモードで連続スキップ可能な最大フレーム数。超過時は自動復帰')
    parser.add_argument('--udp_jump_conf_thresh', type=float, default=0.35, help='ジャンプ判定の最小conf')
    parser.add_argument('--udp_jump_motion_cm', type=float, default=20.0, help='ジャンプ判定の移動量閾値(cm)')
    parser.add_argument('--udp_jump_pose_motion', type=float, default=0.02, help='ジャンプ判定のpose_motion閾値')
    parser.add_argument('--udp_jump_debounce', type=int, default=2, help='ジャンプ判定の連続成立フレーム数')
    parser.add_argument('--udp_jump_hold_frames', type=int, default=4, help='ジャンプ確定後に外れ値抑制をバイパスするフレーム数')
    parser.add_argument('--udp_jump_cooldown', type=int, default=6, help='ジャンプ終了後に再判定を抑制するフレーム数')
    return parser.parse_args()


def process_frame(model, frame, input_size, conf_thresh, device, no_render=False, amp=False):
    """1フレームを処理"""
    times = {}
    
    # 前処理
    t0 = time.time()
    img_tensor, img_size, orig_size = preprocess_frame(frame, input_size, use_sat=True)
    img_tensor = img_tensor.to(device)
    
    targets = [{
        'img_size': torch.tensor(img_size).to(device),
        'img_path': 'webcam_frame'
    }]
    times['preprocess'] = time.time() - t0
    
    # 推論
    t0 = time.time()
    with torch.no_grad():
        if amp and device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model([img_tensor], targets)
        else:
            outputs = model([img_tensor], targets)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    times['inference'] = time.time() - t0
    
    # 後処理
    t0 = time.time()
    idx = 0
    confs = outputs['pred_confs'][idx].reshape(-1)
    select_idx = torch.where(confs > conf_thresh)[0]
    num_persons = int(select_idx.numel())
    times['postprocess'] = time.time() - t0

    udp_candidates = {
        'poses': np.empty((0, 72), dtype=np.float32),
        'transls': np.empty((0, 3), dtype=np.float32),
        'confs': np.empty((0,), dtype=np.float32),
        'query_ids': np.empty((0,), dtype=np.int32),
    }
    if num_persons > 0:
        pose_t = outputs['pred_poses'][idx][select_idx].detach().cpu().numpy().astype(np.float32)
        transl_t = outputs['pred_transl'][idx][select_idx].detach().cpu().numpy().astype(np.float32)
        conf_t = confs[select_idx].detach().cpu().numpy().astype(np.float32)
        qid_t = select_idx.detach().cpu().numpy().astype(np.int32)
        udp_candidates = {
            'poses': pose_t,
            'transls': transl_t,
            'confs': conf_t,
            'query_ids': qid_t,
        }
    
    if no_render:
        return frame, num_persons, times, udp_candidates
    
    # 描画
    t0 = time.time()
    pred_intrinsics = outputs['pred_intrinsics'][idx].reshape(3, 3).detach().cpu()
    
    # 正規化を戻した画像を取得
    ori_img = tensor_to_BGR(unNormalize(img_tensor).cpu())
    
    # メッシュを描画
    if num_persons > 0:
        pred_verts = outputs['pred_verts'][idx][select_idx].detach().cpu().numpy()
        colors = get_colors_rgb(num_persons)
        result_img = vis_meshes_img(
            img=ori_img.copy(),
            verts=pred_verts,
            smpl_faces=model.human_model.faces,
            cam_intrinsics=pred_intrinsics,
            colors=colors,
            padding=True
        )
    else:
        result_img = pad_img(ori_img, max(ori_img.shape[0], ori_img.shape[1]))
    
    # 有効領域を切り出し
    result_img = result_img[:img_size[0], :img_size[1]]
    
    # 元のサイズにリサイズ
    result_img = cv2.resize(result_img, (orig_size[1], orig_size[0]))
    
    times['render'] = time.time() - t0
    
    return result_img, num_persons, times, udp_candidates


def main():
    args = get_args()
    
    if args.input_size % 56 != 0:
        print(f"エラー: input_size ({args.input_size}) は56の倍数である必要があります")
        sys.exit(1)
    if args.render_every < 1:
        print("エラー: render_every は1以上で指定してください")
        sys.exit(1)
    if args.udp_only:
        args.no_render = True
        args.no_display = True
        args.show_fps = False
    
    # Webカメラを開く
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"エラー: カメラを開けません (ID: {args.camera})")
        sys.exit(1)
    
    # カメラ解像度を設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # 実際の解像度を取得
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"カメラID: {args.camera}")
    print(f"解像度: {actual_width}x{actual_height}, FPS: {fps:.2f}")
    
    # 録画設定
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save, fourcc, 30.0, (actual_width, actual_height))
        print(f"録画ファイル: {args.save}")
    
    # モデル読み込み
    print("モデルを読み込み中...")
    model_args = load_config(args.input_size)
    model, _ = build_sat_model(model_args, set_criterion=False)
    
    print(f"重みを読み込み中: {args.weights}")
    state_dict = torch.load(args.weights, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict, strict=False)

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARN] CUDAが利用できないため device=cpu にフォールバックします")
        args.device = 'cpu'

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    print(f"デバイス: {device}")
    print(f"入力サイズ: {args.input_size}")
    print(f"信頼度閾値: {args.conf_thresh}")
    print(f"AMP: {'ON' if args.amp else 'OFF'}")
    if args.udp_only:
        print("UDP専用モード: ON (描画/表示を無効化)")
    if args.no_render:
        print("メッシュ描画: OFF")
    elif args.render_every > 1:
        print(f"メッシュ描画間隔: {args.render_every}フレーム毎")
    if args.udp and not args.no_render:
        print("[WARN] 描画ONだとrender処理が重く、FPSが大きく低下します。実運用は --udp_only または --no_render を推奨")
    if args.udp:
        print(f"UDP送信: ON (local:{args.local_port} -> ue:{args.ue_port})")
        if args.udp_trans_jump_cm > 0:
            print(
                f"外れ値抑制: ON (mode={args.udp_outlier_mode}, "
                f"trans_jump_cm={args.udp_trans_jump_cm:.1f}, max_skip={args.udp_outlier_max_skip})"
            )
        else:
            print("外れ値抑制: OFF")
        print(
            "ジャンプ例外: "
            f"conf>={args.udp_jump_conf_thresh:.2f}, "
            f"motion>={args.udp_jump_motion_cm:.1f}cm or "
            f"pose>={args.udp_jump_pose_motion:.4f}, "
            f"debounce={args.udp_jump_debounce}, "
            f"hold={args.udp_jump_hold_frames}, cooldown={args.udp_jump_cooldown}"
        )
    print("\n=== 操作方法 ===")
    if not args.no_display:
        print("  'q' または ESC: 終了")
        if args.save:
            print("  'r': 録画開始/停止")
    else:
        print("  Ctrl+C: 終了")
    print("==================\n")
    
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
        sock.bind(("0.0.0.0", args.local_port))
        print(f"[WAIT] HELLO on 0.0.0.0:{args.local_port}")
        _, addr = sock.recvfrom(2048)
        ue_host = addr[0]
        print(f"[HELLO] from {addr} -> dst={(ue_host, args.ue_port)}")
        time_connection_success = time.perf_counter()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("カメラからフレームを取得できません")
                break
            
            start_time = time.time()
            if jump_cooldown_left > 0:
                jump_cooldown_left -= 1
            if jump_hold_left > 0:
                jump_hold_left -= 1
            
            # フレーム処理
            render_this_frame = (not args.no_render) and (frame_count % args.render_every == 0)
            result_frame, num_persons, times, udp_candidates = process_frame(
                model, frame, args.input_size, args.conf_thresh, device,
                no_render=not render_this_frame,
                amp=args.amp
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
                        print(f"UDP frame={frame_count} skip ({track_reason}) persons={num_persons}")
                else:
                    quaternions = smpl_pose_to_ue_quaternions(udp_data['pose'])
                    trans = transl_to_ue(udp_data['transl'])
                    if np.isfinite(quaternions).all() and np.isfinite(trans).all():
                        raw_trans_motion = 0.0 if prev_trans is None else float(np.linalg.norm(trans - prev_trans))
                        raw_pose_motion = 0.0 if prev_quat is None else float(np.mean(np.abs(quaternions - prev_quat)))
                        outlier_note = ""
                        send_trans = trans.copy()
                        skip_send = False

                        # ジャンプ判定: conf + (平行移動 or pose変化) を連続で満たしたときのみ確定
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
                            outlier_note = f" jump_bypass(hold={jump_hold_left})"

                        if (not jump_armed) and prev_trans is not None and args.udp_trans_jump_cm > 0 and raw_trans_motion > args.udp_trans_jump_cm:
                            if args.udp_outlier_mode == 'skip':
                                outlier_skip_streak += 1
                                if outlier_skip_streak <= args.udp_outlier_max_skip:
                                    skip_send = True
                                    outlier_note = (
                                        f"trans_outlier {raw_trans_motion:.2f}>{args.udp_trans_jump_cm:.2f}cm "
                                        f"(skip {outlier_skip_streak}/{args.udp_outlier_max_skip})"
                                    )
                                else:
                                    outlier_note = (
                                        f" outlier=rebase({raw_trans_motion:.2f}cm "
                                        f"after {outlier_skip_streak} skips)"
                                    )
                                    outlier_skip_streak = 0
                            # clamp: 1フレーム移動量を閾値内に制限
                            elif args.udp_outlier_mode == 'clamp':
                                delta = trans - prev_trans
                                scale = args.udp_trans_jump_cm / (raw_trans_motion + 1e-9)
                                send_trans = prev_trans + delta * scale
                                outlier_note = (
                                    f" outlier=clamp({raw_trans_motion:.2f}"
                                    f"->{args.udp_trans_jump_cm:.2f}cm)"
                                )
                                outlier_skip_streak = 0
                        else:
                            outlier_skip_streak = 0

                        if skip_send:
                            if frame_count % 30 == 0:
                                print(
                                    f"UDP frame={frame_count} skip ({outlier_note}) "
                                    f"persons={num_persons}"
                                )
                        else:
                            pose_str = ",".join(f"{v:.4f}" for v in quaternions.flatten())
                            trans_str = ",".join(f"{v:.4f}" for v in send_trans.flatten())
                            payload = f"{pose_str}|{trans_str}\n".encode("ascii")
                            sent = sock.sendto(payload, (ue_host, args.ue_port))

                            if not first_send_done:
                                startup_lag = time.perf_counter() - time_connection_success
                                print(f"[INFO] Startup Lag: {startup_lag:.4f} seconds")
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
                                    f"UDP frame={frame_count} conf={udp_data['conf']:.3f} "
                                    f"persons={udp_data['num_persons']} query={udp_data['query_id']} "
                                    f"reason={track_reason} pose_motion={pose_motion:.5f} "
                                    f"trans_motion={trans_motion:.2f}{outlier_note} sent={sent}"
                                )
                    else:
                        if frame_count % 30 == 0:
                            print(f"UDP frame={frame_count} skip (invalid payload) persons={num_persons}")
            
            # FPS計算
            elapsed = time.time() - start_time
            current_fps = 1.0 / elapsed if elapsed > 0 else 0
            fps_queue.append(current_fps)
            if len(fps_queue) > 30:
                fps_queue.pop(0)
            avg_fps = np.mean(fps_queue)
            
            # FPS表示
            if args.show_fps:
                cv2.putText(result_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(result_frame, f"Persons: {num_persons}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 処理時間の詳細表示
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
            
            # 録画中表示
            if recording and writer is not None:
                cv2.circle(result_frame, (actual_width - 30, 30), 10, (0, 0, 255), -1)
                cv2.putText(result_frame, "REC", (actual_width - 80, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 録画
            if recording and writer is not None:
                writer.write(result_frame)
            
            # コンソールにFPS表示（no_displayモード）
            if args.no_display:
                if frame_count % 30 == 0:  # 30フレームごとに表示
                    render_ms = f"{times['render']*1000:.0f}ms" if 'render' in times else "skip"
                    print(f"\rFPS: {avg_fps:.1f} | Persons: {num_persons} | "
                          f"Preprocess: {times['preprocess']*1000:.0f}ms | "
                          f"Inference: {times['inference']*1000:.0f}ms | "
                          f"Postprocess: {times['postprocess']*1000:.0f}ms | "
                          f"Render: {render_ms}")
            
            # 表示
            if not args.no_display:
                cv2.imshow('SAT-HMR Webcam Demo', result_frame)
            
                # キー入力処理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("\n終了します")
                    break
                elif key == ord('r') and args.save:  # 'r'
                    recording = not recording
                    if recording:
                        
                        print("録画開始")
                    else:
                        print("録画停止")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n中断されました")
    
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if sock is not None:
            sock.close()
        cv2.destroyAllWindows()
    
    print(f"\n処理完了!")
    print(f"処理フレーム数: {frame_count}")
    if len(fps_queue) > 0:
        print(f"平均FPS: {np.mean(fps_queue):.2f}")
    
    if args.save:
        print(f"録画ファイル: {args.save}")


if __name__ == '__main__':
    main()

"""
SAT-HMR Video Demo (ONNX Runtime)
- Keep original video_demo.py (Torch) intact.
- Run inference from video file with ONNX Runtime.
"""

import argparse
import csv
import math
import os
import socket
import sys
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from torchvision import transforms

from configs.paths import smpl_model_path
from models.human_models import SMPL_Layer
from utils.transforms import unNormalize

ARRAY2TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_args():
    parser = argparse.ArgumentParser('SAT-HMR Video Demo (ONNX Runtime)')
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', default=None, help='Output video path')
    parser.add_argument('--onnx', default='datas/onnx/sat_hmr_sat_runtimefix.onnx', help='ONNX model path')
    parser.add_argument('--ort_provider', choices=['cuda', 'cpu'], default='cuda', help='ONNX Runtime execution provider')
    parser.add_argument('--conf_thresh', type=float, default=0.3, help='Detection confidence threshold')
    parser.add_argument('--input_size', type=int, default=1288, help='SAT-HMR base input size (for preprocessing)')

    parser.add_argument('--no_render', action='store_true', help='Skip mesh rendering')
    parser.add_argument('--render_every', type=int, default=1, help='Render every N frames')
    parser.add_argument('--show_fps', action='store_true', help='Overlay FPS text')
    parser.add_argument('--show', action='store_true', help='Show preview window')
    parser.add_argument('--no_display', action='store_true', help='Do not display window (also disables rendering)')
    parser.add_argument('--no_save', action='store_true', help='Do not save result video')
    parser.add_argument('--metrics_csv', default=None, help='Per-frame metrics CSV output path')
    parser.add_argument('--udp_only', action='store_true', help='UE send only mode (no render, no display)')

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


def process_img_like_base(img, input_size):
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
    img_resized, img_size, _ = process_img_like_base(frame, input_size)

    patch_size = 56 if use_sat else 14
    pad_h = (img_size[0] + patch_size - 1) // patch_size * patch_size
    pad_w = (img_size[1] + patch_size - 1) // patch_size * patch_size

    pad_img = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
    pad_img[:img_size[0], :img_size[1]] = img_resized

    pil_img = Image.fromarray(pad_img[:, :, ::-1].copy())
    norm_img = ARRAY2TENSOR(pil_img)

    return norm_img, img_size, frame.shape[:2]


def smpl_pose_to_ue_quaternions(pose):
    rotations = Rotation.from_rotvec(pose.reshape(-1, 3))
    quaternions = rotations.as_quat().astype(np.float32)
    quaternions[:, 0] *= -1.0
    quaternions[:, 2] *= -1.0
    flip_root = np.array([1., 0., 0., 0.], dtype=np.float32)
    quaternions[0] = (
        Rotation.from_quat(flip_root) * Rotation.from_quat(quaternions[0])
    ).as_quat().astype(np.float32)
    return quaternions


def transl_to_ue(transl):
    trans = transl.copy().astype(np.float32)
    trans[1] *= -1.0
    trans[2] = 0.0
    return trans * 100.0


def select_udp_target(cands, tracker_state, max_dist_cm, reinit_conf):
    if len(cands['confs']) == 0:
        return None, tracker_state, None

    confs = cands['confs']
    poses = cands['poses']
    transls = cands['transls']
    query_ids = cands['query_ids']

    best_idx = int(np.argmax(confs))
    reason = 'max_conf'

    if tracker_state['active'] and tracker_state['transl'] is not None:
        dists_cm = np.linalg.norm(transls - tracker_state['transl'][None, :], axis=1) * 100.0
        near_idx = int(np.argmin(dists_cm))
        near_dist = float(dists_cm[near_idx])
        near_conf = float(confs[near_idx])
        best_conf = float(confs[best_idx])

        if near_dist <= max_dist_cm:
            best_idx = near_idx
            reason = f'nearest:{near_dist:.1f}cm'
        elif best_conf >= reinit_conf:
            reason = f'reinit_conf:{best_conf:.3f}'
        else:
            return None, tracker_state, f'skip_jump nearest={near_dist:.1f}cm conf={near_conf:.3f}'

    selected = {
        'pose': poses[best_idx],
        'transl': transls[best_idx],
        'conf': float(confs[best_idx]),
        'query_id': int(query_ids[best_idx]),
        'num_persons': len(confs),
    }

    tracker_state['active'] = True
    tracker_state['transl'] = transls[best_idx].copy()
    tracker_state['query_id'] = int(query_ids[best_idx])
    tracker_state['missed'] = 0
    return selected, tracker_state, reason


def build_cam_intrinsics_from_img_size(img_size_hw: Tuple[int, int], input_size: int) -> np.ndarray:
    h, w = img_size_hw
    focal = input_size / (2.0 * math.tan(math.pi / 6.0))
    K = np.array(
        [
            [focal, 0.0, w / 2.0],
            [0.0, focal, h / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return K


class OnnxRunner:
    def __init__(self, onnx_path: str, provider: str = 'cuda'):
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                'ONNX Runtime is not available. '
                'Install onnxruntime-gpu (or onnxruntime for CPU). '
                f'(detail: {exc})'
            )

        available = ort.get_available_providers()
        if provider == 'cuda':
            if 'CUDAExecutionProvider' in available:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                print('[WARN] CUDAExecutionProvider is unavailable. Fallback to CPUExecutionProvider.')
                providers = ['CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.ort = ort
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.providers = self.session.get_providers()

        self.input_metas = {x.name: x for x in self.session.get_inputs()}
        self.output_names = [x.name for x in self.session.get_outputs()]
        self.input_names = list(self.input_metas.keys())
        if not self.input_names:
            raise RuntimeError('ONNX model has no inputs')

        self.image_input_name = None
        self.img_size_input_name = None
        for name, meta in self.input_metas.items():
            shape = tuple(meta.shape)
            typ = meta.type
            if len(shape) == 4 and 'float' in typ:
                self.image_input_name = name
            elif len(shape) <= 2 and ('int64' in typ or 'int32' in typ):
                if 'img_size' in name:
                    self.img_size_input_name = name

        if self.image_input_name is None:
            self.image_input_name = self.input_names[0]

        if self.img_size_input_name is None:
            for name in self.input_names:
                if name != self.image_input_name:
                    self.img_size_input_name = name
                    break

    def _expected_dtype(self, name: str):
        typ = self.input_metas[name].type
        if 'int64' in typ:
            return np.int64
        if 'int32' in typ:
            return np.int32
        return np.float32

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        feed: Dict[str, np.ndarray] = {}
        for name in self.input_names:
            if name not in inputs:
                raise KeyError(f'Missing input tensor: {name}')
            arr = np.ascontiguousarray(inputs[name])
            expected = self._expected_dtype(name)
            if arr.dtype != expected:
                arr = arr.astype(expected, copy=False)
            feed[name] = arr

        outs = self.session.run(self.output_names, feed)
        return {name: arr for name, arr in zip(self.output_names, outs)}

    def describe(self):
        print(f'[INFO] ONNX Runtime providers: {self.providers}')
        print('[INFO] ONNX IO tensors:')
        for name in self.input_names:
            meta = self.input_metas[name]
            print(f'  - INPUT  {name:20s} type={meta.type:16s} shape={tuple(meta.shape)}')
        for out in self.session.get_outputs():
            print(f'  - OUTPUT {out.name:20s} type={out.type:16s} shape={tuple(out.shape)}')

    def get_model_image_shape(self):
        shape = tuple(self.input_metas[self.image_input_name].shape)
        if len(shape) != 4:
            return None
        static: list[int] = []
        for v in shape:
            if isinstance(v, int) and v > 0:
                static.append(v)
            else:
                return None
        return tuple(static)

    def close(self):
        # Keep symmetry with TensorRTRunner API.
        return


def process_frame_onnx(
    runner: OnnxRunner,
    frame: np.ndarray,
    input_size: int,
    conf_thresh: float,
    no_render: bool,
    smpl_layer: Optional[SMPL_Layer],
    smpl_device: Optional[torch.device],
    render_utils: Optional[Dict[str, object]],
):
    times: Dict[str, float] = {}

    t0 = time.time()
    img_tensor, img_size, orig_size = preprocess_frame(frame, input_size, use_sat=True)
    image_np = np.ascontiguousarray(img_tensor.unsqueeze(0).numpy().astype(np.float32))
    times['preprocess'] = time.time() - t0

    expected = runner.get_model_image_shape()
    if expected is not None and tuple(image_np.shape) != expected:
        raise RuntimeError(
            'Preprocess output shape does not match ONNX static input shape: '
            f'{tuple(image_np.shape)} vs {expected}. '
            'Use camera resolution/aspect that yields the engine shape.'
        )

    ort_inputs = {runner.image_input_name: image_np}
    if runner.img_size_input_name is not None:
        ort_inputs[runner.img_size_input_name] = np.asarray(img_size, dtype=np.int64)

    t0 = time.time()
    outputs = runner.infer(ort_inputs)
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

        if smpl_device is None:
            smpl_device = torch.device('cpu')
        with torch.no_grad():
            poses_t = torch.from_numpy(pred_poses).to(device=smpl_device, non_blocking=True)
            betas_t = torch.from_numpy(pred_betas).to(device=smpl_device, non_blocking=True)
            verts, _ = smpl_layer(poses=poses_t, betas=betas_t)
            trans_t = torch.from_numpy(pred_transl).to(device=smpl_device, non_blocking=True).unsqueeze(1)
            verts_cam = verts + trans_t
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


def setup_render_imports(no_render: bool):
    if no_render:
        return True, None

    try:
        # Keep same workaround order as webcam_demo.py
        os.environ.setdefault('PYOPENGL_PLATFORM', 'pyglet')
        import pyrender  # noqa: F401

        def force_windows_create(self):
            from pyrender.platforms import pyglet_platform
            from pyrender.renderer import Renderer
            self._platform = pyglet_platform.PygletPlatform(self.viewport_width, self.viewport_height)
            self._platform.init_context()
            self._renderer = Renderer(self.viewport_width, self.viewport_height)

        pyrender.OffscreenRenderer._create = force_windows_create

        from utils.visualization import tensor_to_BGR, pad_img, vis_meshes_img, get_colors_rgb

        return True, {
            'tensor_to_BGR': tensor_to_BGR,
            'pad_img': pad_img,
            'vis_meshes_img': vis_meshes_img,
            'get_colors_rgb': get_colors_rgb,
        }
    except Exception as exc:
        print(f'[WARN] Render modules import failed. Continue with no_render=True. ({exc})')
        return False, None


def main():
    args = get_args()

    if args.input_size % 56 != 0:
        print(f'エラー: input_size ({args.input_size}) は56の倍数である必要があります')
        sys.exit(1)
    if args.render_every < 1:
        print('エラー: render_every は1以上で指定してください')
        sys.exit(1)
    if args.show:
        args.no_display = False
    if args.no_display:
        args.no_render = True

    if args.udp_only:
        args.no_render = True
        args.no_display = True
        args.show_fps = False

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

    # Recorder
    writer = None
    if not args.no_save:
        if args.output is None:
            base, _ = os.path.splitext(args.input)
            args.output = f'{base}_onnx.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, video_fps, (actual_width, actual_height))
        if writer.isOpened():
            print(f'出力動画: {args.output}')
        else:
            writer = None
            print('[WARN] VideoWriter の初期化に失敗したため保存を無効化します')

    # ONNX Runtime
    print(f'[INFO] ONNX model load: {args.onnx}')
    try:
        runner = OnnxRunner(args.onnx, provider=args.ort_provider)
    except Exception as exc:
        print(f'[ERROR] ONNX runner init failed: {exc}')
        print('ヒント: `pip install onnxruntime-gpu` (CPUなら `pip install onnxruntime`)')
        sys.exit(1)

    runner.describe()

    # Render setup
    render_ok, render_utils = setup_render_imports(args.no_render)
    if not render_ok:
        args.no_render = True

    smpl_layer = None
    smpl_device = None
    if not args.no_render:
        print('[INFO] Initialize SMPL layer for mesh rendering...')
        smpl_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        smpl_layer = SMPL_Layer(model_path=smpl_model_path, with_genders=False).to(smpl_device).eval()

    print(f'ONNX: {args.onnx}')
    print(f'ORT Provider: {args.ort_provider}')
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
        print("  'q' または ESC: 終了 (早期終了)")
    else:
        print('  Ctrl+C: 終了')
    print('==================\n')

    frame_count = 0
    fps_queue = []
    recording = writer is not None
    total_persons = 0
    all_times = {'preprocess': [], 'inference': [], 'postprocess': [], 'render': []}
    metrics_fp = None
    metrics_writer = None
    if args.metrics_csv:
        metrics_dir = os.path.dirname(args.metrics_csv)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        metrics_fp = open(args.metrics_csv, 'w', newline='', encoding='utf-8')
        metrics_writer = csv.writer(metrics_fp)
        metrics_writer.writerow([
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
        ])

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
                print('動画の終端に到達しました')
                break

            start_time = time.time()
            if jump_cooldown_left > 0:
                jump_cooldown_left -= 1
            if jump_hold_left > 0:
                jump_hold_left -= 1

            render_this_frame = (not args.no_render) and (frame_count % args.render_every == 0)

            result_frame, num_persons, times, udp_candidates = process_frame_onnx(
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
            for k, v in times.items():
                if k in all_times:
                    all_times[k].append(v)

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
                        f"\rFrame: {frame_count}/{total_frames} | FPS: {avg_fps:.1f} | Persons: {num_persons} | "
                        f"Preprocess: {times['preprocess']*1000:.0f}ms | "
                        f"Inference: {times['inference']*1000:.0f}ms | "
                        f"Postprocess: {times['postprocess']*1000:.0f}ms | "
                        f"Render: {render_ms}"
                    )
            else:
                cv2.imshow('SAT-HMR Video Demo (ONNX Runtime)', result_frame)
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
        if sock is not None:
            sock.close()
        if metrics_fp is not None:
            metrics_fp.close()
        runner.close()
        cv2.destroyAllWindows()

    print('\n処理完了!')
    print(f'処理フレーム数: {frame_count}')
    print(f'平均検出人数: {total_persons / max(frame_count, 1):.2f}')
    print('\n--- 処理時間 (平均) ---')
    total_time_ms = 0.0
    for k, v in all_times.items():
        if len(v) > 0:
            avg_ms = float(np.mean(v) * 1000.0)
            total_time_ms += avg_ms
            print(f'  {k}: {avg_ms:.1f} ms')
    if total_time_ms > 0:
        print(f'  合計: {total_time_ms:.1f} ms ({1000.0 / total_time_ms:.1f} FPS)')
    if writer is not None and args.output:
        print(f'出力動画: {args.output}')
    if args.metrics_csv:
        print(f'測定CSV: {args.metrics_csv}')


if __name__ == '__main__':
    main()

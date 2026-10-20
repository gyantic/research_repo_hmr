"""Helpers shared by the SAT-HMR demo / benchmark scripts.

Used by video_demo.py, video_demo_trt.py, webcam_demo_trt.py and
engines/engine.py (and, through video_demo.py, by ue/infer_sequential.py).

Only dependencies that every one of those scripts already imports are loaded
here. Optional heavy backends (TensorRT, cuda-python, Triton) are NOT imported:
the TensorRT runner lives in trt_runner.py and imports them lazily.
"""

import math
import os
import random
import socket
import time
from typing import Tuple

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation
from torchvision import transforms


BF16_DTYPE_NAMES = {"bfloat16", "bf16", "bfp16"}

# DINOv2 / ImageNet normalization used by SAT-HMR's dataset pipeline.
IMAGENET_TO_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def is_bf16(dtype_cfg) -> bool:
    return str(dtype_cfg).lower() in BF16_DTYPE_NAMES


# ---------------------------------------------------------------------------
# Reproducibility / config
# ---------------------------------------------------------------------------

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
    # warn_only: some CUDA kernels used by SAT-HMR have no deterministic variant.
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_config(input_size):
    """Build the inference config from configs/run/demo.yaml + its model yaml.

    Paths are relative, so the caller must run from the sat-hmr/ directory.
    """
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
    # demo.yaml overrides the model yaml.
    for k, v in demo_config.items():
        setattr(args, k, v)

    args.mode = 'infer'
    args.pretrain = False
    args.pretrain_path = None
    args.input_size = input_size

    return args


def resolve_device(mode: str) -> torch.device:
    """Map "cpu" / "cuda" / anything-else ("auto") to a torch.device, falling back to CPU."""
    mode = mode.lower()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            print("[WARN] DEVICE_MODE=cuda ですが CUDA が利用できないため CPU にフォールバックします")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dequantize_int8_weight_state_dict_if_needed(state_dict):
    """Expand a tools/quantize_weights_int8.py checkpoint back to float weights.

    `<name>.weight_int8` + `<name>.weight_scale` (per output channel) become
    `<name>.weight`. A regular checkpoint is returned unchanged.
    """
    int8_keys = [k for k in state_dict if k.endswith(".weight_int8")]
    if not int8_keys:
        return state_dict, {"loaded_int8_weight_file": False, "dequantized_tensors": 0}

    restored = {}
    for key, tensor in state_dict.items():
        if key.endswith(".weight_int8"):
            base = key[: -len("_int8")]
            scale_key = base + "_scale"
            if scale_key not in state_dict:
                raise KeyError(f"Missing INT8 weight scale for {key}: expected {scale_key}")
            scale = state_dict[scale_key].float().unsqueeze(1)
            restored[base] = tensor.float() * scale
        elif key.endswith(".weight_scale"):
            continue
        else:
            restored[key] = tensor

    return restored, {
        "loaded_int8_weight_file": True,
        "dequantized_tensors": len(int8_keys),
    }


# ---------------------------------------------------------------------------
# Image preprocessing (mirrors datasets BASE.process_img / __getitem__)
# ---------------------------------------------------------------------------

def process_img_like_base(img, input_size):
    """Resize so that the longer side equals input_size. Returns (img, (h, w), resize_rate)."""
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
    """BGR frame -> normalized CHW tensor zero-padded to a multiple of the patch size.

    Returns (norm_img, resized (h, w), original (h, w)).
    """
    img_resized, img_size, _resize_rate = process_img_like_base(frame, input_size)

    # SAT works on 56 px (4 x 14 px DINOv2 patches) cells; plain ViT on 14 px patches.
    patch_size = 56 if use_sat else 14
    pad_h = (img_size[0] + patch_size - 1) // patch_size * patch_size
    pad_w = (img_size[1] + patch_size - 1) // patch_size * patch_size

    pad_img = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
    pad_img[:img_size[0], :img_size[1]] = img_resized

    # BGR -> RGB
    pil_img = Image.fromarray(pad_img[:, :, ::-1].copy())
    norm_img = IMAGENET_TO_TENSOR(pil_img)
    return norm_img, img_size, frame.shape[:2]


def build_cam_intrinsics_from_img_size(img_size_hw: Tuple[int, int], input_size: int) -> np.ndarray:
    """Default SAT-HMR pinhole camera (60 deg FOV over input_size px) centred on the resized image."""
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


def setup_render_imports(no_render: bool):
    """Import pyrender-based mesh rendering. Returns (ok, render_fns or None).

    ok is False when the import failed; the caller then disables rendering
    instead of aborting the run.
    """
    if no_render:
        return True, None

    try:
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


# ---------------------------------------------------------------------------
# SMPL -> Unreal Engine conversion and UDP target tracking
# ---------------------------------------------------------------------------

def smpl_pose_to_ue_quaternions(pose):
    """SMPL axis-angle pose (72,) -> (24, 4) float32 quaternions (x, y, z, w) in UE's frame."""
    rotations = Rotation.from_rotvec(pose.reshape(-1, 3))
    quaternions = rotations.as_quat().astype(np.float32)
    # Right-handed SMPL camera frame -> left-handed UE frame: mirror x and z components.
    quaternions[:, 0] *= -1.0
    quaternions[:, 2] *= -1.0
    # Rotate the root 180 deg about x so the body stands upright in UE.
    flip_root = np.array([1., 0., 0., 0.], dtype=np.float32)
    quaternions[0] = (
        Rotation.from_quat(flip_root) * Rotation.from_quat(quaternions[0])
    ).as_quat().astype(np.float32)
    return quaternions


def transl_to_ue(transl):
    """SMPL camera-space translation in metres -> UE translation in centimetres.

    y is flipped for UE's axis convention and depth (z) is zeroed.
    """
    trans = transl.copy().astype(np.float32)
    trans[1] *= -1.0
    trans[2] = 0.0
    return trans * 100.0


def select_udp_target(cands, tracker_state, max_dist_cm, reinit_conf):
    """Pick the one person to stream to UE, preferring the one tracked last frame.

    Returns (selected or None, tracker_state, reason). The nearest candidate within
    max_dist_cm of the tracked person keeps the track; otherwise the most confident
    one re-initialises it only if its confidence >= reinit_conf.
    """
    if len(cands['confs']) == 0:
        return None, tracker_state, None

    confs = cands['confs']
    poses = cands['poses']
    transls = cands['transls']
    query_ids = cands['query_ids']

    best_idx = int(np.argmax(confs))
    reason = "max_conf"

    if tracker_state['active'] and tracker_state['transl'] is not None:
        dists_cm = np.linalg.norm(transls - tracker_state['transl'][None, :], axis=1) * 100.0
        near_idx = int(np.argmin(dists_cm))
        near_dist = float(dists_cm[near_idx])
        near_conf = float(confs[near_idx])
        best_conf = float(confs[best_idx])

        if near_dist <= max_dist_cm:
            best_idx = near_idx
            reason = f"nearest:{near_dist:.1f}cm"
        elif best_conf >= reinit_conf:
            reason = f"reinit_conf:{best_conf:.3f}"
        else:
            return None, tracker_state, f"skip_jump nearest={near_dist:.1f}cm conf={near_conf:.3f}"

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


def empty_udp_candidates():
    """select_udp_target() candidates for a frame with no detection."""
    return {
        'poses': np.empty((0, 72), dtype=np.float32),
        'transls': np.empty((0, 3), dtype=np.float32),
        'confs': np.empty((0,), dtype=np.float32),
        'query_ids': np.empty((0,), dtype=np.int32),
    }


def make_udp_candidates(poses, transls, confs, select_idx):
    """select_udp_target() candidates. confs covers every query; select_idx are the detected ones."""
    return {
        'poses': poses,
        'transls': transls,
        'confs': confs[select_idx].astype(np.float32),
        'query_ids': select_idx.astype(np.int32),
    }


# ---------------------------------------------------------------------------
# UDP streaming to UE (video_demo_trt.py / webcam_demo_trt.py)
# ---------------------------------------------------------------------------

def add_udp_args(parser):
    """Add the --udp / --ue_port / --local_port / --udp_* options used by UdpPoseStreamer."""
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


def print_udp_settings(args):
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


class UdpPoseStreamer:
    """Send one tracked person per frame to UE as an ASCII '<quaternions>|<translation>' line.

    A translation step larger than --udp_trans_jump_cm is treated as a detection
    outlier and skipped or clamped. A confident, sustained large motion is instead
    accepted as a real jump: it opens a "jump hold" window that bypasses outlier
    suppression, followed by a cooldown before another jump can be detected.
    """

    LOG_EVERY_N_FRAMES = 30

    def __init__(self, args):
        self.args = args
        self.sock = None
        self.ue_addr = None
        self.connected_at = None
        self.first_send_done = False

        self.prev_quat = None
        self.prev_trans = None
        self.outlier_skip_streak = 0
        self.jump_candidate_streak = 0
        self.jump_hold_left = 0
        self.jump_cooldown_left = 0
        self.tracker_state = {
            'active': False,
            'transl': None,
            'query_id': None,
            'missed': 0,
        }

    def wait_for_hello(self):
        """Bind the local port and block until UE sends HELLO; its host becomes the destination."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', self.args.local_port))
        print(f'[WAIT] HELLO on 0.0.0.0:{self.args.local_port}')
        _, addr = self.sock.recvfrom(2048)
        self.ue_addr = (addr[0], self.args.ue_port)
        print(f'[HELLO] from {addr} -> dst={self.ue_addr}')
        self.connected_at = time.perf_counter()

    def close(self):
        if self.sock is not None:
            self.sock.close()

    def process(self, frame_count, num_persons, candidates):
        """Track, filter and (maybe) send one frame. Must be called once per frame."""
        args = self.args
        log_this_frame = frame_count % self.LOG_EVERY_N_FRAMES == 0
        self._tick_jump_timers()

        target, self.tracker_state, track_reason = select_udp_target(
            candidates,
            self.tracker_state,
            args.udp_track_max_dist_cm,
            args.udp_reinit_conf,
        )
        if target is None:
            self._on_target_missing()
            if log_this_frame:
                print(f'UDP frame={frame_count} skip ({track_reason}) persons={num_persons}')
            return

        quaternions = smpl_pose_to_ue_quaternions(target['pose'])
        trans = transl_to_ue(target['transl'])
        if not (np.isfinite(quaternions).all() and np.isfinite(trans).all()):
            if log_this_frame:
                print(f'UDP frame={frame_count} skip (invalid payload) persons={num_persons}')
            return

        raw_trans_motion = 0.0 if self.prev_trans is None else float(np.linalg.norm(trans - self.prev_trans))
        raw_pose_motion = 0.0 if self.prev_quat is None else float(np.mean(np.abs(quaternions - self.prev_quat)))

        self._update_jump_detection(target['conf'], raw_trans_motion, raw_pose_motion)
        jump_armed = self.jump_hold_left > 0
        if jump_armed:
            self.outlier_skip_streak = 0
            send_trans, skip_send = trans.copy(), False
            outlier_note = f' jump_bypass(hold={self.jump_hold_left})'
        else:
            send_trans, skip_send, outlier_note = self._suppress_trans_outlier(trans, raw_trans_motion)

        if skip_send:
            if log_this_frame:
                print(f'UDP frame={frame_count} skip ({outlier_note}) persons={num_persons}')
            return

        sent = self._send(quaternions, send_trans)

        if self.prev_quat is None:
            pose_motion = 0.0
            trans_motion = 0.0
        else:
            pose_motion = raw_pose_motion
            trans_motion = float(np.linalg.norm(send_trans - self.prev_trans))

        self.prev_quat = quaternions.copy()
        self.prev_trans = send_trans.copy()

        # The last frame of a jump hold starts the cooldown.
        if jump_armed and self.jump_hold_left == 1:
            self.jump_cooldown_left = max(0, args.udp_jump_cooldown)

        if log_this_frame:
            print(
                f'UDP frame={frame_count} conf={target["conf"]:.3f} '
                f'persons={target["num_persons"]} query={target["query_id"]} '
                f'reason={track_reason} pose_motion={pose_motion:.5f} '
                f'trans_motion={trans_motion:.2f}{outlier_note} sent={sent}'
            )

    def _tick_jump_timers(self):
        if self.jump_cooldown_left > 0:
            self.jump_cooldown_left -= 1
        if self.jump_hold_left > 0:
            self.jump_hold_left -= 1

    def _on_target_missing(self):
        self.outlier_skip_streak = 0
        self.jump_candidate_streak = 0
        self.tracker_state['missed'] += 1
        if self.tracker_state['missed'] > self.args.udp_hold_frames:
            self.tracker_state['active'] = False
            self.tracker_state['transl'] = None
            self.tracker_state['query_id'] = None

    def _update_jump_detection(self, conf, raw_trans_motion, raw_pose_motion):
        """Open the jump hold once a jump-like motion has lasted --udp_jump_debounce frames."""
        args = self.args
        if self.prev_trans is None or self.jump_cooldown_left != 0:
            self.jump_candidate_streak = 0
            return

        looks_like_jump = (
            conf >= args.udp_jump_conf_thresh and
            (
                raw_trans_motion >= args.udp_jump_motion_cm or
                raw_pose_motion >= args.udp_jump_pose_motion
            )
        )
        if not looks_like_jump:
            self.jump_candidate_streak = 0
            return

        self.jump_candidate_streak += 1
        if self.jump_candidate_streak >= max(1, args.udp_jump_debounce):
            self.jump_hold_left = max(self.jump_hold_left, args.udp_jump_hold_frames)
            self.jump_candidate_streak = 0

    def _suppress_trans_outlier(self, trans, raw_trans_motion):
        """Returns (send_trans, skip_send, outlier_note)."""
        args = self.args
        is_outlier = (
            self.prev_trans is not None and
            args.udp_trans_jump_cm > 0 and
            raw_trans_motion > args.udp_trans_jump_cm
        )
        if not is_outlier:
            self.outlier_skip_streak = 0
            return trans.copy(), False, ''

        if args.udp_outlier_mode != 'skip':
            # clamp: move at most udp_trans_jump_cm towards the new position.
            delta = trans - self.prev_trans
            scale = args.udp_trans_jump_cm / (raw_trans_motion + 1e-9)
            note = f' outlier=clamp({raw_trans_motion:.2f}->{args.udp_trans_jump_cm:.2f}cm)'
            self.outlier_skip_streak = 0
            return self.prev_trans + delta * scale, False, note

        self.outlier_skip_streak += 1
        if self.outlier_skip_streak <= args.udp_outlier_max_skip:
            note = (
                f'trans_outlier {raw_trans_motion:.2f}>{args.udp_trans_jump_cm:.2f}cm '
                f'(skip {self.outlier_skip_streak}/{args.udp_outlier_max_skip})'
            )
            return trans.copy(), True, note

        # Skipped too long: accept the new position as the new baseline.
        note = f' outlier=rebase({raw_trans_motion:.2f}cm after {self.outlier_skip_streak} skips)'
        self.outlier_skip_streak = 0
        return trans.copy(), False, note

    def _send(self, quaternions, trans):
        pose_str = ','.join(f'{v:.4f}' for v in quaternions.flatten())
        trans_str = ','.join(f'{v:.4f}' for v in trans.flatten())
        payload = f'{pose_str}|{trans_str}\n'.encode('ascii')
        sent = self.sock.sendto(payload, self.ue_addr)

        if not self.first_send_done:
            startup_lag = time.perf_counter() - self.connected_at
            print(f'[INFO] Startup Lag: {startup_lag:.4f} seconds')
            self.first_send_done = True
        return sent

"""Helpers shared by the SAT-HMR demo / benchmark scripts.

Used by video_demo*.py, webcam_demo*.py, sathmr_{tcp,udp}_benchmark.py and
engines/engine.py (and, through video_demo.py, by ue/infer_sequential.py).

Only dependencies that every one of those scripts already imports are loaded
here. Optional heavy backends (TensorRT, cuda-python, Triton) are NOT imported:
the TensorRT runner lives in trt_runner.py and imports them lazily.
"""

import math
import os
import random
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


# ---------------------------------------------------------------------------
# Single-person inference used by sathmr_{tcp,udp}_benchmark.py
# ---------------------------------------------------------------------------

def infer_frame(model, frame, input_size, conf_thresh, device):
    """Run one frame and return the SMPL parameters of the most confident person."""
    img_tensor, img_size, _orig_size = preprocess_frame(frame, input_size, use_sat=True)
    img_tensor = img_tensor.to(device)

    targets = [{
        'img_size': torch.tensor(img_size).to(device),
        'img_path': 'video_frame'
    }]

    with torch.no_grad():
        outputs = model([img_tensor], targets)

    idx = 0
    confs = outputs['pred_confs'][idx].squeeze(-1)
    select_idx = torch.where(confs > conf_thresh)[0]

    if len(select_idx) == 0:
        return {'detection_flag': False}

    best_idx = select_idx[confs[select_idx].argmax()]

    return {
        'detection_flag': True,
        'pose': outputs['pred_poses'][idx][best_idx].detach().cpu().numpy(),
        'betas': outputs['pred_betas'][idx][best_idx].detach().cpu().numpy(),
        'transl': outputs['pred_transl'][idx][best_idx].detach().cpu().numpy(),
        'conf': confs[best_idx].detach().cpu().item(),
    }

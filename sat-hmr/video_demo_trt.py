"""
SAT-HMR Video Demo (TensorRT)
- Keep original video_demo.py (Torch) intact.
- Run inference from video file with TensorRT engine (.engine).
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


class TensorRTRunner:
    def __init__(self, engine_path: str):
        try:
            import tensorrt as trt  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                'TensorRT Python bindings are not available. '
                'Install tensorrt in the active environment. '
                f'(detail: {exc})'
            )

        cuda_import_exc = None
        try:
            from cuda import cudart as cuda_rt  # type: ignore
        except Exception as exc:
            cuda_import_exc = exc
            try:
                # cuda-python>=13 may expose runtime here instead of cuda.cudart
                from cuda.bindings import runtime as cuda_rt  # type: ignore
            except Exception:
                raise RuntimeError(
                    'CUDA runtime bindings are not available. '
                    'Install cuda-python in this environment. '
                    f'(detail: {cuda_import_exc})'
                )

        self.trt = trt
        self.cudart = cuda_rt

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        # Required for some version-compatible engines that embed lean-runtime host code.
        if hasattr(runtime, 'engine_host_code_allowed'):
            runtime.engine_host_code_allowed = True

        with open(engine_path, 'rb') as f:
            engine_bytes = f.read()

        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError(
                f'Failed to deserialize engine: {engine_path} '
                f'(TensorRT runtime={self.trt.__version__}). '
                'The engine was likely built by an incompatible TensorRT version.'
            )

        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError('Failed to create TensorRT execution context')

        self.engine = engine
        self.context = context
        self.runtime = runtime
        self.stream = self._cuda(self.cudart.cudaStreamCreate(), 'cudaStreamCreate')

        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == self.trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        if not self.input_names:
            raise RuntimeError('Engine has no inputs')

        self.image_input_name = None
        self.img_size_input_name = None
        for name in self.input_names:
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            if len(shape) == 4 and self.trt.nptype(dtype) == np.float32:
                self.image_input_name = name
            elif len(shape) <= 2 and self.trt.nptype(dtype) in (np.int32, np.int64):
                if 'img_size' in name:
                    self.img_size_input_name = name

        if self.image_input_name is None:
            # fallback to first input
            self.image_input_name = self.input_names[0]

        if self.img_size_input_name is None:
            for name in self.input_names:
                if name != self.image_input_name:
                    self.img_size_input_name = name
                    break

        self.device_ptrs: Dict[str, int] = {}
        self.output_host: Dict[str, np.ndarray] = {}
        self.allocated_shapes: Dict[str, Tuple[int, ...]] = {}

    def _cuda(self, ret, op_name: str):
        if isinstance(ret, tuple):
            status = int(ret[0])
            values = ret[1:]
        else:
            status = int(ret)
            values = ()
        if status != 0:
            hint = ''
            if status == 35:
                hint = (
                    ' | cudaErrorInsufficientDriver: CUDA runtime and NVIDIA driver are incompatible. '
                    'If cuda-python 13.x is installed on a CUDA 12.x driver, downgrade cuda-python to 12.x.'
                )
            raise RuntimeError(f'{op_name} failed (cudaError={status}){hint}')
        if len(values) == 0:
            return None
        if len(values) == 1:
            return values[0]
        return values

    def _free_buffers(self):
        for ptr in self.device_ptrs.values():
            try:
                self._cuda(self.cudart.cudaFree(ptr), 'cudaFree')
            except Exception:
                pass
        self.device_ptrs.clear()
        self.output_host.clear()
        self.allocated_shapes.clear()

    def _ensure_allocated(self, inputs: Dict[str, np.ndarray]):
        # Set dynamic input shapes first.
        for name in self.input_names:
            arr = inputs[name]
            self.context.set_input_shape(name, tuple(arr.shape))

        needs_alloc = False
        current_shapes: Dict[str, Tuple[int, ...]] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(int(x) for x in self.context.get_tensor_shape(name))
            if any(x < 0 for x in shape):
                raise RuntimeError(f'Unresolved dynamic shape for tensor: {name} -> {shape}')
            current_shapes[name] = shape
            if self.allocated_shapes.get(name) != shape:
                needs_alloc = True

        if not needs_alloc and len(self.device_ptrs) == self.engine.num_io_tensors:
            return

        self._free_buffers()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = self.trt.nptype(self.engine.get_tensor_dtype(name))
            shape = current_shapes[name]
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            ptr = self._cuda(self.cudart.cudaMalloc(nbytes), f'cudaMalloc({name})')
            self.device_ptrs[name] = int(ptr)
            self.context.set_tensor_address(name, int(ptr))
            self.allocated_shapes[name] = shape

            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.OUTPUT:
                self.output_host[name] = np.empty(shape, dtype=dtype)

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for name in self.input_names:
            if name not in inputs:
                raise KeyError(f'Missing input tensor: {name}')
            arr = np.ascontiguousarray(inputs[name])
            expected_dtype = self.trt.nptype(self.engine.get_tensor_dtype(name))
            if arr.dtype != expected_dtype:
                arr = arr.astype(expected_dtype, copy=False)
            inputs[name] = arr

        self._ensure_allocated(inputs)

        for name in self.input_names:
            arr = inputs[name]
            self._cuda(
                self.cudart.cudaMemcpyAsync(
                    self.device_ptrs[name],
                    arr.ctypes.data,
                    arr.nbytes,
                    self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self.stream,
                ),
                f'cudaMemcpyAsync H2D ({name})',
            )

        ok = self.context.execute_async_v3(stream_handle=self.stream)
        if not ok:
            raise RuntimeError('TensorRT execution failed (execute_async_v3 returned False)')

        for name in self.output_names:
            out = self.output_host[name]
            self._cuda(
                self.cudart.cudaMemcpyAsync(
                    out.ctypes.data,
                    self.device_ptrs[name],
                    out.nbytes,
                    self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self.stream,
                ),
                f'cudaMemcpyAsync D2H ({name})',
            )

        self._cuda(self.cudart.cudaStreamSynchronize(self.stream), 'cudaStreamSynchronize')
        return {k: v.copy() for k, v in self.output_host.items()}

    def describe(self):
        print('[INFO] TensorRT IO tensors:')
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            mode_s = 'INPUT' if mode == self.trt.TensorIOMode.INPUT else 'OUTPUT'
            dtype = self.trt.nptype(self.engine.get_tensor_dtype(name)).__name__
            shape = tuple(self.engine.get_tensor_shape(name))
            print(f'  - {mode_s:6} {name:20s} dtype={dtype:8s} shape={shape}')

    def get_engine_image_shape(self):
        shape = tuple(self.engine.get_tensor_shape(self.image_input_name))
        if len(shape) == 4 and shape[0] > 0 and shape[2] > 0 and shape[3] > 0:
            return tuple(int(v) for v in shape)
        return None

    def close(self):
        self._free_buffers()
        try:
            self._cuda(self.cudart.cudaStreamDestroy(self.stream), 'cudaStreamDestroy')
        except Exception:
            pass


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
    pred_record = {
        'joints2d': np.zeros((0, 24, 2), dtype=np.float32),
        'joints3d': np.zeros((0, 24, 3), dtype=np.float32),
        'verts': np.zeros((0, 0, 3), dtype=np.float32),
        'transl': np.zeros((0, 3), dtype=np.float32),
        'bboxes': np.zeros((0, 4), dtype=np.float32),
        'focal_length': np.zeros((0,), dtype=np.float32),
    }

    pred_poses = np.zeros((0, 72), dtype=np.float32)
    pred_betas = np.zeros((0, 10), dtype=np.float32)
    pred_transl = np.zeros((0, 3), dtype=np.float32)
    pred_verts = np.zeros((0, 0, 3), dtype=np.float32)
    pred_j3d_24 = np.zeros((0, 24, 3), dtype=np.float32)
    pred_j2d_24 = np.zeros((0, 24, 2), dtype=np.float32)
    pred_intrinsics_np = build_cam_intrinsics_from_img_size(img_size, input_size)
    if 'pred_intrinsics' in outputs:
        pred_intrinsics_np = outputs['pred_intrinsics'][0].reshape(3, 3).astype(np.float32)

    if num_persons > 0:
        pred_poses = pick('pred_poses')[0][select_idx].astype(np.float32)
        pred_betas = pick('pred_betas')[0][select_idx].astype(np.float32)
        pred_transl = pick('pred_transl')[0][select_idx].astype(np.float32)
        pred_boxes = pick('pred_boxes')[0][select_idx].astype(np.float32)
        conf_t = confs[select_idx].astype(np.float32)
        qids = select_idx.astype(np.int32)
        udp_candidates = {
            'poses': pred_poses,
            'transls': pred_transl,
            'confs': conf_t,
            'query_ids': qids,
        }

        if smpl_layer is not None:
            if smpl_device is None:
                smpl_device = torch.device('cpu')
            with torch.no_grad():
                poses_t = torch.from_numpy(pred_poses).to(device=smpl_device, non_blocking=True)
                betas_t = torch.from_numpy(pred_betas).to(device=smpl_device, non_blocking=True)
                verts, joints = smpl_layer(poses=poses_t, betas=betas_t)
                trans_t = torch.from_numpy(pred_transl).to(device=smpl_device, non_blocking=True).unsqueeze(1)
                verts_cam = verts + trans_t
                joints_cam = joints + trans_t
                pred_verts = verts_cam.cpu().numpy().astype(np.float32)
                pred_j3d_24 = joints_cam[:, :24, :].cpu().numpy().astype(np.float32)
                pred_j2d_24 = np.zeros((num_persons, 24, 2), dtype=np.float32)

                z = np.clip(pred_j3d_24[..., 2:3], 1e-6, None)
                pred_j2d_24[..., 0] = pred_j3d_24[..., 0] * pred_intrinsics_np[0, 0] / z[..., 0] + pred_intrinsics_np[0, 2]
                pred_j2d_24[..., 1] = pred_j3d_24[..., 1] * pred_intrinsics_np[1, 1] / z[..., 0] + pred_intrinsics_np[1, 2]

        pred_record = {
            'joints2d': pred_j2d_24,
            'joints3d': pred_j3d_24,
            'verts': pred_verts,
            'transl': pred_transl,
            'bboxes': pred_boxes,
            'focal_length': np.full((num_persons,), float(pred_intrinsics_np[0, 0]), dtype=np.float32),
        }

    times['postprocess'] = time.time() - t0

    if no_render or render_utils is None or smpl_layer is None:
        return frame, num_persons, times, udp_candidates, pred_record

    t0 = time.time()
    tensor_to_BGR = render_utils['tensor_to_BGR']
    pad_img = render_utils['pad_img']
    vis_meshes_img = render_utils['vis_meshes_img']
    get_colors_rgb = render_utils['get_colors_rgb']

    ori_img = tensor_to_BGR(unNormalize(img_tensor).cpu())

    if num_persons > 0:
        pred_intrinsics = torch.from_numpy(pred_intrinsics_np)
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

    return result_img, num_persons, times, udp_candidates, pred_record


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
            args.output = f'{base}_trt.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, video_fps, (actual_width, actual_height))
        if writer.isOpened():
            print(f'出力動画: {args.output}')
        else:
            writer = None
            print('[WARN] VideoWriter の初期化に失敗したため保存を無効化します')

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
    smpl_device = None
    if (not args.no_render) or args.pred_npz:
        if args.no_render and args.pred_npz:
            print('[INFO] Initialize SMPL layer for NPZ export...')
        else:
            print('[INFO] Initialize SMPL layer for mesh rendering...')
        smpl_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        smpl_layer = SMPL_Layer(model_path=smpl_model_path, with_genders=False).to(smpl_device).eval()

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
        print("  'q' または ESC: 終了 (早期終了)")
    else:
        print('  Ctrl+C: 終了')
    print('==================\n')

    frame_count = 0
    fps_queue = []
    recording = writer is not None
    total_persons = 0
    all_times = {'preprocess': [], 'inference': [], 'postprocess': [], 'render': []}
    pred_frame_ids = []
    pred_image_paths = []
    pred_person_counts = []
    pred_joints2d = []
    pred_joints3d = []
    pred_verts = []
    pred_transl = []
    pred_bboxes = []
    pred_focal_length = []
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
            if args.pred_npz:
                pred_frame_ids.append(frame_count)
                pred_image_paths.append(f"{args.input}#frame_{frame_count:06d}")
                pred_person_counts.append(num_persons)
                pred_joints2d.append(pred_record['joints2d'])
                pred_joints3d.append(pred_record['joints3d'])
                pred_verts.append(pred_record['verts'])
                pred_transl.append(pred_record['transl'])
                pred_bboxes.append(pred_record['bboxes'])
                pred_focal_length.append(pred_record['focal_length'])
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
                cv2.putText(result_frame, f'FPS: {current_fps:.1f}', (10, 30),
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
                        f"Frame: {frame_count}/{total_frames} | FPS: {current_fps:.1f} | Persons: {num_persons} | "
                        f"Preprocess: {times['preprocess']*1000:.0f}ms | "
                        f"Inference: {times['inference']*1000:.0f}ms | "
                        f"Postprocess: {times['postprocess']*1000:.0f}ms | "
                        f"Render: {render_ms}"
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
    if args.pred_npz:
        pred_dir = os.path.dirname(args.pred_npz)
        if pred_dir:
            os.makedirs(pred_dir, exist_ok=True)
        np.savez_compressed(
            args.pred_npz,
            frame_ids=np.asarray(pred_frame_ids, dtype=np.int64),
            image_paths=np.asarray(pred_image_paths, dtype=object),
            person_counts=np.asarray(pred_person_counts, dtype=np.int32),
            joints2d=np.asarray(pred_joints2d, dtype=object),
            joints3d=np.asarray(pred_joints3d, dtype=object),
            verts=np.asarray(pred_verts, dtype=object),
            transl=np.asarray(pred_transl, dtype=object),
            bboxes=np.asarray(pred_bboxes, dtype=object),
            focal_length=np.asarray(pred_focal_length, dtype=object),
            failed_frames=np.int32(0),
        )
        print(f'予測NPZ: {args.pred_npz}')


if __name__ == '__main__':
    main()

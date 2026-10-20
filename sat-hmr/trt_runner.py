"""TensorRT engine runner and per-frame helpers.

TensorRTRunner / pick_output / select_confident_queries are shared by
video_demo_trt.py, webcam_demo_trt.py and tools/eval_3dpw_trt.py; the
"demo helpers" section is shared by the two demos only.

tensorrt and cuda-python are imported lazily in TensorRTRunner.__init__, so
importing this module does not require them.
"""

import sys
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from sathmr_common import preprocess_frame, print_udp_settings
from utils.transforms import unNormalize


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
        for name in self._tensor_names():
            if self._is_input(name):
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        if not self.input_names:
            raise RuntimeError('Engine has no inputs')

        self.image_input_name = None
        self.img_size_input_name = None
        for name in self.input_names:
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = self._np_dtype(name)
            if len(shape) == 4 and dtype == np.float32:
                self.image_input_name = name
            elif len(shape) <= 2 and dtype in (np.int32, np.int64) and 'img_size' in name:
                self.img_size_input_name = name

        if self.image_input_name is None:
            self.image_input_name = self.input_names[0]

        if self.img_size_input_name is None:
            for name in self.input_names:
                if name != self.image_input_name:
                    self.img_size_input_name = name
                    break

        self.device_ptrs: Dict[str, int] = {}
        self.output_host: Dict[str, np.ndarray] = {}
        self.allocated_shapes: Dict[str, Tuple[int, ...]] = {}

    def _tensor_names(self):
        return [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]

    def _is_input(self, name: str) -> bool:
        return self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT

    def _np_dtype(self, name: str):
        return self.trt.nptype(self.engine.get_tensor_dtype(name))

    def _cuda(self, ret, op_name: str):
        """Raise on a non-zero cudaError and unwrap the remaining return values."""
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
        """Set the input shapes and (re)allocate device/host buffers when any IO shape changed."""
        for name in self.input_names:
            self.context.set_input_shape(name, tuple(inputs[name].shape))

        needs_alloc = False
        current_shapes: Dict[str, Tuple[int, ...]] = {}
        for name in self._tensor_names():
            shape = tuple(int(x) for x in self.context.get_tensor_shape(name))
            if any(x < 0 for x in shape):
                raise RuntimeError(f'Unresolved dynamic shape for tensor: {name} -> {shape}')
            current_shapes[name] = shape
            if self.allocated_shapes.get(name) != shape:
                needs_alloc = True

        if not needs_alloc and len(self.device_ptrs) == self.engine.num_io_tensors:
            return

        self._free_buffers()

        for name in self._tensor_names():
            dtype = self._np_dtype(name)
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
            expected_dtype = self._np_dtype(name)
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
        for name in self._tensor_names():
            mode_s = 'INPUT' if self._is_input(name) else 'OUTPUT'
            dtype = self._np_dtype(name).__name__
            shape = tuple(self.engine.get_tensor_shape(name))
            print(f'  - {mode_s:6} {name:20s} dtype={dtype:8s} shape={shape}')

    def get_engine_image_shape(self):
        """Static (N, C, H, W) of the image input, or None if any of N/H/W is dynamic."""
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


def pick_output(outputs: Dict[str, np.ndarray], name: str) -> np.ndarray:
    if name not in outputs:
        raise KeyError(f'Output tensor not found: {name}. got={list(outputs.keys())}')
    return outputs[name]


def select_confident_queries(outputs: Dict[str, np.ndarray], conf_thresh: float):
    """Returns (confs of every query, indices of the queries with conf > conf_thresh)."""
    confs = pick_output(outputs, 'pred_confs')[0].reshape(-1)
    return confs, np.where(confs > conf_thresh)[0]


# ---------------------------------------------------------------------------
# Demo helpers (video_demo_trt.py / webcam_demo_trt.py)
# ---------------------------------------------------------------------------

def normalize_demo_args(args):
    """Validate the shared demo options and resolve the flags that imply others."""
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


def open_runner_or_exit(engine_path: str) -> TensorRTRunner:
    print(f'[INFO] TensorRT engine load: {engine_path}')
    try:
        runner = TensorRTRunner(engine_path)
    except Exception as exc:
        print(f'[ERROR] TensorRT runner init failed: {exc}')
        print('ヒント: `pip install tensorrt cuda-python` (環境に合わせて)')
        sys.exit(1)

    runner.describe()
    return runner


def print_demo_settings(args):
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
        print_udp_settings(args)


def infer_frame(runner: TensorRTRunner, frame: np.ndarray, input_size: int, times: Dict[str, float]):
    """Preprocess one BGR frame and run the engine; records times['preprocess'] / ['inference'].

    Returns (img_tensor, resized (h, w), original (h, w), outputs).
    """
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
    return img_tensor, img_size, orig_size, outputs


def render_mesh_frame(
    render_utils: Dict[str, object],
    img_tensor,
    img_size,
    orig_size,
    verts: Optional[np.ndarray],
    smpl_faces,
    cam_intrinsics,
) -> np.ndarray:
    """Draw camera-space SMPL meshes on the preprocessed image, resized back to the frame size.

    verts=None (nobody detected) returns the bare image.
    """
    ori_img = render_utils['tensor_to_BGR'](unNormalize(img_tensor).cpu())

    if verts is None:
        result_img = render_utils['pad_img'](ori_img, max(ori_img.shape[0], ori_img.shape[1]))
    else:
        colors = render_utils['get_colors_rgb'](len(verts))
        result_img = render_utils['vis_meshes_img'](
            img=ori_img.copy(),
            verts=verts,
            smpl_faces=smpl_faces,
            cam_intrinsics=cam_intrinsics,
            colors=colors,
            padding=True,
        )

    # Drop the patch padding, then undo the preprocess resize.
    result_img = result_img[:img_size[0], :img_size[1]]
    return cv2.resize(result_img, (orig_size[1], orig_size[0]))


def update_fps_window(fps_queue, elapsed: float, window: int = 30):
    """Append this frame's FPS to fps_queue (keeping the last `window`). Returns (current, average)."""
    current_fps = 1.0 / elapsed if elapsed > 0 else 0
    fps_queue.append(current_fps)
    if len(fps_queue) > window:
        fps_queue.pop(0)
    avg_fps = float(np.mean(fps_queue)) if fps_queue else 0.0
    return current_fps, avg_fps


def draw_timing_overlay(frame: np.ndarray, fps: float, num_persons: int, times: Dict[str, float]):
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Persons: {num_persons}', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    y_offset = 110
    for label, key in (('Preprocess', 'preprocess'), ('Inference', 'inference'),
                       ('Postprocess', 'postprocess'), ('Render', 'render')):
        if key not in times:
            continue
        cv2.putText(frame, f'{label}: {times[key]*1000:.1f}ms', (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 30


def draw_rec_marker(frame: np.ndarray, frame_width: int):
    cv2.circle(frame, (frame_width - 30, 30), 10, (0, 0, 255), -1)
    cv2.putText(frame, 'REC', (frame_width - 80, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def format_timing_status(times: Dict[str, float]) -> str:
    render_ms = f"{times['render']*1000:.0f}ms" if 'render' in times else 'skip'
    return (
        f"Preprocess: {times['preprocess']*1000:.0f}ms | "
        f"Inference: {times['inference']*1000:.0f}ms | "
        f"Postprocess: {times['postprocess']*1000:.0f}ms | "
        f"Render: {render_ms}"
    )

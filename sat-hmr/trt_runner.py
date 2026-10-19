"""TensorRT engine runner shared by video_demo_trt.py and webcam_demo_trt.py.

tensorrt and cuda-python are imported lazily in TensorRTRunner.__init__, so
importing this module does not require them.
"""

from typing import Dict, Tuple

import numpy as np


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

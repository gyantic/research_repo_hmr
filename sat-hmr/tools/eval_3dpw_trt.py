#!/usr/bin/env python3
"""Evaluate SAT-HMR TensorRT engine on 3DPW (GT) with MPJPE/PA-MPJPE/PVE/Detection metrics.

This script follows the same metric definition as `engines/funcs/eval_funcs.py:evaluate_3dpw`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.paths import smpl_model_path
from datasets.pw3d import PW3D
from models.human_models import SMPL_Layer
from tools.sat_hmr_export_utils import load_config
from utils import misc
from utils.constants import H36M_EVAL_JOINTS
from utils.evaluation import cal_3d_position_error, compute_prf1, get_matching_dict, match_2d_greedy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 3DPW with TensorRT inference")
    parser.add_argument("--engine", required=True, help="TensorRT engine path")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="3DPW split")
    parser.add_argument("--input_size", type=int, default=1288, help="SAT-HMR input size")
    parser.add_argument("--conf_thresh", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--iou_thresh", type=float, default=0.1, help="2D matching IoU threshold")
    parser.add_argument("--batch_size", type=int, default=1, help="Evaluation batch size (TRT is usually 1)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--max_images", type=int, default=-1, help="Limit number of images for quick run")
    parser.add_argument("--smpl_device", choices=["cuda", "cpu"], default="cuda", help="Device for SMPL reconstruction")
    parser.add_argument("--summary_csv", default="plots/eval_3dpw_trt_summary.csv", help="Summary CSV output")
    parser.add_argument("--details_csv", default="", help="Per-image details CSV output (optional)")
    parser.add_argument("--summary_json", default="", help="Summary JSON output (optional)")
    return parser.parse_args()


def ensure_parent(path: str) -> None:
    if not path:
        return
    parent = Path(path).parent
    if str(parent) and str(parent) != ".":
        parent.mkdir(parents=True, exist_ok=True)


def project_joints(j3d: np.ndarray, cam_intrinsics: np.ndarray) -> np.ndarray:
    # j3d: [N, J, 3], K: [3, 3]
    proj = np.matmul(j3d, cam_intrinsics.T)
    z = np.clip(proj[..., 2:3], 1e-8, None)
    return proj[..., :2] / z


def pick_output(outputs: Dict[str, np.ndarray], name: str) -> np.ndarray:
    if name not in outputs:
        raise KeyError(f"Output tensor not found: {name}. got={list(outputs.keys())}")
    return outputs[name]


class TensorRTRunner:
    def __init__(self, engine_path: str):
        try:
            import tensorrt as trt  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "TensorRT Python bindings are not available. "
                "Install tensorrt in the active environment. "
                f"(detail: {exc})"
            )

        cuda_import_exc = None
        try:
            from cuda import cudart as cuda_rt  # type: ignore
        except Exception as exc:
            cuda_import_exc = exc
            try:
                from cuda.bindings import runtime as cuda_rt  # type: ignore
            except Exception:
                raise RuntimeError(
                    "CUDA runtime bindings are not available. "
                    "Install cuda-python in this environment. "
                    f"(detail: {cuda_import_exc})"
                )

        self.trt = trt
        self.cudart = cuda_rt

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        if hasattr(runtime, "engine_host_code_allowed"):
            runtime.engine_host_code_allowed = True

        with open(engine_path, "rb") as f:
            engine_bytes = f.read()

        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError(
                f"Failed to deserialize engine: {engine_path} "
                f"(TensorRT runtime={self.trt.__version__}). "
                "The engine was likely built by an incompatible TensorRT version."
            )

        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.engine = engine
        self.context = context
        self.runtime = runtime
        self.stream = self._cuda(self.cudart.cudaStreamCreate(), "cudaStreamCreate")

        self.input_names: List[str] = []
        self.output_names: List[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == self.trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        if not self.input_names:
            raise RuntimeError("Engine has no inputs")

        self.image_input_name = None
        self.img_size_input_name = None
        for name in self.input_names:
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            if len(shape) == 4 and self.trt.nptype(dtype) == np.float32:
                self.image_input_name = name
            elif len(shape) <= 2 and self.trt.nptype(dtype) in (np.int32, np.int64):
                if "img_size" in name:
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

    def _cuda(self, ret, op_name: str):
        if isinstance(ret, tuple):
            status = int(ret[0])
            values = ret[1:]
        else:
            status = int(ret)
            values = ()
        if status != 0:
            hint = ""
            if status == 35:
                hint = (
                    " | cudaErrorInsufficientDriver: CUDA runtime and NVIDIA driver are incompatible. "
                    "If cuda-python 13.x is installed on a CUDA 12.x driver, downgrade cuda-python to 12.x."
                )
            raise RuntimeError(f"{op_name} failed (cudaError={status}){hint}")
        if len(values) == 0:
            return None
        if len(values) == 1:
            return values[0]
        return values

    def _free_buffers(self):
        for ptr in self.device_ptrs.values():
            try:
                self._cuda(self.cudart.cudaFree(ptr), "cudaFree")
            except Exception:
                pass
        self.device_ptrs.clear()
        self.output_host.clear()
        self.allocated_shapes.clear()

    def _ensure_allocated(self, inputs: Dict[str, np.ndarray]):
        for name in self.input_names:
            arr = inputs[name]
            self.context.set_input_shape(name, tuple(arr.shape))

        needs_alloc = False
        current_shapes: Dict[str, Tuple[int, ...]] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(int(x) for x in self.context.get_tensor_shape(name))
            if any(x < 0 for x in shape):
                raise RuntimeError(f"Unresolved dynamic shape for tensor: {name} -> {shape}")
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
            ptr = self._cuda(self.cudart.cudaMalloc(nbytes), f"cudaMalloc({name})")
            self.device_ptrs[name] = int(ptr)
            self.context.set_tensor_address(name, int(ptr))
            self.allocated_shapes[name] = shape

            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.OUTPUT:
                self.output_host[name] = np.empty(shape, dtype=dtype)

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for name in self.input_names:
            if name not in inputs:
                raise KeyError(f"Missing input tensor: {name}")
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
                f"cudaMemcpyAsync H2D ({name})",
            )

        ok = self.context.execute_async_v3(stream_handle=self.stream)
        if not ok:
            raise RuntimeError("TensorRT execution failed (execute_async_v3 returned False)")

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
                f"cudaMemcpyAsync D2H ({name})",
            )

        self._cuda(self.cudart.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")
        return {k: v.copy() for k, v in self.output_host.items()}

    def get_engine_image_shape(self):
        shape = tuple(self.engine.get_tensor_shape(self.image_input_name))
        if len(shape) == 4 and shape[0] > 0 and shape[2] > 0 and shape[3] > 0:
            return tuple(int(v) for v in shape)
        return None

    def close(self):
        self._free_buffers()
        try:
            self._cuda(self.cudart.cudaStreamDestroy(self.stream), "cudaStreamDestroy")
        except Exception:
            pass


def evaluate(args: argparse.Namespace) -> Dict[str, float]:
    cfg = load_config(args.input_size)
    ds = PW3D(
        split=args.split,
        input_size=args.input_size,
        aug=False,
        mode="eval",
        human_type="smpl",
        sat_cfg=cfg.sat_cfg,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=misc.collate_fn,
    )

    smpl_device = torch.device("cpu")
    if args.smpl_device == "cuda" and torch.cuda.is_available():
        smpl_device = torch.device("cuda")

    smpl_layer = SMPL_Layer(model_path=smpl_model_path, with_genders=False).to(smpl_device).eval()
    smpl2h36m_dev = torch.from_numpy(smpl_layer.smpl2h36m_regressor).float().to(smpl_device)
    smpl2h36m_cpu = smpl2h36m_dev.detach().cpu()

    runner = TensorRTRunner(args.engine)
    expected_shape = runner.get_engine_image_shape()
    if expected_shape is not None:
        print(f"[INFO] Engine static image shape: {expected_shape}")

    total_count = 0
    total_miss = 0
    total_fp = 0

    mpjpe_vals: List[float] = []
    pa_mpjpe_vals: List[float] = []
    pve_vals: List[float] = []
    pa_pve_vals: List[float] = []
    infer_ms_list: List[float] = []

    details_writer = None
    details_fp = None
    if args.details_csv:
        ensure_parent(args.details_csv)
        details_fp = open(args.details_csv, "w", newline="", encoding="utf-8")
        details_writer = csv.writer(details_fp)
        details_writer.writerow([
            "image_index",
            "image_path",
            "gt_persons",
            "pred_persons",
            "misses",
            "false_positives",
            "matched",
            "infer_ms",
            "mpjpe_mm_mean",
            "pa_mpjpe_mm_mean",
            "pve_mm_mean",
            "pa_pve_mm_mean",
        ])

    start_t = time.perf_counter()
    processed_images = 0

    progress = tqdm(total=len(loader), desc="evaluate_3dpw_trt")
    for samples, targets in loader:
        for sample, target in zip(samples, targets):
            if args.max_images > 0 and processed_images >= args.max_images:
                break

            image_np = np.ascontiguousarray(sample.unsqueeze(0).numpy().astype(np.float32))
            if expected_shape is not None and tuple(image_np.shape) != expected_shape:
                # Pad or crop spatial dims to match engine static shape
                _, _, eh, ew = expected_shape
                _, _, ih, iw = image_np.shape
                if ih != eh or iw != ew:
                    padded = np.zeros((1, 3, eh, ew), dtype=np.float32)
                    ch = min(ih, eh)
                    cw = min(iw, ew)
                    padded[:, :, :ch, :cw] = image_np[:, :, :ch, :cw]
                    image_np = padded

            trt_inputs = {runner.image_input_name: image_np}
            if runner.img_size_input_name is not None:
                trt_inputs[runner.img_size_input_name] = np.asarray(target["img_size"].cpu().numpy(), dtype=np.int64)

            t0 = time.perf_counter()
            outputs = runner.infer(trt_inputs)
            infer_ms = (time.perf_counter() - t0) * 1000.0
            infer_ms_list.append(infer_ms)

            confs = pick_output(outputs, "pred_confs")[0].reshape(-1)
            sel = np.where(confs > args.conf_thresh)[0]

            if len(sel) > 0:
                pred_poses = pick_output(outputs, "pred_poses")[0][sel].astype(np.float32)
                pred_betas = pick_output(outputs, "pred_betas")[0][sel].astype(np.float32)
                pred_transl = pick_output(outputs, "pred_transl")[0][sel].astype(np.float32)

                with torch.no_grad():
                    poses_t = torch.from_numpy(pred_poses).to(device=smpl_device, non_blocking=True)
                    betas_t = torch.from_numpy(pred_betas).to(device=smpl_device, non_blocking=True)
                    transl_t = torch.from_numpy(pred_transl).to(device=smpl_device, non_blocking=True)

                    verts_t, smpl_joints_t = smpl_layer(poses=poses_t, betas=betas_t)
                    verts_t = verts_t + transl_t.unsqueeze(1)
                    smpl_joints_t = smpl_joints_t + transl_t.unsqueeze(1)
                    j3ds_t = (
                        torch.einsum("bik,ji->bjk", [verts_t - transl_t.unsqueeze(1), smpl2h36m_dev])
                        + transl_t.unsqueeze(1)
                    )

                pred_verts = verts_t.detach().cpu().numpy()
                pred_j3ds = j3ds_t.detach().cpu().numpy()
                pred_smpl_j3ds = smpl_joints_t.detach().cpu().numpy()
            else:
                pred_verts = np.zeros((0, 6890, 3), dtype=np.float32)
                pred_j3ds = np.zeros((0, 17, 3), dtype=np.float32)
                pred_smpl_j3ds = np.zeros((0, 24, 3), dtype=np.float32)

            cam_intrinsics = target["cam_intrinsics"]
            if cam_intrinsics.ndim == 3:
                cam_k = cam_intrinsics[0].cpu().numpy()
            else:
                cam_k = cam_intrinsics.cpu().numpy()
            pred_j2ds = project_joints(pred_smpl_j3ds[:, :24, :], cam_k)

            gt_verts_t = target["verts"].cpu()
            gt_transl_t = target["transl"].cpu()
            gt_j3ds_t = (
                torch.einsum("bik,ji->bjk", [gt_verts_t - gt_transl_t[:, None, :], smpl2h36m_cpu])
                + gt_transl_t[:, None, :]
            )
            gt_verts = gt_verts_t.numpy()
            gt_j3ds = gt_j3ds_t.numpy()
            gt_j2ds = target["j2ds"].cpu().numpy()[:, :24, :]

            greedy_match = match_2d_greedy(pred_j2ds, gt_j2ds, iou_thresh=args.iou_thresh)
            match_dict, false_positive_count = get_matching_dict(greedy_match)

            gt_count = int(len(gt_j3ds))
            misses = 0
            matched = 0
            sample_mpjpe: List[float] = []
            sample_pa_mpjpe: List[float] = []
            sample_pve: List[float] = []
            sample_pa_pve: List[float] = []

            for gt_idx in range(gt_count):
                pred_idx = match_dict.get(str(gt_idx), "miss")
                if pred_idx in ("miss", "invalid"):
                    misses += 1
                    continue

                pred_idx = int(pred_idx)
                matched += 1

                gt3d = gt_j3ds[gt_idx].reshape(-1, 3)
                pd3d = pred_j3ds[pred_idx].reshape(-1, 3)
                gtv = gt_verts[gt_idx].reshape(-1, 3)
                pdv = pred_verts[pred_idx].reshape(-1, 3)

                gt_pelvis = gt3d[[0], :].copy()
                pd_pelvis = pd3d[[0], :].copy()

                gt3d = (gt3d - gt_pelvis)[H36M_EVAL_JOINTS, :].copy()
                pd3d = (pd3d - pd_pelvis)[H36M_EVAL_JOINTS, :].copy()
                gtv = (gtv - gt_pelvis).copy()
                pdv = (pdv - pd_pelvis).copy()

                e_j, pa_e_j = cal_3d_position_error(pd3d, gt3d)
                e_v, pa_e_v = cal_3d_position_error(pdv, gtv)

                sample_mpjpe.append(float(e_j))
                sample_pa_mpjpe.append(float(pa_e_j))
                sample_pve.append(float(e_v))
                sample_pa_pve.append(float(pa_e_v))

                mpjpe_vals.append(float(e_j))
                pa_mpjpe_vals.append(float(pa_e_j))
                pve_vals.append(float(e_v))
                pa_pve_vals.append(float(pa_e_v))

            total_count += gt_count
            total_miss += misses
            total_fp += int(false_positive_count)

            if details_writer is not None:
                details_writer.writerow([
                    processed_images,
                    target.get("img_path", ""),
                    gt_count,
                    int(len(sel)),
                    misses,
                    int(false_positive_count),
                    matched,
                    round(infer_ms, 4),
                    round(float(np.mean(sample_mpjpe)), 4) if sample_mpjpe else "",
                    round(float(np.mean(sample_pa_mpjpe)), 4) if sample_pa_mpjpe else "",
                    round(float(np.mean(sample_pve)), 4) if sample_pve else "",
                    round(float(np.mean(sample_pa_pve)), 4) if sample_pa_pve else "",
                ])

            processed_images += 1

        progress.update(1)
        if args.max_images > 0 and processed_images >= args.max_images:
            break

    progress.close()
    total_sec = time.perf_counter() - start_t

    precision, recall, f1 = compute_prf1(total_count, total_miss, total_fp)

    def safe_mean(vals: List[float]) -> float:
        if not vals:
            return float("nan")
        return float(np.mean(vals))

    def safe_p95(vals: List[float]) -> float:
        if not vals:
            return float("nan")
        return float(np.percentile(vals, 95))

    summary = {
        "backend": "tensorrt",
        "engine": args.engine,
        "split": args.split,
        "input_size": args.input_size,
        "conf_thresh": args.conf_thresh,
        "iou_thresh": args.iou_thresh,
        "images_evaluated": int(processed_images),
        "gt_instances": int(total_count),
        "tp_instances": int(total_count - total_miss),
        "fp_instances": int(total_fp),
        "fn_instances": int(total_miss),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "detection_rate": float(recall),
        "MPJPE_mm": round(safe_mean(mpjpe_vals), 4),
        "PA_MPJPE_mm": round(safe_mean(pa_mpjpe_vals), 4),
        "MVE_mm": round(safe_mean(pve_vals), 4),
        "PA_MVE_mm": round(safe_mean(pa_pve_vals), 4),
        "infer_ms_mean": round(safe_mean(infer_ms_list), 4),
        "infer_ms_p95": round(safe_p95(infer_ms_list), 4),
        "wall_time_sec": round(total_sec, 4),
    }

    if details_fp is not None:
        details_fp.close()
    runner.close()
    return summary


def main() -> int:
    args = parse_args()
    ensure_parent(args.summary_csv)
    if args.summary_json:
        ensure_parent(args.summary_json)

    summary = evaluate(args)

    print("\n=== 3DPW TRT Evaluation Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    fieldnames = list(summary.keys())
    with open(args.summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary)
    print(f"[DONE] Summary CSV: {args.summary_csv}")

    if args.summary_json:
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[DONE] Summary JSON: {args.summary_json}")

    if args.details_csv:
        print(f"[DONE] Details CSV: {args.details_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

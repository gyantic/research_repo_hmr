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
from typing import Dict, List, Optional, Tuple

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
from trt_runner import TensorRTRunner, pick_output, select_confident_queries
from utils import misc
from utils.constants import H36M_EVAL_JOINTS
from utils.evaluation import cal_3d_position_error, compute_prf1, get_matching_dict, match_2d_greedy


DETAILS_CSV_COLUMNS = [
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
]


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


def fit_to_engine_shape(image_np: np.ndarray, expected_shape: Optional[Tuple[int, ...]]) -> np.ndarray:
    """Zero-pad / crop H and W to the engine's static input shape (3DPW image sizes vary)."""
    if expected_shape is None or tuple(image_np.shape) == expected_shape:
        return image_np
    _, _, eh, ew = expected_shape
    _, _, ih, iw = image_np.shape
    if ih == eh and iw == ew:
        return image_np
    padded = np.zeros((1, 3, eh, ew), dtype=np.float32)
    ch = min(ih, eh)
    cw = min(iw, ew)
    padded[:, :, :ch, :cw] = image_np[:, :, :ch, :cw]
    return padded


def to_h36m_joints(verts: torch.Tensor, transl: torch.Tensor, smpl2h36m: torch.Tensor) -> torch.Tensor:
    """Regress H36M joints from root-relative vertices, then move them back by transl ([B, 1, 3])."""
    return torch.einsum("bik,ji->bjk", [verts - transl, smpl2h36m]) + transl


def predict_people(
    outputs: Dict[str, np.ndarray],
    sel: np.ndarray,
    smpl_layer: SMPL_Layer,
    smpl_device: torch.device,
    smpl2h36m_dev: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns camera-space (verts [N,6890,3], H36M joints [N,17,3], SMPL joints [N,24,3])."""
    if len(sel) == 0:
        return (
            np.zeros((0, 6890, 3), dtype=np.float32),
            np.zeros((0, 17, 3), dtype=np.float32),
            np.zeros((0, 24, 3), dtype=np.float32),
        )

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
        j3ds_t = to_h36m_joints(verts_t, transl_t.unsqueeze(1), smpl2h36m_dev)

    return (
        verts_t.detach().cpu().numpy(),
        j3ds_t.detach().cpu().numpy(),
        smpl_joints_t.detach().cpu().numpy(),
    )


def pelvis_aligned_errors(
    pred_j3d: np.ndarray, gt_j3d: np.ndarray, pred_verts: np.ndarray, gt_verts: np.ndarray
) -> Tuple[float, float, float, float]:
    """Returns (MPJPE, PA-MPJPE, PVE, PA-PVE) after aligning both sides at their pelvis joint."""
    gt3d = gt_j3d.reshape(-1, 3)
    pd3d = pred_j3d.reshape(-1, 3)
    gtv = gt_verts.reshape(-1, 3)
    pdv = pred_verts.reshape(-1, 3)

    gt_pelvis = gt3d[[0], :].copy()
    pd_pelvis = pd3d[[0], :].copy()

    gt3d = (gt3d - gt_pelvis)[H36M_EVAL_JOINTS, :].copy()
    pd3d = (pd3d - pd_pelvis)[H36M_EVAL_JOINTS, :].copy()
    gtv = (gtv - gt_pelvis).copy()
    pdv = (pdv - pd_pelvis).copy()

    e_j, pa_e_j = cal_3d_position_error(pd3d, gt3d)
    e_v, pa_e_v = cal_3d_position_error(pdv, gtv)
    return float(e_j), float(pa_e_j), float(e_v), float(pa_e_v)


def safe_mean(vals: List[float]) -> float:
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def safe_p95(vals: List[float]) -> float:
    if not vals:
        return float("nan")
    return float(np.percentile(vals, 95))


def rounded_mean_or_blank(vals: List[float]):
    return round(float(np.mean(vals)), 4) if vals else ""


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
        details_writer.writerow(DETAILS_CSV_COLUMNS)

    start_t = time.perf_counter()
    processed_images = 0

    progress = tqdm(total=len(loader), desc="evaluate_3dpw_trt")
    for samples, targets in loader:
        for sample, target in zip(samples, targets):
            if args.max_images > 0 and processed_images >= args.max_images:
                break

            image_np = np.ascontiguousarray(sample.unsqueeze(0).numpy().astype(np.float32))
            image_np = fit_to_engine_shape(image_np, expected_shape)

            trt_inputs = {runner.image_input_name: image_np}
            if runner.img_size_input_name is not None:
                trt_inputs[runner.img_size_input_name] = np.asarray(target["img_size"].cpu().numpy(), dtype=np.int64)

            t0 = time.perf_counter()
            outputs = runner.infer(trt_inputs)
            infer_ms = (time.perf_counter() - t0) * 1000.0
            infer_ms_list.append(infer_ms)

            _confs, sel = select_confident_queries(outputs, args.conf_thresh)
            pred_verts, pred_j3ds, pred_smpl_j3ds = predict_people(
                outputs, sel, smpl_layer, smpl_device, smpl2h36m_dev
            )

            cam_intrinsics = target["cam_intrinsics"]
            if cam_intrinsics.ndim == 3:
                cam_k = cam_intrinsics[0].cpu().numpy()
            else:
                cam_k = cam_intrinsics.cpu().numpy()
            pred_j2ds = project_joints(pred_smpl_j3ds[:, :24, :], cam_k)

            gt_verts_t = target["verts"].cpu()
            gt_transl_t = target["transl"].cpu()
            gt_j3ds_t = to_h36m_joints(gt_verts_t, gt_transl_t[:, None, :], smpl2h36m_cpu)
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

                e_j, pa_e_j, e_v, pa_e_v = pelvis_aligned_errors(
                    pred_j3ds[pred_idx], gt_j3ds[gt_idx], pred_verts[pred_idx], gt_verts[gt_idx]
                )

                sample_mpjpe.append(e_j)
                sample_pa_mpjpe.append(pa_e_j)
                sample_pve.append(e_v)
                sample_pa_pve.append(pa_e_v)

                mpjpe_vals.append(e_j)
                pa_mpjpe_vals.append(pa_e_j)
                pve_vals.append(e_v)
                pa_pve_vals.append(pa_e_v)

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
                    rounded_mean_or_blank(sample_mpjpe),
                    rounded_mean_or_blank(sample_pa_mpjpe),
                    rounded_mean_or_blank(sample_pve),
                    rounded_mean_or_blank(sample_pa_pve),
                ])

            processed_images += 1

        progress.update(1)
        if args.max_images > 0 and processed_images >= args.max_images:
            break

    progress.close()
    total_sec = time.perf_counter() - start_t

    precision, recall, f1 = compute_prf1(total_count, total_miss, total_fp)

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

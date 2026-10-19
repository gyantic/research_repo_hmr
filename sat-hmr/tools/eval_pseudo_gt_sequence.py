#!/usr/bin/env python3
"""Evaluate SAT-HMR predictions against pseudo-GT sequences (e.g., SAM3D).

Expected input:
- GT npz and Pred npz each contain per-frame person arrays.
- Supports both fixed-shape arrays and object arrays.

Primary metrics:
- Pseudo_MPJPE / Pseudo_PA_MPJPE (if joints3d exists on both sides)
- Pseudo_PVE / Pseudo_PA_PVE (if verts exists on both sides)
- precision / recall / f1 / detection_rate from frame-wise person matching
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.constants import H36M_EVAL_JOINTS
from utils.evaluation import cal_3d_position_error, compute_prf1, get_matching_dict, match_2d_greedy, select_and_align


AUTO_KEY = "auto"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate pseudo-GT sequence vs prediction sequence")
    p.add_argument("--gt_npz", required=True, help="Pseudo-GT .npz path (e.g., SAM3D output converted)")
    p.add_argument("--pred_npz", required=True, help="Prediction .npz path (SAT-HMR/TensorRT/ONNX)")

    p.add_argument("--match_mode", choices=["auto", "joints2d", "transl", "pelvis"], default="joints2d")
    p.add_argument("--iou_thresh", type=float, default=0.1, help="IoU threshold used by 2D greedy matching")
    p.add_argument(
        "--eval_style",
        choices=["auto", "3dpw", "agora"],
        default="auto",
        help="Metric style aligned with eval_funcs.py. auto: 3dpw style unless vert counts differ, then agora style.",
    )
    p.add_argument(
        "--transl_max_dist",
        type=float,
        default=-1.0,
        help="Optional max distance (meter) for transl/pelvis matching. -1 disables gating.",
    )

    p.add_argument("--gt_frame_key", default=AUTO_KEY)
    p.add_argument("--gt_joints2d_key", default=AUTO_KEY)
    p.add_argument("--gt_joints3d_key", default=AUTO_KEY)
    p.add_argument("--gt_verts_key", default=AUTO_KEY)
    p.add_argument("--gt_transl_key", default=AUTO_KEY)

    p.add_argument("--pred_frame_key", default=AUTO_KEY)
    p.add_argument("--pred_joints2d_key", default=AUTO_KEY)
    p.add_argument("--pred_joints3d_key", default=AUTO_KEY)
    p.add_argument("--pred_verts_key", default=AUTO_KEY)
    p.add_argument("--pred_transl_key", default=AUTO_KEY)

    p.add_argument("--summary_csv", default="plots/pseudo_gt_eval_summary.csv")
    p.add_argument("--append_summary", action="store_true", help="Append summary row to summary_csv instead of overwrite")
    p.add_argument(
        "--run_name",
        default="",
        help="Run label for summary comparison (e.g., torch, tensorrt). If empty, pred_npz stem is used.",
    )
    p.add_argument(
        "--compare_csv",
        default="",
        help="Optional output path for 2-run comparison CSV generated from summary_csv rows.",
    )
    p.add_argument("--compare_base_run", default="", help="Base run_name for comparison (optional)")
    p.add_argument("--compare_target_run", default="", help="Target run_name for comparison (optional)")
    # Backward-compatible aliases.
    p.add_argument("--compare_left_run", default="", help=argparse.SUPPRESS)
    p.add_argument("--compare_right_run", default="", help=argparse.SUPPRESS)
    p.add_argument("--details_csv", default="plots/pseudo_gt_eval_details.csv")
    p.add_argument("--summary_json", default="")
    p.add_argument(
        "--allow_truncate_points",
        action="store_true",
        help="Allow point-set truncation when GT/Pred joint or vertex counts differ (disabled by default).",
    )
    p.add_argument("--inspect_only", action="store_true", help="Print detected keys and exit")
    return p.parse_args()


def ensure_parent(path: str) -> None:
    if not path:
        return
    parent = Path(path).parent
    if str(parent) and str(parent) != ".":
        parent.mkdir(parents=True, exist_ok=True)


def _resolve_key(npz: np.lib.npyio.NpzFile, preferred: str, candidates: List[str]) -> Optional[str]:
    keys = set(npz.files)
    if preferred != AUTO_KEY:
        if preferred not in keys:
            raise KeyError(f"Key '{preferred}' not found. available={sorted(keys)}")
        return preferred
    for k in candidates:
        if k in keys:
            return k
    return None


def _to_frame_list(arr: np.ndarray) -> List[Any]:
    if arr.dtype == object:
        return [arr[i] for i in range(len(arr))]
    return [arr[i] for i in range(arr.shape[0])]


def _normalize_person_array(x: Any, field: str) -> Optional[np.ndarray]:
    if x is None:
        return None
    a = np.asarray(x)
    if a.size == 0:
        return np.zeros((0,), dtype=np.float32)

    if field == "transl":
        # [P,3] or [3]
        if a.ndim == 1 and a.shape[0] == 3:
            a = a[None, :]
    elif field == "joints2d":
        # [P,J,2] or [J,2]
        if a.ndim == 2 and a.shape[-1] == 2:
            a = a[None, ...]
    elif field == "joints3d":
        # [P,J,3] or [J,3]
        if a.ndim == 2 and a.shape[-1] == 3:
            a = a[None, ...]
    elif field == "verts":
        # [P,V,3] or [V,3]
        if a.ndim == 2 and a.shape[-1] == 3:
            a = a[None, ...]

    return a.astype(np.float32, copy=False)


def _frame_ids(npz: np.lib.npyio.NpzFile, key: Optional[str], n_fallback: int) -> List[int]:
    if key is None:
        return list(range(n_fallback))
    vals = np.asarray(npz[key]).reshape(-1)
    out: List[int] = []
    for v in vals:
        try:
            out.append(int(v))
        except Exception:
            out.append(len(out))
    return out


def load_sequence(
    path: str,
    frame_key: str,
    joints2d_key: str,
    joints3d_key: str,
    verts_key: str,
    transl_key: str,
) -> Tuple[Dict[int, Dict[str, Optional[np.ndarray]]], Dict[str, Optional[str]], int]:
    npz = np.load(path, allow_pickle=True)

    key_map = {
        "frame": _resolve_key(npz, frame_key, ["frame_ids", "frame_id", "frames", "frame"]),
        "joints2d": _resolve_key(npz, joints2d_key, ["joints2d", "pred_j2ds", "j2d"]),
        "joints3d": _resolve_key(npz, joints3d_key, ["joints3d", "pred_j3ds", "j3d"]),
        "verts": _resolve_key(npz, verts_key, ["verts", "pred_verts"]),
        "transl": _resolve_key(npz, transl_key, ["transl", "pred_transl", "translations"]),
    }

    n_candidates = []
    for k in ["joints2d", "joints3d", "verts", "transl"]:
        kk = key_map[k]
        if kk is not None:
            n_candidates.append(np.asarray(npz[kk]).shape[0])
    n = min(n_candidates) if n_candidates else 0

    if n == 0:
        return {}, key_map, 0

    frame_ids = _frame_ids(npz, key_map["frame"], n)
    if len(frame_ids) < n:
        frame_ids.extend(list(range(len(frame_ids), n)))
    frame_ids = frame_ids[:n]

    per_field: Dict[str, List[Optional[np.ndarray]]] = {}
    for field in ["joints2d", "joints3d", "verts", "transl"]:
        k = key_map[field]
        if k is None:
            per_field[field] = [None] * n
            continue
        raw = _to_frame_list(np.asarray(npz[k]))[:n]
        per_field[field] = [_normalize_person_array(x, field) for x in raw]

    frames: Dict[int, Dict[str, Optional[np.ndarray]]] = {}
    for i, fid in enumerate(frame_ids):
        if fid in frames:
            continue
        frames[fid] = {
            "joints2d": per_field["joints2d"][i],
            "joints3d": per_field["joints3d"][i],
            "verts": per_field["verts"][i],
            "transl": per_field["transl"][i],
        }
    return frames, key_map, n


def _num_persons(frame_data: Dict[str, Optional[np.ndarray]]) -> int:
    counts: List[int] = []
    for k in ["joints2d", "joints3d", "verts", "transl"]:
        arr = frame_data.get(k)
        if arr is None:
            continue
        if arr.ndim == 0:
            counts.append(0)
            continue
        counts.append(int(arr.shape[0]))
    if not counts:
        return 0
    # Some files contain empty joints2d but valid 3D/transl. Use the largest
    # available person count so empty optional fields do not force 0 persons.
    return int(max(counts))


def _greedy_min_dist_match(
    pred_feat: np.ndarray,
    gt_feat: np.ndarray,
    max_dist: Optional[float] = None,
) -> Tuple[Dict[int, int | str], int]:
    pred_n = int(pred_feat.shape[0])
    gt_n = int(gt_feat.shape[0])
    if gt_n == 0:
        return {}, pred_n
    if pred_n == 0:
        return {i: "miss" for i in range(gt_n)}, 0

    d = np.linalg.norm(pred_feat[:, None, :] - gt_feat[None, :, :], axis=-1)
    match: Dict[int, int | str] = {}
    used_pred = np.zeros(pred_n, dtype=bool)
    used_gt = np.zeros(gt_n, dtype=bool)

    while True:
        d_masked = d.copy()
        d_masked[used_pred, :] = np.inf
        d_masked[:, used_gt] = np.inf
        flat_idx = int(np.argmin(d_masked))
        min_val = float(np.min(d_masked))
        if not np.isfinite(min_val):
            break
        if max_dist is not None and min_val > max_dist:
            break
        p = flat_idx // gt_n
        g = flat_idx % gt_n
        used_pred[p] = True
        used_gt[g] = True
        match[g] = p

    for g in range(gt_n):
        if g not in match:
            match[g] = "miss"
    fp = int((~used_pred).sum())
    return match, fp


def _match_frame(
    gt: Dict[str, Optional[np.ndarray]],
    pred: Dict[str, Optional[np.ndarray]],
    mode: str,
    iou_thresh: float,
    max_dist: Optional[float],
) -> Tuple[str, Dict[int, int | str], int]:
    gt_j2d = gt.get("joints2d")
    pd_j2d = pred.get("joints2d")
    gt_t = gt.get("transl")
    pd_t = pred.get("transl")
    gt_j3d = gt.get("joints3d")
    pd_j3d = pred.get("joints3d")

    def _has_persons(arr: Optional[np.ndarray]) -> bool:
        return arr is not None and arr.ndim >= 1 and int(arr.shape[0]) > 0

    chosen = mode
    if mode == "auto":
        if _has_persons(gt_j2d) and _has_persons(pd_j2d):
            chosen = "joints2d"
        elif _has_persons(gt_t) and _has_persons(pd_t):
            chosen = "transl"
        elif _has_persons(gt_j3d) and _has_persons(pd_j3d):
            chosen = "pelvis"
        else:
            chosen = "none"

    if chosen == "joints2d":
        if gt_j2d is None or pd_j2d is None:
            return "none", {}, 0
        gt_n = int(gt_j2d.shape[0]) if gt_j2d.ndim >= 1 else 0
        pd_n = int(pd_j2d.shape[0]) if pd_j2d.ndim >= 1 else 0
        if gt_n == 0:
            return chosen, {}, pd_n
        if pd_n == 0:
            return chosen, {g: "miss" for g in range(gt_n)}, 0
        greedy = match_2d_greedy(pd_j2d, gt_j2d, iou_thresh=iou_thresh)
        md, fp = get_matching_dict(greedy)
        match = {}
        for g in range(gt_n):
            v = md.get(str(g), "miss")
            if isinstance(v, (int, np.integer)):
                match[g] = int(v)
            elif str(v).isdigit():
                match[g] = int(v)
            else:
                match[g] = "miss"
        return chosen, match, int(fp)

    if chosen == "transl":
        if gt_t is None or pd_t is None:
            return "none", {}, 0
        return chosen, *_wrap_match_tuple(_greedy_min_dist_match(pd_t, gt_t, max_dist=max_dist))

    if chosen == "pelvis":
        if gt_j3d is None or pd_j3d is None:
            return "none", {}, 0
        gt_p = gt_j3d[:, 0, :]
        pd_p = pd_j3d[:, 0, :]
        return chosen, *_wrap_match_tuple(_greedy_min_dist_match(pd_p, gt_p, max_dist=max_dist))

    return "none", {}, 0


def _wrap_match_tuple(v: Tuple[Dict[int, int | str], int]) -> Tuple[Dict[int, int | str], int]:
    return v


def _safe_mean(vals: List[float]) -> float:
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    if not Path(path).exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _write_csv_rows(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    ensure_parent(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(out)


def _append_summary_row(path: str, row: Dict[str, Any]) -> None:
    existing = _read_csv_rows(path)
    if not existing:
        _write_csv_rows(path, [row], list(row.keys()))
        return

    old_fields = list(existing[0].keys())
    new_fields = list(row.keys())
    if old_fields == new_fields:
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=old_fields)
            writer.writerow({k: row.get(k, "") for k in old_fields})
        return

    merged_fields = old_fields + [k for k in new_fields if k not in old_fields]
    merged_rows: List[Dict[str, Any]] = [{k: r.get(k, "") for k in merged_fields} for r in existing]
    merged_rows.append({k: row.get(k, "") for k in merged_fields})
    _write_csv_rows(path, merged_rows, merged_fields)


def _find_compare_rows(
    rows: List[Dict[str, str]], base_run: str, target_run: str
) -> Tuple[Optional[Dict[str, str]], Optional[Dict[str, str]]]:
    if not rows:
        return None, None

    if base_run and target_run:
        base = None
        target = None
        for r in rows:
            if r.get("run_name", "") == base_run:
                base = r
            if r.get("run_name", "") == target_run:
                target = r
        return base, target

    if len(rows) < 2:
        return None, None
    return rows[-2], rows[-1]


def _is_fp16_run(run_name: str) -> bool:
    s = (run_name or "").lower()
    return "fp16" in s


def _is_int8_run(run_name: str) -> bool:
    s = (run_name or "").lower()
    return "int8" in s


def _build_compare_row(base: Dict[str, str], target: Dict[str, str]) -> Dict[str, Any]:
    metrics = [
        "precision",
        "recall",
        "f1",
        "detection_rate",
        "Pseudo_MPJPE_mm",
        "Pseudo_PA_MPJPE_mm",
        "Pseudo_PVE_mm",
        "Pseudo_PA_PVE_mm",
        "tp_instances",
        "fn_instances",
        "fp_instances",
    ]

    base_name = base.get("run_name", "")
    target_name = target.get("run_name", "")
    base_is_fp16 = _is_fp16_run(base_name)
    target_is_fp16 = _is_fp16_run(target_name)
    base_is_int8 = _is_int8_run(base_name)
    target_is_int8 = _is_int8_run(target_name)

    # Prefer explicit fp16/int8 mapping when available.
    if base_is_fp16 and target_is_int8:
        fp16_row, int8_row = base, target
    elif base_is_int8 and target_is_fp16:
        fp16_row, int8_row = target, base
    else:
        # Fallback to argument order if run_name does not include fp16/int8.
        fp16_row, int8_row = base, target

    out: Dict[str, Any] = {
        "fp16_run": fp16_row.get("run_name", ""),
        "int8_run": int8_row.get("run_name", ""),
        "fp16_pred_npz": fp16_row.get("pred_npz", ""),
        "int8_pred_npz": int8_row.get("pred_npz", ""),
    }
    for m in metrics:
        fp16_v = _to_float(fp16_row.get(m, "nan"))
        int8_v = _to_float(int8_row.get(m, "nan"))
        out[f"fp16_{m}"] = fp16_v
        out[f"int8_{m}"] = int8_v
        out[f"delta_int8_minus_fp16_{m}"] = int8_v - fp16_v
    return out


def _load_body_verts_ind() -> np.ndarray:
    from configs.paths import smpl_model_path

    path = Path(smpl_model_path) / "smpl" / "body_verts_smpl.npy"
    if not path.exists():
        raise FileNotFoundError(f"body vertex index file not found: {path}")
    return np.asarray(np.load(path), dtype=np.int64).reshape(-1)


def _prepare_eval_pair_3dpw(
    pred_j3d: np.ndarray,
    gt_j3d: np.ndarray,
    pred_v: np.ndarray,
    gt_v: np.ndarray,
    allow_truncate_points: bool,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, bool]]:
    # Mirror evaluate_3dpw: pelvis align then H36M_EVAL_JOINTS, and pelvis-aligned full verts.
    if pred_j3d.ndim != 2 or gt_j3d.ndim != 2:
        return None
    if pred_j3d.shape[1] != 3 or gt_j3d.shape[1] != 3:
        return None
    if pred_v.ndim != 2 or gt_v.ndim != 2:
        return None
    if pred_v.shape[1] != 3 or gt_v.shape[1] != 3:
        return None

    pred_j = pred_j3d[:24].copy()
    gt_j = gt_j3d[:24].copy()
    if pred_j.shape[0] <= max(H36M_EVAL_JOINTS) or gt_j.shape[0] <= max(H36M_EVAL_JOINTS):
        return None

    gt_pelvis = gt_j[[0], :].copy()
    pred_pelvis = pred_j[[0], :].copy()
    gt_j_eval = (gt_j - gt_pelvis)[H36M_EVAL_JOINTS, :].copy()
    pred_j_eval = (pred_j - pred_pelvis)[H36M_EVAL_JOINTS, :].copy()

    truncated_j = bool(pred_j3d.shape[0] != gt_j3d.shape[0])
    truncated_v = False
    pred_v_eval = pred_v
    gt_v_eval = gt_v
    if pred_v.shape[0] != gt_v.shape[0]:
        if not allow_truncate_points:
            return None
        n = min(int(pred_v.shape[0]), int(gt_v.shape[0]))
        if n < 3:
            return None
        truncated_v = True
        pred_v_eval = pred_v[:n]
        gt_v_eval = gt_v[:n]

    pred_v_eval = (pred_v_eval - pred_pelvis).copy()
    gt_v_eval = (gt_v_eval - gt_pelvis).copy()
    return pred_j_eval, gt_j_eval, pred_v_eval, gt_v_eval, truncated_j, truncated_v


def _prepare_eval_pair_agora(
    pred_j3d: np.ndarray,
    gt_j3d: np.ndarray,
    pred_v: np.ndarray,
    gt_v: np.ndarray,
    body_verts_ind: np.ndarray,
    allow_truncate_points: bool,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, bool]]:
    # Mirror evaluate_agora: select_and_align on 24 joints + body verts subset.
    if pred_j3d.ndim != 2 or gt_j3d.ndim != 2:
        return None
    if pred_j3d.shape[1] != 3 or gt_j3d.shape[1] != 3:
        return None
    if pred_v.ndim != 2 or gt_v.ndim != 2:
        return None
    if pred_v.shape[1] != 3 or gt_v.shape[1] != 3:
        return None
    if int(pred_v.shape[0]) <= int(np.max(body_verts_ind)) or int(gt_v.shape[0]) <= int(np.max(body_verts_ind)):
        return None

    pred_j = pred_j3d[:24].copy()
    gt_j = gt_j3d[:24].copy()
    if pred_j.shape[0] < 24 or gt_j.shape[0] < 24:
        return None

    pred_j_eval, pred_v_eval = select_and_align(pred_j, pred_v, body_verts_ind)
    gt_j_eval, gt_v_eval = select_and_align(gt_j, gt_v, body_verts_ind)

    truncated_j = bool(pred_j3d.shape[0] != gt_j3d.shape[0])
    truncated_v = bool(pred_v.shape[0] != gt_v.shape[0])
    if truncated_v and allow_truncate_points:
        n = min(int(pred_v_eval.shape[0]), int(gt_v_eval.shape[0]))
        if n < 3:
            return None
        pred_v_eval = pred_v_eval[:n]
        gt_v_eval = gt_v_eval[:n]

    return pred_j_eval, gt_j_eval, pred_v_eval, gt_v_eval, truncated_j, truncated_v


def run_eval(args: argparse.Namespace) -> Dict[str, float]:
    gt_frames, gt_keys, gt_n = load_sequence(
        args.gt_npz,
        frame_key=args.gt_frame_key,
        joints2d_key=args.gt_joints2d_key,
        joints3d_key=args.gt_joints3d_key,
        verts_key=args.gt_verts_key,
        transl_key=args.gt_transl_key,
    )
    pred_frames, pred_keys, pred_n = load_sequence(
        args.pred_npz,
        frame_key=args.pred_frame_key,
        joints2d_key=args.pred_joints2d_key,
        joints3d_key=args.pred_joints3d_key,
        verts_key=args.pred_verts_key,
        transl_key=args.pred_transl_key,
    )

    print("[INFO] GT keys:", gt_keys)
    print("[INFO] Pred keys:", pred_keys)
    print(f"[INFO] GT frames loaded: {len(gt_frames)} (raw_n={gt_n})")
    print(f"[INFO] Pred frames loaded: {len(pred_frames)} (raw_n={pred_n})")

    if args.match_mode == "joints2d":
        pred_has_nonempty_j2d = False
        for frame in pred_frames.values():
            j2d = frame.get("joints2d")
            if j2d is not None and j2d.ndim >= 1 and int(j2d.shape[0]) > 0:
                pred_has_nonempty_j2d = True
                break
        if not pred_has_nonempty_j2d:
            print("[WARN] match_mode=joints2d but prediction joints2d is empty for all frames.")
            print("[WARN] Use --match_mode transl (or regenerate pred npz with valid joints2d).")

    if args.inspect_only:
        return {}

    common_ids = sorted(set(gt_frames.keys()) & set(pred_frames.keys()))
    if not common_ids:
        raise RuntimeError("No common frame ids between GT and Pred.")

    total_gt = 0
    total_miss = 0
    total_fp = 0

    mpjpe_vals: List[float] = []
    pa_mpjpe_vals: List[float] = []
    pve_vals: List[float] = []
    pa_pve_vals: List[float] = []
    j3d_truncated_pairs = 0
    verts_truncated_pairs = 0
    j3d_skipped_pairs = 0
    verts_skipped_pairs = 0
    eval_style_counts = {"3dpw": 0, "agora": 0}
    body_verts_ind: Optional[np.ndarray] = None
    if args.eval_style in ("auto", "agora"):
        body_verts_ind = _load_body_verts_ind()

    ensure_parent(args.details_csv)
    with open(args.details_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "frame_id",
                "match_mode_used",
                "gt_persons",
                "pred_persons",
                "matched",
                "misses",
                "false_positives",
                "pseudo_mpjpe_mm",
                "pseudo_pa_mpjpe_mm",
                "pseudo_pve_mm",
                "pseudo_pa_pve_mm",
            ]
        )

        for fid in common_ids:
            gt = gt_frames[fid]
            pd = pred_frames[fid]
            gt_num = _num_persons(gt)
            pd_num = _num_persons(pd)

            max_dist = args.transl_max_dist if args.transl_max_dist > 0 else None
            mode_used, match, fp = _match_frame(
                gt,
                pd,
                mode=args.match_mode,
                iou_thresh=args.iou_thresh,
                max_dist=max_dist,
            )

            misses = 0
            matched = 0
            f_mpjpe: List[float] = []
            f_pa_mpjpe: List[float] = []
            f_pve: List[float] = []
            f_pa_pve: List[float] = []

            gt_j3d = gt.get("joints3d")
            pd_j3d = pd.get("joints3d")
            gt_v = gt.get("verts")
            pd_v = pd.get("verts")

            for g in range(gt_num):
                pidx = match.get(g, "miss")
                if pidx == "miss" or (isinstance(pidx, int) and pidx < 0):
                    misses += 1
                    continue
                if not isinstance(pidx, int):
                    misses += 1
                    continue
                if pidx >= pd_num:
                    misses += 1
                    continue

                matched += 1
                if gt_j3d is None or pd_j3d is None or gt_v is None or pd_v is None:
                    j3d_skipped_pairs += 1
                    verts_skipped_pairs += 1
                    continue
                if g >= gt_j3d.shape[0] or pidx >= pd_j3d.shape[0] or g >= gt_v.shape[0] or pidx >= pd_v.shape[0]:
                    j3d_skipped_pairs += 1
                    verts_skipped_pairs += 1
                    continue

                gt_j3d_pair = gt_j3d[g]
                pd_j3d_pair = pd_j3d[pidx]
                gt_v_pair = gt_v[g]
                pd_v_pair = pd_v[pidx]

                # Align with eval_funcs.py styles:
                # - 3dpw: pelvis + H36M_EVAL_JOINTS + full verts
                # - agora: select_and_align(body_verts_ind) for joints/verts
                styles_try: List[str]
                if args.eval_style == "auto":
                    if (
                        pd_v_pair.ndim == 2
                        and gt_v_pair.ndim == 2
                        and pd_v_pair.shape[0] != gt_v_pair.shape[0]
                    ):
                        styles_try = ["agora", "3dpw"]
                    else:
                        styles_try = ["3dpw", "agora"]
                else:
                    styles_try = [args.eval_style]

                aligned = None
                used_style = "none"
                for style in styles_try:
                    if style == "3dpw":
                        aligned = _prepare_eval_pair_3dpw(
                            pd_j3d_pair,
                            gt_j3d_pair,
                            pd_v_pair,
                            gt_v_pair,
                            allow_truncate_points=args.allow_truncate_points,
                        )
                    elif style == "agora" and body_verts_ind is not None:
                        aligned = _prepare_eval_pair_agora(
                            pd_j3d_pair,
                            gt_j3d_pair,
                            pd_v_pair,
                            gt_v_pair,
                            body_verts_ind=body_verts_ind,
                            allow_truncate_points=args.allow_truncate_points,
                        )
                    if aligned is not None:
                        used_style = style
                        break

                if aligned is None:
                    j3d_skipped_pairs += 1
                    verts_skipped_pairs += 1
                    continue

                pd_j3d_cmp, gt_j3d_cmp, pd_v_cmp, gt_v_cmp, truncated_j, truncated_v = aligned
                if truncated_j:
                    j3d_truncated_pairs += 1
                if truncated_v:
                    verts_truncated_pairs += 1
                if used_style in eval_style_counts:
                    eval_style_counts[used_style] += 1

                e, pa = cal_3d_position_error(pd_j3d_cmp, gt_j3d_cmp)
                f_mpjpe.append(float(e))
                f_pa_mpjpe.append(float(pa))
                mpjpe_vals.append(float(e))
                pa_mpjpe_vals.append(float(pa))

                e, pa = cal_3d_position_error(pd_v_cmp, gt_v_cmp)
                f_pve.append(float(e))
                f_pa_pve.append(float(pa))
                pve_vals.append(float(e))
                pa_pve_vals.append(float(pa))

            total_gt += gt_num
            total_miss += misses
            total_fp += fp

            w.writerow(
                [
                    fid,
                    mode_used,
                    gt_num,
                    pd_num,
                    matched,
                    misses,
                    fp,
                    round(_safe_mean(f_mpjpe), 4) if f_mpjpe else "",
                    round(_safe_mean(f_pa_mpjpe), 4) if f_pa_mpjpe else "",
                    round(_safe_mean(f_pve), 4) if f_pve else "",
                    round(_safe_mean(f_pa_pve), 4) if f_pa_pve else "",
                ]
            )

    precision, recall, f1 = compute_prf1(total_gt, total_miss, total_fp)
    run_name = args.run_name.strip() if args.run_name.strip() else Path(args.pred_npz).stem
    summary = {
        "run_name": run_name,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "gt_npz": args.gt_npz,
        "pred_npz": args.pred_npz,
        "match_mode": args.match_mode,
        "eval_style": args.eval_style,
        "iou_thresh": args.iou_thresh,
        "transl_max_dist": args.transl_max_dist,
        "frames_gt": int(len(gt_frames)),
        "frames_pred": int(len(pred_frames)),
        "frames_common": int(len(common_ids)),
        "gt_instances": int(total_gt),
        "tp_instances": int(total_gt - total_miss),
        "fn_instances": int(total_miss),
        "fp_instances": int(total_fp),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "detection_rate": float(recall),
        "Pseudo_MPJPE_mm": round(_safe_mean(mpjpe_vals), 4),
        "Pseudo_PA_MPJPE_mm": round(_safe_mean(pa_mpjpe_vals), 4),
        "Pseudo_PVE_mm": round(_safe_mean(pve_vals), 4),
        "Pseudo_PA_PVE_mm": round(_safe_mean(pa_pve_vals), 4),
        "j3d_truncated_pairs": int(j3d_truncated_pairs),
        "verts_truncated_pairs": int(verts_truncated_pairs),
        "j3d_skipped_pairs": int(j3d_skipped_pairs),
        "verts_skipped_pairs": int(verts_skipped_pairs),
        "eval_style_3dpw_pairs": int(eval_style_counts["3dpw"]),
        "eval_style_agora_pairs": int(eval_style_counts["agora"]),
    }
    return summary


def main() -> int:
    args = parse_args()
    ensure_parent(args.summary_csv)
    if args.summary_json:
        ensure_parent(args.summary_json)
    if args.compare_csv:
        ensure_parent(args.compare_csv)

    summary = run_eval(args)
    if args.inspect_only:
        print("[DONE] inspect_only")
        return 0

    print("\n=== Pseudo-GT Evaluation Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if args.append_summary:
        _append_summary_row(args.summary_csv, summary)
    else:
        _write_csv_rows(args.summary_csv, [summary], list(summary.keys()))
    print(f"[DONE] Summary CSV: {args.summary_csv}")
    print(f"[DONE] Details CSV: {args.details_csv}")

    if args.summary_json:
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[DONE] Summary JSON: {args.summary_json}")

    if args.compare_csv:
        rows = _read_csv_rows(args.summary_csv)
        base_name = args.compare_base_run.strip() or args.compare_left_run.strip()
        target_name = args.compare_target_run.strip() or args.compare_right_run.strip()
        base, target = _find_compare_rows(rows, base_name, target_name)
        if base is None or target is None:
            print(
                "[WARN] Comparison not generated: need at least two summary rows "
                "(or both --compare_base_run and --compare_target_run found)."
            )
        else:
            cmp_row = _build_compare_row(base, target)
            _write_csv_rows(args.compare_csv, [cmp_row], list(cmp_row.keys()))
            print(f"[DONE] Compare CSV: {args.compare_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

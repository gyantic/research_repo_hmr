#!/usr/bin/env python3
"""FastSAM3DBody 3DPW accuracy under the H36M14 (SAT-HMR-aligned) protocol.

Computes MPJPE / PA-MPJPE / PVE / MPVAE / MVE on the H36M14 joint set.
MHR->SMPL is the barycentric mapping (no MLP). GT is gendered 3DPW SMPL
transformed to the camera frame via cam_poses.

  - MPJPE : H36M14 joints, root (H36M joint 0) subtracted, mean L2 [mm]
  - PA-MPJPE: H36M14 joints, similarity Procrustes aligned, mean L2 [mm]
  - PVE / MVE : all SMPL vertices, pelvis(H36M joint 0) subtracted, mean L2 [mm]
               (PVE and MVE are the same per-vertex error; both reported)
  - MPVAE : all SMPL vertices, similarity Procrustes aligned, mean L2 [mm]
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import smplx
import torch
import trimesh
from tqdm import tqdm

H36M14 = [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--samples_dir", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--data_dir", default="/mnt/e/sathmr_data/3dpw")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--smpl_model", default="/home/kyuta/ROMP/model_data/parameters/SMPL_NEUTRAL.pkl")
    p.add_argument("--smpl_model_dir", default="/home/kyuta/ROMP/model_data/parameters")
    p.add_argument("--mapping", default="/home/kyuta/Fast-SAM-3D-Body/mhr2smpl/data/mhr2smpl_mapping.npz")
    p.add_argument("--mhr_face_mesh", default="/home/kyuta/MHR/tools/mhr_smpl_conversion/assets/mhr_face_mask.ply")
    p.add_argument("--h36m_regressor", default="/home/kyuta/SAT-HMR/weights/smpl_data/smpl/J_regressor_h36m_correct.npy")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def rigid_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    ca, cb = a.mean(0), b.mean(0)
    h = (a - ca).T @ (b - cb) / n
    u, s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        s[-1] = -s[-1]
        vt[2] = -vt[2]
        r = vt.T @ u.T
    c = s.sum() / np.var(a, axis=0).sum()
    t = cb - c * r @ ca
    return (c * r @ a.T).T + t


def mean_l2_mm(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, axis=-1).mean() * 1000.0)


def summarize(v: list[float]) -> dict[str, float | int]:
    a = np.asarray(v, dtype=np.float64)
    q1, q3 = np.percentile(a, [25, 75])
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))            # median absolute deviation
    return {"mean": float(a.mean()), "median": med, "mad": mad,
            "mad_scaled": mad * 1.4826,                # normal-consistent estimator
            "std": float(a.std()), "iqr": float(q3 - q1), "count": int(a.size)}


def make_smpl(path: str, gender: str, dev) -> smplx.SMPL:
    m = smplx.SMPL(model_path=path, gender=gender).to(dev)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def load_cam_pose(cache, data_dir: Path, seq: str, frame: int) -> np.ndarray:
    if seq not in cache:
        with (data_dir / "sequenceFiles" / "test" / f"{seq}.pkl").open("rb") as fp:
            cache[seq] = pickle.load(fp, encoding="latin1")
    return cache[seq]["cam_poses"][frame].T.astype(np.float32)


def apply_camera(pts: np.ndarray, cam_t: np.ndarray) -> np.ndarray:
    hom = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    return (hom @ cam_t)[:, :3]


def main() -> None:
    args = parse_args()
    dev = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    files = sorted(Path(args.samples_dir).glob("sample_*.npz"))
    if args.max_samples > 0:
        files = files[: args.max_samples]
    if not files:
        raise SystemExit(f"No sample_*.npz in {args.samples_dir}")

    mp = np.load(args.mapping)
    tri = torch.from_numpy(mp["triangle_ids"].astype(np.int64)).to(dev)
    bary = torch.from_numpy(mp["baryc_coords"].astype(np.float32)).to(dev)
    faces = torch.from_numpy(
        np.asarray(trimesh.load(args.mhr_face_mesh, process=False).faces, dtype=np.int64)
    ).to(dev)

    neutral = make_smpl(args.smpl_model, "neutral", dev)
    gmodels = {"neutral": neutral, "male": make_smpl(args.smpl_model_dir, "male", dev),
               "female": make_smpl(args.smpl_model_dir, "female", dev)}
    h36m = np.load(args.h36m_regressor).astype(np.float32)

    cache: dict = {}
    metrics: dict[str, list[float]] = {k: [] for k in ("MPJPE", "PA-MPJPE", "PVE", "MPVAE", "MVE")}
    data_dir = Path(args.data_dir)

    for s in tqdm(range(0, len(files), args.batch_size), desc="H36M14 accuracy"):
        fs = files[s:s + args.batch_size]
        loaded = [dict(np.load(p, allow_pickle=True)) for p in fs]

        # pred SMPL verts via barycentric MHR->SMPL (no MLP)
        mhr = torch.from_numpy(np.stack([
            d["pred_vertices"].astype(np.float32) + d["pred_cam_t"].astype(np.float32)[None, :] for d in loaded
        ])).float().to(dev)
        with torch.no_grad():
            pred_v = (mhr[:, faces[tri], :] * bary[None, :, :, None]).sum(2).cpu().numpy()

        # GT gendered SMPL verts in camera frame
        gt_v = [None] * len(loaded)
        for gk in ("male", "female", "neutral"):
            idx = [i for i, d in enumerate(loaded)
                   if ("female" if "f" in str(d["gt_gender"]).lower() else ("male" if "m" in str(d["gt_gender"]).lower() else "neutral")) == gk]
            if not idx:
                continue
            poses = torch.from_numpy(np.stack([loaded[i]["gt_poses"].astype(np.float32) for i in idx])).to(dev)
            betas = torch.from_numpy(np.stack([loaded[i]["gt_betas"].astype(np.float32) for i in idx])).to(dev)
            trans = torch.from_numpy(np.stack([loaded[i]["gt_trans"].astype(np.float32) for i in idx])).to(dev)
            with torch.no_grad():
                o = gmodels[gk](betas=betas, body_pose=poses[:, 3:], global_orient=poses[:, :3], transl=trans)
            vv = o.vertices.cpu().numpy()
            for k, i in enumerate(idx):
                cam_t = load_cam_pose(cache, data_dir, str(loaded[i]["sequence"]), int(loaded[i]["frame_idx"]))
                gt_v[i] = apply_camera(vv[k], cam_t)
        gt_v = np.stack(gt_v).astype(np.float32)

        pred_j = np.einsum("jv,bvd->bjd", h36m, pred_v)
        gt_j = np.einsum("jv,bvd->bjd", h36m, gt_v)
        for i in range(len(loaded)):
            pj, gj = pred_j[i], gt_j[i]
            pv, gv = pred_v[i], gt_v[i]
            # joints, root = H36M joint 0
            pj14 = (pj - pj[[0]])[H36M14]
            gj14 = (gj - gj[[0]])[H36M14]
            metrics["MPJPE"].append(mean_l2_mm(pj14, gj14))
            metrics["PA-MPJPE"].append(mean_l2_mm(rigid_align(pj14, gj14), gj14))
            # vertices, root = H36M joint 0 (pelvis)
            pvr = pv - pj[[0]]
            gvr = gv - gj[[0]]
            ve = mean_l2_mm(pvr, gvr)
            metrics["PVE"].append(ve)
            metrics["MVE"].append(ve)
            metrics["MPVAE"].append(mean_l2_mm(rigid_align(pv, gv), gv))

    out = {
        "num_samples": len(files),
        "samples_dir": str(args.samples_dir),
        "joint_set": "H36M14",
        "conversion": "barycentric MHR->SMPL (no MLP)",
        "metrics_mm": {k: summarize(v) for k, v in metrics.items()},
    }
    Path(args.output_json).write_text(json.dumps(out, indent=2))
    with open(args.output_csv, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["metric", "mean", "median", "mad", "mad_scaled", "std", "iqr", "count"])
        for k in ("MPJPE", "PA-MPJPE", "PVE", "MPVAE", "MVE"):
            st = out["metrics_mm"][k]
            w.writerow([k, st["mean"], st["median"], st["mad"], st["mad_scaled"], st["std"], st["iqr"], st["count"]])
    print(json.dumps(out["metrics_mm"], indent=2))


if __name__ == "__main__":
    main()

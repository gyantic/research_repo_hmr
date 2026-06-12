#!/usr/bin/env python3
"""FastSAM3DBody 3DPW paper-reproduction metric path.

The FastSAM3DBody paper's 3DPW PA values match 3DPW's stored 14 body joints
(`gt_joints_3d[:14]`) rather than H36M14 joints regressed from SMPL vertices.
MPJPE and PVE are kept on the camera-frame SMPL path used by the earlier
diagnostic because those reproduce the reported raw errors.
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


H36M_EVAL_JOINTS = [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16]
THREEDPW_PA14 = list(range(14))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--vertices_npz", default="", help="Optional fitted SMPL vertices npz with pred_smpl_vertices")
    parser.add_argument("--data_dir", default="/home/kyuta/Fast-SAM-3D-Body/data/3DPW")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--smpl_model", default="/home/kyuta/ROMP/model_data/parameters/SMPL_NEUTRAL.pkl")
    parser.add_argument("--smpl_model_dir", default="/home/kyuta/ROMP/model_data/parameters")
    parser.add_argument("--mapping", default="/home/kyuta/Fast-SAM-3D-Body/mhr2smpl/data/mhr2smpl_mapping.npz")
    parser.add_argument("--mhr_face_mesh", default="/home/kyuta/MHR/tools/mhr_smpl_conversion/assets/mhr_face_mask.ply")
    parser.add_argument("--h36m_regressor", default="/home/kyuta/SAT-HMR/weights/smpl_data/smpl/J_regressor_h36m_correct.npy")
    parser.add_argument(
        "--pa_protocol",
        choices=["paper_3dpw14", "h36m14_global_orient"],
        default="paper_3dpw14",
        help="PA-MPJPE joint definition: paper 3DPW stored 14 joints, or SAT-HMR-style H36M14 from global-orient SMPL GT.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def procrustes_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_t = source.T
    target_t = target.T
    mu_source = source_t.mean(axis=1, keepdims=True)
    mu_target = target_t.mean(axis=1, keepdims=True)
    source0 = source_t - mu_source
    target0 = target_t - mu_target
    var_source = np.sum(source0**2)
    if var_source <= 1e-12:
        return source.copy()
    k = source0 @ target0.T
    u, _, vh = np.linalg.svd(k)
    v = vh.T
    z = np.eye(3)
    z[-1, -1] *= np.sign(np.linalg.det(u @ v.T))
    rot = v @ z @ u.T
    scale = np.trace(rot @ k) / var_source
    trans = mu_target - scale * (rot @ mu_source)
    return (scale * rot @ source_t + trans).T


def mean_l2_mm(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.linalg.norm(pred - gt, axis=-1).mean() * 1000.0)


def summarize(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    q1, q3 = np.percentile(arr, [25, 75])
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "variance": float(arr.var()),
        "std": float(arr.std()),
        "iqr": float(q3 - q1),
        "count": int(arr.size),
    }


def make_smpl(model_path: str, gender: str, device: torch.device) -> smplx.SMPL:
    model = smplx.SMPL(model_path=model_path, gender=gender).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_cam_pose(seq_cache: dict[str, dict], data_dir: Path, sequence: str, frame_idx: int) -> np.ndarray:
    if sequence not in seq_cache:
        with (data_dir / "sequenceFiles" / "test" / f"{sequence}.pkl").open("rb") as fp:
            seq_cache[sequence] = pickle.load(fp, encoding="latin1")
    return seq_cache[sequence]["cam_poses"][frame_idx].T.astype(np.float32)


def apply_camera(points: np.ndarray, cam_pose_t: np.ndarray) -> np.ndarray:
    hom = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    return (hom @ cam_pose_t)[:, :3]


def load_pred_vertices(
    args: argparse.Namespace,
    sample_files: list[Path],
    device: torch.device,
) -> np.ndarray | None:
    if args.vertices_npz:
        pred_v = np.load(args.vertices_npz)["pred_smpl_vertices"].astype(np.float32)
        return pred_v[: len(sample_files)]

    mapping = np.load(args.mapping)
    triangle_ids = torch.from_numpy(mapping["triangle_ids"].astype(np.int64)).to(device)
    barycentric = torch.from_numpy(mapping["baryc_coords"].astype(np.float32)).to(device)
    mhr_faces = torch.from_numpy(
        np.asarray(trimesh.load(args.mhr_face_mesh, process=False).faces, dtype=np.int64)
    ).to(device)

    pred_vertices = []
    for start in tqdm(range(0, len(sample_files), args.batch_size), desc="Barycentric pred"):
        files = sample_files[start : start + args.batch_size]
        pred_mhr = []
        for path in files:
            data = np.load(path, allow_pickle=True)
            pred_mhr.append(data["pred_vertices"].astype(np.float32) + data["pred_cam_t"].astype(np.float32)[None, :])
        pred_mhr_t = torch.from_numpy(np.stack(pred_mhr)).float().to(device)
        with torch.no_grad():
            pred_v = (
                pred_mhr_t[:, mhr_faces[triangle_ids], :]
                * barycentric[None, :, :, None]
            ).sum(dim=2).detach().cpu().numpy()
        pred_vertices.append(pred_v)
    return np.concatenate(pred_vertices, axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    samples_dir = Path(args.samples_dir)
    sample_files = sorted(samples_dir.glob("sample_*.npz"))
    if not sample_files:
        raise SystemExit(f"No sample_*.npz files found in {samples_dir}")

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    neutral_model = make_smpl(args.smpl_model, "neutral", device)
    gender_models = {
        "neutral": neutral_model,
        "male": make_smpl(args.smpl_model_dir, "male", device),
        "female": make_smpl(args.smpl_model_dir, "female", device),
    }
    j_regressor = neutral_model.J_regressor.detach().cpu().numpy().astype(np.float32)
    h36m_regressor = np.load(args.h36m_regressor).astype(np.float32)
    pred_v_all = load_pred_vertices(args, sample_files, device)

    seq_cache: dict[str, dict] = {}
    metrics = {"MPJPE": [], "PA-MPJPE": [], "PVE": [], "MPVAE": []}

    for start in tqdm(range(0, len(sample_files), args.batch_size), desc="Paper repro eval"):
        files = sample_files[start : start + args.batch_size]
        loaded = [np.load(path, allow_pickle=True) for path in files]
        pred_v = pred_v_all[start : start + len(files)]

        gt_v_list: list[np.ndarray | None] = [None] * len(loaded)
        for gender_key in ("male", "female", "neutral"):
            idxs, poses, betas, trans = [], [], [], []
            for idx, data in enumerate(loaded):
                gender = str(data["gt_gender"]).lower()
                sample_gender = "female" if "f" in gender else ("male" if "m" in gender else "neutral")
                if sample_gender != gender_key:
                    continue
                idxs.append(idx)
                poses.append(data["gt_poses"].astype(np.float32))
                betas.append(data["gt_betas"].astype(np.float32))
                trans.append(data["gt_trans"].astype(np.float32))
            if not idxs:
                continue
            poses_t = torch.from_numpy(np.stack(poses)).float().to(device)
            with torch.no_grad():
                out = gender_models[gender_key](
                    betas=torch.from_numpy(np.stack(betas)).float().to(device),
                    body_pose=poses_t[:, 3:],
                    global_orient=poses_t[:, :3],
                    transl=torch.from_numpy(np.stack(trans)).float().to(device),
                )
            verts = out.vertices.detach().cpu().numpy()
            for local_idx, sample_idx in enumerate(idxs):
                gt_v_list[sample_idx] = verts[local_idx]

        gt_v = []
        gt_pa14 = []
        for data, verts in zip(loaded, gt_v_list):
            if verts is None:
                raise RuntimeError("Internal gender batching error")
            cam_pose_t = load_cam_pose(seq_cache, data_dir, str(data["sequence"]), int(data["frame_idx"]))
            gt_v.append(apply_camera(verts, cam_pose_t))
            gt_pa14.append(apply_camera(data["gt_joints_3d"].astype(np.float32), cam_pose_t)[THREEDPW_PA14])
        gt_v = np.stack(gt_v).astype(np.float32)
        gt_pa14 = np.stack(gt_pa14).astype(np.float32)

        pred_h36m = np.einsum("jv,bvd->bjd", h36m_regressor, pred_v)[:, H36M_EVAL_JOINTS]
        gt_h36m = np.einsum("jv,bvd->bjd", h36m_regressor, gt_v)[:, H36M_EVAL_JOINTS]
        pred_j24 = np.einsum("jv,bvd->bjd", j_regressor, pred_v)
        gt_j24 = np.einsum("jv,bvd->bjd", j_regressor, gt_v)

        pred_pa14 = pred_j24[:, THREEDPW_PA14]
        for pred_j, gt_j, pred_vertices, gt_vertices, pred_root, gt_root, pred_pa_j, gt_pa_j in zip(
            pred_h36m,
            gt_h36m,
            pred_v,
            gt_v,
            pred_j24[:, 0],
            gt_j24[:, 0],
            pred_pa14,
            gt_pa14,
        ):
            pred_rooted = pred_j - pred_j[0:1]
            gt_rooted = gt_j - gt_j[0:1]
            metrics["MPJPE"].append(mean_l2_mm(pred_rooted, gt_rooted))
            if args.pa_protocol == "h36m14_global_orient":
                metrics["PA-MPJPE"].append(mean_l2_mm(procrustes_align(pred_rooted, gt_rooted), gt_rooted))
            else:
                metrics["PA-MPJPE"].append(mean_l2_mm(procrustes_align(pred_pa_j, gt_pa_j), gt_pa_j))

            pred_vertices_rooted = pred_vertices - pred_root[None, :]
            gt_vertices_rooted = gt_vertices - gt_root[None, :]
            metrics["PVE"].append(mean_l2_mm(pred_vertices_rooted, gt_vertices_rooted))
            metrics["MPVAE"].append(
                mean_l2_mm(procrustes_align(pred_vertices_rooted, gt_vertices_rooted), gt_vertices_rooted)
            )

    summary = {name: summarize(values) for name, values in metrics.items()}
    summary["num_samples"] = len(sample_files)
    summary["samples_dir"] = str(samples_dir)
    summary["vertices_npz"] = args.vertices_npz
    summary["pa_protocol"] = args.pa_protocol
    summary["metric_protocol"] = {
        "MPJPE": "H36M14 from SMPL vertices, root-relative, camera-frame GT",
        "PA-MPJPE": (
            "H36M14 from SMPL vertices, root-relative, camera-frame GT, Procrustes aligned"
            if args.pa_protocol == "h36m14_global_orient"
            else "SMPL J_regressor joints 0:14 vs 3DPW stored gt_joints_3d[0:14], Procrustes aligned"
        ),
        "PVE": "SMPL vertices, pelvis-root-relative, camera-frame GT",
        "MPVAE": "SMPL vertices, pelvis-root-relative, Procrustes aligned",
    }

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with output_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["metric", "mean", "median", "variance", "std", "iqr", "count"])
        for metric in ("MPJPE", "PA-MPJPE", "PVE", "MPVAE"):
            stats = summary[metric]
            writer.writerow([
                metric,
                stats["mean"],
                stats["median"],
                stats["variance"],
                stats["std"],
                stats["iqr"],
                stats["count"],
            ])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

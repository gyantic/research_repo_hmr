#!/usr/bin/env python3
"""Compare PA-MPJPE while changing only the final H36M joint subset."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import smplx
import torch
import trimesh
from tqdm import tqdm


JOINT_SETS = {
    "sathmr_h36m14": (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16),
    "fastsam3dbody_front14": tuple(range(14)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--data_dir", default="/mnt/e/sathmr_data/3dpw")
    parser.add_argument("--fastsam_dir", default="/home/kyuta/Fast-SAM-3D-Body")
    parser.add_argument(
        "--model_path",
        default="/home/kyuta/Fast-SAM-3D-Body/mhr2smpl/experiments/multiview_n30000_e500/best_model.pth",
    )
    parser.add_argument(
        "--sample_idx",
        default="/home/kyuta/Fast-SAM-3D-Body/mhr2smpl/experiments/multiview_n30000_e500/sample_idx.npy",
    )
    parser.add_argument("--mapping", default="/home/kyuta/Fast-SAM-3D-Body/mhr2smpl/data/mhr2smpl_mapping.npz")
    parser.add_argument("--mhr_face_mesh", default="/home/kyuta/MHR/tools/mhr_smpl_conversion/assets/mhr_face_mask.ply")
    parser.add_argument("--smpl_model", default="/home/kyuta/ROMP/model_data/parameters/SMPL_NEUTRAL.pkl")
    parser.add_argument("--smpl_model_dir", default="/home/kyuta/ROMP/model_data/parameters")
    parser.add_argument("--h36m_regressor", default="/home/kyuta/SAT-HMR/weights/smpl_data/smpl/J_regressor_h36m_correct.npy")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def rigid_transform_3d(a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    n, _ = a.shape
    centroid_a = np.mean(a, axis=0)
    centroid_b = np.mean(b, axis=0)
    h = np.dot(np.transpose(a - centroid_a), b - centroid_b) / n
    u, s, v = np.linalg.svd(h)
    r = np.dot(np.transpose(v), np.transpose(u))
    if np.linalg.det(r) < 0:
        s[-1] = -s[-1]
        v[2] = -v[2]
        r = np.dot(np.transpose(v), np.transpose(u))
    c = (1.0 / np.var(a, axis=0).sum()) * np.sum(s)
    t = -np.dot(c * r, np.transpose(centroid_a)) + np.transpose(centroid_b)
    return c, r, t


def rigid_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    c, r, t = rigid_transform_3d(a, b)
    return np.transpose(np.dot(c * r, np.transpose(a))) + t


def pa_mpjpe_mm(pred: np.ndarray, gt: np.ndarray) -> float:
    aligned = rigid_align(pred, gt)
    return float(np.sqrt(np.sum((aligned - gt) ** 2, axis=1)).mean() * 1000.0)


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


def build_mlp_features(
    loaded: list[dict[str, np.ndarray | str | int]],
    mhr_faces: torch.Tensor,
    triangle_ids: torch.Tensor,
    barycentric: torch.Tensor,
    sample_idx: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    pred_mhr = []
    for data in loaded:
        pred_raw = data["pred_vertices"].astype(np.float32).copy()
        pred_cam_t = data["pred_cam_t"].astype(np.float32).copy()
        pred_raw[:, [1, 2]] *= -1
        pred_cam_t[[1, 2]] *= -1
        pred_mhr.append(pred_raw + pred_cam_t[None, :])
    pred_mhr_t = torch.from_numpy(np.stack(pred_mhr)).float().to(device)
    with torch.no_grad():
        smpl_target = (
            pred_mhr_t[:, mhr_faces[triangle_ids], :]
            * barycentric[None, :, :, None]
        ).sum(dim=2)
        smpl_sub = smpl_target[:, sample_idx, :]
        smpl_sub = smpl_sub - smpl_sub.mean(dim=1, keepdim=True)
    return smpl_sub.reshape(len(loaded), -1)


def pad_body_pose(bp_21: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(bp_21.shape[0], 69, dtype=bp_21.dtype, device=bp_21.device)
    out[:, :63] = bp_21
    return out


def load_sample(path: Path) -> dict[str, np.ndarray | str | int]:
    with np.load(path, allow_pickle=True) as data:
        return {
            "pred_vertices": data["pred_vertices"].copy(),
            "pred_cam_t": data["pred_cam_t"].copy(),
            "gt_poses": data["gt_poses"].copy(),
            "gt_betas": data["gt_betas"].copy(),
            "gt_trans": data["gt_trans"].copy(),
            "gt_gender": str(data["gt_gender"]),
            "sequence": str(data["sequence"]),
            "frame_idx": int(data["frame_idx"]),
        }


def main() -> None:
    args = parse_args()
    samples_dir = Path(args.samples_dir)
    sample_files = sorted(samples_dir.glob("sample_*.npz"))
    if args.max_samples > 0:
        sample_files = sample_files[: args.max_samples]
    if not sample_files:
        raise SystemExit(f"No sample_*.npz files found in {samples_dir}")

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    mapping = np.load(args.mapping)
    triangle_ids = torch.from_numpy(mapping["triangle_ids"].astype(np.int64)).to(device)
    barycentric = torch.from_numpy(mapping["baryc_coords"].astype(np.float32)).to(device)
    mhr_faces = torch.from_numpy(
        np.asarray(trimesh.load(args.mhr_face_mesh, process=False).faces, dtype=np.int64)
    ).to(device)

    neutral_model = make_smpl(args.smpl_model, "neutral", device)
    gender_models = {
        "neutral": neutral_model,
        "male": make_smpl(args.smpl_model_dir, "male", device),
        "female": make_smpl(args.smpl_model_dir, "female", device),
    }
    h36m = np.load(args.h36m_regressor).astype(np.float32)
    seq_cache: dict[str, dict] = {}
    pa_values: dict[str, list[float]] = {name: [] for name in JOINT_SETS}

    for start in tqdm(range(0, len(sample_files), args.batch_size), desc="Joint-set PA-MPJPE A/B"):
        files = sample_files[start : start + args.batch_size]
        loaded = [load_sample(path) for path in files]
        # Barycentric MHR->SMPL conversion (no MLP)
        pred_mhr = np.stack([
            data["pred_vertices"].astype(np.float32)
            + data["pred_cam_t"].astype(np.float32)[None, :]
            for data in loaded
        ])
        pred_mhr_t = torch.from_numpy(pred_mhr).float().to(device)
        with torch.no_grad():
            pred_v = (
                pred_mhr_t[:, mhr_faces[triangle_ids], :]
                * barycentric[None, :, :, None]
            ).sum(dim=2).detach().cpu().numpy()

        gt_camera: list[np.ndarray | None] = [None] * len(loaded)
        for gender_key in ("male", "female", "neutral"):
            idxs, poses, betas, trans = [], [], [], []
            for local_idx, data in enumerate(loaded):
                gender = str(data["gt_gender"]).lower()
                sample_gender = "female" if "f" in gender else ("male" if "m" in gender else "neutral")
                if sample_gender != gender_key:
                    continue
                idxs.append(local_idx)
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
            for k, local_idx in enumerate(idxs):
                data = loaded[local_idx]
                cam_pose_t = load_cam_pose(seq_cache, Path(args.data_dir), str(data["sequence"]), int(data["frame_idx"]))
                gt_camera[local_idx] = apply_camera(verts[k], cam_pose_t)
        gt_v = np.stack([v for v in gt_camera if v is not None]).astype(np.float32)
        if len(gt_v) != len(files):
            raise RuntimeError("Internal GT camera batching error")

        pred_h36m_all = np.einsum("jv,bvd->bjd", h36m, pred_v)
        gt_h36m_all = np.einsum("jv,bvd->bjd", h36m, gt_v)
        for pred_j_all, gt_j_all in zip(pred_h36m_all, gt_h36m_all):
            pred_rooted = pred_j_all - pred_j_all[[0], :]
            gt_rooted = gt_j_all - gt_j_all[[0], :]
            for name, joint_ids in JOINT_SETS.items():
                pa_values[name].append(
                    pa_mpjpe_mm(pred_rooted[list(joint_ids), :], gt_rooted[list(joint_ids), :])
                )

    summary = {
        "num_samples": len(sample_files),
        "samples_dir": str(samples_dir),
        "data_dir": str(Path(args.data_dir)),
        "fixed_protocol": {
            "prediction": "FastSAM3DBody sample_*.npz -> barycentric MHR2SMPL vertices (no MLP)",
            "ground_truth": "3DPW gendered SMPL vertices with gt global_orient/body_pose/betas/transl transformed by cam_poses",
            "regressor": str(Path(args.h36m_regressor)),
            "root": "H36M joint 0 subtracted before selecting final joints",
            "alignment": "SAT-HMR rigid_transform_3d similarity alignment",
        },
        "only_changed_between_rows": "final H36M joint indices",
        "joint_sets": {name: list(ids) for name, ids in JOINT_SETS.items()},
        "pa_mpjpe_mm": {name: summarize(values) for name, values in pa_values.items()},
    }

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with output_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["joint_set", "indices", "mean", "median", "variance", "std", "iqr", "count"])
        for name, values in summary["pa_mpjpe_mm"].items():
            writer.writerow([
                name,
                " ".join(map(str, summary["joint_sets"][name])),
                values["mean"],
                values["median"],
                values["variance"],
                values["std"],
                values["iqr"],
                values["count"],
            ])
    print(json.dumps(summary["pa_mpjpe_mm"], indent=2))


if __name__ == "__main__":
    main()

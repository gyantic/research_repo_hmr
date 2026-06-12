#!/usr/bin/env python3
"""Evaluate FastSAM3DBody 3DPW sample_*.npz outputs on H36M14 joints."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import smplx
import torch
import trimesh
from tqdm import tqdm


H36M_EVAL_JOINTS = [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16]


def procrustes_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    mu_source = source.mean(axis=0)
    mu_target = target.mean(axis=0)
    source0 = source - mu_source
    target0 = target - mu_target
    var_source = (source0 ** 2).sum()
    if var_source <= 1e-12:
        return source.copy()
    u, s, vt = np.linalg.svd(source0.T @ target0)
    z = np.eye(3)
    z[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rot = vt.T @ z @ u.T
    scale = s.sum() / var_source
    return scale * (source @ rot.T) + (mu_target - scale * (rot @ mu_source))


def mean_l2_mm(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(((pred - gt) ** 2).sum(axis=1)).mean() * 1000.0)


def summarize(values: list[float]) -> dict[str, float]:
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


def mhr_vertices_to_smpl(
    pred_vertices: torch.Tensor,
    source_faces: torch.Tensor,
    triangle_ids: torch.Tensor,
    barycentric_coords: torch.Tensor,
) -> torch.Tensor:
    source_vertices = pred_vertices * 0.01
    triangles = source_vertices[:, source_faces[triangle_ids], :]
    return (triangles * barycentric_coords[None, :, :, None]).sum(dim=2)


def make_smpl(model_path: str, gender: str, device: torch.device) -> smplx.SMPL:
    model = smplx.SMPL(model_path=model_path, gender=gender).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--smpl_model", default="/home/kyuta/ROMP/model_data/parameters/SMPL_NEUTRAL.pkl")
    parser.add_argument("--smpl_model_dir", default="/home/kyuta/ROMP/model_data/parameters")
    parser.add_argument("--mapping", default="/home/kyuta/Fast-SAM-3D-Body/mhr2smpl/data/mhr2smpl_mapping.npz")
    parser.add_argument("--mhr_face_mesh", default="/home/kyuta/MHR/tools/mhr_smpl_conversion/assets/mhr_face_mask.ply")
    parser.add_argument("--h36m_regressor", default="/home/kyuta/SAT-HMR/weights/smpl_data/smpl/J_regressor_h36m_correct.npy")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples_dir = Path(args.samples_dir)
    sample_files = sorted(samples_dir.glob("sample_*.npz"))
    if not sample_files:
        raise SystemExit(f"No sample_*.npz files found in {samples_dir}")

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    mapping = np.load(args.mapping)
    triangle_ids = torch.from_numpy(mapping["triangle_ids"].astype(np.int64)).to(device)
    barycentric = torch.from_numpy(mapping["baryc_coords"].astype(np.float32)).to(device)
    mhr_mesh = trimesh.load(args.mhr_face_mesh, process=False)
    source_faces = torch.from_numpy(np.asarray(mhr_mesh.faces, dtype=np.int64)).to(device)
    h36m_regressor = torch.from_numpy(np.load(args.h36m_regressor).astype(np.float32)).to(device)

    neutral_model = make_smpl(args.smpl_model, "neutral", device)
    gender_models = {
        "neutral": neutral_model,
        "male": make_smpl(args.smpl_model_dir, "male", device),
        "female": make_smpl(args.smpl_model_dir, "female", device),
    }

    metrics = {
        "H36M14_MPJPE_neutralGT": [],
        "H36M14_PA-MPJPE_neutralGT": [],
        "H36M14_MPJPE_genderGT": [],
        "H36M14_PA-MPJPE_genderGT": [],
    }

    for start in tqdm(range(0, len(sample_files), args.batch_size), desc="H36M14 eval"):
        batch_files = sample_files[start : start + args.batch_size]
        pred_mhr_cm = []
        gt_poses = []
        gt_betas = []
        gt_genders = []

        for path in batch_files:
            data = np.load(path, allow_pickle=True)
            pred_raw = data["pred_vertices"].astype(np.float32).copy()
            pred_raw[:, [1, 2]] *= -1
            pred_cam_t = data["pred_cam_t"].astype(np.float32).copy()
            pred_cam_t[[1, 2]] *= -1
            pred_mhr_cm.append((pred_raw + pred_cam_t[None, :]) * 100.0)
            gt_poses.append(data["gt_poses"].astype(np.float32))
            gt_betas.append(data["gt_betas"].astype(np.float32))
            gt_genders.append(str(data["gt_gender"]).lower())

        pred_mhr_t = torch.from_numpy(np.stack(pred_mhr_cm)).float().to(device)
        gt_poses_t = torch.from_numpy(np.stack(gt_poses)).float().to(device)
        gt_betas_t = torch.from_numpy(np.stack(gt_betas)).float().to(device)

        with torch.no_grad():
            pred_verts = mhr_vertices_to_smpl(pred_mhr_t, source_faces, triangle_ids, barycentric)
            pred_h36m = torch.einsum("jv,bvd->bjd", h36m_regressor, pred_verts)[:, H36M_EVAL_JOINTS]

            neutral_out = neutral_model(
                betas=gt_betas_t,
                body_pose=gt_poses_t[:, 3:],
                global_orient=gt_poses_t[:, :3],
            )
            gt_h36m_neutral = torch.einsum("jv,bvd->bjd", h36m_regressor, neutral_out.vertices)[:, H36M_EVAL_JOINTS]

            gt_h36m_gender = []
            for i, gender in enumerate(gt_genders):
                model_key = "female" if "female" in gender else ("male" if "male" in gender else "neutral")
                out = gender_models[model_key](
                    betas=gt_betas_t[i : i + 1],
                    body_pose=gt_poses_t[i : i + 1, 3:],
                    global_orient=gt_poses_t[i : i + 1, :3],
                )
                gt_h36m_gender.append(
                    torch.einsum("jv,bvd->bjd", h36m_regressor, out.vertices)[:, H36M_EVAL_JOINTS]
                )
            gt_h36m_gender = torch.cat(gt_h36m_gender, dim=0)

        pred_np = pred_h36m.detach().cpu().numpy()
        neutral_np = gt_h36m_neutral.detach().cpu().numpy()
        gender_np = gt_h36m_gender.detach().cpu().numpy()

        for pred, gt_neutral, gt_gender in zip(pred_np, neutral_np, gender_np):
            pred_root = pred - pred[0:1]
            gt_neutral_root = gt_neutral - gt_neutral[0:1]
            gt_gender_root = gt_gender - gt_gender[0:1]

            metrics["H36M14_MPJPE_neutralGT"].append(mean_l2_mm(pred_root, gt_neutral_root))
            metrics["H36M14_PA-MPJPE_neutralGT"].append(
                mean_l2_mm(procrustes_align(pred_root, gt_neutral_root), gt_neutral_root)
            )
            metrics["H36M14_MPJPE_genderGT"].append(mean_l2_mm(pred_root, gt_gender_root))
            metrics["H36M14_PA-MPJPE_genderGT"].append(
                mean_l2_mm(procrustes_align(pred_root, gt_gender_root), gt_gender_root)
            )

    summary = {name: summarize(values) for name, values in metrics.items()}
    summary["samples_dir"] = str(samples_dir)
    summary["regressor"] = args.h36m_regressor
    summary["H36M_EVAL_JOINTS"] = H36M_EVAL_JOINTS

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with output_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["metric", "mean", "median", "variance", "std", "iqr", "count"])
        for name in metrics:
            stats = summary[name]
            writer.writerow([
                name,
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

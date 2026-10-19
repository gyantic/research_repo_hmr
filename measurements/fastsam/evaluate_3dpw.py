#!/usr/bin/env python3
"""Fast-SAM-3D-Body の 3DPW 精度評価（論文 図2 の Fast-SAM-3D-Body と脚注1の値）。

step0_infer_3dpw.py が書いた sample_*.npz（MHR メッシュの推定）を、重心座標の対応表で SMPL 頂点へ
変換し、性別ごとの 3DPW 正解 SMPL（カメラ座標）と比べて MPJPE / PA-MPJPE / PVE / MPVAE(PA-PVE) を出す。

--pa_protocol で PA-MPJPE の関節集合を選ぶ:
  h36m14_global_orient  SMPL 頂点から H36M 回帰行列で求めた 14 関節（本論文の評価、図2）
  paper_3dpw14          SMPL 関節列の先頭 14 関節と 3DPW の gt_joints_3d[:14]（原論文の関節集合、脚注1）

    python measurements/fastsam/evaluate_3dpw.py --samples_dir <step0 の出力>/samples         --output_json metrics.json --output_csv metrics.csv --pa_protocol h36m14_global_orient
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import pickle
from pathlib import Path

import numpy as np
import smplx
import torch
import trimesh
from tqdm import tqdm

# SMPL はリポジトリ直下の body_models/smpl/ に1か所だけ置く（SAT-HMR・ROMP と共用）
SMPL_DIR = Path(__file__).resolve().parents[2] / "body_models" / "smpl"

# H36M-17 regressor joints used by SAT-HMR for 14-joint evaluation (drops joints 0, 7 and 9).
H36M14_EVAL_JOINTS = [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16]

SMPL_GENDERS_IN_BATCH_ORDER = ("male", "female", "neutral")


def _patch_chumpy_compat() -> None:
    """公式 SMPL の .pkl は chumpy 形式のため、Python 3.11 / NumPy 1.24+ で消えた名前を戻す。"""
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    for alias, target in (
        ("bool", np.bool_), ("int", np.int_), ("float", np.float64),
        ("complex", np.complex128), ("object", np.object_), ("str", np.str_),
        ("unicode", np.str_),
    ):
        if alias not in np.__dict__:
            setattr(np, alias, target)


_patch_chumpy_compat()


def resolve_device(requested: str) -> torch.device:
    # Any value other than exactly "cuda" (including "cuda:1") falls back to CPU, as in the original scripts.
    return torch.device(requested if requested == "cuda" and torch.cuda.is_available() else "cpu")


def make_smpl(model_path: str, gender: str, device: torch.device) -> smplx.SMPL:
    model = smplx.SMPL(model_path=model_path, gender=gender).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def make_gender_models(
    neutral_model_path: str, gendered_model_dir: str, device: torch.device
) -> dict[str, smplx.SMPL]:
    """Neutral model from an explicit .pkl path; male/female from the SMPL directory."""
    return {
        "neutral": make_smpl(neutral_model_path, "neutral", device),
        "male": make_smpl(gendered_model_dir, "male", device),
        "female": make_smpl(gendered_model_dir, "female", device),
    }


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


def mean_l2_mm(pred_m: np.ndarray, gt_m: np.ndarray) -> float:
    """Mean per-point Euclidean distance; inputs in meters, result in millimeters."""
    return float(np.linalg.norm(pred_m - gt_m, axis=-1).mean() * 1000.0)


def load_barycentric_mapping(
    mapping_path: str, mhr_face_mesh_path: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (mhr_faces, triangle_ids, barycentric) for the MHR->SMPL barycentric mapping."""
    mapping = np.load(mapping_path)
    triangle_ids = torch.from_numpy(mapping["triangle_ids"].astype(np.int64)).to(device)
    barycentric = torch.from_numpy(mapping["baryc_coords"].astype(np.float32)).to(device)
    mhr_faces = torch.from_numpy(
        np.asarray(trimesh.load(mhr_face_mesh_path, process=False).faces, dtype=np.int64)
    ).to(device)
    return mhr_faces, triangle_ids, barycentric


def mhr_to_smpl_vertices(
    mhr_vertices: torch.Tensor,
    mhr_faces: torch.Tensor,
    triangle_ids: torch.Tensor,
    barycentric: torch.Tensor,
) -> torch.Tensor:
    """[B, V_mhr, 3] -> [B, 6890, 3]: each SMPL vertex is a barycentric blend of one MHR triangle (no MLP)."""
    triangles = mhr_vertices[:, mhr_faces[triangle_ids], :]
    return (triangles * barycentric[None, :, :, None]).sum(dim=2)


def translated_pred_mhr_vertices_m(sample) -> np.ndarray:
    """Predicted MHR vertices shifted by pred_cam_t (meters, no axis flip)."""
    return sample["pred_vertices"].astype(np.float32) + sample["pred_cam_t"].astype(np.float32)[None, :]


def smpl_gender_key(raw_gender) -> str:
    # 3DPW stores "m"/"f"; test "f" first because "female" also contains "m".
    gender = str(raw_gender).lower()
    if "f" in gender:
        return "female"
    if "m" in gender:
        return "male"
    return "neutral"


def gendered_smpl_vertices(
    samples: list, gender_models: dict[str, smplx.SMPL], device: torch.device
) -> list[np.ndarray]:
    """GT SMPL vertices (world frame) per sample, evaluated in one SMPL batch per gender.

    Batch composition and order (male, female, neutral) match the original scripts,
    so GPU results are reproduced exactly.
    """
    vertices: list[np.ndarray | None] = [None] * len(samples)
    for gender_key in SMPL_GENDERS_IN_BATCH_ORDER:
        idxs = [i for i, sample in enumerate(samples) if smpl_gender_key(sample["gt_gender"]) == gender_key]
        if not idxs:
            continue
        poses = torch.from_numpy(np.stack([samples[i]["gt_poses"].astype(np.float32) for i in idxs])).float().to(device)
        betas = torch.from_numpy(np.stack([samples[i]["gt_betas"].astype(np.float32) for i in idxs])).float().to(device)
        trans = torch.from_numpy(np.stack([samples[i]["gt_trans"].astype(np.float32) for i in idxs])).float().to(device)
        with torch.no_grad():
            out = gender_models[gender_key](
                betas=betas,
                body_pose=poses[:, 3:],
                global_orient=poses[:, :3],
                transl=trans,
            )
        batch_vertices = out.vertices.detach().cpu().numpy()
        for batch_idx, sample_idx in enumerate(idxs):
            vertices[sample_idx] = batch_vertices[batch_idx]
    if any(v is None for v in vertices):
        raise RuntimeError("Internal gender batching error")
    return vertices


def load_cam_pose(seq_cache: dict[str, dict], data_dir: Path, sequence: str, frame_idx: int) -> np.ndarray:
    """Transposed 4x4 world->camera pose from 3DPW sequenceFiles/test (cached per sequence)."""
    if sequence not in seq_cache:
        with (data_dir / "sequenceFiles" / "test" / f"{sequence}.pkl").open("rb") as fp:
            seq_cache[sequence] = pickle.load(fp, encoding="latin1")
    return seq_cache[sequence]["cam_poses"][frame_idx].T.astype(np.float32)


def apply_camera(points: np.ndarray, cam_pose_t: np.ndarray) -> np.ndarray:
    hom = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    return (hom @ cam_pose_t)[:, :3]


# First 14 of the 24 SMPL joints, compared against 3DPW's stored gt_joints_3d[:14].
THREEDPW_PA14 = list(range(14))
METRIC_NAMES = ("MPJPE", "PA-MPJPE", "PVE", "MPVAE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--vertices_npz", default="", help="Optional fitted SMPL vertices npz with pred_smpl_vertices")
    parser.add_argument("--data_dir", default=str(Path(__file__).resolve().parents[2] / "datasets" / "3dpw"))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--smpl_model", default=str(SMPL_DIR / "SMPL_NEUTRAL.pkl"))
    parser.add_argument("--smpl_model_dir", default=str(SMPL_DIR))
    parser.add_argument("--mapping", default=str(Path(__file__).resolve().parents[2] / "fast-sam-3d-body" / "mhr2smpl" / "data" / "mhr2smpl_mapping.npz"))
    parser.add_argument("--mhr_face_mesh", default=str(Path(__file__).resolve().parents[2] / "fast-sam-3d-body" / "mhr2smpl" / "data" / "mhr_face_mask.ply"))
    parser.add_argument("--h36m_regressor", default=str(SMPL_DIR / "J_regressor_h36m_correct.npy"))
    parser.add_argument(
        "--pa_protocol",
        choices=["paper_3dpw14", "h36m14_global_orient"],
        default="paper_3dpw14",
        help="PA-MPJPE joint definition: paper 3DPW stored 14 joints, or SAT-HMR-style H36M14 from global-orient SMPL GT.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


# Umeyama similarity alignment with scale = trace(R K) / var. Not shared with the other
# scripts because their alignment variants differ in the last floating-point bits.
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


def load_pred_vertices(
    args: argparse.Namespace,
    sample_files: list[Path],
    device: torch.device,
) -> np.ndarray:
    """[N, 6890, 3] predicted SMPL vertices: from --vertices_npz if given, else barycentric MHR->SMPL."""
    if args.vertices_npz:
        pred_v = np.load(args.vertices_npz)["pred_smpl_vertices"].astype(np.float32)
        return pred_v[: len(sample_files)]

    mhr_faces, triangle_ids, barycentric = load_barycentric_mapping(args.mapping, args.mhr_face_mesh, device)
    pred_vertices = []
    for start in tqdm(range(0, len(sample_files), args.batch_size), desc="Barycentric pred"):
        files = sample_files[start : start + args.batch_size]
        pred_mhr = [translated_pred_mhr_vertices_m(np.load(path, allow_pickle=True)) for path in files]
        pred_mhr_t = torch.from_numpy(np.stack(pred_mhr)).float().to(device)
        with torch.no_grad():
            pred_v = mhr_to_smpl_vertices(pred_mhr_t, mhr_faces, triangle_ids, barycentric).detach().cpu().numpy()
        pred_vertices.append(pred_v)
    return np.concatenate(pred_vertices, axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    samples_dir = Path(args.samples_dir)
    sample_files = sorted(samples_dir.glob("sample_*.npz"))
    if not sample_files:
        raise SystemExit(f"No sample_*.npz files found in {samples_dir}")

    device = resolve_device(args.device)
    data_dir = Path(args.data_dir)
    gender_models = make_gender_models(args.smpl_model, args.smpl_model_dir, device)
    smpl24_regressor = gender_models["neutral"].J_regressor.detach().cpu().numpy().astype(np.float32)
    h36m_regressor = np.load(args.h36m_regressor).astype(np.float32)
    pred_v_all = load_pred_vertices(args, sample_files, device)

    seq_cache: dict[str, dict] = {}
    metrics: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}

    for start in tqdm(range(0, len(sample_files), args.batch_size), desc="Paper repro eval"):
        files = sample_files[start : start + args.batch_size]
        loaded = [np.load(path, allow_pickle=True) for path in files]
        pred_v = pred_v_all[start : start + len(files)]

        gt_v = []
        gt_pa14 = []
        for data, world_verts in zip(loaded, gendered_smpl_vertices(loaded, gender_models, device)):
            cam_pose_t = load_cam_pose(seq_cache, data_dir, str(data["sequence"]), int(data["frame_idx"]))
            gt_v.append(apply_camera(world_verts, cam_pose_t))
            gt_pa14.append(apply_camera(data["gt_joints_3d"].astype(np.float32), cam_pose_t)[THREEDPW_PA14])
        gt_v = np.stack(gt_v).astype(np.float32)
        gt_pa14 = np.stack(gt_pa14).astype(np.float32)

        pred_h36m = np.einsum("jv,bvd->bjd", h36m_regressor, pred_v)[:, H36M14_EVAL_JOINTS]
        gt_h36m = np.einsum("jv,bvd->bjd", h36m_regressor, gt_v)[:, H36M14_EVAL_JOINTS]
        pred_j24 = np.einsum("jv,bvd->bjd", smpl24_regressor, pred_v)
        gt_j24 = np.einsum("jv,bvd->bjd", smpl24_regressor, gt_v)
        pred_pa14 = pred_j24[:, THREEDPW_PA14]

        for pred_j, gt_j, pred_vertices, gt_vertices, pred_pelvis, gt_pelvis, pred_pa_j, gt_pa_j in zip(
            pred_h36m,
            gt_h36m,
            pred_v,
            gt_v,
            pred_j24[:, 0],
            gt_j24[:, 0],
            pred_pa14,
            gt_pa14,
        ):
            # Root = first of the selected H36M14 joints (H36M joint 1), as in the original diagnostic.
            pred_rooted = pred_j - pred_j[0:1]
            gt_rooted = gt_j - gt_j[0:1]
            metrics["MPJPE"].append(mean_l2_mm(pred_rooted, gt_rooted))
            if args.pa_protocol == "h36m14_global_orient":
                metrics["PA-MPJPE"].append(mean_l2_mm(procrustes_align(pred_rooted, gt_rooted), gt_rooted))
            else:
                metrics["PA-MPJPE"].append(mean_l2_mm(procrustes_align(pred_pa_j, gt_pa_j), gt_pa_j))

            pred_vertices_rooted = pred_vertices - pred_pelvis[None, :]
            gt_vertices_rooted = gt_vertices - gt_pelvis[None, :]
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
        for metric in METRIC_NAMES:
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

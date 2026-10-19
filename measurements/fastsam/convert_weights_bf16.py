"""Fast-SAM-3D-Body の fp32 重みを bf16 に変換して保存し、ファイルサイズを記録する。

autocast（実行時キャスト）ではなく、チェックポイント内の fp32 テンソルを
実際に bf16 へ変換した重みファイルを生成する。fp32 以外（fp16, int 等）は変更しない。

論文表2のサイズ合算レシピは results/measurements/benchmark_manifest_all.csv に従う
（MB = 1e6 bytes）:
    fp32: model.ckpt 2109.129 + mhr_model.pt 696.110
          + yolo11m-pose.engine 27.864 + moge-2-vits blobs 140.555 = 2973.7 MB
    bf16: model_bf16.ckpt 1907.421 + mhr_model_bf16.pt 350.308
          + yolo11m-pose.engine 27.864（FP16 エンジンのため変換対象外）
          + moge model_bf16.pt 70.316 = 2355.9 MB

Usage:
    python measurements/fastsam/convert_weights_bf16.py
"""

import os
from pathlib import Path

import torch

FASTSAM_ROOT = Path(__file__).resolve().parents[2] / "fast-sam-3d-body"
HF_HUB = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
OUT_DIR = FASTSAM_ROOT / "checkpoints" / "bf16"
BYTES_PER_MB = 1e6  # 表2・manifest と同じ定義

TARGETS = [
    FASTSAM_ROOT / "checkpoints/sam-3d-body-dinov3/model.ckpt",
    FASTSAM_ROOT / "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt",
    FASTSAM_ROOT / "checkpoints/yolo/yolo11m-pose.pt",
    HF_HUB
    / "models--Ruicheng--moge-2-vits-normal/snapshots/679230677b4d282c6f304189a93e98e14f085902/model.pt",
]


def cast_fp32_to_bf16(obj, stats):
    """オブジェクトを再帰的に走査し、fp32 テンソルのみ bf16 へ変換する。"""
    if isinstance(obj, torch.Tensor):
        if obj.dtype == torch.float32:
            stats["converted"] += 1
            return obj.to(torch.bfloat16)
        stats["kept"] += 1
        return obj
    if isinstance(obj, dict):
        return {k: cast_fp32_to_bf16(v, stats) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        seq = [cast_fp32_to_bf16(v, stats) for v in obj]
        return type(obj)(seq) if isinstance(obj, tuple) else seq
    if isinstance(obj, torch.nn.Module):
        for p in obj.parameters():
            if p.dtype == torch.float32:
                p.data = p.data.to(torch.bfloat16)
                stats["converted"] += 1
            else:
                stats["kept"] += 1
        for b in obj.buffers():
            if b.dtype == torch.float32:
                b.data = b.data.to(torch.bfloat16)
                stats["converted"] += 1
            else:
                stats["kept"] += 1
        return obj
    return obj


def size_mb(path):
    return path.stat().st_size / BYTES_PER_MB


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total_src_mb = 0.0
    total_dst_mb = 0.0
    for src in TARGETS:
        dst = OUT_DIR / (src.stem + "_bf16" + src.suffix)
        if dst.exists():
            print(f"{src.name}: {size_mb(src):.1f} MB -> {dst.name}: {size_mb(dst):.1f} MB (既存、スキップ)")
        else:
            stats = {"converted": 0, "kept": 0}
            obj = cast_fp32_to_bf16(torch.load(src, map_location="cpu", weights_only=False), stats)
            # torch.save cannot serialize a TorchScript module; it needs torch.jit.save.
            if isinstance(obj, torch.jit.ScriptModule):
                torch.jit.save(obj, str(dst))
            else:
                torch.save(obj, dst)
            print(
                f"{src.name}: {size_mb(src):.1f} MB -> {dst.name}: {size_mb(dst):.1f} MB "
                f"(fp32->bf16: {stats['converted']} tensors, unchanged: {stats['kept']})"
            )
        total_src_mb += size_mb(src)
        total_dst_mb += size_mb(dst)

    print(f"\n合計: fp32 {total_src_mb:.1f} MB -> bf16 {total_dst_mb:.1f} MB")


if __name__ == "__main__":
    main()

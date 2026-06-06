#!/usr/bin/env python3
"""Per-component timing breakdown of the model forward pass."""

import os, sys, time
import torch

# Same settings as run_publisher_ue.py
os.environ["SAM3D_QUIET"] = "0"   # Enable timing logs
os.environ["TORCHINDUCTOR_CUDAGRAPH_TREES"] = "0"
os.environ["USE_TRT_BACKBONE"] = "1"
os.environ["TRT_BACKBONE_PATH"] = "./checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_int8.engine"
os.environ["GPU_HAND_PREP"] = "1"
os.environ["LAYER_DTYPE"] = "fp32"
os.environ["SKIP_KEYPOINT_PROMPT"] = "1"
os.environ["KEYPOINT_PROMPT_INTERM_INTERVAL"] = "999"
os.environ["IMG_SIZE"] = "512"
os.environ["MHR_NO_CORRECTIVES"] = "1"
os.environ["FOV_TRT"] = "0"   # FOVなし（カメラ内部パラメータは固定値を使う）
os.environ["BODY_INTERM_PRED_LAYERS"] = "0,1,2"
os.environ["HAND_INTERM_PRED_LAYERS"] = "0,1"
os.environ["MHR_USE_CUDA_GRAPH"] = "1"
os.environ["USE_COMPILE"] = "1"
os.environ["USE_COMPILE_BACKBONE"] = "1"
os.environ["DECODER_COMPILE"] = "1"
os.environ["COMPILE_MODE"] = "reduce-overhead"
os.environ["COMPILE_WARMUP_BATCH_SIZES"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.expanduser("~/.cache/torchinductor")
os.environ["TRITON_CACHE_DIR"] = os.path.expanduser("~/.cache/triton")

parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

from sam_3d_body.build_models import load_sam_3d_body

CHECKPOINT_DIR = "./checkpoints/sam-3d-body-dinov3"

print("Loading model...")
model, cfg = load_sam_3d_body(
    checkpoint_path=os.path.join(CHECKPOINT_DIR, "model.ckpt"),
    mhr_path=os.path.join(CHECKPOINT_DIR, "assets", "mhr_model.pt"),
)
model.cuda().eval()
model.apply_compile(mode="reduce-overhead", dtype=torch.float16)

# ダミー入力（body 1人 + 両手 2人 = 3クロップ）
B, N = 1, 3
img = torch.randn(B, N, 3, 512, 512, device="cuda", dtype=torch.float32)
cam_int = torch.tensor([
    [1000., 0., 256.],
    [0., 1000., 256.],
    [0., 0., 1.],
], device="cuda").unsqueeze(0)

batch = {
    "img": img,
    "cam_int": cam_int,
    "has_mask": torch.zeros(B, N, device="cuda"),
    "person_valid": torch.ones(B, N, device="cuda"),
    "img_ori": [type("obj", (), {"data": (img[0, 0].permute(1,2,0).cpu().numpy() * 255).astype("uint8")})()],
}
model._initialize_batch(batch)

model.body_batch_idx = [0]
model.hand_batch_idx = [1, 2]

print("\nWarming up (5 iterations, logs suppressed)...")
os.environ["SAM3D_QUIET"] = "1"
import sam_3d_body.models.meta_arch.sam3d_body as _m
_m._QUIET = True

with torch.no_grad():
    for _ in range(5):
        model.forward_pose_branch(batch)
torch.cuda.synchronize()

print("Warmup done.\n")
print("=" * 60)
print("Profiling (1 iteration, full logs):")
print("=" * 60)

# ログを有効化
_m._QUIET = False

with torch.no_grad():
    t_total_start = time.perf_counter()
    torch.cuda.synchronize()
    model.forward_pose_branch(batch)
    torch.cuda.synchronize()
    t_total = (time.perf_counter() - t_total_start) * 1000

print(f"\nforward_pose_branch TOTAL: {t_total:.1f} ms")

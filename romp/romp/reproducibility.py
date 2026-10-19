"""Seeding and deterministic-backend settings shared by the ROMP video entry points.

Applied once when ``romp.predict.video`` is imported, so every script that builds a
``Video_processor`` (romp_video_demo.py, the UE generators, make_proxy_obj.py) runs
with the same settings before the model is built.
"""
import os
import random

import cv2
import numpy as np
import torch

SEED = 42


def setup_reproducibility(seed: int = SEED):
    # Must be set before the first cuBLAS call for deterministic matmuls.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    # warn_only: some ops have no deterministic kernel; warn instead of failing.
    torch.use_deterministic_algorithms(True, warn_only=True)

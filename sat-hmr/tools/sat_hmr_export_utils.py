#!/usr/bin/env python3
"""Common helpers for SAT-HMR ONNX/TRT export related scripts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import torch
import yaml


def load_config(input_size: int, sat_num_lvls_override: int = -1) -> SimpleNamespace:
    demo_cfg_path = Path("configs/run/demo.yaml")
    with demo_cfg_path.open("r", encoding="utf-8") as f:
        demo_config = yaml.safe_load(f)

    model_name = demo_config.get("model", "default")
    model_cfg_path = Path(f"configs/models/{model_name}.yaml")
    with model_cfg_path.open("r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)

    cfg = SimpleNamespace()
    for k, v in model_config.items():
        setattr(cfg, k, v)
    for k, v in demo_config.items():
        setattr(cfg, k, v)

    cfg.mode = "infer"
    cfg.pretrain = False
    cfg.pretrain_path = None
    cfg.input_size = input_size
    if sat_num_lvls_override > 0 and hasattr(cfg, "sat_cfg") and isinstance(cfg.sat_cfg, dict):
        cfg.sat_cfg = dict(cfg.sat_cfg)
        cfg.sat_cfg["num_lvls"] = int(sat_num_lvls_override)
    return cfg


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def build_infer_model(
    input_size: int,
    weights_path: str,
    device_name: str,
    sat_num_lvls_override: int = -1,
) -> Tuple[torch.nn.Module, Dict[str, int]]:
    # Delay heavy import so callers can set env flags (e.g., XFORMERS_DISABLED) first.
    from models import build_sat_model

    device = resolve_device(device_name)
    cfg = load_config(input_size, sat_num_lvls_override=sat_num_lvls_override)
    model, _ = build_sat_model(cfg, set_criterion=False)

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    if sat_num_lvls_override > 0:
        # Export-only SAT level override can change level_embed shape (e.g. 3 -> 2).
        # Drop incompatible level_embed from checkpoint and keep model-initialized value.
        if isinstance(state_dict, dict) and "level_embed" in state_dict:
            ckpt_v = state_dict["level_embed"]
            model_v = model.state_dict().get("level_embed", None)
            if model_v is not None and tuple(ckpt_v.shape) != tuple(model_v.shape):
                state_dict = dict(state_dict)
                del state_dict["level_embed"]

    incompatible = model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()
    return model, {
        "missing_keys": len(incompatible.missing_keys),
        "unexpected_keys": len(incompatible.unexpected_keys),
    }


class SatHmrOnnxWrapper(torch.nn.Module):
    """ONNX export wrapper with plain tensor inputs/outputs."""

    def __init__(self, model: torch.nn.Module, keep_img_size_input: bool = True):
        super().__init__()
        self.model = model
        self.keep_img_size_input = keep_img_size_input

    def forward(self, image: torch.Tensor, img_size: torch.Tensor):  # type: ignore[override]
        # image: [1,3,H,W], img_size: [2] => [h,w]
        if image.dim() != 4 or image.shape[0] != 1:
            raise RuntimeError("image must be shaped [1, 3, H, W]")
        # Use squeeze instead of advanced indexing to avoid exporting a top-level Gather(image, 0).
        sample = image.squeeze(0)
        targets = [{"img_size": img_size, "img_path": "onnx_frame"}]
        outputs = self.model([sample], targets)
        base_outputs = (
            outputs["pred_poses"],
            outputs["pred_betas"],
            outputs["pred_transl"],
            outputs["pred_confs"],
            outputs["pred_boxes"],
        )
        if self.keep_img_size_input:
            # Keep img_size as a real graph input by exposing an img_size-dependent output.
            return base_outputs + (outputs["pred_intrinsics"],)
        return base_outputs


def build_dummy_inputs(
    img_h: int, img_w: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Inputs are already normalized image tensors in SAT-HMR pipeline.
    image = torch.randn(1, 3, img_h, img_w, device=device, dtype=torch.float32)
    img_size = torch.tensor([img_h, img_w], device=device, dtype=torch.int64)
    return image, img_size

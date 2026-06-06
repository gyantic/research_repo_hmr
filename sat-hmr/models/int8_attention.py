import os
from typing import Optional

import torch


def int8_attention_enabled(module: Optional[torch.nn.Module] = None) -> bool:
    if module is not None and hasattr(module, "_sat_int8_attention_enabled"):
        return bool(getattr(module, "_sat_int8_attention_enabled"))
    return os.environ.get("SAT_HMR_INT8_ATTENTION") == "1"


def int8_attention_cpu_fallback_allowed() -> bool:
    return os.environ.get("SAT_HMR_INT8_ATTENTION_ALLOW_CPU_FALLBACK") == "1"


def _next_multiple_of_8(length: int) -> int:
    return ((length + 7) // 8) * 8


def _min_supported_rows(length: int) -> int:
    return max(length, 17)


def _pad_dim0(tensor: torch.Tensor, target: int, value) -> torch.Tensor:
    if tensor.shape[0] >= target:
        return tensor
    pad_shape = (target - tensor.shape[0],) + tuple(tensor.shape[1:])
    pad = torch.full(pad_shape, value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=0)


def _pad_dim0_zero(tensor: torch.Tensor, target: int) -> torch.Tensor:
    return _pad_dim0(tensor, target, 0)


def _pad_dim0_one(tensor: torch.Tensor, target: int) -> torch.Tensor:
    return _pad_dim0(tensor, target, 1.0)


def _pad_last_dim(tensor: torch.Tensor, target: int, value) -> torch.Tensor:
    if tensor.shape[-1] >= target:
        return tensor
    pad_shape = tuple(tensor.shape[:-1]) + (target - tensor.shape[-1],)
    pad = torch.full(pad_shape, value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=-1)


def _quantize_per_output_channel(weight: torch.Tensor):
    scale = weight.abs().amax(dim=0).clamp_min(1e-5) / 127.0
    quant = torch.clamp((weight / scale).round(), -127, 127).to(torch.int8)
    return quant.contiguous(), scale.contiguous()


def _resolve_attn_bias(
    attn_bias: Optional[torch.Tensor],
    leading_shape,
    q_len: int,
    kv_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if attn_bias is None:
        return None

    bias = attn_bias.to(device=device, dtype=dtype)
    target_shape = tuple(leading_shape) + (q_len, kv_len)

    while bias.dim() > len(target_shape) and bias.shape[0] == 1:
        bias = bias.squeeze(0)
    while bias.dim() < len(target_shape):
        bias = bias.unsqueeze(0)

    bias = torch.broadcast_to(bias, target_shape)
    return bias.reshape(-1, q_len, kv_len).contiguous()


def _ensure_int8_mm_supported(m: int, k: int, n: int, stage: str) -> None:
    if int8_attention_cpu_fallback_allowed():
        return
    if m <= 16 or k <= 0 or n <= 0 or (k % 8) != 0 or (n % 8) != 0:
        raise RuntimeError(
            "torchao int8 attention would fall back to CPU for "
            f"{stage}: mm({m}x{k}, {k}x{n}). "
            "Required for GPU int mm: m > 16, k % 8 == 0, n % 8 == 0. "
            "Use `--allow_int8_attention_cpu_fallback` to force the slow path."
        )


def torchao_int8_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    from torchao.quantization import (
        quant_int8_per_token_matmul,
        quantize_activation_per_token_absmax,
    )

    if out_dtype is None:
        out_dtype = query.dtype

    if query.device.type != "cuda" and not int8_attention_cpu_fallback_allowed():
        raise RuntimeError(
            "torchao int8 attention requires CUDA for a usable path. "
            "Use `--allow_int8_attention_cpu_fallback` to force the slow CPU fallback."
        )

    q_len = query.shape[-2]
    kv_len = key.shape[-2]
    value_dim = value.shape[-1]
    padded_q_len = _min_supported_rows(q_len)
    padded_kv_len = _next_multiple_of_8(kv_len)
    leading_shape = query.shape[:-2]

    query_flat = query.reshape(-1, q_len, query.shape[-1]).contiguous()
    key_flat = key.reshape(-1, kv_len, key.shape[-1]).contiguous()
    value_flat = value.reshape(-1, kv_len, value_dim).contiguous()
    bias_flat = _resolve_attn_bias(
        attn_bias,
        leading_shape,
        q_len,
        kv_len,
        dtype=torch.float32,
        device=query.device,
    )
    if bias_flat is not None and padded_kv_len != kv_len:
        bias_flat = _pad_last_dim(bias_flat, padded_kv_len, float("-inf"))

    outputs = []
    for idx in range(query_flat.shape[0]):
        _ensure_int8_mm_supported(padded_q_len, query_flat[idx].shape[-1], padded_kv_len, "QK^T")
        q_i8, q_scale = quantize_activation_per_token_absmax(query_flat[idx])
        if padded_q_len != q_len:
            q_i8 = _pad_dim0_zero(q_i8, padded_q_len)
            q_scale = _pad_dim0_one(q_scale, padded_q_len)
        k_i8, k_scale = quantize_activation_per_token_absmax(key_flat[idx])
        if padded_kv_len != kv_len:
            k_i8 = _pad_dim0_zero(k_i8, padded_kv_len)
            k_scale = _pad_dim0_one(k_scale, padded_kv_len)
        logits = quant_int8_per_token_matmul(
            q_i8,
            q_scale,
            k_i8.transpose(0, 1).contiguous(),
            k_scale,
            output_dtype=torch.float32,
        )
        logits = logits[:q_len]
        if bias_flat is not None:
            logits = logits + bias_flat[idx]
        elif padded_kv_len != kv_len:
            logits[:, kv_len:] = float("-inf")

        attn = logits.softmax(dim=-1).to(out_dtype)
        attn_i8, attn_scale = quantize_activation_per_token_absmax(attn)
        if padded_q_len != q_len:
            attn_i8 = _pad_dim0_zero(attn_i8, padded_q_len)
            attn_scale = _pad_dim0_one(attn_scale, padded_q_len)
        v_i8, v_scale = _quantize_per_output_channel(value_flat[idx])
        if padded_kv_len != kv_len:
            v_i8 = _pad_dim0_zero(v_i8, padded_kv_len)
        _ensure_int8_mm_supported(padded_q_len, padded_kv_len, value_dim, "AV")
        out = quant_int8_per_token_matmul(
            attn_i8,
            attn_scale,
            v_i8,
            v_scale,
            output_dtype=torch.float32,
        )
        outputs.append(out[:q_len].to(out_dtype))

    return torch.stack(outputs, dim=0).reshape(*leading_shape, q_len, value_dim)

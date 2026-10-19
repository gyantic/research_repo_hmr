"""Grouped Linear reference/Triton kernels used only by tools/benchmark_grouped_bbox_int8.py.

Not wired into the SAT-HMR model (the grouped decoder/bbox paths that used them were removed).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


def _prod(values: Sequence[int]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


if TRITON_AVAILABLE:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def grouped_int8_gemm_kernel(
        x_ptr,
        w_ptr,
        out_ptr,
        x_scale_ptr,
        w_scale_ptr,
        bias_ptr,
        M,
        N,
        K,
        stride_xg,
        stride_xm,
        stride_xk,
        stride_wg,
        stride_wk,
        stride_wn,
        stride_og,
        stride_om,
        stride_on,
        stride_xsg,
        stride_xsm,
        stride_wsg,
        stride_wsn,
        stride_bg,
        stride_bn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        pid_g = tl.program_id(axis=2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

        for k_start in range(0, K, BLOCK_K):
            k_offsets = k_start + offs_k
            x_ptrs = (
                x_ptr
                + pid_g * stride_xg
                + offs_m[:, None] * stride_xm
                + k_offsets[None, :] * stride_xk
            )
            w_ptrs = (
                w_ptr
                + pid_g * stride_wg
                + k_offsets[:, None] * stride_wk
                + offs_n[None, :] * stride_wn
            )
            x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (k_offsets[None, :] < K), other=0)
            w = tl.load(w_ptrs, mask=(k_offsets[:, None] < K) & (offs_n[None, :] < N), other=0)
            acc += tl.dot(x, w, out_dtype=tl.int32)

        x_scale_ptrs = x_scale_ptr + pid_g * stride_xsg + offs_m * stride_xsm
        w_scale_ptrs = w_scale_ptr + pid_g * stride_wsg + offs_n * stride_wsn
        bias_ptrs = bias_ptr + pid_g * stride_bg + offs_n * stride_bn

        x_scale = tl.load(x_scale_ptrs, mask=offs_m < M, other=1.0)
        w_scale = tl.load(w_scale_ptrs, mask=offs_n < N, other=1.0)
        bias = tl.load(bias_ptrs, mask=offs_n < N, other=0.0)

        out = acc.to(tl.float32) * x_scale[:, None] * w_scale[None, :] + bias[None, :]
        out_ptrs = (
            out_ptr
            + pid_g * stride_og
            + offs_m[:, None] * stride_om
            + offs_n[None, :] * stride_on
        )
        tl.store(out_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def grouped_int8_gemm_relu_requant_kernel(
        x_ptr,
        w_ptr,
        out_q_ptr,
        out_scale_ptr,
        x_scale_ptr,
        w_scale_ptr,
        bias_ptr,
        M,
        N,
        K,
        stride_xg,
        stride_xm,
        stride_xk,
        stride_wg,
        stride_wk,
        stride_wn,
        stride_oqg,
        stride_oqm,
        stride_oqn,
        stride_osg,
        stride_osm,
        stride_xsg,
        stride_xsm,
        stride_wsg,
        stride_wsn,
        stride_bg,
        stride_bn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_g = tl.program_id(axis=1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)

        x_scale_ptrs = x_scale_ptr + pid_g * stride_xsg + offs_m * stride_xsm
        x_scale = tl.load(x_scale_ptrs, mask=offs_m < M, other=1.0)

        row_max = tl.zeros((BLOCK_M,), dtype=tl.float32)

        # Pass 1: find per-row post-ReLU max after dequant + bias.
        for n_start in range(0, N, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
            for k_start in range(0, K, BLOCK_K):
                k_offsets = k_start + offs_k
                x_ptrs = (
                    x_ptr
                    + pid_g * stride_xg
                    + offs_m[:, None] * stride_xm
                    + k_offsets[None, :] * stride_xk
                )
                w_ptrs = (
                    w_ptr
                    + pid_g * stride_wg
                    + k_offsets[:, None] * stride_wk
                    + offs_n[None, :] * stride_wn
                )
                x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (k_offsets[None, :] < K), other=0)
                w = tl.load(w_ptrs, mask=(k_offsets[:, None] < K) & (offs_n[None, :] < N), other=0)
                acc += tl.dot(x, w, out_dtype=tl.int32)

            w_scale_ptrs = w_scale_ptr + pid_g * stride_wsg + offs_n * stride_wsn
            bias_ptrs = bias_ptr + pid_g * stride_bg + offs_n * stride_bn
            w_scale = tl.load(w_scale_ptrs, mask=offs_n < N, other=1.0)
            bias = tl.load(bias_ptrs, mask=offs_n < N, other=0.0)

            out = acc.to(tl.float32) * x_scale[:, None] * w_scale[None, :] + bias[None, :]
            out = tl.maximum(out, 0.0)
            tile_row_max = tl.max(out, axis=1)
            row_max = tl.maximum(row_max, tile_row_max)

        q_scale = tl.maximum(row_max / 127.0, 1e-5)
        out_scale_ptrs = out_scale_ptr + pid_g * stride_osg + offs_m * stride_osm
        tl.store(out_scale_ptrs, q_scale, mask=offs_m < M)

        # Pass 2: recompute and requantize.
        for n_start in range(0, N, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
            for k_start in range(0, K, BLOCK_K):
                k_offsets = k_start + offs_k
                x_ptrs = (
                    x_ptr
                    + pid_g * stride_xg
                    + offs_m[:, None] * stride_xm
                    + k_offsets[None, :] * stride_xk
                )
                w_ptrs = (
                    w_ptr
                    + pid_g * stride_wg
                    + k_offsets[:, None] * stride_wk
                    + offs_n[None, :] * stride_wn
                )
                x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (k_offsets[None, :] < K), other=0)
                w = tl.load(w_ptrs, mask=(k_offsets[:, None] < K) & (offs_n[None, :] < N), other=0)
                acc += tl.dot(x, w, out_dtype=tl.int32)

            w_scale_ptrs = w_scale_ptr + pid_g * stride_wsg + offs_n * stride_wsn
            bias_ptrs = bias_ptr + pid_g * stride_bg + offs_n * stride_bn
            w_scale = tl.load(w_scale_ptrs, mask=offs_n < N, other=1.0)
            bias = tl.load(bias_ptrs, mask=offs_n < N, other=0.0)

            out = acc.to(tl.float32) * x_scale[:, None] * w_scale[None, :] + bias[None, :]
            out = tl.maximum(out, 0.0)
            out_q = tl.clamp(tl.floor(out / q_scale[:, None] + 0.5), 0.0, 127.0).to(tl.int8)

            out_q_ptrs = (
                out_q_ptr
                + pid_g * stride_oqg
                + offs_m[:, None] * stride_oqm
                + offs_n[None, :] * stride_oqn
            )
            tl.store(out_q_ptrs, out_q, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def grouped_fp_input_int8_gemm_relu_requant_kernel(
        x_ptr,
        w_ptr,
        out_q_ptr,
        out_scale_ptr,
        w_scale_ptr,
        bias_ptr,
        M,
        N,
        K,
        stride_xg,
        stride_xm,
        stride_xk,
        stride_wg,
        stride_wk,
        stride_wn,
        stride_oqg,
        stride_oqm,
        stride_oqn,
        stride_osg,
        stride_osm,
        stride_wsg,
        stride_wsn,
        stride_bg,
        stride_bn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_g = tl.program_id(axis=1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)

        x_row_max = tl.zeros((BLOCK_M,), dtype=tl.float32)

        # Pass 1: exact row-wise absmax for input quantization.
        for k_start in range(0, K, BLOCK_K):
            k_offsets = k_start + offs_k
            x_ptrs = (
                x_ptr
                + pid_g * stride_xg
                + offs_m[:, None] * stride_xm
                + k_offsets[None, :] * stride_xk
            )
            x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (k_offsets[None, :] < K), other=0.0)
            tile_absmax = tl.max(tl.abs(x), axis=1)
            x_row_max = tl.maximum(x_row_max, tile_absmax)

        x_scale = tl.maximum(x_row_max / 127.0, 1e-5)

        row_max = tl.zeros((BLOCK_M,), dtype=tl.float32)

        # Pass 2: GEMM + dequant + bias + ReLU max scan.
        for n_start in range(0, N, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
            for k_start in range(0, K, BLOCK_K):
                k_offsets = k_start + offs_k
                x_ptrs = (
                    x_ptr
                    + pid_g * stride_xg
                    + offs_m[:, None] * stride_xm
                    + k_offsets[None, :] * stride_xk
                )
                w_ptrs = (
                    w_ptr
                    + pid_g * stride_wg
                    + k_offsets[:, None] * stride_wk
                    + offs_n[None, :] * stride_wn
                )
                x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (k_offsets[None, :] < K), other=0.0)
                scaled_x = x / x_scale[:, None]
                x_q = tl.where(
                    scaled_x >= 0,
                    tl.floor(scaled_x + 0.5),
                    tl.ceil(scaled_x - 0.5),
                )
                x_q = tl.clamp(x_q, -127.0, 127.0).to(tl.int8)
                w = tl.load(w_ptrs, mask=(k_offsets[:, None] < K) & (offs_n[None, :] < N), other=0)
                acc += tl.dot(x_q, w, out_dtype=tl.int32)

            w_scale_ptrs = w_scale_ptr + pid_g * stride_wsg + offs_n * stride_wsn
            bias_ptrs = bias_ptr + pid_g * stride_bg + offs_n * stride_bn
            w_scale = tl.load(w_scale_ptrs, mask=offs_n < N, other=1.0)
            bias = tl.load(bias_ptrs, mask=offs_n < N, other=0.0)

            out = acc.to(tl.float32) * x_scale[:, None] * w_scale[None, :] + bias[None, :]
            out = tl.maximum(out, 0.0)
            tile_row_max = tl.max(out, axis=1)
            row_max = tl.maximum(row_max, tile_row_max)

        q_scale = tl.maximum(row_max / 127.0, 1e-5)
        out_scale_ptrs = out_scale_ptr + pid_g * stride_osg + offs_m * stride_osm
        tl.store(out_scale_ptrs, q_scale, mask=offs_m < M)

        # Pass 3: recompute GEMM and requantize output.
        for n_start in range(0, N, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
            for k_start in range(0, K, BLOCK_K):
                k_offsets = k_start + offs_k
                x_ptrs = (
                    x_ptr
                    + pid_g * stride_xg
                    + offs_m[:, None] * stride_xm
                    + k_offsets[None, :] * stride_xk
                )
                w_ptrs = (
                    w_ptr
                    + pid_g * stride_wg
                    + k_offsets[:, None] * stride_wk
                    + offs_n[None, :] * stride_wn
                )
                x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (k_offsets[None, :] < K), other=0.0)
                scaled_x = x / x_scale[:, None]
                x_q = tl.where(
                    scaled_x >= 0,
                    tl.floor(scaled_x + 0.5),
                    tl.ceil(scaled_x - 0.5),
                )
                x_q = tl.clamp(x_q, -127.0, 127.0).to(tl.int8)
                w = tl.load(w_ptrs, mask=(k_offsets[:, None] < K) & (offs_n[None, :] < N), other=0)
                acc += tl.dot(x_q, w, out_dtype=tl.int32)

            w_scale_ptrs = w_scale_ptr + pid_g * stride_wsg + offs_n * stride_wsn
            bias_ptrs = bias_ptr + pid_g * stride_bg + offs_n * stride_bn
            w_scale = tl.load(w_scale_ptrs, mask=offs_n < N, other=1.0)
            bias = tl.load(bias_ptrs, mask=offs_n < N, other=0.0)

            out = acc.to(tl.float32) * x_scale[:, None] * w_scale[None, :] + bias[None, :]
            out = tl.maximum(out, 0.0)
            out_q = tl.clamp(tl.floor(out / q_scale[:, None] + 0.5), 0.0, 127.0).to(tl.int8)

            out_q_ptrs = (
                out_q_ptr
                + pid_g * stride_oqg
                + offs_m[:, None] * stride_oqm
                + offs_n[None, :] * stride_oqn
            )
            tl.store(out_q_ptrs, out_q, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class GroupedInt8LinearTriton(nn.Module):
    """
    Grouped INT8 Linear backed by a Triton grouped GEMM kernel.

    Weights are statically quantized per output channel. Activations are
    quantized per token on the fly; the Triton kernel performs grouped int8 GEMM
    with fused dequant + bias.
    """

    def __init__(
        self,
        weight_q_t: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("weight_q_t", weight_q_t.contiguous())
        self.register_buffer("weight_scale", weight_scale.contiguous())
        self.register_buffer("bias", bias.contiguous())

    @staticmethod
    def is_supported() -> bool:
        return TRITON_AVAILABLE

    @staticmethod
    def _quantize_weight_per_output_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = weight.abs().amax(dim=-1).clamp_min(1e-5) / 127.0
        quant = torch.clamp((weight / scale.unsqueeze(-1)).round(), -127, 127).to(torch.int8)
        return quant.transpose(1, 2).contiguous(), scale.contiguous()

    @staticmethod
    def _quantize_activation_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = x.abs().amax(dim=-1).clamp_min(1e-5) / 127.0
        quant = torch.clamp((x / scale.unsqueeze(-1)).round(), -127, 127).to(torch.int8)
        return quant.contiguous(), scale.contiguous()

    @classmethod
    def from_linears(cls, linears: Sequence[nn.Linear]):
        modules = list(linears)
        if not modules:
            raise ValueError("linears must not be empty")
        first_in = int(modules[0].in_features)
        first_out = int(modules[0].out_features)
        if any(int(m.in_features) != first_in or int(m.out_features) != first_out for m in modules):
            raise ValueError("All grouped linears must share the same input/output shape")

        weight = torch.stack([module.weight.detach().to(torch.float32) for module in modules], dim=0)
        bias = torch.stack([module.bias.detach().to(torch.float32) for module in modules], dim=0)
        weight_q_t, weight_scale = cls._quantize_weight_per_output_channel(weight)
        return cls(weight_q_t=weight_q_t, weight_scale=weight_scale, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available in this environment")
        if x.device.type != "cuda":
            raise RuntimeError("GroupedInt8LinearTriton requires CUDA")
        if x.ndim < 3:
            raise ValueError(f"Expected x.ndim >= 3, got shape {tuple(x.shape)}")
        if x.shape[0] != self.weight_q_t.shape[0]:
            raise ValueError(
                f"Leading dim G must match grouped weight count: {x.shape[0]} vs {self.weight_q_t.shape[0]}"
            )

        batch_shape = tuple(int(v) for v in x.shape[1:-1])
        x_q, x_scale = self.quantize_activation(x)
        return self.forward_prequantized(x_q, x_scale, batch_shape=batch_shape)

    def quantize_activation(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.device.type != "cuda":
            raise RuntimeError("GroupedInt8LinearTriton requires CUDA")
        group_count = int(x.shape[0])
        batch_shape = tuple(int(v) for v in x.shape[1:-1])
        in_dim = int(x.shape[-1])
        flat_m = _prod(batch_shape)
        x_flat = x.reshape(group_count, flat_m, in_dim).to(torch.float32)
        return self._quantize_activation_per_token(x_flat)

    def forward_prequantized(
        self,
        x_q: torch.Tensor,
        x_scale: torch.Tensor,
        *,
        batch_shape: tuple[int, ...] | None = None,
        out_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available in this environment")
        if x_q.device.type != "cuda":
            raise RuntimeError("GroupedInt8LinearTriton requires CUDA")
        if x_q.ndim != 3:
            raise ValueError(f"Expected prequantized x_q.ndim == 3, got {tuple(x_q.shape)}")

        group_count = int(x_q.shape[0])
        flat_m = int(x_q.shape[1])
        in_dim = int(x_q.shape[2])
        out_dim = int(self.weight_q_t.shape[-1])
        if batch_shape is None:
            batch_shape = (flat_m,)

        weight_q_t = self.weight_q_t.to(device=x_q.device)
        weight_scale = self.weight_scale.to(device=x_q.device, dtype=torch.float32)
        bias = self.bias.to(device=x_q.device, dtype=torch.float32)
        out = torch.empty((group_count, flat_m, out_dim), device=x_q.device, dtype=torch.float32)

        grid = lambda meta: (
            triton.cdiv(flat_m, meta["BLOCK_M"]),
            triton.cdiv(out_dim, meta["BLOCK_N"]),
            group_count,
        )
        grouped_int8_gemm_kernel[grid](
            x_q,
            weight_q_t,
            out,
            x_scale,
            weight_scale,
            bias,
            flat_m,
            out_dim,
            in_dim,
            x_q.stride(0),
            x_q.stride(1),
            x_q.stride(2),
            weight_q_t.stride(0),
            weight_q_t.stride(1),
            weight_q_t.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            x_scale.stride(0),
            x_scale.stride(1),
            weight_scale.stride(0),
            weight_scale.stride(1),
            bias.stride(0),
            bias.stride(1),
        )
        return out.to(dtype=out_dtype).reshape(group_count, *batch_shape, out_dim)

    def forward_prequantized_relu_requant(
        self,
        x_q: torch.Tensor,
        x_scale: torch.Tensor,
        *,
        batch_shape: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available in this environment")
        if x_q.device.type != "cuda":
            raise RuntimeError("GroupedInt8LinearTriton requires CUDA")
        if x_q.ndim != 3:
            raise ValueError(f"Expected prequantized x_q.ndim == 3, got {tuple(x_q.shape)}")

        group_count = int(x_q.shape[0])
        flat_m = int(x_q.shape[1])
        in_dim = int(x_q.shape[2])
        out_dim = int(self.weight_q_t.shape[-1])
        if batch_shape is None:
            batch_shape = (flat_m,)

        weight_q_t = self.weight_q_t.to(device=x_q.device)
        weight_scale = self.weight_scale.to(device=x_q.device, dtype=torch.float32)
        bias = self.bias.to(device=x_q.device, dtype=torch.float32)
        out_q = torch.empty((group_count, flat_m, out_dim), device=x_q.device, dtype=torch.int8)
        out_scale = torch.empty((group_count, flat_m), device=x_q.device, dtype=torch.float32)

        grid = lambda meta: (
            triton.cdiv(flat_m, meta["BLOCK_M"]),
            group_count,
        )
        grouped_int8_gemm_relu_requant_kernel[grid](
            x_q,
            weight_q_t,
            out_q,
            out_scale,
            x_scale,
            weight_scale,
            bias,
            flat_m,
            out_dim,
            in_dim,
            x_q.stride(0),
            x_q.stride(1),
            x_q.stride(2),
            weight_q_t.stride(0),
            weight_q_t.stride(1),
            weight_q_t.stride(2),
            out_q.stride(0),
            out_q.stride(1),
            out_q.stride(2),
            out_scale.stride(0),
            out_scale.stride(1),
            x_scale.stride(0),
            x_scale.stride(1),
            weight_scale.stride(0),
            weight_scale.stride(1),
            bias.stride(0),
            bias.stride(1),
        )
        return (
            out_q.reshape(group_count, *batch_shape, out_dim),
            out_scale.reshape(group_count, *batch_shape),
        )

    def forward_fused_input_relu_requant(
        self,
        x: torch.Tensor,
        *,
        batch_shape: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available in this environment")
        if x.device.type != "cuda":
            raise RuntimeError("GroupedInt8LinearTriton requires CUDA")
        if x.ndim < 3:
            raise ValueError(f"Expected x.ndim >= 3, got shape {tuple(x.shape)}")
        if x.shape[0] != self.weight_q_t.shape[0]:
            raise ValueError(
                f"Leading dim G must match grouped weight count: {x.shape[0]} vs {self.weight_q_t.shape[0]}"
            )

        group_count = int(x.shape[0])
        batch_shape = tuple(int(v) for v in x.shape[1:-1]) if batch_shape is None else batch_shape
        in_dim = int(x.shape[-1])
        flat_m = _prod(batch_shape)
        out_dim = int(self.weight_q_t.shape[-1])

        x_flat = x.reshape(group_count, flat_m, in_dim).to(torch.float32)
        weight_q_t = self.weight_q_t.to(device=x.device)
        weight_scale = self.weight_scale.to(device=x.device, dtype=torch.float32)
        bias = self.bias.to(device=x.device, dtype=torch.float32)
        out_q = torch.empty((group_count, flat_m, out_dim), device=x.device, dtype=torch.int8)
        out_scale = torch.empty((group_count, flat_m), device=x.device, dtype=torch.float32)

        grid = lambda meta: (
            triton.cdiv(flat_m, meta["BLOCK_M"]),
            group_count,
        )
        grouped_fp_input_int8_gemm_relu_requant_kernel[grid](
            x_flat,
            weight_q_t,
            out_q,
            out_scale,
            weight_scale,
            bias,
            flat_m,
            out_dim,
            in_dim,
            x_flat.stride(0),
            x_flat.stride(1),
            x_flat.stride(2),
            weight_q_t.stride(0),
            weight_q_t.stride(1),
            weight_q_t.stride(2),
            out_q.stride(0),
            out_q.stride(1),
            out_q.stride(2),
            out_scale.stride(0),
            out_scale.stride(1),
            weight_scale.stride(0),
            weight_scale.stride(1),
            bias.stride(0),
            bias.stride(1),
        )
        return (
            out_q.reshape(group_count, *batch_shape, out_dim),
            out_scale.reshape(group_count, *batch_shape),
        )

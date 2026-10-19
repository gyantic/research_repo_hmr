"""Grouped Linear reference/Triton kernels used only by tools/benchmark_grouped_bbox_int8.py.

Not wired into the SAT-HMR model (the grouped decoder/bbox paths that used them were removed).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def _prod(values: Sequence[int]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def loop_grouped_linear_forward(linears: Sequence[nn.Module], x: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation using the original per-linear loop.

    Args:
        linears: sequence of Linear modules
        x: tensor shaped [G, ..., C], where G == len(linears)
    """
    if x.ndim < 2:
        raise ValueError(f"Expected x.ndim >= 2, got shape {tuple(x.shape)}")
    if x.shape[0] != len(linears):
        raise ValueError(
            f"Leading dim G must match len(linears): {x.shape[0]} vs {len(linears)}"
        )
    return torch.stack([linears[i](x[i]) for i in range(len(linears))], dim=0)


class GroupedLinearReference(nn.Module):
    """
    Grouped reference implementation for same-shape Linear modules.

    It stacks one weight/bias pair per group element and applies grouped batched
    matmul across the leading G dimension.
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, impl: str = "bmm") -> None:
        super().__init__()
        self.impl = impl
        self.register_buffer("weight", weight.contiguous())
        self.register_buffer("bias", bias.contiguous())

    @classmethod
    def from_linears(cls, linears: Sequence[nn.Linear], impl: str = "bmm"):
        modules = list(linears)
        if not modules:
            raise ValueError("linears must not be empty")
        first_in = int(modules[0].in_features)
        first_out = int(modules[0].out_features)
        if any(int(m.in_features) != first_in or int(m.out_features) != first_out for m in modules):
            raise ValueError("All grouped linears must share the same input/output shape")
        weight = torch.stack([module.weight.detach() for module in modules], dim=0)
        bias = torch.stack([module.bias.detach() for module in modules], dim=0)
        return cls(weight=weight, bias=bias, impl=impl)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [G, ..., K]
        Returns:
            y: [G, ..., N]
        """
        if x.ndim < 3:
            raise ValueError(f"Expected x.ndim >= 3, got shape {tuple(x.shape)}")
        if x.shape[0] != self.weight.shape[0]:
            raise ValueError(
                f"Leading dim G must match grouped weight count: {x.shape[0]} vs {self.weight.shape[0]}"
            )

        group_count = int(x.shape[0])
        batch_shape = tuple(int(v) for v in x.shape[1:-1])
        in_dim = int(x.shape[-1])
        flat_m = _prod(batch_shape)

        y = x.reshape(group_count, flat_m, in_dim)
        weight = self.weight.to(dtype=y.dtype, device=y.device)
        bias = self.bias.to(dtype=y.dtype, device=y.device)

        if self.impl == "bmm":
            y = torch.bmm(y, weight.transpose(1, 2)) + bias[:, None, :]
        elif self.impl == "einsum":
            y = torch.einsum("gmk,gnk->gmn", y, weight) + bias[:, None, :]
        else:
            raise ValueError(f"Unsupported impl: {self.impl}")

        out_dim = int(y.shape[-1])
        return y.reshape(group_count, *batch_shape, out_dim)


class GroupedInt8LinearReference(nn.Module):
    """
    Grouped INT8 reference for same-shape Linear modules.

    Weights are pre-quantized once per output channel. Activations are quantized
    per token at runtime, then multiplied with `torch._int_mm` batched via
    `torch.vmap`.
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
        """
        Args:
            x: [G, ..., K]
        Returns:
            y: [G, ..., N]
        """
        if x.ndim < 3:
            raise ValueError(f"Expected x.ndim >= 3, got shape {tuple(x.shape)}")
        if x.shape[0] != self.weight_q_t.shape[0]:
            raise ValueError(
                f"Leading dim G must match grouped weight count: {x.shape[0]} vs {self.weight_q_t.shape[0]}"
            )

        group_count = int(x.shape[0])
        batch_shape = tuple(int(v) for v in x.shape[1:-1])
        in_dim = int(x.shape[-1])
        flat_m = _prod(batch_shape)

        x_flat = x.reshape(group_count, flat_m, in_dim).to(torch.float32)
        x_q, x_scale = self._quantize_activation_per_token(x_flat)

        weight_q_t = self.weight_q_t.to(device=x.device)
        weight_scale = self.weight_scale.to(device=x.device, dtype=torch.float32)
        bias = self.bias.to(device=x.device, dtype=torch.float32)

        # Batched grouped int32 GEMM.
        y_int = torch.vmap(torch._int_mm)(x_q, weight_q_t)
        y = y_int.to(torch.float32) * x_scale.unsqueeze(-1) * weight_scale[:, None, :]
        y = y + bias[:, None, :]
        out_dim = int(y.shape[-1])
        return y.to(dtype=x.dtype).reshape(group_count, *batch_shape, out_dim)

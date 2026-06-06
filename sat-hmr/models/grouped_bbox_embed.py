from __future__ import annotations

import math
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .grouped_linear import GroupedLinearReference
from .triton_grouped_int8 import GroupedInt8LinearTriton


def _prod(values: Sequence[int]) -> int:
    out = 1
    for v in values:
        out *= int(v)
    return out


def loop_bbox_embed_forward(bbox_embed: Sequence[nn.Module], x: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation using the original per-MLP loop.

    Args:
        bbox_embed: sequence of MLP modules, one per decoder layer
        x: tensor shaped [G, ..., C], where G == len(bbox_embed)
    """
    if x.ndim < 2:
        raise ValueError(f"Expected x.ndim >= 2, got shape {tuple(x.shape)}")
    if x.shape[0] != len(bbox_embed):
        raise ValueError(
            f"Leading dim G must match len(bbox_embed): {x.shape[0]} vs {len(bbox_embed)}"
        )
    return torch.stack([bbox_embed[i](x[i]) for i in range(len(bbox_embed))], dim=0)


class GroupedBBoxEmbedReference(nn.Module):
    """
    Grouped reference implementation for decoder.bbox_embed.

    It keeps one stacked weight/bias tensor per MLP stage and applies either
    batched matmul or einsum across the grouped dimension G.
    """

    def __init__(
        self,
        stage_weights: Iterable[torch.Tensor],
        stage_biases: Iterable[torch.Tensor],
        impl: str = "bmm",
    ) -> None:
        super().__init__()
        self.impl = impl

        weights = list(stage_weights)
        biases = list(stage_biases)
        if len(weights) != len(biases):
            raise ValueError("stage_weights and stage_biases must have the same length")
        if not weights:
            raise ValueError("At least one stage is required")

        self.stage_weights = nn.ParameterList(
            [nn.Parameter(w.contiguous(), requires_grad=False) for w in weights]
        )
        self.stage_biases = nn.ParameterList(
            [nn.Parameter(b.contiguous(), requires_grad=False) for b in biases]
        )

    @classmethod
    def from_bbox_embed(cls, bbox_embed: Sequence[nn.Module], impl: str = "bmm"):
        mlps = list(bbox_embed)
        if not mlps:
            raise ValueError("bbox_embed must not be empty")
        num_layers = len(mlps[0].layers)
        if any(len(mlp.layers) != num_layers for mlp in mlps):
            raise ValueError("All bbox_embed MLPs must have the same number of layers")

        stage_weights: List[torch.Tensor] = []
        stage_biases: List[torch.Tensor] = []
        for stage_idx in range(num_layers):
            stage_weights.append(torch.stack([mlp.layers[stage_idx].weight.detach() for mlp in mlps], dim=0))
            stage_biases.append(torch.stack([mlp.layers[stage_idx].bias.detach() for mlp in mlps], dim=0))
        return cls(stage_weights, stage_biases, impl=impl)

    def _grouped_linear(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        if self.impl == "bmm":
            return torch.bmm(x, weight.transpose(1, 2)) + bias[:, None, :]
        if self.impl == "einsum":
            return torch.einsum("gmk,gnk->gmn", x, weight) + bias[:, None, :]
        raise ValueError(f"Unsupported impl: {self.impl}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [G, ..., K]
        Returns:
            y: [G, ..., out_dim]
        """
        if x.ndim < 3:
            raise ValueError(f"Expected x.ndim >= 3, got shape {tuple(x.shape)}")
        group_count = x.shape[0]
        batch_shape = tuple(int(v) for v in x.shape[1:-1])
        in_dim = int(x.shape[-1])
        flat_m = _prod(batch_shape)

        y = x.reshape(group_count, flat_m, in_dim)
        for stage_idx, (weight, bias) in enumerate(zip(self.stage_weights, self.stage_biases)):
            y = self._grouped_linear(y, weight.to(dtype=y.dtype, device=y.device), bias.to(dtype=y.dtype, device=y.device))
            if stage_idx < len(self.stage_weights) - 1:
                y = F.relu(y)

        out_dim = int(y.shape[-1])
        return y.reshape(group_count, *batch_shape, out_dim)


class GroupedBBoxEmbedFusedInt8(nn.Module):
    """
    Grouped bbox_embed implementation with:
    - stage 0 (`768->768`): Triton grouped INT8 with fused input quant + ReLU + requant
    - stage 1 (`768->768`): Triton grouped INT8 consuming prequantized stage-0 output
    - stage 2 (`768->4`): grouped FP16/FP32 projection

    This matches the current strongest microbenchmark path for bbox hidden layers
    while keeping the final narrow head simple and numerically stable.
    """

    def __init__(
        self,
        stage0_int8: GroupedInt8LinearTriton,
        stage1_int8: GroupedInt8LinearTriton,
        stage2_fp: GroupedLinearReference,
    ) -> None:
        super().__init__()
        self.stage0_int8 = stage0_int8
        self.stage1_int8 = stage1_int8
        self.stage2_fp = stage2_fp

    @classmethod
    def from_bbox_embed(cls, bbox_embed: Sequence[nn.Module]):
        mlps = list(bbox_embed)
        if not mlps:
            raise ValueError("bbox_embed must not be empty")
        if not GroupedInt8LinearTriton.is_supported():
            raise RuntimeError("GroupedBBoxEmbedFusedInt8 requires Triton support")
        num_layers = len(mlps[0].layers)
        if num_layers != 3:
            raise ValueError(f"Expected bbox_embed MLPs with 3 layers, got {num_layers}")
        if any(len(mlp.layers) != num_layers for mlp in mlps):
            raise ValueError("All bbox_embed MLPs must have the same number of layers")

        stage0 = GroupedInt8LinearTriton.from_linears([mlp.layers[0] for mlp in mlps])
        stage1 = GroupedInt8LinearTriton.from_linears([mlp.layers[1] for mlp in mlps])
        stage2 = GroupedLinearReference.from_linears([mlp.layers[2] for mlp in mlps], impl="bmm")
        return cls(stage0_int8=stage0, stage1_int8=stage1, stage2_fp=stage2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 3:
            raise ValueError(f"Expected x.ndim >= 3, got shape {tuple(x.shape)}")
        batch_shape = tuple(int(v) for v in x.shape[1:-1])

        y_q, y_scale = self.stage0_int8.forward_fused_input_relu_requant(
            x,
            batch_shape=batch_shape,
        )
        y_q = y_q.reshape(y_q.shape[0], -1, y_q.shape[-1])
        y_scale = y_scale.reshape(y_scale.shape[0], -1)
        y = self.stage1_int8.forward_prequantized(
            y_q,
            y_scale,
            batch_shape=batch_shape,
            out_dtype=x.dtype,
        )
        y = F.relu(y)
        return self.stage2_fp(y)

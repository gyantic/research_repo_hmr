from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class PatchConv2dAsLinear(nn.Module):
    def __init__(self, conv: nn.Conv2d) -> None:
        super().__init__()
        if conv.groups != 1:
            raise ValueError("Only groups=1 Conv2d is supported")

        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = tuple(conv.kernel_size)
        self.stride = tuple(conv.stride)
        self.padding = tuple(conv.padding)
        self.dilation = tuple(conv.dilation)
        self.groups = conv.groups

        in_features = conv.in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.linear = nn.Linear(in_features, conv.out_channels, bias=conv.bias is not None)
        self.linear = self.linear.to(device=conv.weight.device, dtype=conv.weight.dtype)
        with torch.no_grad():
            self.linear.weight.copy_(conv.weight.reshape(conv.out_channels, -1))
            if conv.bias is not None:
                self.linear.bias.copy_(conv.bias)

    def _output_hw(self, height: int, width: int) -> Tuple[int, int]:
        out_h = (
            height
            + 2 * self.padding[0]
            - self.dilation[0] * (self.kernel_size[0] - 1)
            - 1
        ) // self.stride[0] + 1
        out_w = (
            width
            + 2 * self.padding[1]
            - self.dilation[1] * (self.kernel_size[1] - 1)
            - 1
        ) // self.stride[1] + 1
        return int(out_h), int(out_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        ).transpose(1, 2)
        out = self.linear(patches)
        out_h, out_w = self._output_hw(height, width)
        return out.transpose(1, 2).reshape(batch, self.out_channels, out_h, out_w)


def _prefix_matches(name: str, prefixes: Sequence[str]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def _is_patch_projection_conv(module: nn.Module) -> bool:
    return isinstance(module, nn.Conv2d) and module.groups == 1 and module.kernel_size == module.stride


def convert_patch_convs_to_linear(model: nn.Module, prefixes: Sequence[str]) -> Iterable[str]:
    converted = []

    def _convert(parent: nn.Module, name_prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            fq_name = f"{name_prefix}{child_name}"
            if _is_patch_projection_conv(child) and _prefix_matches(fq_name, prefixes):
                setattr(parent, child_name, PatchConv2dAsLinear(child))
                converted.append(fq_name)
                continue
            _convert(child, fq_name + ".")

    _convert(model)
    return converted


def count_patch_projection_modules(model: nn.Module) -> Tuple[int, int]:
    total = 0
    converted = 0
    for _, module in model.named_modules():
        if isinstance(module, PatchConv2dAsLinear):
            total += 1
            converted += 1
        elif _is_patch_projection_conv(module):
            total += 1
    return total, converted

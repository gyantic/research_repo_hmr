"""
Static W8A8 fused MLP/FFN kernels for SAT-HMR hot path.

Replaces torchao dynamic INT8 on encoder DINO MLPs and decoder FFNs.
- Static (calibrated scalar) activation scale → eliminates per-token absmax reduction.
- Per-output-channel weight scale.
- 2 Triton kernels per MLP:
    K1: bf16 x → int8 quant → int8 GEMM (fc1) → dequant + bias + GELU/ReLU → int8 quant
    K2: int8 h → int8 GEMM (fc2) → dequant + bias → bf16 y
- Intermediate stays as int8 (no bf16 round-trip).
"""

from __future__ import annotations

from typing import Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

if TRITON_AVAILABLE:

    @triton.jit
    def _static_int8_fc1_act_quant_kernel(
        x_ptr,           # bf16 (M, K)
        w_ptr,           # int8 (K, N) — pre-transposed
        h_ptr,           # int8 (M, N) output
        w_scale_ptr,     # fp32 (N,)
        b_ptr,           # fp32 (N,) bias
        input_scale,     # fp32 scalar (kernel arg)
        inter_scale,     # fp32 scalar (kernel arg)
        M, N, K,
        stride_xm, stride_xk,
        stride_wk, stride_wn,
        stride_hm, stride_hn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        ACTIVATION: tl.constexpr,  # 0 = gelu (tanh approx), 1 = relu
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        m_mask = offs_m < M
        n_mask = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        inv_input_scale = 1.0 / input_scale

        for k_start in range(0, K, BLOCK_K):
            k_off = k_start + offs_k
            k_mask = k_off < K

            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_off[None, :] * stride_xk
            x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

            # Static activation quant: x_int8 = clamp(round(x * inv_input_scale), -127, 127)
            xq = x.to(tl.float32) * inv_input_scale
            xq = tl.extra.cuda.libdevice.rint(xq)
            xq = tl.minimum(tl.maximum(xq, -127.0), 127.0)
            x_int = xq.to(tl.int8)

            w_ptrs = w_ptr + k_off[:, None] * stride_wk + offs_n[None, :] * stride_wn
            w_int = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)

            acc += tl.dot(x_int, w_int, out_dtype=tl.int32)

        w_scale = tl.load(w_scale_ptr + offs_n, mask=n_mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)

        # Dequant: int32 → fp32 with scales
        out = acc.to(tl.float32) * (input_scale * w_scale[None, :]) + b[None, :]

        # Activation
        if ACTIVATION == 0:
            # GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            x3 = out * out * out
            inner = 0.7978845608028654 * (out + 0.044715 * x3)
            t = tl.extra.cuda.libdevice.tanh(inner)
            out = 0.5 * out * (1.0 + t)
        else:
            # ReLU
            out = tl.maximum(out, 0.0)

        # Static requant
        inv_inter_scale = 1.0 / inter_scale
        oq = out * inv_inter_scale
        oq = tl.extra.cuda.libdevice.rint(oq)
        oq = tl.minimum(tl.maximum(oq, -127.0), 127.0)
        h_out = oq.to(tl.int8)

        h_ptrs = h_ptr + offs_m[:, None] * stride_hm + offs_n[None, :] * stride_hn
        tl.store(h_ptrs, h_out, mask=m_mask[:, None] & n_mask[None, :])


    @triton.jit
    def _static_int8_fc2_kernel(
        h_ptr,           # int8 (M, K)
        w_ptr,           # int8 (K, N) — pre-transposed
        y_ptr,           # bf16 (M, N) output
        w_scale_ptr,     # fp32 (N,)
        b_ptr,           # fp32 (N,)
        inter_scale,     # fp32 scalar
        M, N, K,
        stride_hm, stride_hk,
        stride_wk, stride_wn,
        stride_ym, stride_yn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        m_mask = offs_m < M
        n_mask = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

        for k_start in range(0, K, BLOCK_K):
            k_off = k_start + offs_k
            k_mask = k_off < K

            h_ptrs = h_ptr + offs_m[:, None] * stride_hm + k_off[None, :] * stride_hk
            w_ptrs = w_ptr + k_off[:, None] * stride_wk + offs_n[None, :] * stride_wn
            h = tl.load(h_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0)
            w = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)

            acc += tl.dot(h, w, out_dtype=tl.int32)

        w_scale = tl.load(w_scale_ptr + offs_n, mask=n_mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)

        out = acc.to(tl.float32) * (inter_scale * w_scale[None, :]) + b[None, :]

        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        tl.store(y_ptrs, out.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


def _materialize_weight(weight: torch.Tensor) -> torch.Tensor:
    """Materialize a torch.Tensor weight that may be wrapped by torchao quantization.

    torchao の LinearActivationQuantizedTensor / AffineQuantizedTensor は .abs() 等を
    実装していないので、dequantize して plain bf16/fp32 に戻す。
    """
    w = weight
    # torchao の AffineQuantizedTensor / LinearActivationQuantizedTensor は dequantize() を持つ
    if hasattr(w, "dequantize"):
        try:
            w = w.dequantize()
        except Exception:
            pass
    # LinearActivationQuantizedTensor は中に original_weight_tensor を持つ
    if hasattr(w, "original_weight_tensor"):
        try:
            inner = w.original_weight_tensor
            if hasattr(inner, "dequantize"):
                inner = inner.dequantize()
            w = inner
        except Exception:
            pass
    if not isinstance(w, torch.Tensor) or type(w) is not torch.Tensor:
        # Tensor subclass のままなら、.to() で plain tensor 化を試す
        try:
            w = w.to(dtype=w.dtype, copy=True).contiguous()
        except Exception:
            pass
    return w.detach()


def _quantize_weight_per_output_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Static per-output-channel symmetric quant. weight shape (out, in)."""
    w = _materialize_weight(weight).to(torch.float32)
    scale = w.abs().amax(dim=1).clamp_min(1e-8) / 127.0  # (out,)
    q = torch.clamp((w / scale.unsqueeze(1)).round(), -127, 127).to(torch.int8)
    return q, scale


class StaticInt8FusedMLP(nn.Module):
    """fc1 -> activation -> fc2 with static (calibrated) W8A8 quant."""

    def __init__(
        self,
        fc1: nn.Linear,
        fc2: nn.Linear,
        input_scale: float,
        intermediate_scale: float,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if activation not in ("gelu", "relu"):
            raise ValueError(f"unsupported activation: {activation}")
        if fc1.in_features != fc2.out_features:
            raise ValueError("expected fc1.in_features == fc2.out_features for residual MLP")
        if fc1.out_features != fc2.in_features:
            raise ValueError("expected fc1.out_features == fc2.in_features (hidden dim consistency)")
        if fc1.bias is None or fc2.bias is None:
            raise ValueError("StaticInt8FusedMLP requires biases on fc1 and fc2")

        # Static per-output-channel weight quant
        w1_q, w1_s = _quantize_weight_per_output_channel(fc1.weight)  # (hidden, in), (hidden,)
        w2_q, w2_s = _quantize_weight_per_output_channel(fc2.weight)  # (in, hidden), (in,)

        # Pre-transpose for tl.dot: kernel needs weight shape (K, N)
        # Kernel 1: x (M, in) @ w1_q.T (in, hidden) → h (M, hidden)
        # Kernel 2: h (M, hidden) @ w2_q.T (hidden, in) → y (M, in)
        self.register_buffer("w1_q_t", w1_q.t().contiguous())  # (in, hidden)
        self.register_buffer("w2_q_t", w2_q.t().contiguous())  # (hidden, in)
        self.register_buffer("w1_scale", w1_s.to(torch.float32).contiguous())
        self.register_buffer("w2_scale", w2_s.to(torch.float32).contiguous())
        self.register_buffer("b1", fc1.bias.detach().to(torch.float32).contiguous())
        self.register_buffer("b2", fc2.bias.detach().to(torch.float32).contiguous())

        self.input_scale = float(input_scale)
        self.inter_scale = float(intermediate_scale)
        self.in_features = int(fc1.in_features)
        self.hidden_features = int(fc1.out_features)
        self.activation = activation
        self._act_id = 0 if activation == "gelu" else 1

    @classmethod
    def is_supported(cls) -> bool:
        return TRITON_AVAILABLE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _static_int8_fused_mlp_call(self, x)


@torch._dynamo.allow_in_graph
def _static_int8_fused_mlp_call(self: "StaticInt8FusedMLP", x: torch.Tensor) -> torch.Tensor:
    """allow_in_graph で Dynamo に opaque op として graph に取り込ませる。
    内部で Triton kernel を呼ぶが Dynamo はトレースしない (graph 内の単一 node 扱い)。
    これで前後の LayerNorm / residual add / dropout の Inductor fusion が維持される。"""
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton not available")
    if x.device.type != "cuda":
        raise RuntimeError("StaticInt8FusedMLP requires CUDA")

    orig_shape = x.shape
    x_flat = x.reshape(-1, self.in_features).contiguous()
    M, K1 = x_flat.shape
    K2 = self.hidden_features

    # Fixed Triton block sizes (autotune は M 変動で再 tune が走り壊滅的に遅くなるので使わない)
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 64

    # Phase 1: bf16 → int8
    h_int = torch.empty((M, K2), dtype=torch.int8, device=x.device)
    grid1 = (triton.cdiv(M, BLOCK_M), triton.cdiv(K2, BLOCK_N))
    _static_int8_fc1_act_quant_kernel[grid1](
        x_flat, self.w1_q_t, h_int,
        self.w1_scale, self.b1,
        self.input_scale, self.inter_scale,
        M, K2, K1,
        x_flat.stride(0), x_flat.stride(1),
        self.w1_q_t.stride(0), self.w1_q_t.stride(1),
        h_int.stride(0), h_int.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        ACTIVATION=self._act_id,
        num_warps=8, num_stages=3,
    )

    # Phase 2: int8 → bf16
    y = torch.empty((M, K1), dtype=x.dtype, device=x.device)
    grid2 = (triton.cdiv(M, BLOCK_M), triton.cdiv(K1, BLOCK_N))
    _static_int8_fc2_kernel[grid2](
        h_int, self.w2_q_t, y,
        self.w2_scale, self.b2,
        self.inter_scale,
        M, K1, K2,
        h_int.stride(0), h_int.stride(1),
        self.w2_q_t.stride(0), self.w2_q_t.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8, num_stages=3,
    )

    return y.reshape(*orig_shape[:-1], K1)


# ---------------------------------------------------------------------------
# Calibration & replacement utilities
# ---------------------------------------------------------------------------


class _MlpScaleCollector:
    """Collect input absmax and post-(fc1+act) absmax via forward pre-hooks."""

    def __init__(self, fc1: nn.Linear, activation: str) -> None:
        self.fc1 = fc1
        self.activation = activation
        self.input_max = 0.0
        self.inter_max = 0.0

    def __call__(self, module: nn.Module, inputs: tuple) -> None:
        if not inputs:
            return
        x = inputs[0]
        if not torch.is_tensor(x):
            return
        # Input to fc1: max abs over the entire tensor
        xm = float(x.detach().abs().max().item())
        if xm > self.input_max:
            self.input_max = xm
        # Compute fc1+activation manually via F.linear (avoids hook recursion when fc1 IS module).
        # Note: torchao が weight を量子化済みの場合、F.linear が dispatch を経由するので
        # forward を直接呼ばないと weight 復元できないケースがある。その場合は元の fc1.forward を一時的に呼び出す。
        with torch.no_grad():
            try:
                w = self.fc1.weight
                b = self.fc1.bias
                h = F.linear(x, w, b)
            except Exception:
                # Fallback: invoke forward directly bypassing the registered hook
                h = nn.Linear.forward(self.fc1, x)
            if self.activation == "gelu":
                h = F.gelu(h)
            elif self.activation == "relu":
                h = F.relu(h)
        hm = float(h.abs().max().item())
        if hm > self.inter_max:
            self.inter_max = hm

    def to_scales(self) -> tuple[float, float]:
        # Symmetric quant: abs_max / 127 (clamp to avoid div-by-zero)
        return (
            max(self.input_max, 1e-8) / 127.0,
            max(self.inter_max, 1e-8) / 127.0,
        )


class CalibrationHooks:
    """Context manager that registers forward pre-hooks for calibration."""

    def __init__(self, targets: Iterable[tuple[str, nn.Module, nn.Module, nn.Linear, nn.Linear, str]]) -> None:
        # targets: list of (key, hook_target, swap_owner, fc1, fc2, activation)
        self.targets = list(targets)
        self.collectors: dict[str, _MlpScaleCollector] = {}
        self._handles: list = []

    def __enter__(self):
        for key, hook_target, _swap_owner, fc1, _fc2, act in self.targets:
            collector = _MlpScaleCollector(fc1, act)
            self.collectors[key] = collector
            handle = hook_target.register_forward_pre_hook(collector)
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def scales(self) -> dict[str, tuple[float, float]]:
        return {k: v.to_scales() for k, v in self.collectors.items()}


def enumerate_hot_targets(model: nn.Module) -> list[tuple[str, nn.Module, nn.Module, nn.Linear, nn.Linear, str]]:
    """Enumerate (key, hook_target, swap_owner, fc1, fc2, activation) for all hot-path MLP/FFN.

    hook_target: module on which to register a forward pre-hook (its forward sees the FFN input).
    swap_owner: module whose .fc1/.fc2 (encoder) or .linear1/.linear2 (decoder) get replaced.

    - Encoder: hook_target = swap_owner = Mlp instance.
    - Decoder: hook_target = layer.linear1 (called with positional arg), swap_owner = XformerDecoderLayer.
    """
    targets: list[tuple[str, nn.Module, nn.Module, nn.Linear, nn.Linear, str]] = []

    # Encoder: dinov2 Mlp instances
    try:
        from models.encoders.dinov2.layers.mlp import Mlp as DinoMlp
    except Exception:
        DinoMlp = None

    if DinoMlp is not None:
        for name, mod in model.named_modules():
            if isinstance(mod, DinoMlp) and isinstance(mod.fc1, nn.Linear) and isinstance(mod.fc2, nn.Linear):
                targets.append((f"encoder/{name}", mod, mod, mod.fc1, mod.fc2, "gelu"))

    # Decoder: XformerDecoderLayer FFN (linear1 + activation + linear2)
    try:
        from models.decoder import XformerDecoderLayer
    except Exception:
        XformerDecoderLayer = None

    if XformerDecoderLayer is not None:
        for name, mod in model.named_modules():
            if isinstance(mod, XformerDecoderLayer):
                if isinstance(mod.linear1, nn.Linear) and isinstance(mod.linear2, nn.Linear):
                    act_fn = mod.activation
                    if act_fn is F.relu:
                        act_str = "relu"
                    elif act_fn is F.gelu:
                        act_str = "gelu"
                    else:
                        continue
                    targets.append((f"decoder/{name}", mod.linear1, mod, mod.linear1, mod.linear2, act_str))

    return targets


def apply_static_int8_fused(
    model: nn.Module,
    scales: Mapping[str, tuple[float, float]],
) -> int:
    """Replace hot MLPs with StaticInt8FusedMLP. Returns count of replaced modules."""
    if not StaticInt8FusedMLP.is_supported():
        raise RuntimeError("Triton not available; cannot apply StaticInt8FusedMLP")

    targets = enumerate_hot_targets(model)
    count = 0

    from models.encoders.dinov2.layers.mlp import Mlp as DinoMlp
    from models.decoder import XformerDecoderLayer

    for key, _hook_target, swap_owner, fc1, fc2, act in targets:
        if key not in scales:
            continue
        in_scale, inter_scale = scales[key]
        fused = StaticInt8FusedMLP(fc1, fc2, in_scale, inter_scale, activation=act)
        fused = fused.to(device=fc1.weight.device)

        if isinstance(swap_owner, DinoMlp):
            # Mlp.forward: fc1 → act → drop → fc2 → drop. fused が最終出力を返すので残りは identity。
            swap_owner.fc1 = fused
            swap_owner.act = nn.Identity()
            swap_owner.fc2 = nn.Identity()
            swap_owner.drop = nn.Identity()
            count += 1
        elif isinstance(swap_owner, XformerDecoderLayer):
            # FFN: linear2(dropout(activation(linear1(norm3(tgt))))) → fused が最終出力。残りは identity (dropout は eval で identity 同等)。
            swap_owner.linear1 = fused
            swap_owner.activation = nn.Identity()
            swap_owner.linear2 = nn.Identity()
            count += 1

    return count


# ---------------------------------------------------------------------------
# L3: compile-native static-scale INT8 Linear (no Triton, no _dynamo.disable)
# ---------------------------------------------------------------------------


class StaticScaleInt8Linear(nn.Module):
    """Static W8A8 Linear that lives inside torch.compile graph.

    - weight は per-output-channel で静的量子化 (init 時 1 回)
    - activation は **calibrated 静的 scalar scale** で量子化 (per-token absmax 不要)
    - 実装は torch native ops + torch._int_mm のみ — Inductor が graph に取り込む
    - `@_dynamo.disable` も custom Triton kernel も使わない
    - 目的は Option K の Inductor fusion を維持しつつ per-token absmax を消すこと
    """

    def __init__(self, linear: nn.Linear, input_scale: float) -> None:
        super().__init__()
        if linear.in_features % 8 != 0 or linear.out_features % 8 != 0:
            raise ValueError("torch._int_mm requires in/out features divisible by 8")
        w_q, w_s = _quantize_weight_per_output_channel(linear.weight)  # (out, in), (out,)
        self.register_buffer("weight_int8", w_q.contiguous())
        # Combined scale: input_scale * weight_scale (broadcast over output dim)
        # 計算は dequant 時に input_scale × weight_scale が必要。それぞれ別 buffer で持つ。
        out_dtype = linear.weight.dtype if linear.weight.dtype != torch.float32 else torch.bfloat16
        self.register_buffer("weight_scale", w_s.to(out_dtype).contiguous())
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().to(out_dtype).contiguous())
        else:
            self.bias = None
        self.register_buffer("input_scale", torch.tensor(float(input_scale), dtype=out_dtype))
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self._out_dtype = out_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features), bf16
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        # Static activation quant: x_int8 = clamp(round(x / input_scale), -127, 127).to(int8)
        # Note: division by static scalar is cheap, no reduction.
        x_scaled = x_flat / self.input_scale
        x_int8 = torch.clamp(torch.round(x_scaled), -127.0, 127.0).to(torch.int8)
        # int8 GEMM: (M, K) @ (N, K).T → (M, N) int32
        out_int = torch._int_mm(x_int8, self.weight_int8.t().contiguous())
        # Dequant: int32 → bf16 with combined scale
        out = out_int.to(self._out_dtype) * (self.input_scale * self.weight_scale)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], self.out_features)


class _LinearScaleCollector:
    """Collect input absmax for a single Linear via forward pre-hook."""

    def __init__(self) -> None:
        self.input_max = 0.0

    def __call__(self, module: nn.Module, inputs: tuple) -> None:
        if not inputs:
            return
        x = inputs[0]
        if not torch.is_tensor(x):
            return
        m = float(x.detach().abs().max().item())
        if m > self.input_max:
            self.input_max = m

    def to_scale(self) -> float:
        return max(self.input_max, 1e-8) / 127.0


def enumerate_decoder_ffn_linears(model: nn.Module) -> list[tuple[str, nn.Module, str]]:
    """Return list of (key, layer, attr_name) for decoder linear1/linear2."""
    targets: list[tuple[str, nn.Module, str]] = []
    try:
        from models.decoder import XformerDecoderLayer
    except Exception:
        return targets
    for name, mod in model.named_modules():
        if isinstance(mod, XformerDecoderLayer):
            for attr in ("linear1", "linear2"):
                lin = getattr(mod, attr, None)
                if isinstance(lin, nn.Linear):
                    targets.append((f"decoder/{name}/{attr}", mod, attr))
    return targets


class LinearCalibrationHooks:
    """Context manager that hooks individual Linear modules to capture input absmax."""

    def __init__(self, targets: Iterable[tuple[str, nn.Module, str]]) -> None:
        # targets: list of (key, layer, attr_name)
        self.targets = list(targets)
        self.collectors: dict[str, _LinearScaleCollector] = {}
        self._handles: list = []

    def __enter__(self):
        for key, layer, attr in self.targets:
            lin = getattr(layer, attr)
            collector = _LinearScaleCollector()
            self.collectors[key] = collector
            handle = lin.register_forward_pre_hook(collector)
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def scales(self) -> dict[str, float]:
        return {k: v.to_scale() for k, v in self.collectors.items()}


def apply_static_scale_int8_decoder_ffn(
    model: nn.Module,
    scales: Mapping[str, float],
) -> int:
    """Replace decoder linear1/linear2 with StaticScaleInt8Linear. Returns count."""
    targets = enumerate_decoder_ffn_linears(model)
    count = 0
    for key, layer, attr in targets:
        if key not in scales:
            continue
        lin = getattr(layer, attr)
        in_scale = scales[key]
        try:
            new = StaticScaleInt8Linear(lin, in_scale)
            new = new.to(device=lin.weight.device)
            setattr(layer, attr, new)
            count += 1
        except Exception as e:
            print(f"[WARN] StaticScaleInt8Linear replace failed for {key}: {e}")
    return count

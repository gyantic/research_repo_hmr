# SAT-HMR Precision Benchmark Summary

## モデルサイズ

| format | サイズ |
|--------|--------|
| FP32 (.pth) | 875 MB（約2.29億パラメータ）|
| FP16 TensorRT engine | 460 MB |
| INT8 TensorRT engine | 325 MB |

---

## 精度（3DPW test set、eval_batch_size=12）

| 設定 | Recall | MPJPE (mm) | PA-MPJPE (mm) | MVE (mm) | PA-MVE (mm) |
|------|--------|-----------|--------------|---------|------------|
| FP32 | 1.00 | 63.6 | 41.6 | 73.7 | 53.4 |
| BF16 | 0.99 | 65.1 | 42.3 | 74.9 | 54.0 |
| TorchAO INT8 | 0.99 | 65.7 | 42.4 | 75.5 | 54.2 |

> 出典: results/eval_3dpw/, results/eval_3dpw_bf16/, results/eval_3dpw_torchao_int8/

---

## TensorRT（`video_demo_trt.py`）

計測: 2026-02-26 / 出典: docs/int8_spec.md

| precision | 推論時間 | FPS (no_udp) | FPS (UDP込み) |
|-----------|---------|-------------|--------------|
| FP32 | 52.32 ms | 14.14 fps | 9.04 fps |
| FP16 | 13.11 ms | 41.91 fps | 30.53 fps |
| INT8 QDQ PTQ | 14.53 ms | 37.60 fps | 28.65 fps |

---

> **注**: 以下のFPS実験（BNB/TorchAO）はすべて `batch_size=1`（リアルタイム推論条件）で計測。評価時は `eval_batch_size=12`。

## PyTorch eager baseline（実験間共通参照値）

| 出典 CSV | dtype | median ms |
|---------|-------|----------|
| bnb_speed_candidates_fp16.csv | fp16 | 28.93 ms ← Best Known |
| bnb_subscope_fp16.csv | fp16 | 29.06 ms |
| bnb_decoder_other_breakdown_fp16.csv | fp16 | 29.55 ms |
| bnb_full_fp16.csv | fp16 | 30.62 ms |
| bnb_scope_fp16.csv | fp16 | 30.11 ms |
| bnb_attn_split_fp16.csv | fp16 | 31.29 ms |
| bnb_near_ties_fp16_long.csv (200fr) | fp16 | 30.23 ms |
| bnb_full.csv | bf16 | 30.10 ms |
| torchao_fast_candidate.csv | bf16 eager | 35.47 ms |
| torchao_qlinear_compiled.csv | bf16 + compile | 35.04 ms |
| torchao_qlinear_max_autotune_cudagraphs.csv | bf16 + max-autotune + cudagraphs | 89.21 ms |

---

## BNB INT8 実験

### full / broad scope（実験 D〜G）

| dtype | scope | median ms | vs baseline |
|-------|-------|----------|------------|
| bf16 | full | 70.46 ms | +134% |
| fp16 | full | 56.93 ms | +86% |
| fp16 compiled | full (fallback) | 58.17 ms | +90% |
| fp16 | encoder.blocks | 34.19 ms | +13% |
| fp16 | decoder | 46.59 ms | +55% |
| fp16 | encoder.blocks+decoder | 53.16 ms | +77% |
| fp16 | full | 61.14 ms | +103% |

### subscope breakdown（実験 H〜K）

| scope | median ms | vs baseline (~29-30ms) |
|-------|----------|----------------------|
| encoder.blocks.mlp | 30.88 ms | +1.3 ms |
| encoder.blocks.attn | 33.50 ms | +2.2 ms |
| decoder.ffn | 29.49 ms | ≈0 |
| decoder.attn | 35.34 ms | +6.0 ms |
| decoder.bbox_embed | 32.76 ms | +3.2 ms |
| decoder.query_scale | 30.02 ms | +0.5 ms |
| decoder.ref_anchor_head | 30.03 ms | +0.5 ms |
| decoder.ref_point_head | 30.19 ms | +0.6 ms |
| decoder.other | 35.65 ms | +6.1 ms |

### speed candidate sweep（実験 L、baseline 28.93ms）

| scope | median ms | vs baseline |
|-------|----------|------------|
| decoder.ffn | 29.49 ms | +0.56 ms |
| decoder.ref_anchor_head | 29.69 ms | +0.76 ms |
| decoder.ref_point_head | 29.73 ms | +0.80 ms |
| decoder.query_scale | 30.37 ms | +1.44 ms |
| encoder.blocks.mlp | 31.15 ms | +2.22 ms |
| encoder.blocks.mlp+decoder.ffn | 32.70 ms | +3.77 ms |
| decoder.ffn+query/ref heads | 33.32 ms | +4.39 ms |
| combined candidate (全部) | 35.74 ms | +6.81 ms |

### longer run 確認（実験 N、200fr、baseline 30.23ms）

| scope | median ms |
|-------|----------|
| decoder.ffn | 32.17 ms |
| decoder.ref_anchor_head | 33.45 ms |
| decoder.ref_point_head | 37.06 ms |

### compiled（実験 F, M）

| scope | median ms |
|-------|----------|
| baseline compiled | 33.24 ms |
| full (compile fallback) | 58.17 ms |
| baseline compiled | 32.09 ms |
| decoder.ffn compiled | 33.58 ms |

---

## TorchAO INT8 実験

### weight_only_int8, full model（.pthロード後にTorchAO weight INT8適用して推論、weight_only_int8_speed.csv）

| scope | median ms | FPS |
|-------|----------|-----|
| baseline (fp16 eager) | 31.66 ms | 31.6 fps |
| full model INT8 | 65.30 ms | **15.3 fps** |

### weight_only_int8, compile_mode=default（実験 A）、baseline bf16 35.47ms / 28.2fps

| scope | median ms | FPS |
|-------|----------|-----|
| encoder.blocks.mlp | 40.33 ms | 24.8 fps |
| decoder.ffn | 37.38 ms | 26.8 fps |
| encoder.blocks.mlp+decoder.ffn | 40.69 ms | 24.6 fps |

### weight_only_int8, compile_quantized_linear（実験 B）、baseline bf16 35.04ms / 28.5fps

| scope | median ms | FPS |
|-------|----------|-----|
| encoder.blocks.mlp | 42.44 ms | 23.6 fps |
| decoder.ffn | 40.16 ms | 24.9 fps |
| encoder.blocks.mlp+decoder.ffn | 44.79 ms | 22.3 fps |

### weight_only_int8, max-autotune + cudagraphs（実験 C）、baseline bf16 89.21ms / 11.2fps

| scope | median ms | FPS |
|-------|----------|-----|
| encoder.blocks.mlp | 129.14 ms | 7.7 fps |
| decoder.ffn | 142.32 ms | 7.0 fps |
| encoder.blocks.mlp+decoder.ffn | 149.47 ms | 6.7 fps |

---

## Linear shape microbench（実験 P）

FP16 vs BNB INT8 単体比較

| shape | FP16 median ms | BNB INT8 median ms | ratio (BNB/FP16) |
|-------|---------------|-------------------|-----------------|
| encoder.attn.qkv | 0.0791 | 0.1279 | 1.617x |
| encoder.attn.proj | 0.0599 | 0.1039 | 1.733x |
| encoder.mlp.fc1 | 0.1001 | 0.1260 | 1.259x |
| encoder.mlp.fc2 | 0.1030 | 0.1199 | 1.164x |
| decoder.attn.proj | 0.0491 | 0.1004 | 2.046x |
| decoder.ffn.fc1 | 0.0588 | 0.1001 | 1.703x |
| decoder.ffn.fc2 | 0.0650 | 0.1010 | 1.554x |
| decoder.bbox.hidden | 0.0455 | 0.1011 | 2.224x |
| decoder.bbox.out | 0.0369 | 0.0987 | 2.678x |

---

## Custom Triton grouped INT8 microbench

### decoder.bbox_embed hidden path（multi-stage）

| 実装 | median ms | vs fp16_loop |
|------|----------|-------------|
| fp16_loop | 0.2897 ms | 1.000x |
| grouped_int8_vmap | 0.7443 ms | 0.389x |
| grouped_int8_triton | 0.4016 ms | 0.721x |
| triton + stage1 quant fused | 0.3470 ms | 0.835x |
| triton + stage0+1 quant fused | **0.2250 ms** | **1.287x** |

### decoder.self_query_pre（single-stage）

| 実装 | median ms | vs loop |
|------|----------|---------|
| loop | 0.4850 ms | 1.000x |
| grouped_bmm | 0.4072 ms | 1.191x |
| fused_int8_triton | 1.2920 ms | 0.375x |
| fused_input_int8_triton | 0.6473 ms | 0.749x |

---

> 出典: plots/int8_experiment_memo_2026-03-14.md, docs/int8_spec.md

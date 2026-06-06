# SAT-HMR INT8 + torch.compile 最適化記録 (2026-04-25)

inference 平均を **260 ms → 50.9 ms (-80%)** まで圧縮した過程。

各ステップごとに「**やったこと → ログのどこを見て発見したか → なぜ効いたか → 結果**」を 1 ブロックでまとめる。

---

## 全体スコア

| Step | 構成 | inference 平均 | 改善率 | ログファイル |
|---|---|---|---|---|
| 初期 (Step 0) | encoder のみ compile、INT8 化 64 個 | 260 ms | — | `compile_full_20260425_1713.log` |
| Step 1 | + `--compile_quantized_linear` + `capture_dynamic_output_shape_ops` + `position_encoding_xy` disable | 189 ms | -27% | `compile_full_20260425_1727.log` |
| Step 2 | + listcomp ヘルパー化 (sat_model + vision_transformer) | **74 ms** | -61% | `compile_full_20260425_1741.log` |
| (脱線) Step A | decoder INT8 のみ、decoder compile なし | 301 ms | 悪化 | `compile_full_stepA.log` |
| (脱線) Step C / C2 | decoder INT8 + decoder compile | 109 ms / 111 ms | 横ばい | `compile_full_stepC.log` / `compile_full_stepC2.log` |
| (脱線) Option A | model 全体 compile + 全 Linear INT8 | 247 ms | warmup 爆発 | `compile_full_optA.log` |
| Step 5 / Option D | encoder + decoder + heads 個別 compile + warmup 5 除外 | 50.9 ms | -31% | `compile_full_optD.log` |
| (脱線) Option D2 | + Inductor 強化 (use_mixed_mm, epilogue_fusion, coordinate_descent_tuning, recommended_inductor_config_setter) | 77.5 ms (warmup 105s で悪化) | 悪化 | `compile_full_optD2.log` |
| (脱線) Option E | D2 + boolean mask → index_select 書き換え | 61.3 ms (まだ悪化) | 悪化 | `compile_full_optE.log` |
| **Step 6 / Option F** | D + boolean mask → index_select + `lvl_pooling` を `unflatten` 化 + `_check_is_size` ヒント追加 (Inductor 強化は revert) | **48.2 ms** | -5.3% (D 比) | `compile_full_optF.log` |
| (脱線) Option G | F + decoder の `memory_efficient_attention` を `F.scaled_dot_product_attention` に置換 | 49.4 ms (微悪化) + **人物検出が壊れた**ので即 revert | 失敗 | `compile_full_optG.log` |
| (脱線) Option H | F の構成で `--compile_dynamic` を外して static shape compile | **457.7 ms (約 10x 悪化)** | 失敗 | `compile_full_optH_static.log` |
| (脱線) Option I_A | `position_encoding_xy` の `@_dynamo.disable` を外す | 77.6 ms (悪化、Inductor `same_meta` SymInt バグ復活) | 失敗 | `plots/profile_int8_blocks_2026-04-30/compile_full_optI_A.log` |
| (脱線) Option I_B | `_drop_register_tokens_concat` の `@_dynamo.disable` を外す | 46.4 ms (微改善だが decoder.py:469 が WON'T CONVERT、健全でない) | 失敗 | `plots/profile_int8_blocks_2026-04-30/compile_full_optI_B.log` |
| (脱線) Option I_C | `_get_attn_bias_from_cache` の `@_dynamo.disable` を外す | **221.5 ms (約 4.6x 悪化)** | 失敗 | `plots/profile_int8_blocks_2026-04-30/compile_full_optI_C.log` |
| Step 7 / Option J | F + `torch._check(sat_idx.size(0) > 0)` + `lvl_pooling` に `_check(N > 0)` / `_check(N % 4 == 0)` 追加 | 45.8 ms (-5%、warmup 同等。BackendCompilerFailed 4 件 warmup 内のみ発生) | 採用 | `plots/profile_int8_blocks_2026-04-30/compile_full_optJ.log` |
| **Step 8 / Option K** | J + Inductor の `pre_grad_fusion_options` から `batch_linear` / `batch_linear_lhs` を除外 (torchao 量子化テンソルで `aten.stack` 失敗するため) | **43.0 ms (-11% vs F、cache hit 時)**。BackendCompilerFailed 0、recompile 13→10 | **採用 (現状最良 = default)** | `plots/profile_int8_blocks_2026-04-30/compile_full_optK.log` (初回 68.8s warmup) / 同 `_v2` (cache hit 29.5s warmup) |
| (実験/失敗) Option L | K + 全 hot MLP/FFN (encoder DINO Mlp 14 + decoder FFN 6 + α = 21 個) を **custom Triton kernel** (`StaticInt8FusedMLP`、static-scale W8A8 fused) に置換、`@_dynamo.disable` でラップ | 50.4 ms (-13% 悪化)、graph break +14 (21 module 分の disable boundary)、recompile +4 | 失敗 (experimental flag `--static_int8_fused_mlp` として残置) | `plots/profile_int8_blocks_2026-04-30/compile_full_optL.log` |
| (実験/失敗) Option L2 | L の `@_dynamo.disable` を `@torch._dynamo.allow_in_graph` に変更 (Inductor fusion 維持狙い) | 47.5 ms (-3 ms vs L だが still 悪化)、recompile 38、BackendCompilerFailed 4 | 失敗 (experimental) | `plots/profile_int8_blocks_2026-04-30/compile_full_optL2.log` |
| (実験/失敗) Option L3 | K + decoder FFN (linear1/linear2 計 12 Linear) のみ **compile-native `StaticScaleInt8Linear`** (torch._int_mm + torch native ops、custom Triton 無し、disable 無し) に置換 | 50.2 ms (悪化)、graph break 6 / recompile 10 / BCF 0 (構造的指標は維持) | 失敗 (experimental flag `--static_scale_int8_decoder_ffn` として残置) | `plots/profile_int8_blocks_2026-04-30/compile_full_optL3.log` |

**累積: 260 → 43.0 ms (-83%、約 6.0x、Option K cache hit 時)** / Option F 維持なら 48.2 ms (-81%)

> bf16 + xformers (compile / INT8 なし) baseline は 29 ms。**まだ 19 ms (+66%) のギャップ**が残っている。
> 19 ms ギャップの推定内訳:
> - 1700 kernel launch/frame による CPU launch overhead: 5-8 ms (nsys API summary より)
> - graph break 6 件の CUDA stream sync: 3-5 ms
> - INT8 quant/dequant 自体の overhead: 残り (~6-10 ms)

---

## A. 最終構成 (Option D)

### A-1. INT8 化の範囲

`nn.Linear` の `Wx + b` (matmul) **だけ** が INT8 計算。それ以外は bf16。

| 項目 | 状態 |
|---|---|
| Linear の重み W | INT8 (静的量子化) |
| Linear への入力 x = activation tensor | 動的 INT8 化 (per-token absmax) |
| Linear 内部の matmul | INT8 × INT8 → INT32 |
| dequant 後の出力 | bf16 |

> 用語注意: torchao の `int8_dynamic_activation_int8_weight` の "activation" は **データテンソル**のこと。**ReLU/GELU 等の活性化関数とは別物**。これら活性化関数は bf16 のまま。

**量子化されないもの**: 活性化関数 (ReLU, GELU, SiLU, sigmoid, softmax)、LayerNorm、Attention の Q@K^T / attn@V (xformers Flash で bf16)、residual add、position encoding、view/permute 等。

**数:**
```
Linear 263 個中、INT8 化 228 個 (86.7%)
  skipped_dim       23 個 (in/out が 8 で割れない、head の最終投影など)
  skipped_nonplain  12 個 (既に他のラッパーで包まれた weight)
```

### A-2. compile 範囲

| 部位 | 状態 |
|---|---|
| `model.forward_encoder` | ✅ compile (`--compile_encoder_only`) |
| `model.decoder` | ✅ compile (`--compile_decoder_only`) |
| pose_head / shape_head / cam_head / conf_head / bbox_embed | ✅ 個別 compile (`--compile_heads`) — 各 instance ごとに wrap (合計 20 個) |
| post-decoder の Python 制御フロー (`for lvl in range`、SMPL_Layer、bbox 後処理) | ✗ eager |
| 前処理 (画像 → patch 化) | ✗ eager |

### A-3. 最終実行コマンド

```bash
TORCH_LOGS="recompiles,graph_breaks" \
TORCHDYNAMO_EXTENDED_DEBUG_GUARD_ADDED="s0 <= 2147483647" \
python -u video_demo.py \
  -i videos/sample_video.mp4 \
  --no_save --no_render \
  --dtype_cfg bfloat16 \
  --ptq_mode int8_dynamic --no_int8_attn \
  --output_dir plots/profile_int8_blocks_2026-04-25 \
  --metrics_csv int8_blocks_metrics.csv \
  --compile --compile_encoder_only --compile_decoder_only --compile_heads --compile_dynamic \
  --compile_quantized_linear \
  --warmup_frames 5
```

---

## B. 段階的改善の経緯

### B-1. Step 0 (初期状態) → 260 ms

- 構成: encoder のみ compile、INT8 化 64 個
- ログ: [`compile_full_20260425_1713.log`](../plots/profile_int8_blocks_2026-04-25/compile_full_20260425_1713.log) (17:13 時点)
- 状態: Dynamo が 40+ 個のサブグラフに断片化、graph break / recompile が大量

### B-2. Step 1 (3 つの torch.compile 詰まり解消) → 189 ms

#### やったこと

3 つ同時に投入:

1. **`--compile_quantized_linear` フラグを追加して使用**
2. **`_dynamo.config.capture_dynamic_output_shape_ops = True`** を `video_demo.py` に追加
3. **`models/position_encoding.py` の `position_encoding_xy` に `@torch._dynamo.disable`** を貼る

#### ログのどこを見て発見したか

Step 0 のログ (`compile_full_20260425_1713.log`) で、`TORCH_LOGS="all"` 経由で出ている **`CompilationMetrics(...)` 行**の `restart_reasons` と `fail_type` が宝の山だった。

**ヒント a (→ `--compile_quantized_linear`)**:
```
restart_reasons={"'inline in skipfiles: Linear.forward |
   _fn /home/kyuta/.../torch/_dynamo/eval_frame.py,
   skipped according trace_rules.lookup SKIP_DIRS'"}
```
これが 17 件出ていた。`_fn` は `torch._dynamo.disable` の wrapper の中身。コードを grep して [`video_demo.py:235 wrap_quantized_linears_for_compile`](../video_demo.py#L235) が量子化 Linear を `_dynamo.disable` で包んでいるのを発見。`--compile_quantized_linear` を付けるとこの wrap が skip され、量子化 Linear が graph に取り込まれる。

**修正内容**: 実行コマンドに `--compile_quantized_linear` を追加するだけ。コード変更不要。
([`video_demo.py:864-866`](../video_demo.py#L864) でこのフラグが True なら `wrap_quantized_linears_for_compile()` を skip)

**ヒント b (→ `capture_dynamic_output_shape_ops`)**:
```
restart_reasons={'dynamic shape operator: aten.nonzero.default;
   to enable, set torch._dynamo.config.capture_dynamic_output_shape_ops = True'}
```
これが 7 件。**メッセージ自体が解決策を提示している**。`co_firstlineno=432` から forward_encoder 内部、`lvl1_img_patches_28[lvl1_sat_mask]` などの boolean mask indexing が原因 (内部で nonzero に展開される)。

**修正内容**: [`video_demo.py`](../video_demo.py) の Dynamo config 設定に 1 行追加:
```python
_dynamo.config.capture_dynamic_output_shape_ops = True
```
このフラグを True にすると、Dynamo は data-dependent shape を symbolic shape として graph 内に取り込み、restart せずに処理を続ける。結果として `forward_encoder` の resume 分裂 (5 frame → 1 frame) が解消される。

**ヒント c (→ `position_encoding_xy` の disable)**:
```
fail_type='BackendCompilerFailed'
fail_reason="backend='inductor' raised:\nAttributeError: 'SymInt' object has no attribute 'size'"
co_name='position_encoding_xy'
co_filename='/home/kyuta/SAT-HMR/models/position_encoding.py'
```
スタックを遡ると `_inductor/fx_passes/post_grad.py:643 same_meta` で `sym_eq(val1.size(), val2.size())` を SymInt スカラに対して呼んでいる Inductor 内部のバグ。位置を限定して `@_dynamo.disable` で Inductor 経路を回避するのが安全。

**修正内容**: [`models/position_encoding.py:12`](../models/position_encoding.py#L12) の `position_encoding_xy` 関数定義の直前に 1 行追加:
```python
@torch._dynamo.disable
def position_encoding_xy(pos_x, pos_y, embedding_dim, temperature = 20, scale = 2*math.pi):
    ...
```
`@torch._dynamo.disable` を貼ると、Dynamo はこの関数を graph に取り込まず eager 実行する。Inductor の post_grad pass を回避できるので BackendCompilerFailed が発生しなくなる。元々この関数は重くないので eager fallback でも実害なし。

#### なぜ効いたか

graph 断片化の主因 25 件 (skipfile 17 + nonzero 7 + Inductor バグ 1) を一掃したため、コンパイル単位が大幅に統合された。

#### 結果

189 ms (-27%)、ログ: [`compile_full_20260425_1727.log`](../plots/profile_int8_blocks_2026-04-25/compile_full_20260425_1727.log)

---

### B-3. Step 2 (listcomp ヘルパー化) → 74 ms ← 今日最大の飛躍

#### やったこと

`models/sat_model.py` と `models/encoders/dinov2/models/vision_transformer.py` の `zip(tensor.split(可変長lens))` パターンを、`@torch._dynamo.disable` で囲ったヘルパー関数に切り出した:

- `_drop_register_tokens_concat`
- `_normalize_pos_per_lvl`
- `_build_x_list_multi`
- `_build_pos_list_multi`
- `_build_lvl_id_list_multi`
- `_post_block_outputs` (vision_transformer.py 側)

#### ログのどこを見て発見したか

Step 1 のログ (`compile_full_20260425_1727.log`) で:
```
fail_type='Unsupported'
fail_reason='call_method UserDefinedObjectVariable(list_iterator) __next__ [] {}'
co_name='<listcomp>'
co_firstlineno=562
fail_user_frame_lineno=562
has_guarded_code=False  ← 完全失敗
```

これが **10 件**。`co_name='<listcomp>'` で listcomp 内部の失敗、`fail_user_frame_lineno` がピンポイントで失敗箇所を特定 (sat_model.py:562, 654, 655, 657, 660, 663, 666 と vision_transformer.py:413, 415)。

`UserDefinedObjectVariable(zip|enumerate|list_iterator)` は Dynamo が `tensor.split(可変長)` の結果を「未知のオブジェクト」として扱っているサイン。`has_guarded_code=False` は完全コンパイル放棄。

#### なぜ効いたか

Dynamo は `tensor.split(可変長 list)` の戻り値を `UserDefinedObjectVariable` 扱いし、その上の `zip/enumerate` を一切トレースできなかった。compile 範囲外に明示的に追い出すと、周辺の graph が細切れにならず大きな塊として保たれ、Inductor の fusion が効くようになった。

#### 結果

74 ms (-61%)、ログ: [`compile_full_20260425_1741.log`](../plots/profile_int8_blocks_2026-04-25/compile_full_20260425_1741.log)

---

### B-4. (脱線) Step A: decoder INT8 のみで compile なし → 301 ms 悪化

#### やったこと

Step 2 構成 (encoder INT8 + encoder compile) をベースに、**decoder の Linear だけ INT8 化を追加**。decoder の compile は意図的に入れず (検証目的)。

| 部位 | INT8 量子化 | compile |
|---|---|---|
| encoder | ✅ ON (64 Linear) | ✅ ON (`--compile_encoder_only`) |
| **decoder** | ✅ ON (新たに 90 Linear 追加) | **✗ OFF** ← ここがまずかった |
| heads | ✗ OFF (`--int8_skip_modules` で除外) | ✗ OFF |

具体的な変更: 実行コマンドの `--int8_skip_modules pose_head shape_head cam_head conf_head bbox_embed decoder` から `decoder` を削除しただけ。コード変更なし。

#### ログのどこを見て学んだか

ログ全体で:
- `quantized=154` (decoder の 90 Linear が新規 INT8 化)
- `target=encoder` (decoder は eager のまま)
- inference 平均 **301 ms** ← 大幅悪化

#### 学び

**INT8 化と compile はセット必須**。decoder INT8 化したのに encoder のみ compile すると、decoder の per-Linear ごとに quant/dequant overhead が乗る。compile しないと Inductor の int8_mm fusion が効かないので、単純に bf16 より遅い。

ログ: [`compile_full_stepA.log`](../plots/profile_int8_blocks_2026-04-25/compile_full_stepA.log)

---

### B-5. (脱線) Step C / C2: decoder も compile に追加 → 109 ms (steady state は 66 ms)

#### やったこと

`--compile_decoder_only` を追加し、encoder と decoder の両方を `torch.compile()` でラップ ([`video_demo.py`](../video_demo.py) の排他制約緩和)。Step C2 では追加で `_dynamo.config.force_parameter_static_shapes = False` を試した。

#### ログのどこを見たか

Step C ログ (`compile_full_stepC.log`):
```
target=decoder+encoder
inference 平均 108.9 ms、warmup 30.4 s
Recompile 9 件 (うち decoder 由来 4 件)
```

Step C2 ログ (`compile_full_stepC2.log`):
```
[16/1] Recompiling function _dispatch__torch_function__ in torchao/utils.py:363
   triggered by:
   - 16/0: tensor 'L['args'][1]' size mismatch at index 0. expected 768, actual 2048.
     Guard failed on a parameter, consider using
     torch._dynamo.config.force_parameter_static_shapes = False
```

メッセージ通り `force_parameter_static_shapes = False` を設定 → **recompile 件数 9 のまま変化なし**。

#### 学び

- decoder compile を加えると steady-state は 66 ms と僅かに速いが、warmup が 30s に伸びて 528 frame では amortize できず平均悪化に見える。
- `force_parameter_static_shapes` フラグは `nn.Parameter` 用で、torchao の `AffineQuantizedTensor` (Tensor subclass) には届かない (= **エラーメッセージが嘘をついていた**)。

ログ: [`compile_full_stepC.log`](../plots/profile_int8_blocks_2026-04-25/compile_full_stepC.log) / [`compile_full_stepC2.log`](../plots/profile_int8_blocks_2026-04-25/compile_full_stepC2.log)

---

### B-6. (脱線) Option A: model 全体 compile + 全 Linear INT8 → 247 ms (warmup 102s)

#### やったこと

「全 Linear INT8 化 + 全部 compile してみる」という最大攻撃を試した。

| 部位 | INT8 量子化 | compile |
|---|---|---|
| encoder | ✅ ON | ✅ ON (model 全体に含まれる) |
| decoder | ✅ ON | ✅ ON (model 全体に含まれる) |
| heads | ✅ ON (新規) | ✅ ON (model 全体に含まれる) |
| post-processing (`for lvl ...`、SMPL_Layer など) | — | ✅ ON ← **これが問題** |

具体的な変更:
- `--int8_skip_modules` を**空に**(= heads も INT8 化、合計 228 個)
- `--compile_encoder_only` と `--compile_decoder_only` と `--compile_heads` を**全部外して**、`--compile` のみ → これで `model.forward` 全体がひとつの `torch.compile()` でラップされる

#### ログのどこを見て学んだか

Option A ログ ([`compile_full_optA.log`](../plots/profile_int8_blocks_2026-04-25/compile_full_optA.log)):
```
quantized=228   (全 Linear が INT8 化された)
target=model    (model.forward 全体が 1 つの compile target)
First frame compile 102 s   ← warmup 爆発
Recompile 14 件
```

特に注目したのが MLP.forward の **3 件 recompile**:

```
Recompiling function forward in /home/kyuta/SAT-HMR/models/sat_model.py:995
   - 26/1: len(L['self']._modules['layers']) == 3
   - 26/2: len(L['self']._modules['layers']) == 6
   - 26/3: len(L['self']._modules['layers']) == 5
```

[`sat_model.py:995`](../models/sat_model.py#L995) は `MLP.forward` (= 全 head が共通で使うクラス)。`L['self']._modules['layers']` は MLP インスタンスの `nn.ModuleList`。`len(...)` がガードされているのは **MLP の層数で specialize 分岐**しているということ。

| head 種別 | MLP の層数 |
|---|---|
| pose_head | 6 層 |
| shape_head | 5 層 |
| cam_head, bbox_embed | 3 層 |

これが 3 つの specialize を要求 → **同じ code object (MLP.forward) を 3 回コンパイル**。各 specialize 約 1-2 秒掛かるので 8 秒分の warmup 増。

#### なぜ失敗 (= 247 ms に悪化) したか

2 つの問題が同時発生:

1. **`--compile` のみだと `model.forward` 全体が 1 つの巨大 graph として扱われる**:
   - encoder (graph 化したい)
   - decoder (graph 化したい)
   - heads ループ (graph 化したい)
   - SMPL_Layer 呼び出し、bbox 計算、`for lvl in range`、`if self.training` 分岐 (これは Python 制御フローで graph 化と相性が悪い)
   - → post-processing 部分で graph break が +4 件発生 (Step C の 6 件 → 8 件に増加)
2. **MLP class の共有 specialize**:
   - head ごとに別 instance だが、`torch.compile(model)` で wrap した内側だと、Dynamo は MLP.forward の **code object 1 つに対してキャッシュを持つ**
   - → 層数違いで 3 回 specialize、warmup 102 秒に膨張

要するに **「`--compile` (model 全体) は荒すぎる、`--compile_heads` で各 instance を個別 wrap すれば specialize が分離する」** という発見につながった。

#### 学び (Step 5 / Option D に直結)

- model 全体 compile は post-processing も graph 化されて余分な graph break が出る
- 共有された MLP class は shape ごとに specialize して compile cost が爆発
- → head **インスタンスごとに `torch.compile()` で wrap** すると、それぞれ独立した compile target になり、MLP.forward の specialize 共有が解消される (Option D で実装)

---

### B-7. Step 5 / Option D (最終解) → 50.9 ms

#### やったこと

video_demo.py に新フラグを 2 つ追加し、CLI で 1 箇所変更:

1. **`--compile_heads` フラグを新規追加**: pose_head/shape_head/cam_head/conf_head/bbox_embed の各 nn.Module instance を **個別に `torch.compile()` で wrap** (合計 20 個。例: pose_head[0], pose_head[1], ..., shape_head[0], ..., bbox_embed[5])
2. **`--warmup_frames N` フラグを新規追加**: inference 平均値の算出から最初の N frame を除外する (実計算には影響しない、計測上の表示だけ)
3. **CLI 構成**: Step C で確立した `--compile_encoder_only --compile_decoder_only` (= encoder と decoder をそれぞれ個別に compile) に、上記 1 で追加した `--compile_heads` と 2 で追加した `--warmup_frames 5` を**追加**。さらに `--int8_skip_modules` の引数を**空に**することで、これまで除外していた head の Linear (~90 個) も INT8 化対象になる (合計 INT8 化 Linear = 154 + 90 = 228)。

#### ログのどこを見て発見したか

Option A ログから「MLP class 共有 specialize」が問題と判明 (B-6 参照) → head ごとに独立した compile target を割り当てる方針に至った。

#### なぜ効いたか

- model 全体 compile を避けたため post-processing の graph break が発生せず — 具体的には Option A で発生していた sat_model.py の 4 箇所 (825 `prepare_static_eval_buffers()`、840 `forward_encoder(...)` 呼び出し、859 `self.decoder(...)` 呼び出し、896 `self.process_smpl(...)` 呼び出し) が、model.forward を compile しないことで自然に消えた

##### 該当ログの実物 ([`compile_full_optA.log`](../plots/profile_int8_blocks_2026-04-25/compile_full_optA.log))

**注意**: ログには `Graph break: from user code at:` の後に **call stack 全体**が表示される。先頭行の `line N` は break の起点 (= resume frame の入口) で、**実際の break 地点 (真因) は最深部のフレーム**。

| break | 起点 (sat_model.py) | call stack の最深部 (真因) | 真因の正体 | ログ行 |
|---|---|---|---|---|
| ① | line 825 `prepare_static_eval_buffers()` | 同所 | **`prepare_static_eval_buffers` が `_fn` skipfile** (`@torch._dynamo.disable` 系) | `:30` |
| ② | line 840 `forward_encoder(...)` | `block.py:187 get_attn_bias_and_cat` の中 | **`_get_attn_bias_from_cache` の `@_dynamo.disable`** (encoder の xformers attn_bias) | `:67` |
| ③ | line 859 `self.decoder(...)` | `decoder.py:264 position_encoding_xy(...)` | **`position_encoding_xy` の `@_dynamo.disable`** (我々が Step 1 で貼った) | `:664` |
| ④ | line 896 `process_smpl(...)` | `smplx/lbs.py:231 lbs()` 内部 | **smplx ライブラリ** (Linear Blend Skinning) 内部の Dynamo 非対応 op | `:971` |

例えば break ③ の実ログ抜粋 (line 664-670 付近):
```
[19/0] [__graph_breaks] Graph break: from user code at:
   File "/home/kyuta/SAT-HMR/models/sat_model.py", line 859, in torch_dynamo_resume_in_forward_at_840
     hs, reference = self.decoder(memory=final_features, ...)
   File "/home/kyuta/SAT-HMR/models/decoder.py", line 154, in forward
     hs, references = self.decoder(memory=memory, ...)
   File "/home/kyuta/SAT-HMR/models/decoder.py", line 264, in forward
     xy_embed = position_encoding_xy(obj_center[:,0], obj_center[:,1], self.d_model)
                ^^^^^^^^^^^^^^^^^^^^
                ← ここが本当の break 地点 (position_encoding_xy が @_dynamo.disable されているため)
```

##### なぜ Option D ではこれら 4 件が消えるのか

| | Option A (model 全体 compile) | Option D (encoder/decoder/heads 個別 compile) |
|---|---|---|
| model.forward の扱い | compile されている、Dynamo が **deep trace** する | **compile されない**、Python で普通に実行 |
| 例: `self.decoder(...)` 呼び出し | Dynamo が decoder.forward の中まで降りて、内部の `position_encoding_xy` (disable) で break → resume frame 859 が生成される | Python が compile された decoder wrapper を呼ぶだけ。break ではなく **「compile target 間の自然な切り替わり」** |

要するに、Option A の 4 件 graph break は **深い call stack の中で発生した break が、外側の resume frame line として記録されている**だけ。Option D は外側 (model.forward) の compile を捨てたので、内側で起きる break が外側に伝播しなくなる。

(ただし decoder 自身を compile target にしているので、decoder 内部の `position_encoding_xy` 由来の break や `_get_attn_bias_from_cache` 由来の break は **decoder の compile 内部で残る** — それが Option D の 6 件中の `decoder.py:154`、`decoder.py:563` に対応)
- 各 head instance が個別の compile target を持つため、`MLP.forward` の class-level specialize 共有 (3 件 recompile) が消えた
- warmup 22s に圧縮 (Option A の 102s から -78%)
- `--warmup_frames 5` で初回 compile cost を平均値から外した (これは benchmark 側の見せ方の調整で、実計算時間は変わらない)

#### 結果

| 指標 | Option A | **Option D** |
|---|---|---|
| First frame compile | 102 s | **22 s** |
| inference 平均 | 247 ms | **50.9 ms** |
| Graph break | 8 | 6 |
| Recompile | 14 | 13 |
| INT8 化 Linear | 228 | 228 |

ログ: [`compile_full_optD.log`](../plots/profile_int8_blocks_2026-04-25/compile_full_optD.log)

#### Option D の残課題

- **6 graph break** (boolean mask × 2、xformers attn_bias × 2、`_drop_register_tokens_concat` disable × 1、forward_encoder boundary × 1) — per-frame で 0.5-1.5 ms 程度。構造的に発生 (解 6 = padded tensor 書き換えで全消可能だが大工事)
- **13 recompile** — タイムスタンプを見るとすべて 19:15:13-21 の 8 秒間に集中 → warmup 内で完結、per-frame 影響ゼロ

---

### B-8. (脱線・失敗) Option G: decoder attention を SDPA に置換 → 人物検出消失で即 revert

#### やったこと

`decoder.py` の attention 呼び出しを xformers から `F.scaled_dot_product_attention` (SDPA) に切り替え:

- `torch_attention()` ヘルパーを SDPA ベースに書き換え
- self-attention (line 519) と cross-attention (line 564) の `memory_efficient_attention(q,k,v,attn_bias=...)` 呼び出しを `torch_attention(...)` に統一
- decoder.py:563 の graph break (xformers `memory_efficient_attention` 直前) を消す狙い

#### 結果

- inference 49.4 ms (Option F 比 +1.2 ms 微悪化)
- graph break 5 件 (decoder.py:563 が消えて -1)
- **人物検出が壊れた** (推論結果が機能不全)

#### 原因 (推定)

`F.scaled_dot_product_attention` の `attn_mask` パラメータと xformers `attn_bias` の意味論が違う可能性。具体的には:
- xformers Flash kernel は `attn_bias` を内部で適切な scale + add で扱う
- SDPA は `attn_mask` を Q@K^T 後に加算する仕様だが、scale の適用順や dtype 変換に微妙な差異あり
- self-attention で使う dense bias (denoising query 用、padded 経由) の扱いが SDPA で正しくマップされなかった

ログ: [`compile_full_optG.log`](../plots/profile_int8_blocks_2026-04-25/compile_full_optG.log)。**全 decoder.py 変更を revert 済み**。

### B-9. (脱線・失敗) Option H: `--compile_dynamic` を外して static shape compile → 10x 悪化

#### やったこと

実行コマンドから `--compile_dynamic` を削除 (= `dynamic=False`、shape ごとに specialize)。

#### 結果

| 指標 | Option F (dynamic=True) | Option H (dynamic=False) |
|---|---|---|
| inference 平均 | 48.2 ms | **457.7 ms** (約 10x 悪化) |
| First frame compile | 22 s | 125 s |
| Recompile | 13 | **65** |
| `cache_size_limit (8)` hit 警告 | なし | 多発 |

#### 原因

ログの警告:
```
[5/8] torch._dynamo hit config.cache_size_limit (8)   ← decoder
[16/8] torch._dynamo hit config.cache_size_limit (8)  ← torchao dispatch
[24/8] torch._dynamo hit config.cache_size_limit (8)  ← memory_efficient_attention
```

SAT は scale-adaptive token 機構で **frame ごとに 8 種類以上の異なる shape** が出る。static で specialize しようとすると cache (デフォルト上限 8) が即溢れて、以降は **graph break + eager fallback** に陥り推論激遅。

#### 学び

**`dynamic=True` (= `--compile_dynamic`) は必須**。SAT の可変 token 数を考えると static は不可能。`dynamic=None` (auto) も実質 static + 自動切替なので、明示的に `True` のままが正解。

ログ: [`compile_full_optH_static.log`](../plots/profile_int8_blocks_2026-04-25/compile_full_optH_static.log)。実行コマンドに `--compile_dynamic` を戻して維持。

### B-10. nsys プロファイル取得 (情報収集) → kernel launch overhead が支配的と判明

#### やったこと

```bash
nsys profile -o plots/profile_int8_blocks_2026-04-25/nsys/sat_optF \
  --trace=cuda,nvtx,cudnn,cublas \
  python -u video_demo.py ...(Option F 同設定)... --warmup_frames 30
```

#### 結果

WSL2 環境のため GPU kernel runtime データは取れず、CUDA API summary のみ取得。

530 frame + warmup 全体での集計:

| 項目 | 回数 | 合計 | per-frame 換算 |
|---|---|---|---|
| `cudaLaunchKernel` | 592,852 | 3.96 s | ~7.5 ms/frame |
| `cuLaunchKernel` (CUDA Graph 経由) | 313,675 | 2.13 s | ~4.0 ms/frame |
| `cudaMemcpyAsync` | 47,142 | 1.44 s | ~2.7 ms/frame |
| `cudaStreamSynchronize` | 35,098 | 0.77 s | ~1.5 ms/frame |
| **合計 (CPU 側 API オーバーヘッド)** | | ~8.4 s | **~16 ms/frame** |

**inference 48.2 ms のうち約 16 ms (33%) が CPU 側 API オーバーヘッド**。1 frame あたり **1700+ kernel launch** という多さ。

INT8 dynamic 量子化は per-Linear で約 4-5 kernel (per-token absmax + quant + int8_mm + dequant) を起動。228 INT8 Linear × 4 ≈ 900+ kernel/frame が INT8 経路だけで発生。

ログ: [`plots/profile_int8_blocks_2026-04-25/nsys/sat_optF.nsys-rep`](../plots/profile_int8_blocks_2026-04-25/nsys/sat_optF.nsys-rep)

#### 学び

bf16 baseline (29 ms) に対して **launch overhead が支配的** な可能性が高い。launch を減らすには:
- graph break を減らして cudagraphs が効く範囲を広げる
- Inductor の fusion を強化 (ただし Option D2 で試して悪化したので慎重に)

---

### B-11. Step 7 / Option J: `torch._check` で size 制約をヒント追加 → 45.8 ms

#### やったこと

3 箇所に `torch._check` を追加して Dynamo の symbolic shape 解析にヒント:

1. [`sat_model.py:566`](../models/sat_model.py#L566): `torch._check(sat_idx.size(0) > 0)` (encoder の SAT 選択 patch 数 > 0)
2. [`sat_model.py:268 lvl_pooling`](../models/sat_model.py#L268): `torch._check(N > 0)` + `torch._check(N % 4 == 0)` (4 で割り切れる)

#### 結果

inference 48.2 → **45.8 ms (-5%)**、warmup 同等。Graph break / recompile 件数は変わらないが、各 graph fragment 内の処理が効率化された (size 制約が明示されたことで unflatten/reshape が早くなった)。

副作用: BackendCompilerFailed 4 件発生。タイムスタンプから warmup 内に集中 → per-frame 影響ゼロだがログにエラーが出る。原因は次の B-12 で判明。

ログ: [`compile_full_optJ.log`](../plots/profile_int8_blocks_2026-04-30/compile_full_optJ.log)

### B-12. Step 8 / Option K: Inductor `batch_linear` 融合を無効化 → 43.0 ms (cache hit 時)

#### やったこと

Option J の BackendCompilerFailed の真因を call stack から特定:

```
File ".../torch/_inductor/fx_passes/group_batch_fusion.py", line 80, in update_stack_example_value
   example_value = torch.stack(metadata, dim=dim)
File ".../torchao/utils.py", line 393, in _dispatch__torch_dispatch__
NotImplementedError: LinearActivationQuantizedTensor dispatch:
   func=<OpOverload(op='aten.stack', overload='default')>
```

→ Inductor の **`group_batch_fusion` pre_grad pass** が複数 Linear を batched bmm に融合しようとして、torchao 量子化テンソルに対して `torch.stack` を呼ぼうとして失敗していた。

[`video_demo.py`](../video_demo.py) の Inductor config 設定箇所に追加:

```python
import torch._inductor.config as _inductor_config
if hasattr(_inductor_config, "pre_grad_fusion_options"):
    opts = dict(_inductor_config.pre_grad_fusion_options or {})
    opts.pop("batch_linear", None)
    opts.pop("batch_linear_lhs", None)
    _inductor_config.pre_grad_fusion_options = opts
```

`pre_grad_fusion_options` のデフォルト:
```python
{'batch_linear': {}, 'batch_linear_lhs': {}, 'batch_layernorm': {}, 'batch_tanh': {}, 'batch_relu': {}, 'batch_sigmoid': {}}
```

ここから torchao と衝突する `batch_linear` / `batch_linear_lhs` 2 つだけ除外。`batch_layernorm` などは維持。

#### 結果

| | Option F | Option J | **Option K (cache hit)** |
|---|---|---|---|
| inference 平均 | 48.2 ms | 45.8 ms | **43.0 ms** (-11% vs F) |
| First frame compile | 22 s | 22.7 s | 初回 68.8 s / **2 回目以降 29.5 s** |
| Recompile | 13 | 13 | **10** (-3) |
| BackendCompilerFailed | 0 | 4 (warmup) | **0** ✓ |
| WON'T CONVERT | 0 | 1 | **0** ✓ |

副次効果として:
- Recompile 3 件削減 (`_dispatch__torch_function__` ×2、`forward_cross_query_proj` ×1) — `batch_linear` がこれらの recompile を誘発していた
- decoder.py:469 (`XformerDecoderLayer.forward`) が真に compile 成功 (J までは BackendCompilerFailed で eager fallback してた)

#### Warmup の緩和

初回コンパイルは 68.8s だが、`torch._inductor.config.fx_graph_cache = True` (デフォルト ON) と `compile_threads = 28` (28 並列) により:

- ディスクキャッシュ `/tmp/torchinductor_kyuta/` (380 MB) に compiled graph を保存
- 同じ構成で 2 回目以降は **29.5 s warmup** に短縮 (-57%)

`/tmp` 飛ぶ運用なら永続化:
```bash
export TORCHINDUCTOR_CACHE_DIR=/home/kyuta/.cache/torch_inductor
```

ログ: [`compile_full_optK.log`](../plots/profile_int8_blocks_2026-04-30/compile_full_optK.log) (初回) / `compile_full_optK_v2.log` (cache hit)

### B-13. (実験/失敗) Option L: 全 hot MLP/FFN を custom Triton kernel に置換

#### やったこと

`models/static_int8_fused.py` を新規作成:
- 2 つの Triton kernel
  - K1: bf16 → int8 quant (static scalar) → int8 GEMM (fc1) → dequant + bias + GELU/ReLU → int8 quant (static scalar)
  - K2: int8 → int8 GEMM (fc2) → dequant + bias → bf16
- 中間活性は **int8 のまま保持** (bf16 round trip 排除)
- `StaticInt8FusedMLP` クラス + calibration utility + 自動置換ヘルパー (encoder DINO Mlp / decoder XformerDecoderLayer FFN を検出)
- `video_demo.py` に `--static_int8_fused_mlp` flag (calibration frames 数 8 default)

実装後の挙動:
1. PTQ (torchao int8) 適用
2. 8 frame の calibration で各 hot module の `input_scale` / `intermediate_scale` (per-tensor scalar) 収集
3. encoder Mlp 14 + decoder XformerDecoderLayer FFN 6 + その他 1 = **21 個を `StaticInt8FusedMLP` に置換**

#### 結果

inference 平均 **50.4 ms** (Option K cache hit の 43-45 ms から **-13% 悪化**)、graph break 6 → 8 (+2)、recompile 10 → 14 (+4)、BackendCompilerFailed 0。

#### 失敗の真因

`@_dynamo.disable` で 21 module を graph 外に追い出した結果:
- 各 MLP の前後 (LayerNorm / residual add / dropout) と Inductor fusion ができなくなった
- 21 個の graph break boundary が新規追加 (encoder block 14 + decoder layer 6 + 1)
- per-MLP の launch 削減 (4-5 → 2 kernel/MLP) よりも fusion 損失のほうが大きかった

途中で Triton autotune が `M, N, K, ACTIVATION` を key にして毎フレーム再 tune する罠、kwarg 経由で autotune の key 計算が壊れる罠、`fc1.weight` が `LinearActivationQuantizedTensor` で `.abs()` 未実装の罠など複数のミスを修正済み。

ログ: [`compile_full_optL.log`](../plots/profile_int8_blocks_2026-04-30/compile_full_optL.log)

### B-14. (実験/失敗) Option L2: `allow_in_graph` で Inductor fusion 復活を狙う

#### やったこと

`StaticInt8FusedMLP.forward` の `@_dynamo.disable` を外し、内部で `@torch._dynamo.allow_in_graph` 付きヘルパー関数を呼ぶ構造に変更。これで Dynamo は MLP を opaque な node として **graph 内に保持**し、前後の Inductor fusion が維持される設計。

#### 結果

inference **47.5 ms** (L の 50.4 ms から -3 ms 改善、ただし Option K の 43-45 ms にはまだ届かず)、graph break 12、recompile **38** (+24)、BackendCompilerFailed 4。

steady-state it/s は ~14.6 で Option K と実質同等 (~67 ms total) だが、warmup の汚さが大幅増 (recompile 38)。custom Triton kernel 化のメリットが実測で確認できなかった。

ログ: [`compile_full_optL2.log`](../plots/profile_int8_blocks_2026-04-30/compile_full_optL2.log)

### B-15. (実験/失敗) Option L3: compile-native `StaticScaleInt8Linear` (decoder FFN のみ)

#### やったこと

「Option L/L2 の失敗理由は `static scale` がダメではなく、`custom Triton kernel` で graph boundary を作ったせいで Option K の Inductor fusion を壊したこと」という仮説を立て、custom Triton を捨てて以下を試行:

- `StaticScaleInt8Linear` クラスを `static_int8_fused.py` に追加
  - 静的 weight quant (per-output-channel)
  - 静的 calibrated activation scale (scalar)
  - 実装は **torch native ops (clamp, round, .to(int8)) + `torch._int_mm` のみ** — Inductor が graph に取り込む
  - **`@_dynamo.disable` も custom Triton kernel も使わない**
- `enumerate_decoder_ffn_linears` / `LinearCalibrationHooks` / `apply_static_scale_int8_decoder_ffn` を追加
- `--static_scale_int8_decoder_ffn` フラグ追加
- decoder の linear1 / linear2 (6 layer × 2 = 12 Linear) のみを `StaticScaleInt8Linear` に置換

#### 結果

| 指標 | Option K (cache hit) | **Option L3** | 評価 |
|---|---|---|---|
| inference 平均 | 43-45 ms | **50.2 ms** | ❌ 速くならず |
| Graph break | 6 | **6** | ✓ 維持 |
| Recompile | 10 | **10** | ✓ 維持 |
| BackendCompilerFailed | 0 | **0** | ✓ 維持 |

ユーザー定義の構造的成功条件 (Graph break/Recompile/BCF) は **3/4 達成**。しかし inference は速くならず。

#### 結論 (重要な仮説の反証)

> "速くならないなら、dynamic scale 計算は主因ではなく、torchao + Inductor がすでに十分強いという判断になります。"

**`per-token absmax` は支配的なオーバーヘッドではない**ことが実測で反証された。Inductor が torchao path で fusion する際に absmax reduction を周辺ops と統合済みで、static scale に置き換えても顕著な利得が出ない。

ログ: [`compile_full_optL3.log`](../plots/profile_int8_blocks_2026-04-30/compile_full_optL3.log)

---

## B-X. 現状の限界点と今後の改善余地

### 現状の壁

- **bf16 + xformers (compile / INT8 なし) で 29 ms** が出ている
- 本書の Option D (INT8 + compile encoder+decoder+heads) は **50.9 ms** で、bf16 baseline に **22 ms (+75%) 負けている**
- INT8 + compile 路線で 29 ms を切ることが目標

### bf16 baseline に負けている要因 (推定)

1. **INT8 quant/dequant overhead** が matmul の savings を上回る (batch=1、dim=768 という小サイズで顕著)
2. **graph break per-frame 6 件** で CUDA stream sync が 1-2 ms 浮く (boolean mask × 2、xformers attn_bias × 2、`@_dynamo.disable` × 1、forward_encoder boundary × 1)
3. **xformers `BlockDiagonalMask` の Python オブジェクト処理** が compile graph と相性悪く、decoder 側で type guard / obj_id guard による遅延
4. **torchao の `int8_mm` Triton kernel** が cuBLAS bf16 GEMM ほど最適化されていない可能性

### Option D に残っている全 graph break (6 件) と recompile (13 件) の内訳

| 由来 | graph break | recompile |
|---|---|---|
| **encoder** | **4 件** | **5 件** |
| **decoder** | **2 件** | **4 件** |
| **heads** | 0 件 | **4 件** |

#### encoder 由来 (graph break 4、recompile 5)

| 種類 | 場所 | 詳細 |
|---|---|---|
| Graph break (per-frame) | [`sat_model.py:463`](../models/sat_model.py#L463) | forward_encoder 内 xformers attn_bias 境界 |
| Graph break (per-frame) | [`sat_model.py:566`](../models/sat_model.py#L566) `lvl1_img_patches_28[lvl1_sat_mask]` | boolean mask indexing (data-dependent shape) |
| Graph break (per-frame) | [`sat_model.py:593`](../models/sat_model.py#L593) `_drop_register_tokens_concat` 呼び出し | 我々が意図的に `@_dynamo.disable` で外した listcomp ヘルパーの境界 (Step 2 の代償) |
| Graph break (per-frame) | [`sat_model.py:650`](../models/sat_model.py#L650) `token_lvls[-1][to_next_lvl]` | boolean mask indexing |
| Recompile (warmup) × 2 | `torch_dynamo_resume_in_forward_encoder_at_566` | stride 768 ↔ 1 反転 (`lvl1_sat_mask.sum()` が 1 / >1 でテンソル stride が変わる) |
| Recompile (warmup) | `get_attn_bias_and_cat` (block.py:173) | `2 <= L['x_list'][0].size()[1]` seqlen guard |
| Recompile (warmup) | `resume_in_get_attn_bias_and_cat_at_187` (block.py:187) | 同上 |
| Recompile (warmup) | `resume_in_forward_nested_at_257` (block.py:257) | `Ne(L['___stack0'][0].q_seqinfo.max_seqlen, 1)` — xformers `BlockDiagonalMask` の Python 属性ガード |

#### decoder 由来 (graph break 2、recompile 4)

| 種類 | 場所 | 詳細 |
|---|---|---|
| Graph break (per-frame) | [`decoder.py:154`](../models/decoder.py#L154) `mask2bias` | xformers `BlockDiagonalMask` 生成で分断 |
| Graph break (per-frame) | [`decoder.py:563`](../models/decoder.py#L563) `memory_efficient_attention` 直前 | `attn_bias` を Python オブジェクトで渡すため |
| Recompile (warmup) × 2 | `_dispatch__torch_function__` (torchao/utils.py:363) | linear1 (768→2048) と linear2 (2048→768) の shape 違い specialize |
| Recompile (warmup) | `memory_efficient_attention` (xformers) | `___check_obj_id(L['attn_bias'], ...)` ガード失敗 |
| Recompile (warmup) | [`forward_cross_query_proj`](../models/decoder.py#L443) | `___check_type_id(L['self']._modules['ca_qpos_proj'], ...)` (torchao 内部の type 切替) |

#### heads 由来 (graph break 0、recompile 4)

| 種類 | 場所 | 詳細 |
|---|---|---|
| Recompile (warmup) × 4 | `MLP.forward` (sat_model.py:995) | 4 種の shape/layer 違い (pose 6層 / shape 5層 / cam 3層 384hidden / bbox 3層 768hidden) で specialize |

### 完了済み (試行結果まとめ)

- ✅ **boolean mask → index_select 書き換え** (Option E/F): 単独では graph break 数を減らせない (内部 lowering が同じ)。`unflatten` + `_check_is_size` と組み合わせて -5.3%
- ✅ **`lvl_pooling` を `unflatten(0, (-1, 4))` 化** (Option F): 上記の改善の主因
- ✅ **`_check_is_size` ヒント** (Option F): nonzero 結果の SymInt を size-like にマーク。部分的に効果
- ✅ **nsys プロファイル取得** (Option F): kernel launch overhead が ~16 ms/frame と判明、これが gap の最大要因
- ✅ **`torch._check(N > 0)` / `torch._check(N % 4 == 0)` 追加** (Option J): -5%、ただし副作用 BackendCompilerFailed 4 件
- ✅ **Inductor `pre_grad_fusion_options` から `batch_linear` / `batch_linear_lhs` 除外** (Option K): BackendCompilerFailed 0、recompile -3、inference 43.0 ms 達成、fx_graph_cache 有効化で warmup 緩和
- ❌ **Inductor 強化 (use_mixed_mm, epilogue_fusion, coordinate_descent_tuning, recommended_inductor_config_setter)** (Option D2): warmup 爆発、inference 悪化、**revert 済み**
- ❌ **decoder attention を SDPA 置換** (Option G): 人物検出が壊れた、**revert 済み**
- ❌ **`--compile_dynamic` を外して static** (Option H): cache 溢れて 10x 悪化、**revert 済み**
- ❌ **`@_dynamo.disable` 外し系 (I_A, I_B, I_C)**: PyTorch 2.5.1 のバグで全部失敗、**revert 済み**
- ❌ **custom Triton kernel `StaticInt8FusedMLP`** (Option L / L2): fusion ロスで Option K に勝てず、experimental flag `--static_int8_fused_mlp` として保留
- ❌ **compile-native `StaticScaleInt8Linear` (decoder FFN のみ、torch._int_mm 利用)** (Option L3): inference 速くならず、experimental flag `--static_scale_int8_decoder_ffn` として保留 → **「per-token absmax は支配的でない」が判明**

### 方針確定 (2026-04-30)

- **default**: Option K (現状コード状態) を維持
- **`--static_int8_fused_mlp`**: experimental only (使わない限り影響なし)
- **`--static_scale_int8_decoder_ffn`**: experimental only (同上)
- **allow_in_graph / `torch.library.custom_op` 路線**: 保留
- **Triton BLOCK サイズ tuning**: やらない
- **次の本線**: 残 graph break / xformers object guard 削減
- **INT8 範囲は減らさない**: selective ablation で「INT8 から除外」する方向は不可、「INT8 化を増やす」方向のみ可

### 試行済み (Option I シリーズ、すべて失敗・revert 済み)

3 つの `@torch._dynamo.disable` を外す試みはすべて壊滅的に失敗:

- **I_A**: `position_encoding_xy` の disable 外し → Inductor `same_meta` SymInt バグ復活、77.6 ms 悪化
- **I_B**: `_drop_register_tokens_concat` の disable 外し → 微改善 -1.8ms だが副作用で decoder.py:469 全体が WON'T CONVERT、健全でない
- **I_C**: `_get_attn_bias_from_cache` の disable 外し → 221.5 ms (4.6x) 壊滅。BlockDiagonalMask の Python オブジェクト処理を Dynamo がトレースできず

**結論**: 3 つの disable はいずれも PyTorch 2.5.1 では「必要悪」として残すしかない。

### 採用済み (J / K)

- **Option J**: `torch._check(size > 0)` + `lvl_pooling` の `_check(N > 0 & N%4==0)` 追加 → -5%
- **Option K**: Inductor `pre_grad_fusion_options` から `batch_linear` / `batch_linear_lhs` 除外 → BackendCompilerFailed 解消、recompile -3、inference -11% (cache hit 時 43.0 ms)
- **fx_graph_cache** 有効化 (default) で 2 回目以降の warmup 大幅短縮 (68.8s → 29.5s)

### 今後やること (優先順、2026-04-30 時点)

#### 1. xformers `MemEffAttention.forward` を `@_dynamo.disable` でラップ (3 行変更、最小コスト)

encoder の attention.py にある `MemEffAttention.forward` を `@_dynamo.disable` で囲む。

**効果**:
- xformers `BlockDiagonalMask` の attribute access (`q_seqinfo.max_seqlen` など) が compile graph 外になる
- xformers obj_id / max_seqlen ガード由来の **recompile 3 件 (block.py:173/187/257) が消える** 期待

**コスト**:
- encoder 14 block × 1 attention call = **graph break +14 件**
- ただし各 break は per-frame で軽量、qkv proj と attention の Inductor fusion ロスは小さいはず

**やらない理由は無い**。warmup が短くなる効果も期待できる。

#### 2. `--int8_attn` を試す (CLI フラグのみ、INT8 範囲拡大)

現在は `--no_int8_attn` で xformers Flash の bf16 を使っている。`--int8_attn` で xformers INT8 attention カーネル (Q@K^T と attn@V を INT8 化) に切替。

過去メモリで「fp16 baseline 越えできず」と記録されているが、Option K + INT8 attn の組み合わせは未検証。INT8 範囲を広げる方針なので試す価値あり。

#### 3. xformers 完全廃止 (solution 6) ← **重い、最終手段**

xformers を全部 `F.scaled_dot_product_attention` + 自前 padded mask に書き換え。

**コスト試算**:
- メモリ: dense block-diagonal mask ≈ 24 MB × 14 layer = 336 MB GPU 追加
- 計算: block-diagonal Flash O(Σ Nᵢ²) → dense Flash O((ΣNᵢ)²) で数倍コスト増
- 開発: SAT scale-adaptive 機構の token 構築を全部書き換え。**3-5 日**
- 検証: 人物検出精度を一致させる試行錯誤必要 (Option G で類似失敗あり)

#### 4. PyTorch アップグレード (2.6+)

PyTorch 2.5.1 で衝突した:
- `same_meta` の SymInt バグ ([`_inductor/fx_passes/post_grad.py:643`](https://github.com/pytorch/pytorch))
- `guard_size_oblivious` が unflatten/reshape で完全には機能しない

これらが 2.6+ で修正されている可能性。アップグレードできるなら I_A, I_C 系を再試行する余地あり。

#### 5. (代替路線) FP8 化検討

GPU が H100 系 (Hopper SM 90+) ならば torchao の `float8_dynamic_activation_float8_weight` で BF16 比 2x throughput、精度劣化は INT8 より小さい。Ampere 以前は不可。

#### (廃案) 旧 「今後やること」

#### 1. PyTorch アップグレード (2.6+)

PyTorch 2.5.1 で衝突した:
- `same_meta` の SymInt バグ ([`_inductor/fx_passes/post_grad.py:643`](https://github.com/pytorch/pytorch))
- `guard_size_oblivious` が unflatten/reshape で完全には機能しない

これらが 2.6+ で修正されている可能性。アップグレードできるなら I_A, I_C 系を再試行する余地あり。

#### 2. (大工事) padded tensor + key_padding_mask への構造的書き換え (数日)

SAT-HMR の scale-adaptive token 機構が boolean mask indexing と可変長 split に依存しているのが根本原因。これを **固定長 padded tensor + mask** に書き換えると残り全ての graph break / recompile が構造的に消える。

#### 5. (代替路線) FP8 化検討

GPU が H100 系 (Hopper SM 90+) ならば torchao の `float8_dynamic_activation_float8_weight` で BF16 比 2x throughput、精度劣化は INT8 より小さい。Ampere 以前は不可。

#### 6. `Ne(4*u1, 4)` (= `u1 != 1`) data-dependent guard の解消 (PyTorch 依存)

sat_model.py:569 と `lvl_pooling` の `unflatten` で残る break。`guard_size_oblivious` が PyTorch 内部に追加されないと解消困難 (2.5.1 では未対応)。

### 採用しないと判断したもの (理由付き)

- **Inductor の `coordinate_descent_tuning` / `use_mixed_mm` 等の強化** (Option D2): warmup 爆発、改善なし (autotune が悪い kernel を選ぶ可能性あり)
- **decoder attention の SDPA 置換** (Option G): 人物検出が壊れる。`F.scaled_dot_product_attention` の `attn_mask` と xformers `attn_bias` の意味論差で出力が変わる
- **`--compile_dynamic` を外す** (Option H): SAT の可変 token 数で cache 溢れて 10x 悪化
- **`--compile_mode reduce-overhead` / `max-autotune-no-cudagraphs`**: モード変更は graph break や launch overhead の根本要因に効かない (ユーザー判断)
- **xformers `BlockDiagonalMask` を dense materialize**: 公式に「for debugging & testing」と明記、production では block-diagonal Flash の最適化を失うため悪手
- **3 つの `@_dynamo.disable` 外し** (Option I_A/B/C): PyTorch 2.5.1 のバグで全部失敗 (SymInt / aten.stack / RecursionError 等)
- **custom Triton kernel `StaticInt8FusedMLP`** (Option L / L2): static-scale fused MLP に置換しても Option K の Inductor fusion を超えられず。`@_dynamo.disable` でも `allow_in_graph` でも parity 程度で warmup が悪化。ただし experimental flag (`--static_int8_fused_mlp`) として残置
- **compile-native `StaticScaleInt8Linear` (decoder FFN のみ)** (Option L3): per-token absmax は torchao + Inductor の fusion で既に消化されており、static scale 化しても改善せず。experimental flag (`--static_scale_int8_decoder_ffn`) として残置
- **selective INT8 ablation (一部 Linear を bf16 に戻す)**: ユーザー方針で禁止。INT8 範囲は減らさない。
- **Triton BLOCK サイズ tuning**: ユーザー方針で深追いしない
- **活性化関数 (ReLU/GELU/SiLU/softmax) の INT8 化**: 利得小、精度劣化大
- **LayerNorm の INT8 化**: 同上

---

## C. 量子化対象を更に広げるとしたら

| 対象 | 方法 | 期待効果 / リスク |
|---|---|---|
| Attention の Q@K^T / attn@V | `--int8_attn` (今は `--no_int8_attn` で無効) | 期待効果中、ただし attn_bias materialize cost あり、過去実験で fp16 baseline 越えできず |
| LayerNorm | カスタム INT8 LN を実装 | 利得小、精度劣化大、推奨せず |
| FP8 化 | torchao の `float8_dynamic_activation_float8_weight` | H100 系なら 2x throughput、Ampere 以前は不可 |

- 活性化関数 (ReLU/GELU/SiLU) や softmax を INT8 化することは原則しない (利得小、精度劣化大)
- 量子化されない 35 個の Linear (出力 2/4 次元など) は head の最終投影で、INT8 化しても全体性能にほぼ影響しない

---

## D. ログ解析テンプレート (今後の再現用)

新しい `compile_full_*.log` を見るときの優先順位:

```bash
LOG=plots/profile_int8_blocks_2026-04-25/compile_full_xxx.log

# 1. PTQ と compile target を確認
grep "PTQ applied" $LOG
grep "torch.compile: target=" $LOG

# 2. graph break / recompile の総数
echo "Graph break:  $(grep -c '__graph_breaks.*Graph break:' $LOG)"
echo "Recompile:    $(grep -c '__recompiles.*Recompiling function' $LOG)"

# 3. graph break 発生位置 (改善対象)
grep -A2 "__graph_breaks.*Graph break:" $LOG \
  | grep -oE 'File "/home/kyuta/SAT-HMR/[^"]+", line [0-9]+' \
  | sort | uniq -c

# 4. recompile 発生関数
grep "__recompiles.*Recompiling function" $LOG \
  | grep -oE "function [a-zA-Z_]+ in [^:]+:[0-9]+" \
  | sort | uniq -c

# 5. CompilationMetrics の restart_reasons / fail_type (TORCH_LOGS="all" のとき有効)
grep -oE "restart_reasons=\{[^}]+\}" $LOG | sort | uniq -c | sort -rn
grep "fail_type='[^N]" $LOG | head

# 6. inference 平均と warmup
tr '\r' '\n' < $LOG | grep -E '^\s+(preprocess|inference|postprocess):|First frame completed'

# 7. recompile タイムスタンプ (warmup 内に収まっているか)
grep "__recompiles" $LOG | grep -oE 'V[0-9]+ [0-9:]+\.[0-9]+' | head -5
grep "__recompiles" $LOG | grep -oE 'V[0-9]+ [0-9:]+\.[0-9]+' | tail -5
```

---

## E. 新フラグ一覧 (2026-04-30 時点)

| フラグ | 状態 | 用途 |
|---|---|---|
| `--compile` | 標準 | torch.compile を有効化 |
| `--compile_encoder_only` | 標準 | model.forward_encoder を compile |
| `--compile_decoder_only` | 標準 | model.decoder を compile |
| `--compile_heads` | 標準 | pose/shape/cam/conf/bbox_embed の各 head を個別に compile |
| `--compile_dynamic` | 標準 | dynamic shape compile (SAT は必須) |
| `--compile_quantized_linear` | 標準 | torchao 量子化 Linear を compile graph に含める |
| `--warmup_frames N` | 標準 | 平均算出から最初 N frame を除外 |
| `--static_int8_fused_mlp` | **experimental** (使うな) | 21 hot MLP/FFN を Triton fused kernel に置換 (Option L 系、効果無し) |
| `--static_int8_calibration_frames N` | experimental | 上記の calibration frames 数 |
| `--static_scale_int8_decoder_ffn` | **experimental** (使うな) | decoder FFN を compile-native StaticScaleInt8Linear に置換 (Option L3、効果無し) |

## F. 関連する変更ファイル一覧

| ファイル | 変更内容 |
|---|---|
| [`video_demo.py`](../video_demo.py) | `--compile_heads`, `--warmup_frames`, `--static_int8_fused_mlp`, `--static_int8_calibration_frames`, `--static_scale_int8_decoder_ffn` フラグ追加、`_dynamo.config.capture_dynamic_output_shape_ops = True` 追加、`pre_grad_fusion_options` から `batch_linear` / `batch_linear_lhs` を除外、encoder/decoder 同時 compile 対応、calibration + 置換ロジック追加 |
| [`models/sat_model.py`](../models/sat_model.py) | `_drop_register_tokens_concat`, `_normalize_pos_per_lvl`, `_build_x_list_multi`, `_build_pos_list_multi`, `_build_lvl_id_list_multi` の 5 ヘルパー関数を追加 (全部 `@torch._dynamo.disable`)、`lvl_pooling` を `unflatten` 化 + `_check_is_size`/`_check`、boolean mask を `index_select` ベースに書き換え |
| [`models/encoders/dinov2/models/vision_transformer.py`](../models/encoders/dinov2/models/vision_transformer.py) | `_post_block_outputs` ヘルパー関数を追加 (`@torch._dynamo.disable`)、`forward_specific_layers_list` と `forward_additional_layers_list` の listcomp を置換 |
| [`models/position_encoding.py`](../models/position_encoding.py) | `position_encoding_xy` に `@torch._dynamo.disable` を追加 (Inductor バグ回避) |
| [`models/static_int8_fused.py`](../models/static_int8_fused.py) | **新規ファイル** (~600 行)。Option L 系の experimental 実装: Triton kernel 2 個 (`_static_int8_fc1_act_quant_kernel`, `_static_int8_fc2_kernel`)、`StaticInt8FusedMLP`、`StaticScaleInt8Linear` (Option L3)、calibration utility (`CalibrationHooks`, `LinearCalibrationHooks`)、置換ヘルパー (`apply_static_int8_fused`, `apply_static_scale_int8_decoder_ffn`)。default では使用されない。 |

---

## G. 2026-05-01 作業ログ (graph break 個別撲滅試行)

graph break #5 → #4 → #2 → #3 → #6 → #1 の優先順で個別撲滅を試行中。

### G.0.0 graph break 番号定義 (#1〜#6)

K (43.9 ms) 時点の 6 graph break を以下の番号で扱う。優先度は section B-12 の per-break コスト推定 + 修正容易性で並べた。

| # | 状態 | 場所 | 内容 | 修正方針 |
|---|---|---|---|---|
| **#1** | ✅ **解消** (Kp2k6a1a) | [block.py:164](../models/encoders/dinov2/layers/block.py#L164) `_get_attn_bias_from_cache` | `@_dynamo.disable` 関数で xformers `BlockDiagonalMask` 生成 → 呼び出し境界 | `get_attn_bias_and_cat` で single batch + single x_list の fast path 追加。`attn_bias=None` で xformers 直接呼び、forward_nested の split() も None 経路で自前 reshape。trim mean -4.5 ms 改善。詳細 G.11 |
| **#2** | ⚠️ **部分解消** (Kp2k) | [sat_model.py:573-590](../models/sat_model.py#L573) reshape 後 `Ne(u1, 1)` | `lvl1_sat_patches_28 = torch.cat([..., [:1]])` 後の `permute → reshape(-1, C)` で SAT patch 数 1 / >1 のガード失敗 | 元 reshape GB は消去成功 (40.0 ms 達成)。**ただし副次効果で line 609 / 616 の listcomp に新規 BCF=2 出現**、GB total は 5→6 で +1。inference は -1.0 ms 改善するが GB 数は減らない |
| **#3** | ❌ **解消不可** | [sat_model.py:609 / 616](../models/sat_model.py#L609) lvl0_pos_embed listcomp / x_list listcomp | Kp2k 採用時に発生した listcomp + split BCF (`Eq(4*u0, 0)`) | **9 案試行全て失敗** (3a〜3k)。BCF を消すと downstream `forward_specific_layers_list` の Inductor compile bug が露呈し output 崩壊。BCF は eager fallback で bug を覆う保護機構として機能している。詳細 G.9 |
| **#4** | ✅ **改善** (Kp4a) | [sat_model.py:268-274](../models/sat_model.py#L268) `lvl_pooling` の `unflatten(0, (-1, 4))` | `tokens.size(0)` SymInt に対し `Eq(u0, 4)` の data-dependent guard | module-level `@_dynamo.disable` 関数 `_lvl_pool4` に切り出し。Kp4a 結果: 41.0 ms (Kp -1.0 ms) |
| **#5** | ✅ **解消** (Kp) | [position_encoding.py:12](../models/position_encoding.py#L12) `position_encoding_xy` | PyTorch 2.5.1 Inductor `same_meta` SymInt バグ (`_inductor/fx_passes/post_grad.py:643` で `node.meta['val']` が SymInt scalar の場合 `.size()` で AttributeError) | `safe_same_meta` monkey-patch で SymInt non-Tensor 値を skip。position_encoding_xy 内部は変えず `@_dynamo.disable` だけ削除 (G.5 Kp 結果: 42.0 ms) |
| **#6** | ✅ **解消** (Kp2k6a) | [decoder.py:563](../models/decoder.py#L563) `get_decoder_block_diag_bias` 呼び出し (call stack 上層は 154 / 289) | `@_dynamo.disable` helper で xformers `BlockDiagonalMask` を生成 → boundary が GB | single_batch (batch=1) なら BlockDiagonalMask 不要 → `attn_bias=None` で xformers 直接呼び出し。decoder.py:154/289 GB が消滅。詳細 G.10 |

> **注記**: #1 〜 #6 の番号は section G 内専用ローカル ID。section A の Step 番号や Option ラベル (F/J/K 等) とは独立。

本日は **#5 (G.5 Kp で解消) と #4 (構造変更を伴うため未着手)** を処理。

### G.0 ⚠️ 本セクションの測定値は全て無効

本日の試行 (G.1 / G.2 の全 Option Q / Q1 / X / 4a / R / K_check 等) は **`--compile_heads` を付けず `--int8_skip_modules` で heads を除外した別環境** で取られた。Option K の正規構成は section A (line 90) に明記の通り:

```bash
--compile --compile_encoder_only --compile_decoder_only --compile_heads --compile_dynamic
# --int8_skip_modules は空 (= heads も INT8 化、quantized=228)
```

しかし本日の測定は誤って下記コマンドで実行:
```bash
--compile --compile_encoder_only --compile_decoder_only --compile_dynamic
--int8_skip_modules pose_head shape_head cam_head conf_head bbox_embed
```

結果: PTQ `quantized=154` (heads 除外), compile target=`decoder+encoder` (heads compile されず)。**Option K の 43.0 ms は再現せず、本日の K_check は 88.2 ms**。本日の Q/Q1/X/4a/R 各 Option の数値は K (43.0 ms) との比較に使えない。

**今後の作業は section A (line 84-95) のコマンドを起点に再測定すること**。

### G.1 #5: `position_encoding_xy` (models/position_encoding.py)

**原因**: PyTorch 2.5.1 の `_inductor/fx_passes/post_grad.py:643 same_meta()` が `node.meta['val']` に SymInt scalar が入った場合に `.size()` を呼んで `AttributeError: 'SymInt' object has no attribute 'size'` で死ぬバグ。`position_encoding_xy` を Dynamo に通すと必ず踏む。

**試行 (全て不採用)**:

| 試行 | 内容 | 結果 |
|---|---|---|
| **Option Q** | `stack(...).flatten(1)` を `torch.empty + out[:, 0::2] = sin / out[:, 1::2] = cos` の strided write に書き換え (`@_dynamo.disable` 削除) | inference 43.7 ms / `BackendCompilerFailed: 4` (同じ same_meta バグを strided write でも踏む) |
| **Option Q1** | `torch.view_as_real(torch.complex(sin, cos)).reshape(-1, half_dim)` で interleave (`@_dynamo.disable` 削除) | inference 44.8 ms / `BackendCompilerFailed: 4` (同じ) |
| **Option X** | Q1 + `torch._inductor.fx_passes.post_grad.same_meta` を monkey-patch して SymInt non-Tensor 値を safe-skip | inference **105.9 ms** に大幅悪化。`BackendCompilerFailed: 0` にはなったが `view_as_real(complex(...))` 経路の compile 結果が Inductor 側で遅い kernel になる |

**確定方針 (Option Y)**: `@torch._dynamo.disable` を維持し eager 実行に固定。graph break #5 として残るが、Q/Q1 比較で速度影響無しが確定 (eager fallback と同等)。**PyTorch 2.5.1 のこのバグは user-side で根絶不能**。

### G.2 #4: `lvl_pooling` の `unflatten(0, (-1, 4))` (models/sat_model.py:268)

**原因**: `tokens.size(0)` が SymInt の状態で `unflatten(0, (-1, 4))` を呼ぶと Dynamo が `Eq(u0, 4)` の data-dependent guard を解決できず graph break。過去に試した `_check_is_size + _check(N > 0) + _check(N % 4 == 0)` は `_check(>1)` 相当が SAT patch=1 の edge case で runtime fail。`max_pool1d` 版は遅い。

**試行 (候補 α、不採用)**: module-level に `@torch._dynamo.disable` 付き `_lvl_pool4(tokens)` ヘルパーを切り出し、`Model.lvl_pooling` をその呼び出しのみに変えた。

**結果**: inference **86.6 ms** (K の 43-45 ms から +90% の大幅悪化)、recompile 5 → 6、graph break 6 のまま。原因推定: メソッド境界が module-level helper への外部関数呼び出しに変わり、周辺の Inductor 融合 / 呼び出しコンテキストが壊れた。**revert 済み**。

**確定方針**: **#4 は触らない**。Option K の `Model.lvl_pooling` のままが最速。「graph break を `@_dynamo.disable` に置換すれば速度影響無し」という #5 の結論は **#4 では成り立たなかった** (= helper 切り出しのコストが想像より大きかった)。

### G.3 残作業

- [ ] **#2** [sat_model.py:573 の `Ne(u1, 1)` reshape](../models/sat_model.py#L573) — #4 と同型 (1 個ケースの判定)
- [ ] **#3** `_drop_register_tokens_concat` の disable — listcomp/zip/split-free な書き換え (e.g., `repeat_interleave`, `segment_id`)
- [ ] **#6** [decoder.py:563 の xformers `BlockDiagonalMask` 境界](../models/decoder.py#L563) — 非 SDPA approach
- [ ] **#1** encoder `_get_attn_bias_from_cache` disable — 最も危険 (I_C precedent)

### G.4 学んだこと

1. **PyTorch 2.5.1 の `same_meta` SymInt バグ** は user-side で書き換えても回避不能 (interleave 書き込みパターンを変えても踏む)。monkey-patch で迂回しても compile された kernel 自体が遅くなる。`@_dynamo.disable` で諦めて eager 実行が最善。
2. **data-dependent guard (`Eq`, `Ne`) の `_check`** は edge case で runtime fail するため危険。「N > 0 かつ N が ある倍数の倍数」のような複合 invariant を `_check` で表現するのは現状無理筋。**幅広く成り立つ invariant 以外は使わない**、もしくは関数全体を `@_dynamo.disable` する方が安全。
3. graph break 個別撲滅は **形を変えるだけで速度に効かない**ケースが多い。`@_dynamo.disable` 化は count を減らさないが速度劣化も無い (Q/Q1 で確定)。「graph break カウント」よりも「eager fallback の中で重い処理が無いか」を計測指標にすべき。

### G.5 proper K 環境での再開 (2026-05-01 後半)

G.0 で確認した通り本日前半の測定は無効。proper K (`--compile_heads --warmup_frames 5`、`--int8_skip_modules` 空、quantized=228) を再現:

| Option | 構成 | inference | 結果 |
|---|---|---|---|
| **K (再現)** | section A の正規コマンドそのまま | **43.9 ms** | K の 43.0 ms と一致 (誤差 +0.9 ms) → baseline 確定 |

#### Kp: K + same_meta monkey-patch のみ

G.1 の Option X は **Q1 (view_as_real) 書き換え + patch** で 105.9 ms と遅かった。今回 (Kp) は **position_encoding_xy は元のまま (stack/flatten)** で `@_dynamo.disable` だけ削除し、video_demo.py に `patch_inductor_same_meta()` を仕込む。Inductor バグ起源を断ちつつ position_encoding_xy 内部は触らないので、view_as_real が遅かった件の影響も無いはず。

変更:
- `models/position_encoding.py`: `@torch._dynamo.disable` の 1 行のみ削除
- `video_demo.py`: import 直後に下記 patch を追加

```python
def patch_inductor_same_meta():
    import torch._inductor.fx_passes.post_grad as pg
    orig_same_meta = pg.same_meta
    def safe_same_meta(node1, node2):
        val1 = node1.meta.get("val")
        val2 = node2.meta.get("val")
        if val1 is None or val2 is None:
            return False
        if not hasattr(val1, "size") or not hasattr(val2, "size"):
            return False
        return orig_same_meta(node1, node2)
    pg.same_meta = safe_same_meta

patch_inductor_same_meta()
```

実行コマンド: section A のものに `--metrics_csv int8_blocks_metrics_Kp.csv` と `> plots/profile_int8_blocks_2026-04-30/compile_full_Kp.log 2>&1` を追加するだけ。

判定基準:
- inference ≦ 45 ms かつ BCF=0 → **patch 採用、#5 解消**
- inference > 50 ms または BCF > 0 → **patch 不採用、`@_dynamo.disable` を維持** (G.1 Option Y と同じ)

#### Kp 結果 (2026-05-01)

| 指標 | K (baseline) | **Kp** | 差分 |
|---|---|---|---|
| inference | 43.9 ms | **42.0 ms** | **-1.9 ms (-4.3%)** |
| Graph break | 6 | 5 | -1 (**#5 解消**) |
| Recompile | 10 | 11 | +1 |
| BackendCompilerFailed | 0 | 0 | 同 |
| PTQ quantized | 228 | 228 | 同 |

**判定**: 採用基準クリア。**Kp が新しい default**。

確定変更:
- `models/position_encoding.py`: `@torch._dynamo.disable` 削除
- `video_demo.py`: import 直後に `patch_inductor_same_meta()` を呼ぶブロックを追加

`G.1` の Option Q/Q1/X が遅かった原因の切り分けが完了:
- Option Q/Q1 (rewrite だけ、patch 無し): same_meta バグで BCF 4 件 → eager fallback で K と同等
- Option X (rewrite + patch): view_as_real(complex(...)) の compile kernel 自体が遅い (105.9 ms)
- **Kp (rewrite なし + patch のみ)**: 元の stack/flatten 形が compile されて K より速い (42.0 ms)

つまり position_encoding_xy は **書き換えるな、Inductor 側だけ直せ** が結論。

### G.6 残作業 (#4 → #2 → #3 → #6 → #1)

- [x] **#4** ✅ **解消** (Kp4a 2026-05-01): `Model.lvl_pooling` を module-level `@_dynamo.disable` 関数 `_lvl_pool4` に切り出し → 41.0 ms (Kp 比 -1.0 ms)。G.2 の wrong env 測定で「helper 化は遅い」と書いたが proper K では逆に **-1.0 ms 改善**。GB/RC/BCF 変化なし。詳細 G.7 参照。
- [ ] **#2** [sat_model.py:573 の `Ne(u1, 1)` reshape](../models/sat_model.py#L573)
- [ ] **#3** `_drop_register_tokens_concat` の disable
- [ ] **#6** [decoder.py:563 の xformers `BlockDiagonalMask` 境界](../models/decoder.py#L563)
- [ ] **#1** encoder `_get_attn_bias_from_cache` disable

### G.7 #4 改善 (Kp4a 採用、GB 解消は不可)

#### 試行 6 案
| 案 | 実装 | inference | GB | 備考 |
|---|---|---|---|---|
| **4a** | `@_dynamo.disable` helper `_lvl_pool4` に切り出し、内部は元の `unflatten` のまま | **41.0 ms** | 5 | **採用** |
| 4b | `tokens[0::4]` 等の stride スライスと `torch.maximum` 連鎖 | 41.2 ms | 5 | slice 長 `(u0+3)//4` の `Eq(..., 1)` で GB 残存 |
| 4c | `view(quart, 4, C).amax(dim=1)` + `_check_is_size(quart)` | 41.1 ms | 5 | view が同型 guard で GB 残存 |
| 4d | caller 側で `group_starts = next_idx[0::4]` を作って `prev_tokens.index_select(0, group_starts + i)` を i=0,1,2,3 で 4 回呼び、`torch.maximum` 連鎖で集約 (`Model.lvl_pooling` メソッド + `_lvl_pool4` helper はインライン化により削除) | 44.2 ms | **4** | **#4 唯一の消去成功**、ただし速度悪化 |
| 4e | `idx4 = group_starts[:, None] + arange(4)` → `index_select(idx4.reshape(-1)).view(G, 4, C).amax(1)` | 42.1 ms | 5 | `view(G, 4, C)` が `Ne(4*u1+4, 4)` で guard、Dynamo は picked.size(0) = G*4 を追跡できず |
| 4f | 同 idx4 を `flat_idx = idx4.flatten()` で 1 回 `index_select`、`picked[0::4]` 等を chunk 化 + maximum 連鎖 | 41.4 ms | 5 | `group_starts[:, None]` の unsqueeze 経路で guard |
| 4g | advanced indexing `token_lvls[-1][idx4]` で直接 (G, 4, C) を作って `amax(dim=1)` | 41.6 ms | 5 | 同様の unsqueeze/broadcast 経路で guard |

#### Kp4a 採用変更
- [models/sat_model.py:73-76](../models/sat_model.py#L73): module-level `@_dynamo.disable` 関数追加
  ```python
  @torch._dynamo.disable
  def _lvl_pool4(tokens):
      assert len(tokens) % 4 == 0
      return torch.max(tokens.unflatten(0, (-1, 4)), dim=1)[0]
  ```
- [models/sat_model.py:274](../models/sat_model.py#L274): `Model.lvl_pooling` を `return _lvl_pool4(tokens)` のみに

#### 累積結果
| 指標 | K | Kp | **Kp4a** | 累積差分 (vs K) |
|---|---|---|---|---|
| inference | 43.9 ms | 42.0 ms | **41.0 ms** | **-2.9 ms (-6.6%)** |
| Graph break | 6 | 5 | 5 | -1 |
| Recompile | 10 | 11 | 11 | +1 |

**結論**: **#4 の GB 自体は除去不可** (`Eq(u0, 4)` / `Eq((u0+3)//4, 1)` / `Ne(4*u1+4, 4)` / view-side 同型、6 案中 5 案で data-dependent guard が残る)。GB を消せるのは 4d (4 個別 index_select) のみだが、Inductor fusion を壊して **+3.2 ms 悪化**。helper boundary を作って eager fallback に逃がすのが最速。

GB 数は減らないが、内部の data-dependent 計算が Inductor の重い guard 経路を踏まないので **41.0 ms に微改善**。

**根本原因**: caller の `next_idx` が `to_next_lvl.nonzero()` から来ているため Dynamo に対して `len(next_idx) % 4 == 0` を証明できない。`unflatten` / `view(G, 4, C)` / stride slice / unsqueeze 経路のいずれでも shape を「N//4」と Dynamo に伝える術がない (本来は `_check(N % 4 == 0)` で済むはずだが PyTorch 2.5.1 の Dynamo はこの divisibility 情報を活用しない)。`@_dynamo.disable` で eager に逃がすのが現実解。

**custom op / Triton kernel 化 (4j) は未着手**: 数百行の実装コストに対し期待ゲインは < 1 ms (per-break ≈ 0.5-0.8 ms)。過去 Option L / L2 / L3 の Triton 試行が全て失敗した経緯もあり保留。

**ログ**: [`plots/profile_int8_blocks_2026-04-30/compile_full_Kp4a.log`](../plots/profile_int8_blocks_2026-04-30/compile_full_Kp4a.log) (4a) / `compile_full_Kp4b.log` 〜 `compile_full_Kp4g.log` (4b 〜 4g)

### G.8 #2 改善試行 (Kp2k 採用、BCF=2 残存)

#### 試行 7 案 (2c〜2kd)

| 案 | 実装 | inference | GB | BCF | 備考 |
|---|---|---|---|---|---|
| 2c | `_patch_proj_norm` helper を `@_dynamo.disable` で切り出し (Conv2d + permute + reshape + LayerNorm 全部) | **75.6 ms** (+34) | 5 | 0 | helper boundary が **周辺の compile cache を完全崩壊**、recompile cascade 多発 (position_encoding_xy stride / decoder forward / MLP forward 全て再 trigger)。first frame compile 28.9s → 146.8s |
| 2d | `reshape(-1, C)` を排除、LayerNorm を 4D `(N+1, 2, 2, 768)` のまま適用 → `[:-1].flatten(0, 2)` | 43.6 ms | 5 | 0 | `flatten(0, 2)` が `Eq(u1*(s7//14)*(s8//14), -1)` の guard で graph break |
| 2h | LayerNorm 4D → `[:-1]` → 明示 `_N * 4` で reshape (`-1` 自動推論を回避、`_check_is_size(_out_size)` 付き) | 44.7 ms | 5 | 0 | reshape の整合性検証で `Eq(768*u1*(s7//14)*(s8//14), 3072*u1)` を踏む。Dynamo は `s7//14 * s8//14 = 4` を静的に証明できない (s7, s8 は dynamic shape) |
| 2i | conv 出力 (N, 768, H, W) を `view(N, 768, 4).transpose(1, 2)` で空間 4 を hard-code → LayerNorm → `[:-1]` → reshape `_N1 * 4` | crash | - | - | `reshape(_N, 768, 4)` が同型の s7/s8 guard で BCF、recompile cascade で `torch._check(sat_n > 0)` が runtime 評価され sat_n=0 frame で raise → script crash |
| 2j | conv 出力の 4 つの空間位置 `[:, :, 0, 0]` 〜 `[:, :, 1, 1]` を整数 indexing で取り出して `torch.stack(dim=1)` (4, 768) → LayerNorm → `[:-1]` → reshape `_N1 * 4` | crash | - | - | 整数 indexing が `Eq(4*u0, 0)` の新規 guard を踏み BCF=4。同じく `_check(sat_n > 0)` で crash |
| 2j2 | 2j + `torch._check(sat_n > 0)` を削除して runtime crash を解消 (`_check_is_size` のみ残す) | 41.9 ms | 6 | 4 | crash は治ったが BCF=4 (4 件 compile 失敗、eager fallback) + GB +1。Kp4a と速度同じだが構造的指標悪化 |
| **2k** | 2j2 + dummy padding を **`sat_idx` レベルで** 行う (`sat_idx_pad = cat([sat_idx, zeros(1)])` で 1 個追加 → index_select で必ず sat_n+1 個取得) → 4 位置 stack → LayerNorm → reshape `M * 4` (M = sat_n+1 >= 1) → 最後に `[:-4]` で dummy 4 token 除去 | **40.0 ms** | 6 | 2 (4 events) | **採用**。reshape の入力が必ず M >= 1 なので shape 計算に 0 が出ない。BCF=2 unique は eager fallback で機能的に問題なし、persons 検出も正しい (528 frames 全て persons=1) |
| 2kb (試行/不採用) | 2k + line 610 listcomp `torch.cat([full_pos_embed[ys,xs] for ...])` を直接 fancy indexing `full_pos_embed[lvl0_pos_y, lvl0_pos_x]` に置換 | 41.3 ms | 5 | 1 | line 610 由来 BCF を 1 件解消、ただし fancy indexing 経路で speed 微悪化 |
| 2kc (試行/不採用) | 2kb + line 620 listcomp `[torch.cat([cr, lvl0]).unsqueeze(0) for ...]` を `@_dynamo.disable` helper `_build_x_list_single` に切り出し | 42.5 ms | 5 | 0 | **BCF=0 だが output 崩壊**: persons 検出が `[0:166, 1:578, 2:35, 3:5, 4:1]` の異常分布 (Kp2k は全て 1)。`@_dynamo.disable` helper による Inductor compile が誤った kernel を生成。**revert** |
| 2kd (試行/不採用) | 2kc + `forward_nested` 内で `for _x in x_list: torch._dynamo.mark_dynamic(_x, 1)` を追加して `2 <= x_list[0].size()[1]` recompile を消そうとした | 55.8 ms (大幅悪化) | 8 (+3) | 0 | **完全に逆効果**。`mark_dynamic` 自体が block 毎に新たな specialize boundary を作り GB +3 / RC +3、対象の `2 <= ...size()[1]` recompile も消えなかった |

#### Kp2k 採用変更

**変更 1**: [models/sat_model.py:569-590](../models/sat_model.py#L569) 周辺 (lvl0_tokens 生成: 2k):

```python
sat_idx = lvl1_sat_mask.nonzero(as_tuple=True)[0]
sat_n = sat_idx.size(0)
torch._check_is_size(sat_n)
# torch._check(sat_n > 0) は削除 (sat_n=0 frame の eager fallback crash を防ぐため)
sat_idx_pad = torch.cat([sat_idx, sat_idx.new_zeros(1)], dim=0)
lvl1_sat_patches_28 = lvl1_img_patches_28.index_select(0, sat_idx_pad)  # (sat_n+1, 3, 28, 28)
_proj_out = self.encoder_patch_proj[0](lvl1_sat_patches_28)  # (sat_n+1, 768, 2, 2)
M = _proj_out.shape[0]
torch._check_is_size(M)
lvl0_tokens = torch.stack(
    [
        _proj_out[:, :, 0, 0],
        _proj_out[:, :, 0, 1],
        _proj_out[:, :, 1, 0],
        _proj_out[:, :, 1, 1],
    ],
    dim=1,
)  # (sat_n+1, 4, 768)
lvl0_tokens = self.encoder_patch_norm[0](lvl0_tokens)
lvl0_tokens = lvl0_tokens.reshape(M * 4, lvl0_tokens.shape[-1])
lvl0_tokens = lvl0_tokens[:-4]
```

**変更 2 / 3 (line 610 fancy index、line 620 helper) は不採用**: 2kb / 2kc を試したが、2kc で persons 検出が崩壊した (BCF=0 と引き換えに correctness 破壊)。**両方 revert**。Kp2k は line 610 / 620 の listcomp を**そのまま維持**。

#### 累積結果 (K → Kp → Kp4a → Kp2k)

| 指標 | K | Kp | Kp4a | **Kp2k** | 累積差分 (vs K) |
|---|---|---|---|---|---|
| inference | 43.9 ms | 42.0 ms | 41.0 ms | **40.0 ms** | **-3.9 ms (-8.9%)** |
| Graph break | 6 | 5 | 5 | 6 | 同 |
| Recompile | 10 | 11 | 11 | 9 | -1 |
| BackendCompilerFailed | 0 | 0 | 0 | **2 unique (4 events)** | +2 |

**Kp2k 採用判断**:
- 速度ベスト (40.0 ms)、persons 検出も正しい (528 frames 全て persons=1)
- BCF=2 は eager fallback で機能的に問題なし (line 620 listcomp `lvl0_tokens.split(lvl0_token_lens)` が `Eq(4*u0, 0)` guard を踏むが、Dynamo が eager fallback で対応するため出力は正しい)
- 2kc で BCF を 0 にしようとしたが output が崩壊したため不採用

**注意 (#2 部分解消)**:

Kp4a → Kp2k の GB 場所変化:

| Kp4a の GB | Kp2k の GB | 状態 |
|---|---|---|
| sat_model.py:576 (= #2 元の reshape) | (消失) | 消去成功 |
| sat_model.py:605 (= #3 _drop_register_tokens_concat) | sat_model.py:622 に移動、GB list 外 | 統合 |
| sat_model.py:668 (= #4 lvl_pooling caller) | sat_model.py:275 (lvl_pooling 内へ移動) | 移動 |
| - | **sat_model.py:609 (新規)** | listcomp `Eq(4*u0, 0)` BCF |
| - | **sat_model.py:616 (新規)** | listcomp `Eq(4*u0, 0)` BCF |

GB count: 5 → 6 (+1)、BCF count: 0 → 2。**「#2 単体は消えたが副次効果で別箇所に新規 GB/BCF 発生」**。
inference は -1.0 ms 改善しているので採用したが、GB 数の意味では **逆行**している。次の #3 で line 609 / 616 を倒す必要あり。

#### 試行した recompile 削減策 (不採用)

**Kp2kd (mark_dynamic 試行)**: PyTorch 2.7+ の `torch.compiler.config.dynamic_sources` API は 2.6.0 で利用不可だったため代替として `torch._dynamo.mark_dynamic(x_list[0], 1)` を `forward_nested` 内 (block.py:257 直前) に挿入。

結果: inference 55.8 ms (大幅悪化)、GB +3、RC +3。`mark_dynamic` 自体が encoder 各 block 呼び出し毎に新たな specialize boundary を作り、対象の `2 <= ...size()[1]` recompile も消えなかった。**完全 revert**。

#### 残課題

**recompile (warmup 中) 残存**:
- `2 <= L['x_list'][0].size()[1]` (block.py:173 / 187 / 254 等で複数発火)
- `Ne(L['attn_bias'].q_seqinfo.max_seqlen, 1)` (xformers `BlockDiagonalMask` の Python 属性、`mark_dynamic` 適用不可)

これらは warmup 中のみで steady-state には影響なし (`--warmup_frames 5` で除外される)。撲滅するには xformers attn_bias の構築経路を変更するか、padding で max_seqlen=1 ケースを構造的に消す必要がある (大工事)。当面保留。

**ログ**: `compile_full_Kp2c.log` 〜 `compile_full_Kp2j2.log` (2c 〜 2j2、不採用) / [`compile_full_Kp2k.log`](../plots/profile_int8_blocks_2026-04-30/compile_full_Kp2k.log) (**採用**) / `compile_full_Kp2kb.log` (不採用) / `compile_full_Kp2kc.log` (output 崩壊で不採用) / `compile_full_Kp2kd.log` (mark_dynamic 試行、不採用)

### G.9 #3 改善試行 (全 9 案失敗、解消不可と確定 2026-05-06)

#### 試行 9 案

| 案 | 実装 | inference | GB | BCF | persons | 備考 |
|---|---|---|---|---|---|---|
| **Kp2k (起点)** | listcomp + split (BCF で eager fallback) | **40.0 ms** | 6 | 2 | ✓ | baseline |
| 3a | `_drop_register_tokens_concat` の `@_dynamo.disable` 削除 + 簡略化 | 71.9 ms | 6 | 2 | ✓ | compile cache 崩壊 |
| 3b | helper 削除 + caller inline (batch=1 fast path) | 41.1 ms | 6 | 2 | ✓ | 効果なし |
| 3c | `forward_specific_layers` (single tensor 経路) | 59.3 ms | 5 | 2 | ✓ | xformers Flash 非適用で大幅悪化 |
| 3d | helper を `@_dynamo.allow_in_graph` に変更 | 40.5 ms | 6 | 2 | ✓ | 効果なし |
| 3e | dummy 4 後送り + padded pos + 1-element list literal | (実測あり) | 3 | 0 | ❌ 崩壊 | 最も改善大、ただし line 616 literal で output 崩壊 |
| 3f | line 609 / 616 を両方 `@_dynamo.disable` helper 化 | (output 崩壊で測定打ち切り) | - | - | ❌ | 推定 output 崩壊 |
| 3g | batch=1 if/else で line 609 fancy + line 616 1-element list | 39.2 ms | 4 | **0** | ❌ 崩壊 | BCF 解消するも output 崩壊 |
| 3h | batch=1 if/else で **line 609 のみ** fancy (line 616 元のまま) | 40.6 ms | 5 | 2 | ✓ | GB -1 のみ、speed +0.6 ms |
| 3k | dummy 4 保持 + padded で listcomp 維持 (line 616 listcomp 自体は維持) | 55.0 ms | 4 | **0** | ❌ 崩壊 | listcomp 維持でも padded data で compile 成功して bug 露呈 |

#### 重要な発見

**「line 616 listcomp BCF は forward_specific_layers_list の Inductor compile bug を eager fallback で覆い隠している保護機構」**:

- BCF を残す (Kp2k / 3a〜3d / 3h): output ✓
- BCF を消す (3e / 3f / 3g / 3k): **output 崩壊**
- listcomp を維持しても padded data で compile 成功してしまえば bug が露呈 (3k で確認)

つまり「BCF=0 にする = forward_specific_layers_list が compile 経路を通る = bug を踏む」が必然。Kp2k の listcomp+split パターンが踏む `Eq(4*u0, 0)` BCF は、**意図せず正しい挙動を保証している副作用**。

#### #3 不採用、Kp2k で確定

**user-side では line 616 由来 BCF を解消不可**。これは PyTorch 2.6 の Inductor compile bug (forward_specific_layers_list 経路での xformers 連鎖時に発火) の修正が必要だが、本体側の対処。当面 Kp2k 状態を default として維持。

3h は GB -1 と微小改善するが speed +0.6 ms で実質トントン、構造的にも line 609 の fancy index が batch=1 限定で他の path に副作用が無いかは未確認のためリスク残る。**Kp2k のまま**を採用。

**ログ**: `compile_full_Kp2k3a.log` 〜 `compile_full_Kp2k3k.log` (3a 〜 3k)

### G.10 #6 改善 (Kp2k6a 採用)

#### 変更内容

[models/decoder.py:561-571](../models/decoder.py#L561) の cross-attention 経路:

```python
single_batch = len(tgt_lens) == 1 and len(memory_lens) == 1
if XFORMERS_AVAILABLE and not int8_attention_enabled(self):
    if single_batch:
        # batch=1 では BlockDiagonalMask 不要、attn_bias=None で xformers 直接呼び
        tgt2 = memory_efficient_attention(q, k, v, attn_bias=None)
    else:
        attn_bias = get_decoder_block_diag_bias(tgt_lens, memory_lens, q.device)
        tgt2 = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
```

batch=1 inference では `get_decoder_block_diag_bias` (`@_dynamo.disable` 関数) の呼び出しを skip → boundary 消滅。

#### 結果

| 指標 | Kp2k | **Kp2k6a** | 差分 |
|---|---|---|---|
| simple mean | 40.0 ms | 66.0 ms | +26 (frame 9 spike で歪む) |
| **trimmed mean (-top 5%)** | 37.49 ms | **35.68 ms** | **-1.8 ms 改善** |
| p50 (median) | 37.39 ms | 35.77 ms | -1.6 |
| p90 | 39.57 | 39.21 | -0.4 |
| p99 | 41.85 | 45.81 | +4.0 |
| max | 1229 ms | **15678 ms** | frame 9 で 15.7 秒 spike |
| Graph break | 6 | **5** | -1 (decoder.py:154/289 が消滅) |
| Recompile | 9 | 6 | -3 |
| BCF | 2 | 2 | 同 |
| persons | 全 1 | **全 1** | ✓ correctness OK |

#### 重要な発見: trimmed mean で見るべき

simple mean だと frame 9 の 15.7 秒 spike (encoder 側 xformers seqlen guard recompile) で大幅悪化に見えるが、median / trimmed mean では **-1.8 ms 改善**。**user の指摘で trimmed mean / 中央値 で見るべきと判明**。

frame 9 の spike は encoder 側 (block.py:173/187/257 の xformers `BlockDiagonalMask` seqlen guard) が原因で、decoder 変更 (#6) とは独立。warmup タイミングのズレで発火。

#### 残課題: warmup 範囲

現状 `--warmup_frames 5` だが frame 9 で recompile spike が起きるため、平均値が歪む。**warmup 範囲を見直すか、warmup 検出を自動化するかの議論が必要** (TBD)。

**ログ**: [`compile_full_Kp2k6a.log`](../plots/profile_int8_blocks_2026-04-30/compile_full_Kp2k6a.log)

### G.11 #1 改善 (Kp2k6a1a 採用)

#### 変更内容

[models/encoders/dinov2/layers/block.py:173-195](../models/encoders/dinov2/layers/block.py#L173) `get_attn_bias_and_cat` に single batch fast path 追加:

```python
def get_attn_bias_and_cat(x_list, branges=None):
    # #1a: single batch + single x_list fast path
    # BlockDiagonalMask 不要 → `_get_attn_bias_from_cache` (@_dynamo.disable) 呼び出し回避
    if branges is None and len(x_list) == 1 and x_list[0].shape[0] == 1:
        cat_tensors = x_list[0].reshape([1, -1, *x_list[0].shape[2:]])
        return None, cat_tensors
    # ... 元の listcomp + cache 経路
```

[block.py:257-263](../models/encoders/dinov2/layers/block.py#L257) `forward_nested` 末尾の split() を None 対応:

```python
attn_bias, x = get_attn_bias_and_cat(x_list)
x = x + attn_residual_func(x, attn_bias=attn_bias)
x = x + ffn_residual_func(x)
if attn_bias is None:
    return [x.reshape([x_list[0].shape[0], -1, *x.shape[2:]])]
return attn_bias.split(x)
```

#### 結果

| 指標 | Kp2k6a | **Kp2k6a1a** | 差分 |
|---|---|---|---|
| **trim mean (-top 5%)** | 35.68 ms | **31.15 ms** | **-4.5 ms 改善** |
| p50 (median) | 35.77 ms | **31.19 ms** | -4.6 |
| p90 | 39.21 ms | 33.11 ms | -6.1 |
| p99 | 45.81 ms | 35.26 ms | -10.5 |
| max | 15678 ms | 17933 ms | +2.3s (frame 9 spike) |
| Graph break | 5 | 8 (+3) | encoder 経路で listcomp 由来の boundary 増 |
| Recompile | 6 | 5 | -1 |
| BCF | 2 | 2 | 同 |
| persons | 全 1 | 全 1 | ✓ |

#### 累積結果 (K → Kp2k6a1a)

| 指標 | K | Kp | Kp4a | Kp2k | Kp2k6a | **Kp2k6a1a** | 累積差分 (vs K) |
|---|---|---|---|---|---|---|---|
| trim mean | ~43 | ~42 | ~41 | 37.49 | 35.68 | **31.15** | **-12 ms (-28%)** |

**bf16 baseline 29 ms との残ギャップ: 2.2 ms** (Kp4a 比 -10 ms 改善)。

#### I_C precedent (過去の失敗) との違い

過去 Option I_C は `_get_attn_bias_from_cache` の `@_dynamo.disable` を**削除**して 221 ms に大幅悪化。今回は disable を**保持**して single batch 経路で skip するアプローチで成功。

**ログ**: [`compile_full_Kp2k6a1a.log`](../plots/profile_int8_blocks_2026-04-30/compile_full_Kp2k6a1a.log)

### G.12 bf16 baseline 調査 (2026-05-07)

#### 過去の「29 ms baseline」と現環境の乖離

[`docs/precision_benchmark_summary.md:43-51`](precision_benchmark_summary.md#L43) に記録:

| 出典 CSV | dtype | median ms |
|---|---|---|
| bnb_speed_candidates_fp16.csv | fp16 | 28.93 ms |
| bnb_full.csv | bf16 | 30.10 ms |
| torchao_fast_candidate.csv | bf16 eager | 35.47 ms |

過去 `docs/int8_compile_optimization_2026-04-25.md` で「bf16 + xformers baseline は 29 ms」と書いていたが、**実際は fp16 best 値 (28.93 ms)** との混同。bf16 baseline は本来 30 ms (eager) 〜 35 ms (compile)。

#### 現環境での再測定 (2026-05-07)

| 構成 | weight | warmup | trim mean | p50 |
|---|---|---|---|---|
| bf16 naked + xformers (#1a あり) | sat_644.pth (fp32→bf16 cast) | 5 | 51.50 ms | 51.35 |
| bf16 naked + xformers (#1a なし) | sat_644.pth | 5 | 51.22 ms | 51.23 |
| bf16 naked + xformers (#1a あり) | sat_644.pth | 10 | 52.62 ms | 52.77 |
| bf16 naked + xformers (#1a あり) | **sat_644_bf16.pth** | 10 | **55.13 ms** | 55.44 |

#### 検証 (compile / INT8 が完全無効)

`compile_full_bf16naked_w10.log` で:
- `torch.compile` メッセージ無し ✓
- `PTQ applied` / `quantized=` メッセージ無し ✓
- `[DTYPE] model dtype: torch.bfloat16` ✓
- First frame 246 ms (compile spike 無し) ✓
- xformers Flash 利用 ✓

設定は完全に bf16 naked。それでも 51-55 ms。

#### 結論

- **過去 29 ms 値は再現不可** (現環境で 51-55 ms)
- 過去測定時のコマンド構成不明、PyTorch / xformers / GPU 環境が変化した可能性
- `sat_644_bf16.pth` (bf16 専用 weight) を使っても改善せず (むしろ +4 ms)

#### 現在の真の比較

| 構成 | trim mean | 備考 |
|---|---|---|
| bf16 naked (現環境真の baseline) | 51-55 ms | xformers Flash のみ |
| Kp2k6a1a (INT8 + compile) | **31.15 ms** | -20 ms = **-39% 改善** |

過去の「目標 29 ms」は環境依存の値で、**現環境での実質的な比較は bf16 naked 51 ms vs Kp2k6a1a 31 ms**。我々の最適化は **bf16 baseline を 39% 上回って勝っている**。

#### 重要な学び (2026-05-06)

**BCF=0 と correctness はトレードオフになる場合がある**:
- 2kc で `_build_x_list_single` を `@_dynamo.disable` helper にして BCF=0 を達成したが、persons 検出が `[0:166, 1:578, 2:35, 3:5, 4:1]` の異常分布になった (Kp2k は全て persons=1)
- 同じ helper パターンを使っても 2kd (`mark_dynamic` 追加) では correctness 維持 (`mark_dynamic` で eager fallback 経路に逃げるため)
- → `@_dynamo.disable` helper の中で Inductor が compile した kernel が誤動作する条件があり、推測しづらい。Kp2k のように **BCF を眺めるだけで disable 化を強行しない方が安全**

**変更前後では必ず persons 検出を検証**:
```bash
LOG=plots/profile_int8_blocks_2026-04-30/compile_full_<TAG>.log
tr '\r' '\n' < $LOG | grep -oE 'persons=[0-9]+' | sort | uniq -c
```
正常時は全て persons=1 (sample_video.mp4 の場合)。分布が変わったら correctness が崩れている。

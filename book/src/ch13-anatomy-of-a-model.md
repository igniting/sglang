# 13. Anatomy of a Model

> *Models are rewritten rather than imported because every layer must cooperate with
> parallelism, quantization, and the KV cache — and llama.py shows exactly how.*

Hugging Face already ships a `LlamaForCausalLM`. SGLang has its own, and so do the other 217
models in this repository. That duplication needs justifying.

The justification is that the reference implementation makes four assumptions a serving
engine cannot accept — about whether weights are whole, whether attention is self-contained,
whether numbers are dense floats, and whether shapes can change between calls. None of them
can be patched from outside, because they are properties of every layer.

This chapter reads the Llama implementation closely enough that you will be able to open any of the other
217 files and know what you are looking at. It is the most concrete chapter in the book: a
single file, top to bottom, with the reason for each decision.

Watch for one thing in particular. Almost nothing in the file is about caching, sharding,
quantization, or graph capture, even though all four are happening. They live in the layers
the model is built from. That is what makes adding a model a bounded task rather than an
expert one, and it is why Chapter 23's checklist is as short as it is.

<!-- objectives:begin -->
<div class="bk-objectives"><p class="bk-objectives-head">What this chapter gives you <span class="bk-objectives-time">· about 9 min</span></p><ul><li>Name the four components of a modern decoder block and what each replaced</li><li>Read llama.py end to end and say what each layer choice is for</li><li>Explain why models are rewritten rather than imported</li><li>Distinguish a model's total head count from this rank's</li></ul></div>
<!-- objectives:end -->

---

## The contract

A model must provide:

```python
def forward(self, input_ids, positions, forward_batch) -> LogitsProcessorOutput
def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]])
```

`forward_batch` is Chapter 7's `ForwardBatch`, and it carries everything the model needs to
know about *how* it is being run. Everything else is optional capability, declared by
defining a method.

---

## The block, before the code

The file we are about to read assumes you know what a modern decoder block contains. It is
four ideas, each of
which replaced something simpler for a reason worth knowing, because each reason shows up
later as a constraint on the engine.

**RMSNorm instead of LayerNorm.** LayerNorm subtracts the mean and divides by the standard
deviation. Zhang and Sennrich (2019) observed that the re-centering does almost nothing for
transformers and dropped it:

```
RMSNorm(x) = x / sqrt(mean(x²) + ε) × g
```

One reduction instead of two, no mean, no bias. For the engine this matters twice: it is one
of the three kernels Chapter 8's determinism work has to make batch-invariant, and being a
single reduction is what makes it cheap enough to fuse with the residual add and the
downstream quantization, as this chapter's `LlamaDecoderLayer` does.

**SwiGLU instead of a plain MLP.** The classic feed-forward block is
`W₂ · act(W₁x)`. A gated linear unit splits the first projection in two and uses one half to
gate the other. With SiLU as the activation (Shazeer, 2020):

```
SwiGLU(x) = W_down · ( SiLU(W_gate · x)  ⊙  (W_up · x) )
```

Three matrices instead of two, so the hidden dimension is usually scaled by 2/3 to keep the
parameter count level. The engine cares because `W_gate` and `W_up` read the *same* input and
have the *same* shape — which is exactly the condition for fusing them into one GEMM, and the
reason `gate_up_proj` exists below.

**RoPE instead of learned position embeddings.** Rather than adding a position vector to the
input, rotary embeddings (Su et al., 2021) *rotate* each pair of dimensions in Q and K by an
angle proportional to the token's position. For position *m* and dimension pair *i* with
frequency *θ_i* = base^(−2*i*/*d*):

```
q̃_m = R(mθ) q_m ,   k̃_n = R(nθ) k_n
```

Because rotations compose, the inner product *q̃_m* · *k̃_n* depends on *m* − *n* and not on *m*
and *n* separately. Absolute positions go in; relative position comes out of the dot product
for free. Three engine consequences follow. RoPE is applied to Q and K *after* projection and
*before* the cache write, so cached keys are already rotated and a KV entry is valid at
whatever position it was written at — which is what makes Chapter 10's prefix sharing sound.
Context extension becomes a matter of rescaling *θ* rather than retraining, which is why
`rope_scaling` is a config key that the loader has to handle several dialects of. And
Chapter 14's MLA has to work around RoPE specifically, because a rotation cannot be commuted
through a low-rank compression.

**GQA instead of MHA.** Chapter 1 showed that KV cache size is the direct limiter on batch
size, and standard multi-head attention stores one K and one V per query head. Shazeer's
multi-query attention (2019) cut that to a single shared KV head — a 64× reduction for a
64-head model — but degraded quality and destabilized training. Ainslie et al. (2023)
interpolated: divide the query heads into *G* groups and give each group one KV head.

```
G = H     →  multi-head attention (full cache)
1 < G < H →  grouped-query attention
G = 1     →  multi-query attention (smallest cache)
```

Their T5-XXL numbers make the case: MHA at 47.2 ROUGE-1 and 1.51 s per inference step, MQA at
46.6 and 0.28 s, GQA-8 at 47.1 and 0.28 s — MHA's quality at MQA's speed. They also showed
you do not need to train from scratch to get it: mean-pool an existing checkpoint's KV
projections down to *G* heads and continue training for about 5% of the original compute.

Llama-3-70B is GQA with 64 query heads and 8 KV heads, so its KV cache is one eighth of what
MHA would need — the difference between Chapter 1's 320 KB per token and 2.5 MB. The cost
appears in this chapter's constructor, where the head counts no longer divide evenly by the
tensor-parallel size.

---

## A full read of llama.py

### `LlamaMLP` (`python/sglang/srt/models/llama.py:70`)

The feed-forward block, and the first appearance of the fusion pattern. A Llama MLP is
`down(silu(gate(x)) * up(x))` — two parallel projections up, one back down. SGLang
constructs `gate_up_proj` as a single `MergedColumnParallelLinear` rather than two
matrices.

The reason is arithmetic intensity again (Chapter 1). `gate` and `up` read the *same*
input and have the same shape; running them as one GEMM loads that input once and doubles
the work per byte. This is why Chapter 12's `stacked_params_mapping` has to split one
checkpoint tensor across two slices — the fusion is SGLang's, not the checkpoint's.

Column-parallel for the up-projections and row-parallel for the down-projection is the
standard Megatron pattern, and Chapter 16 explains why that specific pairing needs exactly
one collective per block.

### `LlamaAttention` (`:138`)

The constructor is where tensor parallelism becomes concrete:

```python
        tp_size = get_parallel().tp_size
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
```

The `total_` prefix convention is worth internalizing: `total_num_heads` is the model's,
`num_heads` is this rank's. Getting these confused is the single most common bug when
writing a model.

The branch is grouped-query attention meeting tensor parallelism. Llama-3-70B has 64 query
heads and 8 KV heads. On 8-way TP each rank gets 8 query heads and 1 KV head — clean. On
16-way TP there are more ranks than KV heads, so KV heads must be **replicated**: two ranks
hold the same KV head and therefore store the same KV cache entries twice. That is real
> [!warning] Past the KV head count, tensor parallelism stops buying cache
> When the parallel size exceeds the number of KV heads, the heads are replicated rather
> than split, so two ranks store identical cache entries. Adding GPUs still adds compute
> and weight capacity — it just stops adding the thing that was scarce.

memory duplication, and it is why raising TP does not reduce KV memory per GPU beyond the
KV head count. Chapter 16's data-parallel attention is one answer.

Then the layers:

```python
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(...)

        self.rotary_emb = get_rope(...)
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )
```

Four things to notice.

`QKVParallelLinear` fuses three projections and knows the GQA head ratio, so it shards Q
and KV correctly despite their different counts.

`quant_config` is threaded into every layer. A layer decides its own kernel; the model just
passes the config down (Chapter 15).

`prefix=add_prefix(...)` builds the parameter's dotted name, which must match the
checkpoint. This is the thread connecting the module tree to Chapter 12's loader — and to
quantization configs that name specific layers to skip.

And `RadixAttention` (Chapter 10) takes `layer_id`. That is the whole cache integration: the
layer knows which slice of the pool is its own, and the rest — which pages to read, where
to write — arrives through `forward_batch`. **The model contains no cache logic**, which is
why adding a model does not require understanding Chapters 9 through 10.

### `LlamaDecoderLayer` (`:283`)

The constructor is mostly config archaeology:

```python
        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is not None:
            rope_theta = rope_parameters.get("rope_theta", 10000)
            rope_scaling = rope_parameters
        else:
            rope_theta = getattr(config, "rope_theta", 10000)
            rope_scaling = getattr(config, "rope_scaling", None)
        ...
        # Support llamafy/Qwen-Qwen2.5-7B-Instruct-llamafied with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False
        )
```

This is what "Llama architecture" means in practice: dozens of checkpoints that are
*nearly* Llama, each differing in a config key. Note this file uses `getattr` with defaults
despite `.claude/rules/no-getattr-defensive.md` — legitimately, because these fields
genuinely may not exist in a third-party config. The rule targets defensive access to
fields that *are* always present; reading foreign configs is the exception it allows.

The forward is where a real optimization hides:

```python
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(
                hidden_states, quant_linear=self.self_attn.qkv_proj
            )
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual, quant_linear=self.self_attn.qkv_proj
            )
```

The `residual` is threaded *through* the layer rather than added at the end, so
`input_layernorm` performs add-and-normalize as one fused kernel. Over 80 layers that saves
80 separate elementwise-add kernel launches and 80 round-trips of the hidden state through
memory — decode is memory-bound, so this is not a micro-optimization.

`quant_linear=self.self_attn.qkv_proj` is subtler: the norm is told which linear layer
consumes its output, so if that layer wants FP8 input the norm can *emit* FP8 directly,
fusing the quantization into the normalization. The layer boundary is deliberately leaky
because the fusion is worth it.

### `LlamaModel` (`:372`) and `LlamaForCausalLM` (`:496`)

`LlamaModel` is embeddings plus the layer stack, and `:640` `start_layer` / `:644`
`end_layer` are its pipeline-parallel bounds (Chapter 16) — this rank builds only its own
layers, which is what Chapter 12's weight filter pairs with.

`LlamaForCausalLM` adds the LM head and the logits processor (Chapter 8), and is where the
optional capabilities live:

| Method | Enables | Chapter |
| --- | --- | --- |
| `:599` `forward_split_prefill` | layer-split execution for PD multiplexing | 17 |
| `:891` `set_eagle3_layers_to_capture` | EAGLE-3 hidden-state capture | 18 |
| `:905` `set_dflash_layers_to_capture` | DFlash capture | 18 |
| `:854` `get_embed_and_head` / `:857` `set_embed_and_head` | RL weight sync | 11 |
| `:888` `load_kv_cache_scales` | quantized KV scales | 14 |
| `:650` `get_module_name_from_weight_name` | LoRA target resolution | 20 |

Six subsystems reaching into the model, each through one narrow method. A model that
defines none of them still works; it just cannot participate in those features. This is the
extension mechanism: **capability by method presence**, which is why the "add a model"
checklist is short.

`:918` `Phi3ForCausalLM` subclasses `LlamaForCausalLM` outright — same architecture,
different config parsing.

---

## The parallel layer vocabulary

Model files are written almost entirely in terms of a small set of layers from
`python/sglang/srt/layers/`:

| Layer | Shards | Communication |
| --- | --- | --- |
| `python/sglang/srt/layers/linear.py:293` `ColumnParallelLinear` | output dim | none (output stays sharded) |
| `python/sglang/srt/layers/linear.py:1392` `RowParallelLinear` | input dim | all-reduce after |
| `python/sglang/srt/layers/linear.py:492` `MergedColumnParallelLinear` | several column-parallel fused | none |
| `python/sglang/srt/layers/linear.py:921` `QKVParallelLinear` | Q/K/V fused, GQA-aware | none |
| `python/sglang/srt/layers/linear.py:195` `ReplicatedLinear` | nothing | none |
| `python/sglang/srt/layers/vocab_parallel_embedding.py:188` `VocabParallelEmbedding` | vocabulary | all-reduce |
| `python/sglang/srt/layers/vocab_parallel_embedding.py:587` `ParallelLMHead` | vocabulary | all-gather (Chapter 8) |

**The collectives live inside these layers.** A model file contains no `all_reduce` call.
Chapter 16 explains why column-then-row needs exactly one; the point here is that a model
author gets it right by choosing the correct layer type.

`python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/layers/activation.py`, and
`python/sglang/srt/layers/rotary_embedding/` supply the rest, mostly as fused kernels.

---

## Three contrast studies

**`python/sglang/srt/models/deepseek_v2.py`** — MLA and MoE. Attention is replaced by
multi-head latent attention, which compresses KV into a single latent vector per token
(Chapter 9's `MLATokenToKVPool`) and needs its own attention backends (Chapter 14). The MLP
is replaced by a mixture of experts (Chapter 17). Almost nothing carries over from Llama
except the residual structure — a useful demonstration that the "contract" really is just
`forward` and `load_weights`.

**A Qwen-VL variant** — a vision tower feeding a text decoder. The extra work is not in the
decoder but in *splicing*: encoder outputs replace placeholder tokens in the embedding
sequence, and positions become 3D (Chapter 21).

**`python/sglang/srt/models/falcon_h1.py`** — a hybrid where some layers are attention and
others are Mamba-style state-space. Different layers need different cache types in the same
model, which is what Chapter 9's `HybridLinearKVPool` and Chapter 14's
`python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` exist for.

Across all three, what stays fixed is the shape of a model file: a config-driven
constructor, layers from the shared vocabulary, a `forward` taking `forward_batch`, and a
`load_weights` mapping checkpoint names to sharded parameters. What varies is everything
else — which is exactly why the abstraction is drawn where it is.

One line of that file has been standing in for the hardest operation in the model. Chapter
14 goes behind it.

---

<!-- summary:begin -->
<div class="bk-card"><p class="bk-card-head">Chapter 13 in one page</p><ol class="bk-card-arg"><li>RMSNorm, SwiGLU, RoPE, and GQA each replaced something simpler, and each replacement shows up as an engine constraint.</li><li>GQA exists because the KV cache is the scarce resource — it is Chapter 1's argument reaching back into the architecture.</li><li>A model file is a `forward` and a `load_weights` over a shared vocabulary of parallel, quantization-aware layers.</li><li>Fusions like `gate_up_proj` are SGLang's, not the checkpoint's, which is why loading has to split one tensor across two slices.</li><li>The model contains no cache logic at all; `RadixAttention` takes a layer id and the rest arrives through the forward batch.</li></ol><p class="bk-card-sub">Numbers worth keeping</p><table class="bk-card-table"><tbody><tr><th scope='row'>Llama-3-70B query heads / KV heads</th><td>64 / 8</td></tr><tr><th scope='row'>KV cache saved by GQA at that ratio</th><td>8×</td></tr></tbody></table><p class="bk-card-sub">Where it lives</p><table class="bk-card-table"><tbody><tr><th scope='row'>The template model</th><td><code>python/sglang/srt/models/llama.py</code></td></tr><tr><th scope='row'>Parallel layers</th><td><code>python/sglang/srt/layers/linear.py</code></td></tr></tbody></table></div>
<!-- summary:end -->

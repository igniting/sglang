# 15. Tensor, Pipeline, and Data Parallelism

> *The three classical parallelism axes differ in what they split and therefore in which
> interconnect they stress; SGLang adds a fourth because MLA broke the assumptions.*

---

## Why there are several

A 70B model in BF16 is 140 GB. No single GPU holds it, so it must be split. But "split"
admits several answers, and they are not interchangeable — each cuts along a different
dimension and pays a different communication bill.

- **Tensor parallelism (TP)** cuts *within* each layer. Every rank does part of every
  operation. Communication is frequent — twice per transformer block — and must be fast.
- **Pipeline parallelism (PP)** cuts *between* layers. Rank 0 runs layers 0–19, rank 1 runs
  20–39. Communication is rare (once per boundary) and tolerates slower links.
- **Data parallelism (DP)** does not split the model at all: each rank holds a full replica
  and serves different requests.
- **Expert parallelism (EP)** splits MoE experts across ranks. Chapter 16.
- **Context parallelism (CP)** splits the *sequence*.

`python/sglang/srt/distributed/parallel_state.py:2285` `initialize_model_parallel` names all
of them in one signature:

```python
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    attention_data_parallel_size: int = 1,
    attention_context_model_parallel_size: int = 1,
    moe_data_model_parallel_size: int = 1,
    decode_context_parallel_size: int = 1,
    ...
```

Seven dimensions. Note that data parallelism appears *twice* — once for attention, once for
MoE — which is the sign that this is not the textbook taxonomy. The docstring for
`decode_context_parallel_size` is candid about its scope:

> number of GPUs used for decode context parallelism, which splits the KV cache across GPUs
> within each tensor-parallel group during decoding. Must be a divisor of
> tensor_model_parallel_size and is currently only supported on the AMD HIP platform.

That is the honest state of a fast-moving area: a real technique, constrained to one
platform.

---

## Groups and collectives

`python/sglang/srt/distributed/parallel_state.py:237` `GroupCoordinator` wraps a process
group with the operations SGLang needs. Every parallelism axis gets one, and a rank belongs
to several at once — a rank is simultaneously in a TP group, a PP group, and possibly a DP
and EP group, each with different membership.

`:2193` `init_distributed_environment` establishes the world; `initialize_model_parallel`
carves the sub-groups. `python/sglang/srt/model_executor/model_runner.py:1037`
`init_torch_distributed` is the caller, and it runs first in Chapter 6's initialization
chain because everything after it may need a collective.

The collectives and their costs:

| Operation | What it does | Cost |
| --- | --- | --- |
| all-reduce | sum across ranks, all receive | 2×(N−1)/N × data |
| all-gather | concatenate across ranks | (N−1)/N × data |
| reduce-scatter | sum, each rank keeps a slice | (N−1)/N × data |
| all-to-all | every rank sends a slice to every other | the MoE bill (Chapter 16) |

The numbers matter against the link. NVLink between GPUs in a node runs at hundreds of
GB/s; InfiniBand between nodes at tens. **That gap is why TP does not cross node boundaries
cheaply** — an all-reduce twice per layer over the slower link would dominate the forward
pass. It is the single most important fact for choosing a parallelism layout.

`python/sglang/srt/distributed/device_communicators/` holds the implementations that beat
the default: custom all-reduce (a hand-written kernel faster than NCCL for the small
messages decode produces), PyNCCL, quick all-reduce, and symmetric memory. Small-message
latency, not bandwidth, is what decode is sensitive to.

---

## TP is already written

Here is the pleasant surprise: after Chapter 12, there is almost nothing left to explain
about tensor parallelism.

The collectives live *inside* the layers. `ColumnParallelLinear` splits its output
dimension, so each rank produces a slice of the output and no communication is needed.
`RowParallelLinear` splits its input dimension, so each rank computes a partial sum, and one
all-reduce completes it.

Chain them — column then row — and the intermediate stays sharded:

```
  x ──► ColumnParallelLinear ──► [sharded activation] ──► RowParallelLinear ──► all-reduce ──► y
```

**One all-reduce per pair.** That is why Chapter 12's `LlamaMLP` uses
`MergedColumnParallelLinear` for gate/up and `RowParallelLinear` for down, and why
`LlamaAttention` uses `QKVParallelLinear` then `RowParallelLinear`. Two collectives per
transformer block, and a model author gets them right by picking the correct layer type.

`python/sglang/srt/layers/communicator.py` is the per-layer strategy object for the cases
where the default pattern is not what you want — sequence-parallel norms, or fusing the
all-reduce into an adjacent operation.

The head-count constraints from Chapter 12 are the practical limit: TP size must divide the
head count, and when TP exceeds the KV head count, KV heads are replicated and cache memory
stops shrinking.

---

## PP splits layers

Pipeline parallelism gives each rank a contiguous range of layers.
`python/sglang/srt/models/llama.py:640` `start_layer` and `:644` `end_layer` are the bounds,
and Chapter 11's weight filter uses them so a rank loads only its own layers.

Communication is a single activation tensor per boundary — cheap enough for
inter-node links, which makes PP the axis you extend across nodes when TP has filled one.

The cost is the **bubble**. With four stages, rank 0 works on microbatch 1 while ranks 1–3
idle; only after the pipeline fills are all ranks busy. Microbatching narrows the bubble
but never closes it, and in decode — where each step is short — the fill and drain are a
larger fraction than in training.

`python/sglang/srt/managers/scheduler_pp_mixin.py` implements microbatch scheduling in
Chapter 4's loop, and `python/sglang/srt/managers/scheduler.py:4136`
`_pp_microbatches_drained` is what "idle" has to mean when microbatches may still be in
flight.

---

## MLA breaks TP

Now the interesting part.

Chapter 8 introduced `MLATokenToKVPool`: DeepSeek's compressed KV cache, an order of
magnitude smaller per token than standard MHA. Chapter 12 explained that when TP exceeds
the KV head count, KV heads get replicated.

MLA has effectively **one** KV head. So under 8-way TP, all eight ranks store *the same* KV
cache. The compression that made the cache small is spent immediately on replication: eight
copies of a cache one-eighth the size is exactly the memory you started with.

That is the problem data-parallel attention exists to solve.

---

## Data-parallel attention

The idea: for the attention part of the model, stop splitting by head and start splitting by
*sequence*. Each rank owns whole requests, computes their attention alone, and stores only
their KV. No replication.

But the MLP is still tensor-parallel — it is large and benefits from splitting. So each
layer does:

```
  attention:  rank r handles its own sequences        (no communication)
      │
      ▼  all-gather: assemble the full batch
  MLP:        tensor-parallel across all ranks
      │
      ▼  scatter: each rank takes its own sequences back
  attention:  next layer
```

`python/sglang/srt/layers/dp_attention.py:338` `initialize_dp_attention` sets it up, `:412`
`get_dp_local_info` computes a rank's slice, and the gathers are `:450`
`_dp_gather_via_all_reduce` and `:494` `_dp_gather_via_all_gather`.

Two gather strategies exist because of `:76` `DpPaddingMode`:

```python
class DpPaddingMode(IntEnum):

    # Padding tokens to max length and then gather tokens using `all_gather_into_tensor`
    MAX_LEN = auto()
    # Padding tokens to sum length and then gather tokens using `all_reduce`
    SUM_LEN = auto()
```

`MAX_LEN` pads every rank to the largest token count and uses all-gather — wasteful when
ranks are unbalanced, but symmetric. `SUM_LEN` pads to the total and uses all-reduce —
better when counts differ. `:91` `get_dp_padding_mode` chooses, and its comment records a
hard-won constraint:

```python
        # (trangdough) pplx-kernels a2a is a symmetric collective: every EP rank
        # must dispatch the same number of tokens or the device-side handshake
        # deadlocks (idle DP ranks with 0 tokens never signal their peers).
        # Force MAX_LEN so all ranks are padded to equal token counts.
```

An MoE all-to-all kernel deadlocks if a rank contributes zero tokens, so the padding mode is
forced. Chapter 16's communication requirements reaching back into Chapter 15's padding
decision, with the failure mode being a *hang* rather than an error.

---

## The price: everyone must agree

DP attention's cost is synchronization. Because ranks share the MLP, they must agree on the
batch shape *before* the layer runs — and that agreement has to happen inside the forward
pass, since it depends on what every rank received.

`python/sglang/srt/model_executor/forward_batch_info.py:1305` `prepare_mlp_sync_batch` is
that agreement, and `:1620` `post_forward_mlp_sync_batch` undoes the padding afterward.
`python/sglang/srt/model_executor/model_runner.py:1416` `_prepare_eager_forward_batch` is
where it is invoked, with the comment Chapter 6 quoted about why the decode CUDA graph path
can skip it.

Now Chapter 6's `IDLE` mode makes sense:

> No sequence to forward. For data parallel attention, some workers will be IDLE if no
> sequence are allocated.

A rank with no requests still has to run the forward pass, because the other ranks need it
in their collectives. It contributes nothing and consumes a full step of GPU time.
`python/sglang/srt/layers/dp_attention.py:302` `set_is_extend_in_batch` and `:317`
`is_dp_max_padding` propagate the shape decisions, and
`python/sglang/srt/layers/logits_processor.py:249` `compute_dp_attention_metadata` carries
them to the output stage (Chapter 7).

Chapter 5's scheduling implication is real: keeping DP ranks *balanced* matters, because an
unbalanced batch means padding, and padding is wasted compute on every rank.

---

## The other DP

`python/sglang/srt/managers/data_parallel_controller.py` implements a completely different
thing that shares the name: whole-replica routing. Each DP group is an independent engine
with its own weights and its own cache; the controller routes requests between them.

The two have nothing in common. DP attention splits *one* model's attention across ranks
that cooperate every layer; the DP controller runs *n* independent models that never
communicate. Chapter 2's `use_dp_controller` branch in `_launch_scheduler_processes` selects
the latter.

When someone says "DP size 8," ask which one they mean.

---

## Choosing a layout

The decision procedure, in order:

1. **Fit the weights.** TP within a node until the model fits. TP 8 on an 8-GPU node is the
   default for large models.
2. **Still too large?** Add PP across nodes. Inter-node links tolerate PP's rare, small
   transfers; they would not tolerate TP's.
3. **MLA model?** Consider DP attention instead of pure TP, to stop replicating the
   compressed cache.
4. **MoE model?** EP is a separate axis on top (Chapter 16).
5. **Enough capacity but need throughput?** DP replicas, via the controller.

`docs/docs/advanced_features/pipeline_parallelism.mdx`,
`docs/docs/advanced_features/dp_dpa_smg_guide.mdx`, and
`docs/docs/advanced_features/dcp.mdx` carry the project's current guidance, and
`.claude/skills/debug-distributed-hang/SKILL.md` is what you want when a layout deadlocks —
which, as the padding comment above shows, is the characteristic failure of this chapter.

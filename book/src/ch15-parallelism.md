# 15. Tensor, Pipeline, and Data Parallelism

> *The three classical parallelism axes differ in what they split and therefore in which
> interconnect they stress; SGLang adds a fourth because MLA broke the assumptions.*

Everything so far has quietly assumed one GPU. That assumption fails early: a 70-billion
parameter model in half precision is 140 GB of weights, and the frontier models are several
times larger.

So the work has to be split. The interesting part is that "split" admits several different
answers, they are not interchangeable, and choosing badly is not recoverable by tuning
anything else.

The classical three cut along different dimensions. Tensor parallelism splits *within* each
layer, so every rank does part of every operation and they must talk twice per transformer
block. Pipeline parallelism splits *between* layers, so communication is rare but ranks
spend time idle. Data parallelism does not split the model at all.

Which to use depends on a fact about hardware that governs this whole chapter: the network
inside a machine is roughly an order of magnitude faster than the network between machines.

Then there is a fourth axis, which SGLang added because DeepSeek's compressed cache broke
the assumptions the classical answers were built on. That story — a memory optimization from
Chapter 8 forcing a new form of parallelism — is the most interesting thing here.

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

### Why that specific pairing

The column-then-row order is not a convention. It is the only arrangement that gets a
transformer block down to one collective per sublayer, and Shoeybi et al. derived it in the
Megatron-LM paper (2019) by asking where the nonlinearity forces a synchronization.

Take the MLP, *Y* = act(*XA*)*B*, and consider splitting *A* the other way first — by rows,
*A* = [*A*₁; *A*₂], with *X* split by columns to match. Each rank computes a partial product,
and the partials must be *summed before the activation*, because act is elementwise and
act(*a* + *b*) ≠ act(*a*) + act(*b*). That is an all-reduce in the middle of the block.

Now split *A* by columns instead, *A* = [*A*₁, *A*₂]. Each rank holds whole columns, so each
computes a complete slice of the output:

```
[Y₁, Y₂] = [act(X A₁), act(X A₂)]
```

The activation is elementwise and each rank owns entire elements, so it applies locally. No
communication.

The second matrix then has to consume a column-sharded input, which means splitting it by
rows, *B* = [*B*₁; *B*₂]. Each rank computes *Y_i B_i* — a partial sum over the full output
shape — and one all-reduce finishes it:

```
Z = Y₁B₁ + Y₂B₂
```

So the pairing is forced: **column-parallel to keep the nonlinearity local, row-parallel to
collapse the result, one all-reduce at the end.** Attention works out the same way for a
different reason — heads are independent, so splitting Q, K, V by head keeps each rank's
attention computation self-contained, and the output projection is row-parallel to gather
them.

Megatron writes the communication as a pair of conjugate operators, `f` and `g`: `f` is
identity forward and all-reduce backward, `g` is all-reduce forward and identity backward.
Inference only ever runs the forward half, so a block costs **two all-reduces** — one for
attention, one for the MLP. Eighty layers is 160 collectives per forward pass.

That count is the thing to keep in mind, because it is what sets TP's scaling limit. Each
all-reduce moves the full activation tensor — batch × hidden elements — and its latency has
a floor set by the interconnect that does not shrink as you add ranks. Inside a node, NVLink
at hundreds of GB/s makes 160 collectives affordable. Across nodes, at a tenth the bandwidth
and several times the latency, it does not. **This is the reason TP is a within-node axis and
pipeline parallelism is the between-node one**: PP moves one activation tensor per stage
boundary, a handful of transfers rather than 160.

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

The reason this is available at all is a property of attention that the MLP does not share.
Attention is **independent across sequences** — sequence 7's output depends on sequence 7's
tokens and nothing else. So any partition of the batch into groups of whole sequences is a
valid partition of the work, requiring no communication whatsoever inside the attention
block. The MLP is the opposite: it is independent across *rows*, but each row must be
multiplied by the whole weight matrix, which is exactly what does not fit on one GPU.

So the two halves of a transformer layer want to be split along different axes, and the
choice is not aesthetic:

| | Attention | MLP |
| --- | --- | --- |
| Parameters | small (projections only) | large |
| Per-request state | the KV cache — large | none |
| Natural split | by sequence | by hidden dimension |
| Cost of splitting the other way | KV replicated per rank | activations all-reduced per row |

Tensor parallelism picks one axis for both and pays for it in whichever half is wrong. Under
MHA or GQA that cost is tolerable, because splitting attention by head also splits the KV
cache. Under MLA there are no heads to split, so TP replicates the cache in full and the cost
becomes the dominant one.

Data-parallel attention refuses the compromise the same way Chapter 17 does at the
deployment level: **split each half along its own best axis, and pay a transition between
them.** Each layer therefore does:

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

This chapter split a dense model. Chapter 16 turns to models that are sparse by
construction, where the arithmetic barely moves but the communication bill changes shape
entirely.

# 7. Executing a Batch

> *The handoff from Python scheduling objects to GPU tensors is where the mode of the batch
> starts determining everything downstream.*

A batch has been chosen. Now it has to run.

This chapter covers the handoff from the scheduler's world — Python objects, lists of
requests, CPU-side bookkeeping — to the GPU's world of tensors and kernels. It is a short
chapter about a boundary, but the boundary matters more than its size suggests, because
this is where a single enum starts determining almost everything downstream.

That enum is `ForwardMode`, and it has nine values. Whether the batch is a prefill or a
decode, whether it is verifying speculative tokens, whether it is a placeholder for a rank
with no work — every one of those changes which attention kernel runs, whether a CUDA graph
can be replayed, and how memory is accounted. When you find yourself lost in unfamiliar code
later in this book, the first useful question is almost always *which modes reach this line*.

We will also meet `ModelRunner`, the object that owns the GPU, and read its initialization
as what it actually is: a dependency chain that explains most of the ways startup can fail.

<!-- objectives:begin -->
<div class="bk-objectives"><p class="bk-objectives-head">What this chapter gives you <span class="bk-objectives-time">· about 8 min</span></p><ul><li>List the forward modes and say what each one changes downstream</li><li>Explain the ragged batch layout and which operators read its offsets</li><li>Describe the purity rule on `ForwardBatch.init_new` and why overlap requires it</li><li>Follow the handoff from scheduler objects to GPU tensors</li></ul></div>
<!-- objectives:end -->

---

## Mode determines the world

`python/sglang/srt/model_executor/forward_batch_info.py:98` `ForwardMode` is a nine-value
enum, and it is the most consequential nine values in the runtime. Almost every branch in
the execution path keys off it:

```python
class ForwardMode(IntEnum):
    # Extend a sequence. The KV cache of the beginning part of the sequence is already computed (e.g., system prompt).
    # It is also called "prefill" in common terminology.
    EXTEND = auto()
    # Decode one token.
    DECODE = auto()
    # Contains both EXTEND and DECODE when doing chunked prefill.
    MIXED = auto()
    # No sequence to forward. For data parallel attention, some workers will be IDLE if no sequence are allocated.
    IDLE = auto()

    # Used in speculative decoding: verify a batch in the target model.
    TARGET_VERIFY = auto()
    # Used in speculative decoding: extend a batch in the draft model.
    DRAFT_EXTEND_V2 = auto()

    # Used in disaggregated decode worker
    # Represent a batch of requests having their KV cache ready to start decoding
    PREBUILT = auto()

    # Split Prefill for PD multiplexing
    SPLIT_PREFILL = auto()

    # Used in dLLM
    DLLM_EXTEND = auto()
```

The comments are a table of contents for the rest of the book. `EXTEND` and `DECODE` are
Chapter 1's two phases. `MIXED` is Chapter 6's chunked prefill sharing a batch with decode.
`IDLE` is Chapter 16's data-parallel attention, where a rank with no work must still
participate in collectives. `TARGET_VERIFY` and `DRAFT_EXTEND_V2` are Chapter 19.
`PREBUILT` is Chapter 18's disaggregated decode, receiving KV computed elsewhere.

What makes the enum interesting is that it is queried through *predicates*, not equality —
and the predicates do not partition cleanly:

```python
    def is_extend(self, include_draft_extend_v2: bool = False):
        return (
            self == ForwardMode.EXTEND
            or self == ForwardMode.MIXED
            or (include_draft_extend_v2 and self == ForwardMode.DRAFT_EXTEND_V2)
            or self == ForwardMode.TARGET_VERIFY
            or self == ForwardMode.SPLIT_PREFILL
            or self == ForwardMode.DLLM_EXTEND
        )
```

Six of nine modes count as "extend," and one of them only conditionally, depending on a
flag the caller passes. `TARGET_VERIFY` is an extend because verifying *k* speculative
tokens has the same shape as prefilling *k* tokens — multiple new positions per sequence —
even though semantically it is a decode step.

The other predicate worth reading is `:175` `is_cuda_graph`:

```python
    def is_cuda_graph(self):
        return (
            self == ForwardMode.DECODE
            or self == ForwardMode.TARGET_VERIFY
            or self == ForwardMode.IDLE
            or self == ForwardMode.DLLM_EXTEND
        )
```

This is the Chapter 15 constraint expressed as a mode property. CUDA graphs need static
shapes; these four modes have a fixed number of tokens per sequence, so they can be
captured. `EXTEND` has a variable prompt length per request and cannot — hence the
piecewise and prefill-graph machinery Chapter 15 covers.

When reading unfamiliar code in `model_executor/` or `layers/`, the first question is
almost always *which modes reach this line*.

---

## The execution view

`python/sglang/srt/model_executor/forward_batch_info.py:412` `ForwardBatch` is what the
model actually consumes: input ids, positions, `out_cache_loc` (where this step's KV
should be written), sequence lengths, the page table, attention metadata, multimodal
inputs, and the speculative-decoding payload.

It is built by `:739` `init_new` from a `ScheduleBatch`, and that construction is governed
by a rule. `.claude/rules/forward-batch-init-new-purity.md`:

> `init_new` (and any ForwardBatch factory) treats the input `ScheduleBatch` as read-only.
> Per-forward overrides go through the kw-only params of `init_new` /
> `TpModelWorker.forward_batch_generation`, never a ScheduleBatch field.

The reason is Chapter 5's overlap loop. `ScheduleBatch` objects are snapshotted and queued;
if constructing a `ForwardBatch` mutated the batch it read from, the queued snapshot would
drift from what the GPU actually ran.

The rule is unusual in listing its own violations:

> Tolerated exceptions (don't add new ones):
> - `seq_lens_sum` backfill — the `seq_lens` family is slated for removal (→ kv-committed lengths).
> - `sampling_info` sub-object writes (grammars, canary ids) — shared object, until the sampling forward-copy op.
> - `_expand_mrope_from_input` memoizing `mrope_position_delta_repeated_cache` — pre-existing.

Each exception names the refactor that would remove it. This is the most honest kind of
architecture documentation: not "the design is clean" but "here are the three places it is
not, and why."

Two `ForwardBatch` methods carry real weight. `:640` `mark_forward_metadata_ready` and
`:657` `needs_forward_metadata_init` manage the attention backend's two-phase contract from
Chapter 14 — metadata is prepared once per forward, and the batch tracks whether that has
happened. `:1305` `prepare_mlp_sync_batch` is Chapter 16's cross-rank agreement, and it is
called from inside the forward path rather than the scheduler because the required padding
depends on what every other rank is doing.

`python/sglang/srt/model_executor/forward_context.py` provides ambient access to the
current forward's context. Model code deep in an attention layer needs the backend and the
current batch, and threading them through every constructor would mean every layer in every
one of the 218 model files takes parameters it does not use. The context is set once per
forward and read where needed.

### Ragged, not rectangular

There is a shape assumption buried in almost every tutorial description of a transformer,
and this is the point where SGLang abandons it.

Textbook batching is rectangular: a `[batch, seq_len, hidden]` tensor, every sequence the
same length, shorter ones padded and masked. It is convenient and, for serving, ruinous. A
batch holding a 4,000-token prefill and thirty 12-token decodes would pad every row to 4,000
and spend 99% of its arithmetic on padding.

Chapter 6's admission policy makes this worse rather than better: a well-packed batch is
*deliberately* heterogeneous — one prefill chunk plus every available decode row. Padding a
batch that was constructed to be uneven defeats the point of constructing it.

So `ForwardBatch` carries a **ragged** layout instead. `input_ids` is a flat 1-D tensor: all
of sequence 0's new tokens, then all of sequence 1's, with no padding and no batch
dimension. The structure lives beside it, in the offsets:

```python
    # For extend
    extend_num_tokens: Optional[int] = None
    extend_seq_lens: Optional[torch.Tensor] = None
    extend_prefix_lens: Optional[torch.Tensor] = None
    extend_start_loc: Optional[torch.Tensor] = None
```

`extend_seq_lens[i]` is how many *new* tokens sequence *i* contributes; `extend_start_loc[i]`
is where they begin in the flat buffer; `extend_prefix_lens[i]` is how many tokens the
sequence already had cached, which is where its new positions start counting from and, in
Chapter 10's terms, how much of it was a prefix hit.

This is the same CSR-style representation used everywhere data is jagged — an offsets array
plus a values array — and every consumer splits cleanly along Orca's line:

- **Linear layers** ignore the offsets entirely. A `[total_tokens, hidden]` matrix times a
  weight matrix is one GEMM; the rows are independent and it does not matter which sequence
  each came from. Every `nn.Linear` in every model file operates on the flat form without
  knowing it is a batch at all.
- **Attention** reads the offsets, because it must not let sequence 3 attend to sequence 4.
  The kernels of Chapter 14 take `extend_start_loc` and `extend_seq_lens` and treat each
  segment as its own attention problem, which is why their signatures carry `indptr` arrays
  rather than a batch dimension.
- **Anything positional** — RoPE in Chapter 13, the sampler in Chapter 8 — reads them to
  recover per-sequence structure from the flat buffer.

Two costs come with it. Every kernel that touches attention must be varlen-aware, which
rules out naive PyTorch implementations and is part of why Chapter 14's backend layer exists
at all. And the `_cpu` mirrors of these arrays — `extend_seq_lens_cpu`,
`extend_prefix_lens_cpu` — exist because host-side logic needs the same structure without
paying for the synchronization Chapter 5 warned about, so both copies are maintained and
must be kept consistent.

The payoff is that a batch costs exactly the tokens it contains. That is the precondition
for Chapter 6's budget being denominated in tokens rather than requests, and for Chapter 1's
critical batch size being reachable at all.

---

## The worker boundary

`python/sglang/srt/managers/tp_worker.py:74` `BaseTpWorker` is deliberately thin — mostly
delegation to `ModelRunner` plus the weight-update and LoRA methods Chapters 12 and 21
cover. `:299` `TpModelWorker` is the standard implementation.

The abstraction exists because the worker is a substitution point. Chapter 19's speculative
decoding replaces it with a worker that runs a draft model and a target model
(`python/sglang/srt/speculative/eagle_worker_v2.py:1008` `EAGLEWorkerV2` implements the same
interface). The scheduler calls `forward_batch_generation` and does not know which is
underneath.

---

## Initialization as an ordered script

`python/sglang/srt/model_executor/model_runner.py:284` `ModelRunner` owns the GPU: weights,
memory pools, attention backend, CUDA graphs. Its `__init__` (`:287`) is a dependency chain,
and reading it in order explains most startup failures:

```
init_torch_distributed        (:1037)   process groups first — everything else may collective
load_model                    (:1057)   weights onto the device
configure_kv_cache_dtype      (:1322)   decide the cache dtype before sizing the pool
alloc_memory_pool             (:807)    whatever memory is left becomes KV cache
init_attention_backends       (:927)    backends need pool layout to build page tables
init_cuda_graphs              (:992)    capture needs a working backend and real buffers
```

Memory is the through-line. `:807` `alloc_memory_pool` runs *after* weights are loaded
because the pool is sized from what remains — that is what `--mem-fraction-static` means,
and why the same flag behaves differently across models. Chapter 9 covers the arithmetic.

CUDA graph capture comes last because it allocates too, and its cost is not always
predictable up front. `:885` `post_capture_resize_kv_pool` exists for exactly that: after
capture, whatever memory the graphs did not need is returned to the KV pool. The startup
sequence over-reserves, then gives back.

The class has also been decomposed.
`python/sglang/srt/model_executor/model_runner_components/` holds extracted
collaborators — `python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py`,
`python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py`,
`python/sglang/srt/model_executor/model_runner_components/weight_updater.py` — each called from the
corresponding `init_*`.

---

## The forward call

`:1505` `forward` is the public entry, and it is mostly instrumentation:

```python
        self.forward_pass_id += 1

        # Try msprob debugger
        if self.msprobe_debugger is not None:
            ...

        # Step span
        step_span_ctx = profile_range(build_step_span_name(forward_batch))
```

```python
        with (
            canary_ctx,
            step_span_ctx,
            get_global_expert_distribution_recorder().with_forward_pass(
                self.forward_pass_id,
                forward_batch,
            ) as recorder_outputs,
        ):
            output = self._forward_raw(...)
```

Three context managers wrap every forward: a numerical-correctness canary, a profiling
span (Chapter 22), and the expert-distribution recorder (Chapter 17). They are stacked
here, at the single chokepoint every forward passes through, rather than sprinkled through
the model.

`:1649` `_forward_raw` is the dispatch, and it reads as a priority list:

```python
            can_run_graph = bool(
                mode_check()
                and self.decode_cuda_graph_runner
                and self.decode_cuda_graph_runner.can_run_graph(forward_batch)
            )
            ...
            # Replay cuda graph if applicable
            if can_run_graph:
                ret = self.decode_cuda_graph_runner.execute(
                    forward_batch,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
                return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)
```

The graph path is checked first and returns immediately — it is the fast path, and taking
it early skips everything below. Note that eligibility is two questions: does the *mode*
allow a graph (`is_cuda_graph`), and does *this particular batch* fit a captured shape
(`can_run_graph`)?

The comment on what follows is worth quoting in full, because it explains an ordering that
looks arbitrary:

```python
            # DP / MLP-sync padding + attn-tp normalization. Only the decode
            # cuda-graph path above pre-pads its static buffers and returns
            # early; split prefill, the prefill cuda graph, and the eager
            # forward all run the live batch and need this first — it sets
            # global_dp_buffer_len / padded token counts that graph eligibility
            # and the collectives depend on.
            self._prepare_eager_forward_batch(forward_batch)
```

The decode graph path pre-pads at capture time, so it can skip this. Everything else must
pad now, because Chapter 16's collectives require every rank to agree on token counts
before any of them starts communicating.

Then the remaining modes are dispatched in turn — split prefill, prefill CUDA graph, eager
extend — each with its own eligibility conditions. The layered structure is the point:
**four execution paths, tried in order of decreasing speed and increasing generality.**

---

## What comes back

`:258` `ModelRunnerOutput` carries the `logits_output` plus whether a graph was used. The
model produced logits; it did not produce tokens.

Sampling is a separate call — `:1766` `sample` — which Chapter 5 showed the overlap loop
deliberately deferring past result processing, because sampling may depend on grammar state
advanced by the previous step.

Chapter 8 picks up there.

---

## The full handoff

Collecting the transformations a request passes through in this chapter and the last:

```
Req                    scheduler's per-request state, Python objects
  │  ScheduleBatch.prepare_for_extend / prepare_for_decode
  ▼
ScheduleBatch          CPU scheduling view: list of Reqs + bookkeeping
  │  (ModelWorkerBatch: transport subset)
  ▼
ForwardBatch           GPU view: tensors, positions, page table, attn metadata
  │  ModelRunner.forward → _forward_raw → graph replay or eager
  ▼
LogitsProcessorOutput  logits, and optionally hidden states / logprobs
  │  ModelRunner.sample
  ▼
next_token_ids         back to process_batch_result in Chapter 5's loop
```

Four representations for one request in one step. Each exists because its consumers need a
different subset with different mutability rules — and the rules files in `.claude/rules/`
exist because keeping those boundaries straight under an overlapping scheduler is where the
bugs are.

The batch has run. What comes back is logits — not tokens, and certainly not text. Chapter 8
covers the last leg, and closes Part II.

---

<!-- summary:begin -->
<div class="bk-card"><p class="bk-card-head">Chapter 7 in one page</p><ol class="bk-card-arg"><li>The batch's mode determines the world: which kernel runs, which metadata is built, whether a graph can replay.</li><li>Batches are ragged, not rectangular — a flat token buffer plus offsets — because a well-packed batch is deliberately uneven.</li><li>Linear layers ignore the offsets; attention reads them. That is Orca's selective batching, inherited as a data layout.</li><li>`init_new` treats its input as read-only, because the overlap loop queues snapshots that must describe what the GPU actually ran.</li><li>The worker boundary is a substitution point — speculative decoding replaces it entirely.</li></ol><p class="bk-card-sub">Where it lives</p><table class="bk-card-table"><tbody><tr><th scope='row'>Forward batch</th><td><code>python/sglang/srt/model_executor/forward_batch_info.py</code></td></tr><tr><th scope='row'>Worker</th><td><code>python/sglang/srt/managers/tp_worker.py</code></td></tr><tr><th scope='row'>Model runner</th><td><code>python/sglang/srt/model_executor/model_runner.py</code></td></tr></tbody></table></div>
<!-- summary:end -->

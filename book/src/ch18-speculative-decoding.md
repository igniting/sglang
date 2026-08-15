# 18. Speculative Decoding

> *Verifying k tokens costs nearly what generating one costs, so the only question is how
> good a draft you can produce cheaply.*

Return to the uncomfortable fact from Chapter 1: during decode, a GPU's arithmetic units are
idle more than 99% of the time. All that silicon is sitting there while the memory bus does
the actual work.

Speculative decoding is the idea of spending it.

The observation is simple once stated. Running the model on five tokens costs almost exactly
what running it on one costs, because the weights get read once either way and the extra
arithmetic lands on units that were idle. So if something cheap could *guess* the next five
tokens, the real model could check all five in a single pass and keep however many were
right.

What makes this more than a heuristic is the acceptance rule. It is a rejection-sampling
argument, and it guarantees the output distribution is identical to what the model would
have produced alone. A wrong guess costs wasted compute — never a wrong token.

The cost is paid elsewhere, and this chapter is largely about where. Speculation reaches
into memory pools, scheduling budgets, forward modes, CUDA graphs, the radix cache, and
grammar state. Almost nothing in the engine is untouched by it, which is why it comes this
late in the book.

---

## The bandwidth argument

Chapter 1 established that decode is memory-bound: the GPU reads every weight to produce one
token per sequence, and its arithmetic units are more than 99% idle.

Now ask what it costs to run the model on *five* tokens instead of one. The weights are read
exactly once either way. The activation matrix has five rows instead of one, so the
arithmetic is five times larger — against units that were idle anyway. In the memory-bound
regime, **a five-token forward pass costs almost the same as a one-token forward pass.**

That is the whole opportunity. If you could guess the next five tokens, you could check all
five in one pass and accept however many were right.

The guess comes from something cheap: a small draft model, a lightweight head, or a lookup
table. The check is a single forward pass of the real model. And the acceptance rule — a
rejection-sampling argument — guarantees the output distribution is *identical* to what the
target model would have produced alone. Speculative decoding is not an approximation. A
wrong draft costs wasted compute, never a wrong token.

The economics: if you draft 5 tokens and accept 3 on average, you produce 3 tokens per
target forward pass instead of 1. Latency per token drops by roughly 3×, while total FLOPs
rise — spending idle arithmetic to buy latency, exactly as Chapter 1's tradeoff table
predicted.

---

## Capabilities before implementations

`python/sglang/srt/speculative/spec_info.py:30` `SpeculativeAlgorithm` is where to start,
and its docstring explains why:

```python
class SpeculativeAlgorithm(Enum):
    """Builtin speculative decoding algorithms. Plugin-registered ones are
    ``CustomSpecAlgo`` instances; ``from_string`` returns either type, and
    both expose the same ``is_*()`` / ``create_worker`` interface so callers
    dispatch uniformly without isinstance checks.
    """

    DFLASH = auto()
    DSPARK = auto()
    EAGLE = auto()
    EAGLE3 = auto()
    FROZEN_KV_MTP = auto()
    STANDALONE = auto()
    NGRAM = auto()
    NONE = auto()
```

Callers never compare against enum members. They ask **capability questions**:

```
:94   is_speculative                    speculation at all?
:152  has_draft_kv                      does the draft model keep its own KV cache?
:158  carries_draft_hidden_states       must hidden states be captured?
:138  supports_ragged_verify            can verification use ragged batching?
:144  supports_grammar_overlap          can grammar advance overlap with drafting?
:127  supports_target_verify_for_draft
:130  is_war_publish_phase
```

This matters because speculation touches nearly every subsystem in this book, and each
subsystem cares about a *property*, not an algorithm name. Chapter 8's memory pool needs to
know whether there is a second KV cache. Chapter 7's logits processor needs to know whether
to capture hidden states. Chapter 19's grammar needs to know whether it can overlap.

Predicates keep those subsystems from enumerating algorithms — which is what makes
`_get_registered_spec` and plugin algorithms possible at all. Read
`.claude/skills/speculative-naming/SKILL.md` before touching any of this; the naming
conventions are enforced precisely because so many subsystems depend on these names.

---

## EAGLE end to end

EAGLE is the default and the one to understand. Its insight: rather than a separate small
model, use a *lightweight head* that consumes the target model's own hidden states. The
draft is better-informed than an independent small model could be, because it sees what the
target model actually computed.

`python/sglang/srt/speculative/eagle_worker_v2.py:1008` `EAGLEWorkerV2` implements Chapter
6's worker interface, so Chapter 4's scheduler calls `forward_batch_generation` (`:1105`)
and never learns that two models are involved.

### Drafting

`:128` `EagleDraftWorker` owns the draft model, and `:494` `draft` produces candidates:

```python
    def draft(self, batch: ScheduleBatch):
        draft_input: EagleDraftInput = batch.spec_info
        forward_batch, can_run_decode_cuda_graph = prepare_for_draft(
            draft_input,
            self.req_to_token_pool,
            batch,
            self.cuda_graph_runner,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )
        ...
        with canary_outside_ctx:
            # Run draft
            if can_run_decode_cuda_graph:
                parent_list, top_scores_index, draft_tokens, draft_probs = (
                    self.cuda_graph_runner.execute(forward_batch)
                )
```

Two parameters define the draft's shape. `speculative_num_steps` is depth — how many tokens
ahead. `topk` is width — how many candidates per step.

Together they make a **tree**, not a chain. At each step the draft keeps its top-*k*
continuations, so after 3 steps with topk 4 there are many candidate sequences. The
returned `parent_list` and `top_scores_index` encode that tree's structure.

Why a tree beats a chain: acceptance probability decays with depth. A chain of 5 tokens is
accepted only as far as the first mistake. A tree hedges — if the top choice at step 2 is
wrong, a sibling may be right — so the *expected* accepted length is higher for the same
number of verified tokens.

The `can_run_decode_cuda_graph` check threads Chapter 14 through here. The draft model gets
its own captured graphs (`python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py`,
`python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py`,
`python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py`) because it runs the same
launch-bound decode loop the target does, at even smaller sizes where launch overhead
matters more.

`:726` `draft_extend` handles the draft model's own prefill, with separate paths for the
prefill and decode cases (`:729` `_draft_extend_for_prefill`, `:856`
`_draft_extend_for_decode`).

### Verifying

`:1497` `verify` delegates, and the call site is the clearest statement of what
verification needs:

```python
    def verify(self, batch: ScheduleBatch, grammar_barrier=None):
        return run_eagle_verify(
            batch,
            target_worker=self.target_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            plan_stream=self.plan_stream,
            plan_stream_ctx=self.plan_stream_ctx,
            topk=self.topk,
            num_steps=self.speculative_num_steps,
            num_draft_tokens=self.speculative_num_draft_tokens,
            device=self.device,
            metadata_ready_pre_pad=False,
            finalize_tree_path=True,
            grammar_barrier=grammar_barrier,
        )
```

The target model runs on the whole draft tree at once — Chapter 6's `TARGET_VERIFY` mode,
which `is_extend()` returns true for because verifying *n* tokens has the same shape as
prefilling *n*. Attention must respect the tree structure: a candidate may attend to its
ancestors but not to its siblings, which is a custom mask
(`python/sglang/srt/layers/attention/verify_mask.py`) rather than the usual causal one.

`grammar_barrier` is Chapter 19 intruding. If output is grammar-constrained, every drafted
token must be legal, and the grammar state advances as tokens are accepted — creating a
dependency between verification and grammar that has to be synchronized.

`python/sglang/srt/speculative/eagle_info.py` holds the metadata, and
`python/sglang/srt/speculative/eagle_utils.py` the tree operations.

---

## Down to the kernel

Verification is parallel tensor work, and the kernels are in-repo.
`python/sglang/kernels/ops/speculative/` holds the Triton side;
`python/sglang/kernels/aot/csrc/speculative/` the CUDA.

Two operations dominate. **Tree mask construction** turns `parent_list` into an attention
mask where each candidate sees its ancestors and nothing else — built on the GPU per step,
since the tree shape changes every iteration. **Verification** compares target
probabilities against draft probabilities under the acceptance rule and finds, for each
sequence, the longest accepted path through the tree. That is a parallel prefix problem
over a tree, which is exactly the kind of thing that must be a kernel rather than a Python
loop.

`python/sglang/srt/speculative/ragged_verify.py` is the batching optimization: sequences
accept different numbers of tokens, so a dense representation wastes work on the ones that
accepted few.

---

## What it costs the rest of the engine

Speculation is not a self-contained feature. It reaches into most of this book:

**Chapter 6 — modes.** `TARGET_VERIFY` and `DRAFT_EXTEND_V2` exist for it, and
`is_cuda_graph()` includes `TARGET_VERIFY` so verification can be captured.

**Chapter 8 — memory.** With `has_draft_kv`, the draft model needs its own KV pool. Both
grow, and Chapter 5's budget must account for both.

**Chapter 5 — scheduling.** A step may produce 1 to *k* tokens per sequence, so memory
demand per step is variable rather than exactly one slot per sequence. The new-token-ratio
estimate becomes correspondingly harder.

**Chapter 9 — the radix cache.** EAGLE's `RadixKey.is_bigram` flag — the one Chapter 9
explained as an O(1) view flip — exists because EAGLE's cached unit is a token *pair*.

**Chapter 7 — logits.** `CaptureHiddenMode` and `_get_hidden_states_to_store` exist so the
draft head can consume target hidden states.

**Chapter 14 — graphs.** A second model means a second set of captured graphs and more
capture memory.

**Chapter 17 — disaggregation.** `draft_token_to_kv_pool` in the bootstrap queue is the
draft cache crossing machines.

That breadth is why `SpeculativeAlgorithm`'s predicates matter so much. Without them, every
one of these subsystems would need to know the algorithm list.

---

## The other algorithms

**N-gram** (`python/sglang/srt/speculative/ngram_worker.py` and `python/sglang/srt/speculative/cpp_ngram/`) has no draft
model at all. It looks for repetitions in the context and drafts the continuation that
followed last time. Nearly free, and remarkably effective on code completion, summarization,
and any task where the output echoes the input.
`python/sglang/srt/speculative/external_corpus_manager.py` extends this to an external
corpus, with runtime endpoints (`python/sglang/srt/managers/scheduler.py:3996`
`add_external_corpus`).

**MTP / frozen-KV MTP** (`python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py`) uses multi-token-prediction heads
that some models ship with — DeepSeek among them. "Frozen KV" means the draft reuses the
target's KV cache rather than keeping its own, which is why `has_draft_kv` is a predicate.

**Standalone** (`python/sglang/srt/speculative/standalone_worker_v2.py`) is the classical form: a genuinely separate small
model. Simple, and generally worse than EAGLE for the same compute, because it does not see
the target's hidden states.

**DFlash and DSpark** (`python/sglang/srt/speculative/dflash_worker_v2.py`,
`python/sglang/srt/speculative/dspark_components/`) are the newest, from the
2026-06 blog post linked in `README.md`.

**Multi-layer EAGLE** (`python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py`) uses several draft layers for
better acceptance at higher draft cost.

---

## Knowing when to stop

Acceptance rate is workload-dependent. Predictable text drafts well; novel content does not.
And when acceptance is low, speculation is *worse* than not speculating — you pay for draft
passes and discard the results.

`python/sglang/srt/speculative/adaptive_runtime_state.py` measures acceptance and adjusts,
with `python/sglang/srt/speculative/eagle_worker_v2.py:1301` `build_adaptive_runtime_state` and
`python/sglang/srt/speculative/eagle_worker_v2.py:1379` `apply_runtime_state` on the worker.
At high batch sizes the calculus shifts again: with 256 sequences, decode is no longer
severely memory-bound (Chapter 1's batching argument), the spare arithmetic speculation was
spending is already in use, and speculation costs more than it saves. Adaptive speculation
turns itself down.

Chapter 4's per-request histograms (`update_spec_correct_drafts_histogram`,
`update_spec_cap_lens_histogram`) and Chapter 21's
`python/sglang/srt/managers/tokenizer_manager.py:2766` `_calculate_spec_decoding_metrics`
are how you see it happening.

`docs/docs/advanced_features/speculative_decoding.mdx` and
`docs/docs/advanced_features/adaptive_speculative_decoding.mdx` carry the tuning guidance.
The parameters that matter are `--speculative-num-steps` (depth),
`--speculative-eagle-topk` (width), and `--speculative-num-draft-tokens` (total verified) —
and the number to watch is accepted length per step, because if it is not comfortably above
1, speculation is costing you.

Speculation changes *how many* tokens the engine produces per step. Chapter 19 changes
*which* tokens it is allowed to produce at all.

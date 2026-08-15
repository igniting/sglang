# 23. Extending SGLang

> *The extension points are the architecture's seams, and walking them is the final check
> that the reader has understood where the boundaries are.*

Every chapter in this book has pointed at a place where something plugs in. This one
collects them.

That is the last useful test of whether the preceding chapters landed. The extension points
are the architecture's seams, and a seam only makes sense once you know what is on both
sides of it. Adding an attention backend is a short checklist — but only because Chapter 14
explained what the two-phase contract is protecting. Adding a model is bounded work — but
only because Chapter 13 showed how little of the work is actually in the model file.

The chapter is ordered by how often people need each one: adding a model, adding a kernel,
adding an attention backend, porting to new hardware. Each gets the checklist and, more
usefully, the part that catches people the first time.

It closes on a problem specific to this kind of software. An inference engine's most
important property — *the model produces correct output* — cannot be checked by unit tests.
A model with a subtly wrong rotary embedding still produces fluent, plausible text. What the
project does about that says more about engineering an inference engine than any amount of
architecture description.

<!-- objectives:begin -->
<div class="bk-objectives"><p class="bk-objectives-head">What this chapter gives you <span class="bk-objectives-time">· about 10 min</span></p><ul><li>Add a model, a kernel, or an attention backend, and know what will bite</li><li>Recognize the registry-plus-capability pattern the codebase uses everywhere</li><li>Explain why an inference engine's test pyramid is upside down</li><li>Localize a wrong-output bug to a layer</li></ul></div>
<!-- objectives:end -->

---

## Adding a model

The most common contribution, and the shortest checklist — because Chapter 13 established
that a model is a `forward` and a `load_weights` over a shared layer vocabulary.

**1. Config.** A class in `python/sglang/srt/configs/` if the HF config needs translation,
consumed by `python/sglang/srt/configs/model_config.py`.

**2. Model.** A file in `python/sglang/srt/models/`, built from Chapter 13's layers.
`python/sglang/srt/models/llama.py` is the template. Copy the closest existing architecture
rather than starting from HF code — the parallel layers, `RadixAttention` wiring, and
`quant_config` threading are the parts that must be right, and they are identical across
models.

**3. Registry.** So `--model-path` resolves the architecture string.

**4. Weight mapping.** `load_weights`, and the piece most likely to be wrong.

**5. Chat template.** `docs/docs/references/custom_chat_template.mdx`.

**6. Tests and docs.** `docs/docs/supported-models/support_new_models.mdx` is the official
guide.

### When the output is wrong

A model that loads and produces fluent nonsense is the characteristic failure, and it is
almost always the weight mapping — a transposed matrix, a QKV fusion split the wrong way, a
sharded parameter sliced along the wrong axis. Chapter 12's `stacked_params_mapping` is the
usual suspect.

`python/sglang/srt/debug_utils/comparator/` is the tool: run the HF reference and the SGLang
implementation on the same input, dump activations layer by layer, and find the first layer
that diverges. The first divergence localizes the bug; everything after it is downstream
noise.

Order of suspicion, from experience: weight mapping, then rotary embedding configuration
(Chapter 13 showed how many config keys govern it), then normalization placement, then
attention mask.

`python/sglang/srt/model_loader/ci_weight_validation.py` catches a class of these in CI, and
`.claude/skills/kl-consistency-test/SKILL.md` is the systematic version — the prefill/decode
logprob test that Chapter 8 described.

---

## Adding a kernel

Two paths, and the choice is about compile time.

**JIT** (`python/sglang/kernels/jit/`, with `python/sglang/kernels/jit/csrc/`) compiles at
first use. Fast to iterate on, no build step for users, a first-call cost at runtime. Right
for kernels that are specialized per shape or dtype.
`.claude/skills/add-jit-kernel/SKILL.md` is the tutorial.

**AOT** (`python/sglang/kernels/aot/`, formerly the top-level `sgl-kernel`) compiles at
build time into a shipped extension. No runtime cost, a real build. Right for kernels used
on every forward. `.claude/skills/add-sgl-kernel/SKILL.md` is the tutorial, and
`python/sglang/kernels/aot/tests/` and `python/sglang/kernels/aot/benchmark/` show the
expected deliverables.

Both register through the same dispatch layer:

```
python/sglang/kernels/spec.py       KernelSpec — what a kernel declares about itself
python/sglang/kernels/registry.py   :17 KernelRegistry, :76 register_kernel
python/sglang/kernels/selector.py   :38 select_kernel, :92 get_kernel
python/sglang/kernels/fused_op.py   fused operation base
```

`python/sglang/kernels/selector.py:34` `_platform` is where hardware detection happens, so
one operation can have several implementations and the right one is chosen at runtime —
Chapter 14's backend registry, generalized to every operation.

**Tests and benchmarks are part of the contribution, not follow-up.** A kernel without a
correctness test against a reference implementation cannot be safely modified by anyone
else, and a kernel without a benchmark cannot be defended when someone proposes replacing
it.

---

## Adding an attention backend

Chapter 14's contract, restated as a checklist:

1. Subclass `python/sglang/srt/layers/attention/base_attn_backend.py:33` `AttentionBackend`.
2. Implement `:62` `init_forward_metadata` — or, for graph support, the out-of-graph and
   in-graph split.
3. Implement `:261` `forward_decode` and `:274` `forward_extend`.
4. Implement `:160` `init_cuda_graph_state` and `:187`
   `get_cuda_graph_seq_len_fill_value`.
5. Register in `python/sglang/srt/layers/attention/attention_registry.py:34`.

Step 4 is what catches every first attempt. The lint contract in
`init_forward_metadata_in_graph`'s docstring — no `.item()`, no `.cpu()`, no dynamic-shape
`torch.empty()` — is not advisory. Violating it produces a graph that captures successfully
and then replays stale values, which is a wrong answer rather than an error.

Test across every forward mode Chapter 7 lists that your backend claims to support.
`ForwardMode` combinations are where backends break: a backend correct for `DECODE` and
`EXTEND` may be wrong for `MIXED`, and `TARGET_VERIFY` (Chapter 19) has its own mask
requirements.

`python/sglang/srt/layers/attention/triton_backend.py` is the reference to read first.

---

## Porting to new hardware

The largest undertaking, and the one that most tests whether the abstractions hold.

`python/sglang/srt/platforms/` holds the platform abstraction,
`python/sglang/srt/hardware_backend/` the per-vendor code, and
`python/sglang/srt/plugins/` the plugin loading mechanism (called from
`python/sglang/launch_server.py`, so out-of-tree backends can register before anything
else runs).

What a new accelerator actually needs:

**Device communicators** (Chapter 16) — `python/sglang/srt/distributed/device_communicators/`.
Without working collectives nothing beyond one device runs.

**An attention backend** (Chapter 14) — the largest piece.

**Memory pool support** (Chapter 9) — usually the least work, since the pools are mostly
device-agnostic tensor allocation.

**Graph capture** (Chapter 15) — or an honest admission that it is unsupported, which costs
decode performance but does not block correctness.

**Kernels** for quantization, MoE, and sampling, or fallbacks to portable implementations.

The case studies show how far the abstraction stretches. **ROCm/AITER** is closest to
NVIDIA and reuses most of the stack. **Ascend NPU** has its own everything — note the
sampler backend registration Chapter 8 mentioned (`_forward_ascend_backend`), which exists
because even sampling needed a vendor path. **Intel XPU and AMX** target CPUs and a
different accelerator model. **TPU** went a different route entirely, as a separate
`sglang-jax` project — evidence that the abstraction has limits.

`docs/docs/hardware-platforms/` covers each, and
`docs/docs/hardware-platforms/plugin.mdx` covers the plugin mechanism.

---

## The same pattern, five times

Reading those four checklists in a row, a shape emerges that is worth naming, because once
you see it the fifth extension point is predictable rather than novel.

Every one of them is the same construction:

1. **An abstract base declaring a contract** — `AttentionBackend`, `QuantizeMethodBase`,
   `BaseGrammarBackend`, `KVCache`, `BaseTokenToKVPoolAllocator`, `BaseTpWorker`.
2. **A registry mapping a name to an implementation** —
   `python/sglang/srt/layers/attention/attention_registry.py`,
   `python/sglang/kernels/registry.py`, the model registry, the platform registry.
3. **A selector that picks one at construction time**, from configuration plus a hardware
   probe — `select_kernel`, `_platform`, `--attention-backend`.
4. **Capability predicates rather than identity checks** at every call site — Chapter 19's
   `has_draft_kv` and `supports_ragged_verify` are the clearest case, but Chapter 14's
   `init_forward_metadata_in_graph` default-no-op and Chapter 9's pool interface do the same
   job.

The fourth point is the one that carries the weight, and it is the difference between a
codebase with plugins and a codebase that is *actually* extensible. If a call site asks
"is this the FlashInfer backend?" then adding a backend means finding and editing every such
site. If it asks "does this backend support graph capture?" then a new backend answers the
question for itself and no existing code changes. Chapter 19's `SpeculativeAlgorithm` is the
purest specimen — an enum whose members are almost never compared against, wrapped in a dozen
predicates that are.

The pattern has a cost, which the codebase pays visibly. Indirection makes control flow
harder to follow: finding what actually runs means resolving a registry lookup at runtime
rather than reading a call. This is why so much of this book is *addresses* — the seams are
where the code stops being readable top-to-bottom, and a book is a reasonable place to keep
the map.

The pattern also tells you when *not* to use an extension point. If your change would need a
new predicate on the base class, and every existing implementation would have to answer it,
the seam is in the wrong place. That is the signal to widen the contract deliberately rather
than to add a special case behind it.

---

## How the project keeps this safe

An inference engine has an unusual testing problem: the most important property — "the model
produces correct output" — cannot be checked by unit tests. A model with a subtly wrong
rotary embedding still produces fluent text.

### Why "correct" is hard to define here

The difficulty is not laziness about testing. It is that the usual oracle does not exist.

For most software, correctness is exact: the function returns the right value or it does not.
Here, the reference implementation is a *different* floating-point program computing the same
mathematical function, and floating-point addition is not associative, so a correct
reimplementation does not produce bit-identical output. It produces output that differs in the
last few bits — and those differences amplify.

They amplify in two specific ways, and knowing which one you are looking at is most of
debugging.

**Through depth.** An error introduced at layer 3 is transformed by 77 more layers. A
relative difference of 1e-7 at the input to a layer can be 1e-3 at the output of the model,
purely through accumulation, with nothing wrong anywhere. So a fixed tolerance on the final
logits is nearly useless — too tight and every correct implementation fails, too loose and
real bugs pass.

**Through argmax.** Generation is a discrete decision on top of continuous values. Two tokens
with logits differing by 1e-6 will be ordered differently by two correct implementations,
after which the sequences diverge completely and every subsequent comparison is meaningless.
One flipped tie makes a passing test and a failing test look identical.

The three practical responses, in the order they are worth reaching for:

**Compare early, not late.** `python/sglang/srt/debug_utils/comparator/` dumps activations
layer by layer, because the *first* divergence is the only informative one. Comparing final
output tells you that something is wrong; comparing layer 4 tells you what.

**Compare distributions, not tokens.** KL divergence between the reference distribution and
the implementation's is continuous where argmax is not — it degrades smoothly with numerical
error instead of flipping. That is what `.claude/skills/kl-consistency-test/SKILL.md`
measures, and its real contribution is separating the two independent conditions Chapter 8
described: whether the operators are batch-invariant, and whether the two paths compute the
same function at all. A non-zero KL can be either, and a test that cannot distinguish them is
a threshold somebody tunes until it passes.

**Measure the property you actually care about.** Nobody deploys a model to match a reference
implementation bit for bit; they deploy it to answer questions correctly. An accuracy
evaluation is immune to every problem above — it does not care about the last bits, it cares
about the answer — at the cost of being slow, noisy at small sample sizes, and unable to
localize a bug.

Which is why the pyramid is upside down here. **Accuracy evaluations are the real safety
net**, not the slow layer on top of a broad base of fast ones. `test/lm_eval_configs/` holds
the evaluation configs, and a change that does not move an eval score is far more trustworthy
than one that passes unit tests.

`test/run_suite.py` is the entry point, `test/README.md` the layout, and `test/registered/`
the registration that puts a test in CI. `python/sglang/test/` provides the harness —
`CustomTestCase`, server fixtures, and `python/sglang/test/kits/` for common patterns.
`.claude/skills/write-sglang-test/SKILL.md` is the guide, and
`.claude/rules/unit-test-admission.md` states what qualifies as a unit test at all — the
project is deliberately restrictive, because a suite of tests that pass while the model is
broken is worse than no suite.

CI orchestration is in `.github/workflows/`, with
`.claude/skills/ci-workflow-guide/SKILL.md` explaining stage ordering, fast-fail, gating,
and partitioning across runners. Multi-GPU tests need multi-GPU runners, which are scarce,
so gating decides what runs on every PR versus nightly.

When something fails only in CI, `.claude/skills/sglang-bisect-ci-regression/SKILL.md` is
the procedure: extract the signature, bisect the commit window, check runner specificity.

---

## Where to start

If you are looking for a first contribution, in increasing order of scope:

1. **Documentation** for something this book found unclear. The gap between what the code
   does and what is written down is real, and you have just read enough to see it.
2. **A model** whose architecture closely matches an existing one. Bounded, well-guided, and
   immediately useful.
3. **A kernel** for an operation with a slow fallback. Self-contained, with a clear
   correctness oracle.
4. **A quantization scheme** or **attention backend**. Larger, but the interfaces are
   well-defined and Chapters 14 and 15 mapped them.
5. **A hardware backend.** Months of work, and a real contribution.

`docs/docs/developer_guide/contribution_guide.mdx` covers process, and
`.claude/rules/` covers the conventions Chapter 3 introduced. Read those five rule files
before your first patch; each one will otherwise cost you a review cycle.

---

That is how you change the engine. Chapter 24 is about the part nobody can change: it has to
start, scale, and fail somewhere, and the last chapter is about where the internals you now
know decide an operational outcome.

---

<!-- summary:begin -->
<div class="bk-card"><p class="bk-card-head">Chapter 23 in one page</p><ol class="bk-card-arg"><li>Every extension point is the same construction: an abstract contract, a registry, a selector, and capability predicates at the call sites.</li><li>Predicates rather than identity checks are what make a new implementation addable without editing existing code.</li><li>Correctness here has no exact oracle — a correct reimplementation differs in the last bits, and those bits amplify through depth and through argmax.</li><li>So compare early rather than late, compare distributions rather than tokens, and trust accuracy evaluations over unit tests.</li><li>A model that loads and produces fluent nonsense is almost always the weight mapping.</li></ol><p class="bk-card-sub">Numbers worth keeping</p><table class="bk-card-table"><tbody><tr><th scope='row'>Model files in the tree</th><td>218</td></tr></tbody></table><p class="bk-card-sub">Where it lives</p><table class="bk-card-table"><tbody><tr><th scope='row'>Attention registry</th><td><code>python/sglang/srt/layers/attention/attention_registry.py</code></td></tr><tr><th scope='row'>Kernel registry</th><td><code>python/sglang/kernels/registry.py</code></td></tr><tr><th scope='row'>Divergence tool</th><td><code>python/sglang/srt/debug_utils/comparator/</code></td></tr></tbody></table></div>
<!-- summary:end -->

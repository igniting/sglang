# Appendix F — Further Reading

---

## The ideas, in their original form

**RadixAttention and the SGLang frontend** — *SGLang: Efficient Execution of Structured
Language Model Programs* (Zheng et al.). The paper that introduced both the DSL of Chapter 2
and the prefix caching of Chapter 9, and the argument that co-designing them is what makes
each worth more.

**PagedAttention** — *Efficient Memory Management for Large Language Model Serving with
PagedAttention* (Kwon et al., vLLM). The paged KV memory of Chapter 8. SGLang's
acknowledgment section credits vLLM directly.

**FlashAttention** — Dao et al., across several papers. The tiling and online-softmax
structure Chapter 13 read in the Triton kernel: never materializing the attention matrix,
combining partial softmaxes by log-sum-exp.

**Continuous batching** — *Orca: A Distributed Serving System for Transformer-Based
Generative Models* (Yu et al., OSDI '22). Iteration-level scheduling, which Chapter 5
implements as an admission-control problem.

**Speculative decoding** — Leviathan et al. and Chen et al. for the acceptance rule that
preserves the target distribution; the EAGLE papers for the hidden-state drafting of
Chapter 18.

**Tensor and pipeline parallelism** — the Megatron-LM papers. The column-then-row pattern of
Chapter 15 is theirs.

**MLA and DeepSeek architecture** — the DeepSeek-V2 and V3 technical reports. Chapter 8's
compressed cache, Chapter 16's grouped top-k routing, and much of what forced Chapter 15's
DP attention.

**Mixture of experts** — the Switch Transformer and GShard papers for the routing
formulation Chapter 16 builds on.

---

## SGLang's own writing

The LMSYS blog (linked throughout `README.md`) is the primary source for how the system
evolved, and the posts most relevant to this book:

- **v0.4** — the zero-overhead batch scheduler of Chapter 4, cache-aware load balancing of
  Chapter 17, and faster structured outputs from Chapter 19.
- **Large-scale expert parallelism** (96 H100s) and the **GB200 rack-scale** posts —
  Chapters 16 and 17 at the scale that motivated them.
- **PD disaggregation** — the measurements behind Chapter 17.
- **Next-generation speculative decoding (DFlash, Spec V2)** — Chapter 18's newest
  algorithms.
- **Day-0 model support** posts — what Chapter 22's "adding a model" looks like under time
  pressure.

`docs/` is the official documentation, and the sections this book leans on most:

- `docs/docs/advanced_features/` — the operational counterpart to Parts IV–VI.
- `docs/docs/developer_guide/` — contribution process, benchmarking, profiling.
- `docs/docs/hardware-platforms/` — Chapter 22's porting targets.
- `docs/docs/references/` — production metrics, tracing, environment variables.

`.claude/skills/` are the maintainers' own playbooks, and several are better than any
external writing on their topics:

| Skill | Topic | Chapter |
| --- | --- | --- |
| `.claude/skills/large-class-style/SKILL.md` | why `Scheduler` looks the way it does | 4 |
| `.claude/skills/speculative-naming/SKILL.md` | speculative decoding vocabulary | 18 |
| `.claude/skills/kl-consistency-test/SKILL.md` | prefill/decode logprob consistency | 7 |
| `.claude/skills/debug-distributed-hang/SKILL.md` | localizing rank divergence | 15 |
| `.claude/skills/llm-torch-profiler-analysis/SKILL.md` | profile triage | 21 |
| `.claude/skills/sglang-prod-incident-triage/SKILL.md` | live-serving incidents | 21 |
| `.claude/skills/compute-mamba-ratio/SKILL.md` | sizing hybrid model pools | 8, 13 |
| `.claude/skills/ci-workflow-guide/SKILL.md` | CI orchestration | 22 |

---

## Neighbouring systems

Reading another engine is the fastest way to see which of SGLang's choices are necessary and
which are preferences.

**vLLM** — the closest comparison, and the origin of paged attention. SGLang's
acknowledgment section lists it first.

**TensorRT-LLM** — NVIDIA's engine. A compilation-first design, which throws SGLang's
runtime-flexibility choices into relief.

**FlashInfer** — the attention kernel library Chapter 13 treats as a contract. Reading its
API explains why SGLang's backend is shaped the way it is.

**LightLLM**, **Guidance**, **Outlines**, **LMQL** — also credited in
`README.md`. Outlines is Chapter 19's original constrained-decoding backend; Guidance and
LMQL are ancestors of the Chapter 2 DSL.

**sglang-jax** — the TPU port, as a separate project. Evidence for where Chapter 22's
hardware abstraction stops.

---

## A note on staying current

This book is pinned to one commit (see the introduction), and SGLang moves quickly — the
`README.md` news section alone spans day-0 support for a dozen model families in under a
year.

What ages well: the cost model of Chapter 1, the process topology of Chapter 2, the request
path of Part II, the two-level memory indirection of Chapter 8, and the radix tree of
Chapter 9. These are the load-bearing structures, and they have been stable across the
project's life.

What ages fast: kernel backends, quantization formats, speculative algorithms, and hardware
support. Chapters 13, 14, and 18 name specific implementations that will change.

Read the stable parts here; read `docs/` and the blog for the rest.

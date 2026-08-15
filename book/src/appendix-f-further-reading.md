# Appendix F — Further Reading

---

## The papers this book is built on

Each chapter takes an idea from the literature and follows it into the code. These are the
sources, in the order the book uses them. Where a chapter states an equation, a bound, or a
measured number, it comes from here.

### Serving systems

**Orca: A Distributed Serving System for Transformer-Based Generative Models** — Yu et al.,
OSDI '22. Iteration-level scheduling and selective batching. Chapter 6's scheduler and
Chapter 7's ragged layout are both downstream of it.

**Efficient Memory Management for Large Language Model Serving with PagedAttention** — Kwon
et al., SOSP '23 (vLLM). The fragmentation taxonomy and the 20.4–38.2% utilization
measurement that Chapter 9 quotes, and the OS-paging analogy it builds on.

**SGLang: Efficient Execution of Structured Language Model Programs** — Zheng et al.,
NeurIPS '24. RadixAttention, the LRU-leaf-first eviction of Chapter 10, and the theorem that
depth-first traversal order is cache-optimal, which Chapter 6 uses to justify LPM.

**SARATHI / Sarathi-Serve: Efficient LLM Inference by Piggybacked Decodes with Chunked
Prefills** — Agrawal et al., 2023 and OSDI '24. Chunked prefill and stall-free batching in
Chapter 6.

**DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving** — Zhong
et al., OSDI '24. Chapter 18's interference argument, its goodput framing, and its
disagreement with Sarathi.

**Roofline: An Insightful Visual Performance Model** — Williams, Waterman, and Patterson,
CACM 2009. The ridge point and critical batch size of Chapter 1, and the per-kernel triage
of Chapter 22.

### Attention and architecture

**Online normalizer calculation for softmax** — Milakov and Gimelshein, 2018. The running
maximum and rescaling recurrence Chapter 14 derives.

**FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** — Dao et al.,
NeurIPS '22, and its successors FlashAttention-2 and -3. The `Θ(N²d²M⁻¹)` HBM bound and the
tiled output rescaling of Chapter 14.

**Fast Transformer Decoding: One Write-Head is All You Need** — Shazeer, 2019. Multi-query
attention, the first attack on KV cache size.

**GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** —
Ainslie et al., EMNLP '23. Chapter 13's interpolation between MQA and MHA, with the
uptraining recipe and the T5-XXL quality/latency numbers.

**RoFormer: Enhanced Transformer with Rotary Position Embedding** — Su et al., 2021.
Chapter 13's rotary embeddings, and the reason Chapter 14's MLA needs a decoupled variant.

**Root Mean Square Layer Normalization** — Zhang and Sennrich, NeurIPS '19, and **GLU
Variants Improve Transformer** — Shazeer, 2020. The other two components of Chapter 13's
block.

**DeepSeek-V2** and **DeepSeek-V3 Technical Report** — 2024. Multi-head latent attention with
its absorption trick and decoupled RoPE (Chapter 14), auxiliary-loss-free load balancing and
node-limited routing (Chapter 17), and the DualPipe overlap Chapter 17's two-batch overlap
parallels.

### Mixture of experts

**GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding** —
Lepikhin et al., 2020. The routing formulation, auxiliary balance loss, and capacity factors
Chapter 17 traces.

**Switch Transformers** — Fedus et al., 2021. Top-1 routing and the scaling case.

**DeepSeekMoE** — Dai et al., 2024. Fine-grained experts plus shared experts, both visible in
Chapter 17's `TopK` constructor.

### Generation techniques

**Fast Inference from Transformers via Speculative Decoding** — Leviathan et al., ICML '23,
and **Accelerating Large Language Model Decoding with Speculative Sampling** — Chen et al.,
2023. The modified rejection-sampling rule Chapter 19 proves, and the capped-geometric
expression for tokens per step.

**EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty** — Li et al., ICML '24,
with EAGLE-2 and EAGLE-3. Feature-level drafting, the shifted token input, and the tree
attention mask of Chapter 19.

**The Curious Case of Neural Text Degeneration** — Holtzman et al., ICLR '20. Nucleus
sampling, and the failure modes it was introduced to fix — Chapter 8.

**Efficient Guided Generation for Large Language Models** — Willard and Louf, 2023
(Outlines). The FSM-indexed vocabulary of Chapter 20.

**XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models** —
Dong et al., 2024. The byte-level pushdown automaton, the context-independent/dependent
token split, and the adaptive mask cache of Chapter 20.

### Adaptation and scale

**LoRA: Low-Rank Adaptation of Large Language Models** — Hu et al., ICLR '22. The low-rank
hypothesis of Chapter 21, and the merged-weight advice that multi-adapter serving has to
refuse.

**Punica: Multi-Tenant LoRA Serving** and **S-LoRA: Serving Thousands of Concurrent LoRA
Adapters** — 2023. The segmented gather primitive Chapter 21's batch arrays construct.

**Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism** —
Shoeybi et al., 2019. The column-then-row derivation and the two-collectives-per-block count
of Chapter 16.

**Visual Instruction Tuning** — Liu et al., NeurIPS '23 (LLaVA). The projector-into-the-token-
stream architecture of Chapter 21.

**Defeating Nondeterminism in LLM Inference** — Thinking Machines, 2025. Batch invariance as
the actual cause of temperature-0 nondeterminism, and which kernels have to be constrained —
Chapters 8 and 22.

---

## SGLang's own writing

The LMSYS blog (linked throughout `README.md`) is the primary source for how the system
evolved, and the posts most relevant to this book:

- **v0.4** — the zero-overhead batch scheduler of Chapter 5, cache-aware load balancing of
  Chapter 18, and faster structured outputs from Chapter 20.
- **Large-scale expert parallelism** (96 H100s) and the **GB200 rack-scale** posts —
  Chapters 17 and 18 at the scale that motivated them.
- **PD disaggregation** — the measurements behind Chapter 18.
- **Next-generation speculative decoding (DFlash, Spec V2)** — Chapter 19's newest
  algorithms.
- **Day-0 model support** posts — what Chapter 23's "adding a model" looks like under time
  pressure.

`docs/` is the official documentation, and the sections this book leans on most:

- `docs/docs/advanced_features/` — the operational counterpart to Parts IV–VI.
- `docs/docs/developer_guide/` — contribution process, benchmarking, profiling.
- `docs/docs/hardware-platforms/` — Chapter 23's porting targets.
- `docs/docs/references/` — production metrics, tracing, environment variables.

`.claude/skills/` are the maintainers' own playbooks, and several are better than any
external writing on their topics:

| Skill | Topic | Chapter |
| --- | --- | --- |
| `.claude/skills/large-class-style/SKILL.md` | why `Scheduler` looks the way it does | 5 |
| `.claude/skills/speculative-naming/SKILL.md` | speculative decoding vocabulary | 19 |
| `.claude/skills/kl-consistency-test/SKILL.md` | prefill/decode logprob consistency | 8 |
| `.claude/skills/debug-distributed-hang/SKILL.md` | localizing rank divergence | 16 |
| `.claude/skills/llm-torch-profiler-analysis/SKILL.md` | profile triage | 22 |
| `.claude/skills/sglang-prod-incident-triage/SKILL.md` | live-serving incidents | 22 |
| `.claude/skills/compute-mamba-ratio/SKILL.md` | sizing hybrid model pools | 9, 14 |
| `.claude/skills/ci-workflow-guide/SKILL.md` | CI orchestration | 23 |
---

## Neighbouring systems

Reading another engine is the fastest way to see which of SGLang's choices are necessary and
which are preferences.

**vLLM** — the closest comparison, and the origin of paged attention. SGLang's
acknowledgment section lists it first.

**TensorRT-LLM** — NVIDIA's engine. A compilation-first design, which throws SGLang's
runtime-flexibility choices into relief.

**FlashInfer** — the attention kernel library Chapter 14 treats as a contract. Reading its
API explains why SGLang's backend is shaped the way it is.

**LightLLM**, **Guidance**, **Outlines**, **LMQL** — also credited in
`README.md`. Outlines is Chapter 20's original constrained-decoding backend; Guidance and
LMQL are ancestors of the Chapter 3 DSL.

**sglang-jax** — the TPU port, as a separate project. Evidence for where Chapter 23's
hardware abstraction stops.

---

## A note on staying current

This book is pinned to one commit (see the introduction), and SGLang moves quickly — the
`README.md` news section alone spans day-0 support for a dozen model families in under a
year.

What ages well: the cost model of Chapter 1, the process topology of Chapter 3, the request
path of Part II, the two-level memory indirection of Chapter 9, and the radix tree of
Chapter 10. These are the load-bearing structures, and they have been stable across the
project's life.

What ages fast: kernel backends, quantization formats, speculative algorithms, and hardware
support. Chapters 14, 15, and 19 name specific implementations that will change.

Read the stable parts here; read `docs/` and the blog for the rest.

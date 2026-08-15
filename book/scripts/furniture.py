#!/usr/bin/env python3
"""Generate each chapter's opening contract and closing summary card.

Two things the book was missing, both about a reader's ability to *use* it
rather than only to read it:

  * **Objectives.** A reader deciding whether to spend twenty minutes on a
    chapter had nothing to go on but a one-line epigraph. Every chapter now
    states, before the body starts, what it will leave you able to do.

  * **A summary card.** The book was read-once: conclusions lived in the middle
    of paragraphs, so returning to it meant re-reading it. Every chapter now
    ends with the argument in five lines, the numbers worth keeping, and the
    files the chapter lives in.

Both are declared here rather than hand-written into the markdown, so they stay
uniform in shape and can be regenerated. Reading time is computed from the real
word count, so it cannot drift.

Usage:  python3 book/scripts/furniture.py [--check]
"""

from __future__ import annotations

import pathlib
import re
import sys

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"

OPEN_BEGIN = "<!-- objectives:begin -->"
OPEN_END = "<!-- objectives:end -->"
SUM_BEGIN = "<!-- summary:begin -->"
SUM_END = "<!-- summary:end -->"

WPM = 220  # a technical read, with code, is slower than prose


# ===========================================================================
# Per-chapter content. `does` is the reader's contract; `argument` is the
# chapter in five lines; `numbers` are the figures worth keeping; `lives_in`
# is where to look in the tree.
# ===========================================================================

CHAPTERS: dict[str, dict] = {
    "ch01-why-serving-engines": dict(
        does=[
            "Compute, for a given model and GPU, how many users it can serve at once",
            "Explain why decode is memory-bound and prefill is not",
            "Derive the critical batch size, and say what it means for a scheduler",
            "Name which of TTFT, ITL, throughput, and goodput a technique helps and which it charges",
        ],
        argument=[
            "Decode reads every weight to produce one token per sequence, so it is bound by memory bandwidth, not arithmetic.",
            "The fix is batching: one weight read serving many tokens. Arithmetic intensity for a weight-bound GEMM is 2B/s — batch size and dtype, nothing else.",
            "Batch size is capped by KV cache memory, so memory capacity is the direct limiter on throughput.",
            "Little's Law turns that capacity into a request rate, and the ceiling moves only by fitting more caches, finishing faster, or storing prefixes once.",
            "Every later chapter is a response to one of those three.",
        ],
        numbers=[
            ("H100 ridge point", "~296 FLOP/byte"),
            ("Critical batch size at BF16", "~296 rows"),
            ("KV per token, Llama-3-70B", "320 KB"),
            ("Concurrent 4k conversations on 4×H100", "~120"),
        ],
        lives_in=[("The smallest thing that runs a real forward pass",
                   "python/sglang/benchmark/one_batch.py")],
    ),
    "ch02-the-machine-underneath": dict(
        does=[
            "Read a GPU spec sheet and say which number binds your workload",
            "Compute the ridge point for any accelerator, and explain why it barely moves across generations",
            "Place a collective on the interconnect ladder and predict what it costs",
            "Find the handful of hardware facts SGLang reads at startup",
        ],
        argument=[
            "An accelerator is three numbers: memory capacity, memory bandwidth, and tensor-core FLOPS. Each binds a different phase.",
            "Their ratio is Chapter 1's ridge point, and it sits between roughly 150 and 600 FLOP/byte on every part in production — nobody is fixing the memory wall.",
            "The interconnect ladder falls by an order of magnitude per step; NVLink to InfiniBand is the cliff that keeps tensor parallelism inside a node.",
            "Spec sheets are chosen, not false: check dense vs sparse, precision, peak vs achieved, and SKU.",
            "SGLang carries no table of GPUs — it asks the device a few questions and computes the rest.",
        ],
        numbers=[
            ("H100 SXM", "80 GB · 3.35 TB/s · 990 TF BF16"),
            ("NVLink vs InfiniBand", "~20× bandwidth gap"),
            ("HBM vs PCIe", "~100× gap"),
            ("Realistic fraction of peak", "60–80% bandwidth"),
        ],
        lives_in=[
            ("Platform abstraction", "python/sglang/srt/platforms/interface.py"),
            ("Capability probes", "python/sglang/srt/utils/common.py"),
            ("Memory fraction default", "python/sglang/srt/server_args.py"),
        ],
    ),
    "ch03-shape-of-sglang": dict(
        does=[
            "Name the four processes and the socket between each pair",
            "Explain why the schedulers are replicated rather than driven by one controller",
            "Choose between the HTTP server, the Engine class, and the DSL for a given job",
            "Predict which process a given kind of bug lives in",
        ],
        argument=[
            "Tokenization, GPU work, and detokenization are split across processes because in one process the GIL would make the GPU wait on string handling.",
            "The schedulers are SPMD: every rank runs the same code over broadcast data and independently reaches the same conclusion.",
            "That is chosen for per-step latency — a single controller would need a round trip per rank inside a ten-millisecond decode step.",
            "The price is that ranks must stay deterministic in lockstep; disagreement deadlocks rather than crashes.",
            "Output travels forward to the tokenizer manager, not back to the scheduler, because that is where the client's future lives.",
        ],
        numbers=[
            ("Processes in the default topology", "4"),
            ("Schedulers receiving from the front end", "rank 0 only"),
        ],
        lives_in=[
            ("Topology construction", "python/sglang/srt/entrypoints/engine.py"),
            ("Port and socket names", "python/sglang/srt/server_args.py"),
            ("HTTP surface", "python/sglang/srt/entrypoints/http_server.py"),
        ],
    ),
    "ch04-http-to-token-ids": dict(
        does=[
            "Follow a request from an HTTP handler to a message on a ZeroMQ socket",
            "Explain what a tokenizer does and which of its properties leak into the engine",
            "Say why validation happens at the front and not at the scheduler",
            "Trace an abort across process boundaries",
        ],
        argument=[
            "The front end is an async/sync boundary: an asyncio event loop on one side, a synchronous scheduler loop on the other.",
            "Most of its complexity is making message passing look like `await` — a future per request, resolved when a message comes back.",
            "Subword tokenization is greedy and not compositional, which is why detokenization cannot be done one token at a time.",
            "Validation is a stability boundary: past it, a malformed request would crash a process holding the model.",
            "Large tensors travel by shared memory, not through the socket.",
        ],
        numbers=[("Rough English tokens per character", "~1 per 4")],
        lives_in=[
            ("Manager and event loop", "python/sglang/srt/managers/tokenizer_manager.py"),
            ("Batched tokenization", "python/sglang/srt/managers/async_dynamic_batch_tokenizer.py"),
            ("Wire types", "python/sglang/srt/managers/io_struct.py"),
        ],
    ),
    "ch05-scheduler-loop": dict(
        does=[
            "Read `event_loop_normal` and say what each of its four steps does",
            "Explain what the overlap loop reorders, and the hazard that reordering creates",
            "Say why the CPU may run ahead of the GPU, and what breaks it",
            "Explain how a batch can be built from tokens that do not exist yet",
        ],
        argument=[
            "One synchronous loop owns the GPU and answers one question per iteration: what runs next.",
            "Kernel launches are asynchronous, so the CPU can describe step N+1 while the GPU executes step N — unless something forces a host sync.",
            "The overlap loop launches batch N, then processes the results of N−1 while N is still running.",
            "That requires referring to tokens before they exist, which `FutureMap` does by passing pool indices and letting the GPU dereference them.",
            "Anything that reads a device value on the host — `.item()`, `.cpu()` — drains the launch queue and gives back everything overlap bought.",
        ],
        numbers=[("Kernel launches per forward, 70B", "~1,000"),
                 ("CPU cost per launch", "5–10 µs")],
        lives_in=[
            ("The loop", "python/sglang/srt/managers/scheduler.py"),
            ("Future tokens", "python/sglang/srt/managers/overlap_utils.py"),
        ],
    ),
    "ch06-deciding-what-runs": dict(
        does=[
            "Explain continuous batching as admission control under a memory budget",
            "Compute the token budget and say what each of its three terms is for",
            "Describe what retraction is and when it fires",
            "Choose between LPM and DFS-weight ordering, and say why they agree on a static batch",
        ],
        argument=[
            "Admitting a request commits memory whose size nobody knows, because output length is unknown.",
            "Orca made scheduling per-iteration; vLLM made the accounting paged; Sarathi made long prefills chunkable. This scheduler is all three plus prefix-aware ordering.",
            "The budget is denominated in tokens, not requests, which is what the ragged batch layout buys.",
            "When the estimate is wrong the scheduler retracts — destroying work to stay alive — so a nonzero retraction rate means admission is too optimistic.",
            "Ordering by longest cached prefix is provably the optimal cache order on a static batch; on a stream it needs the tree's structure to hold groups together.",
        ],
        numbers=[
            ("SGLang's cache hit rate vs an oracle", "~96%"),
            ("Sarathi's decode-time improvement", "12.5 ms → 1.2 ms per token"),
        ],
        lives_in=[
            ("The decision", "python/sglang/srt/managers/scheduler.py"),
            ("Admission and policy", "python/sglang/srt/managers/schedule_policy.py"),
        ],
    ),
    "ch07-executing-a-batch": dict(
        does=[
            "List the forward modes and say what each one changes downstream",
            "Explain the ragged batch layout and which operators read its offsets",
            "Describe the purity rule on `ForwardBatch.init_new` and why overlap requires it",
            "Follow the handoff from scheduler objects to GPU tensors",
        ],
        argument=[
            "The batch's mode determines the world: which kernel runs, which metadata is built, whether a graph can replay.",
            "Batches are ragged, not rectangular — a flat token buffer plus offsets — because a well-packed batch is deliberately uneven.",
            "Linear layers ignore the offsets; attention reads them. That is Orca's selective batching, inherited as a data layout.",
            "`init_new` treats its input as read-only, because the overlap loop queues snapshots that must describe what the GPU actually ran.",
            "The worker boundary is a substitution point — speculative decoding replaces it entirely.",
        ],
        numbers=[],
        lives_in=[
            ("Forward batch", "python/sglang/srt/model_executor/forward_batch_info.py"),
            ("Worker", "python/sglang/srt/managers/tp_worker.py"),
            ("Model runner", "python/sglang/srt/model_executor/model_runner.py"),
        ],
    ),
    "ch08-sampling-and-return": dict(
        does=[
            "Say what temperature, top-k, top-p, and min-p each do and why their order matters",
            "Explain why the same prompt at temperature 0 can give different output",
            "Describe why incremental detokenization cannot decode one token at a time",
            "Follow a token from a logit to a streamed chunk of text",
        ],
        argument=[
            "Sampling is batched tensor work because every request in the batch asked for different parameters.",
            "Truncation methods differ in what they threshold on: a fixed count, cumulative mass, or a fraction of the peak.",
            "Order matters — temperature is applied before the truncations, so it changes which tokens survive them.",
            "Nondeterminism at temperature 0 is not the sampler; it is batch-size-dependent reduction order in the kernels.",
            "Detokenization is incremental and stateful because tokens are byte sequences that can split a character.",
        ],
        numbers=[("Batch-invariant kernels' cost", "~1.6–2× slower")],
        lives_in=[
            ("Sampler", "python/sglang/srt/layers/sampler.py"),
            ("Per-request parameters", "python/sglang/srt/sampling/sampling_batch_info.py"),
            ("Detokenizer", "python/sglang/srt/managers/detokenizer_manager.py"),
        ],
    ),
    "ch09-kv-pools": dict(
        does=[
            "Follow the two-level indirection from a request to a KV tensor",
            "Explain what paging borrows from operating systems and which wastes it removes",
            "Say what page size trades off, and why SGLang's default is 1",
            "Compute how much memory the pool actually gets",
        ],
        argument=[
            "Level one maps a request and a position to a KV index; level two maps that index to storage. Neither is contiguous per request.",
            "Contiguous per-request allocation left 60–80% of KV memory unused, in three distinct ways.",
            "Paging removes all three: uniform pages cannot fragment externally, on-demand allocation cannot reserve, and internal waste is bounded to one page.",
            "The price is indirection, which every attention kernel must then perform itself — that is what 'paged attention' names.",
            "Running out of KV memory is an expected condition, so the allocator returns `None` rather than raising.",
        ],
        numbers=[
            ("KV memory actually used, pre-paging", "20–38%"),
            ("SGLang default page size", "1 token"),
        ],
        lives_in=[
            ("Pools", "python/sglang/srt/mem_cache/memory_pool.py"),
            ("Paged allocator", "python/sglang/srt/mem_cache/allocator/paged.py"),
        ],
    ),
    "ch10-radixattention": dict(
        does=[
            "Explain what a radix tree over token sequences buys, and why the key is token ids",
            "Walk a match, a split, and an insert",
            "Explain reference counting and why eviction goes leaf-first",
            "Say what makes prefix sharing sound rather than merely fast",
        ],
        argument=[
            "Real workloads share long prefixes — system prompts, few-shot examples, conversation history — and recompute them per request.",
            "A radix tree keyed on token ids stores each shared prefix once and finds the longest match in one walk.",
            "Nobody declares the branch points; splitting discovers them from the workload's own shape.",
            "Reference counting is the invariant that makes it safe: a node in use by a running request cannot be evicted.",
            "Eviction runs leaf-first, so the most-shared prefixes survive longest — which is LRU with the tree's structure as the tiebreak.",
        ],
        numbers=[
            ("Shared prefix, Chapter 1's deployment", "1,800 of 2,550 tokens"),
            ("Cache that duplicating it would cost", "105 GB of 150 GB"),
            ("Concurrency once it is shared once", "192 → far more"),
        ],
        lives_in=[
            ("The tree", "python/sglang/srt/mem_cache/radix_cache.py"),
            ("Where requests meet it", "python/sglang/srt/managers/schedule_batch.py"),
        ],
    ),
    "ch11-beyond-hbm": dict(
        does=[
            "Decide whether fetching a cached prefix beats recomputing it",
            "Explain how the radix tree gains tiers without changing its shape",
            "Say what the write-back invariant is and why it exists",
            "Name what a remote KV store buys and what it costs",
        ],
        argument=[
            "GPU memory holds a small cache; host DRAM and NVMe hold far more at far less bandwidth.",
            "Whether to fetch or recompute is arithmetic: the tier's bandwidth against the model's prefill throughput in KV bytes per second.",
            "Fixed costs dominate for short prefixes, so every tier carries a minimum size.",
            "The win is real only when the transfer overlaps compute — a synchronous fetch pays its full latency.",
            "The same idea as Chapter 14's SRAM-versus-HBM tiling, one level of the hierarchy down.",
        ],
        numbers=[
            ("Host DRAM over PCIe 5", "~50 GB/s"),
            ("NVMe", "2–14 GB/s"),
        ],
        lives_in=[("Tiered tree", "python/sglang/srt/mem_cache/hiradix_cache.py")],
    ),
    "ch12-loading-weights": dict(
        does=[
            "Explain how a rank loads only its own shard without materializing the whole model",
            "Say why safetensors' layout is what makes that free",
            "Read the two generations of the weight-loading protocol against each other",
            "Choose a weight-update path for a reinforcement-learning loop",
        ],
        argument=[
            "Weight loading looks like file I/O and is really distributed sharding: no rank ever holds a complete matrix.",
            "Each parameter carries its own loader, so the checkpoint is iterated once and each parameter takes the slice it wants.",
            "Safetensors is memory-mappable, so a rank never faults in the pages it does not need — which is what makes per-rank slicing free.",
            "The legacy path spells the fusion mapping out per model; the v2 path factors it into a registry, because 218 models cannot each re-implement it.",
            "The same machinery serves reinforcement learning, where weights are replaced every few minutes without a restart.",
        ],
        numbers=[
            ("Llama-3-70B in BF16", "130 GB"),
            ("Copies of the weights: disk path vs IPC path", "3 vs 0"),
        ],
        lives_in=[
            ("Loaders", "python/sglang/srt/model_loader/loader.py"),
            ("A model's own loader", "python/sglang/srt/models/llama.py"),
        ],
    ),
    "ch13-anatomy-of-a-model": dict(
        does=[
            "Name the four components of a modern decoder block and what each replaced",
            "Read llama.py end to end and say what each layer choice is for",
            "Explain why models are rewritten rather than imported",
            "Distinguish a model's total head count from this rank's",
        ],
        argument=[
            "RMSNorm, SwiGLU, RoPE, and GQA each replaced something simpler, and each replacement shows up as an engine constraint.",
            "GQA exists because the KV cache is the scarce resource — it is Chapter 1's argument reaching back into the architecture.",
            "A model file is a `forward` and a `load_weights` over a shared vocabulary of parallel, quantization-aware layers.",
            "Fusions like `gate_up_proj` are SGLang's, not the checkpoint's, which is why loading has to split one tensor across two slices.",
            "The model contains no cache logic at all; `RadixAttention` takes a layer id and the rest arrives through the forward batch.",
        ],
        numbers=[
            ("Llama-3-70B query heads / KV heads", "64 / 8"),
            ("KV cache saved by GQA at that ratio", "8×"),
        ],
        lives_in=[
            ("The template model", "python/sglang/srt/models/llama.py"),
            ("Parallel layers", "python/sglang/srt/layers/linear.py"),
        ],
    ),
    "ch14-attention-backends": dict(
        does=[
            "State the two-phase metadata/kernel contract and why it exists",
            "Derive FlashAttention's tiling and online-softmax rescaling",
            "Read a paged attention kernel signature and name what each argument is",
            "Explain what MLA changes about attention's memory layout",
        ],
        argument=[
            "Hardware, sequence shape, attention variant, and kernel maturity vary independently, so no single kernel fills the matrix.",
            "Metadata is prepared once per forward and kernels run once per layer; nearly every backend bug violates that split.",
            "FlashAttention never materializes the score matrix: online softmax lets a tiled pass produce the exact same answer.",
            "Its `Θ(N²d²M⁻¹)` HBM bound is an optimality result about memory traffic, which is why every serious kernel has this shape.",
            "MLA compresses KV to a latent and folds the up-projections into neighbouring matrices — except for RoPE, which needs a decoupled path.",
        ],
        numbers=[
            ("FlashAttention HBM accesses", "Θ(N²d²M⁻¹) vs Θ(N² + Nd)"),
            ("MLA KV reduction, DeepSeek-V2", "93.3%"),
        ],
        lives_in=[
            ("The contract", "python/sglang/srt/layers/attention/base_attn_backend.py"),
            ("Readable reference", "python/sglang/srt/layers/attention/triton_backend.py"),
            ("The kernel", "python/sglang/kernels/ops/attention/decode_attention.py"),
        ],
    ),
    "ch15-cheap-forward-pass": dict(
        does=[
            "Say which of three taxes a given workload is paying",
            "Explain what quantization is, and the difference between weight-only and weight-and-activation",
            "Say what a CUDA graph removes and what it freezes",
            "Predict which models cannot be fully captured, and what happens then",
        ],
        argument=[
            "Quantization attacks bytes moved, CUDA graphs attack launch overhead, and compilation attacks kernel count. They compose because they attack different things.",
            "Quantization is not a transformation applied to a model — it is a different implementation of every layer, chosen at construction.",
            "Scale granularity is the central axis: per-tensor, per-channel, or per-block, trading accuracy against where the scale can be applied.",
            "Weight-only quantization wins on decode, weight-and-activation also wins on prefill — different halves of the roofline.",
            "A graph records pointers and grid dimensions, so everything that follows — static buffers, bucketing, the ban on `.item()` — is a consequence of that.",
        ],
        numbers=[
            ("Typical gain per precision step", "30–50%"),
            ("Kernel time in decode", "~20 µs, against 5–10 µs to launch"),
        ],
        lives_in=[
            ("Quantization base", "python/sglang/srt/layers/quantization/base_config.py"),
            ("FP8 end to end", "python/sglang/srt/layers/quantization/fp8.py"),
            ("Graph runner", "python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py"),
        ],
    ),
    "ch16-parallelism": dict(
        does=[
            "Derive Megatron's column-then-row split and count the collectives it costs",
            "Choose a parallelism axis for a given interconnect",
            "Explain why MLA breaks tensor parallelism's assumptions",
            "Say what data-parallel attention buys and what it demands in return",
        ],
        argument=[
            "The three classical axes differ in what they split and therefore in which interconnect they stress.",
            "Column-parallel then row-parallel is forced, not conventional: it is the only order that keeps the nonlinearity local and needs one all-reduce.",
            "Two all-reduces per block times eighty layers is 160 collectives per forward — which is why TP stays inside a node.",
            "MLA has effectively one KV head, so TP replicates the cache instead of splitting it, spending the compression immediately.",
            "Data-parallel attention splits attention by sequence and the MLP by hidden dimension, paying a gather between them.",
        ],
        numbers=[
            ("All-reduces per forward, 80 layers", "160"),
            ("NVLink vs InfiniBand", "~20×"),
        ],
        lives_in=[
            ("Groups", "python/sglang/srt/distributed/parallel_state.py"),
            ("DP attention", "python/sglang/srt/layers/dp_attention.py"),
        ],
    ),
    "ch17-moe": dict(
        does=[
            "Explain how MoE changes the cost structure without changing FLOPs per token",
            "Trace the lineage from GShard's auxiliary loss to bias-based balancing",
            "Say why the all-to-all cost is latency rather than bandwidth",
            "Describe what EPLB measures and what it does about it",
        ],
        argument=[
            "Parameter count grows; active parameters per token do not. So MoE needs more GPUs to hold weights without needing more compute.",
            "Tokens in a batch go to different experts, so every MoE layer ships tokens across the network twice.",
            "That cost is dominated by the count of synchronizations, not the bytes — which is why the kernels are hand-written and why routing is constrained.",
            "Routing is learned and therefore unbalanced; the slowest rank sets the step time.",
            "The answers are rebalancing, redundant experts, and overlapping the transfer with compute.",
        ],
        numbers=[
            ("DeepSeek-V3", "256 experts, 8 active, 671B total / 37B active"),
            ("Collectives per forward, 60 MoE layers", "120"),
        ],
        lives_in=[
            ("Router", "python/sglang/srt/layers/moe/topk.py"),
            ("Dispatch", "python/sglang/srt/layers/moe/token_dispatcher/deepep.py"),
            ("Balancing", "python/sglang/srt/eplb/eplb_manager.py"),
        ],
    ),
    "ch18-disaggregation": dict(
        does=[
            "State the interference argument and the goodput framing it needs",
            "Explain what the prefill and decode sides each do at the handshake",
            "Say when disaggregation beats chunked prefill and when it does not",
            "Describe what a cache-aware router decides and on what evidence",
        ],
        argument=[
            "Prefill and decode want opposite hardware, opposite parallelism, and are graded on different metrics.",
            "Colocating them means one long prompt taxes every decoding user in the batch, and the damage is uneven.",
            "Separating the pools lets each be configured for its own job — including different parallelism, which one machine cannot do.",
            "The KV transfer is on the critical path, so the conclusion inverts on a slow interconnect.",
            "This is the direct counter-argument to chunked prefill; both are right in different regimes, and SGLang implements both.",
        ],
        numbers=[
            ("DistServe request-rate gain", "7.4×"),
            ("Transfer as a share of latency, fast link", "<0.1%"),
        ],
        lives_in=[
            ("Prefill and decode sides", "python/sglang/srt/disaggregation/prefill.py"),
            ("Transfer", "python/sglang/srt/disaggregation/nixl/conn.py"),
        ],
    ),
    "ch19-speculative-decoding": dict(
        does=[
            "Prove that speculation leaves the output distribution unchanged",
            "Predict the speedup from an acceptance rate and a draft depth",
            "Explain why a tree beats a chain",
            "Name the subsystems speculation reaches into and why",
        ],
        argument=[
            "Running the model on five tokens costs nearly what running it on one costs, because the weights are read once either way.",
            "The acceptance rule is modified rejection sampling, and it yields the target distribution exactly — for any draft, including a bad one.",
            "Tokens per step is a capped geometric, so depth has sharply diminishing returns and width does not.",
            "EAGLE drafts at the feature level from the target's own hidden states, which is why its acceptance rate is high and its head is tiny.",
            "The cost is coupling: memory pools, scheduling budgets, forward modes, CUDA graphs, the radix cache, and grammar state all learn about it.",
        ],
        numbers=[
            ("EAGLE accepted length", "3.6–3.9 tokens/pass"),
            ("EAGLE speedup on a 70B target", "2.7–3.5×"),
        ],
        lives_in=[
            ("Capabilities", "python/sglang/srt/speculative/spec_info.py"),
            ("EAGLE worker", "python/sglang/srt/speculative/eagle_worker_v2.py"),
        ],
    ),
    "ch20-shaping-output": dict(
        does=[
            "Explain why a grammar over characters is hard to apply to tokens",
            "Say why JSON needs a pushdown automaton rather than a finite one",
            "Describe how XGrammar makes a per-step vocabulary mask affordable",
            "Explain what jump-forward decoding skips and what it complicates",
        ],
        argument=[
            "A guarantee about output shape is enforced by masking logits, so an illegal token has probability zero rather than low probability.",
            "The mismatch is that grammars are defined over characters and models emit tokens, which may span grammar boundaries.",
            "JSON is not regular — matching brackets requires a stack — so the right machine is a byte-level pushdown automaton.",
            "Most tokens' legality depends only on the current node, so their masks are precomputed and cached; under 1% need the full stack.",
            "The mask is computed on the CPU while the GPU runs the forward, which is Chapter 5's argument reappearing.",
        ],
        numbers=[
            ("XGrammar mask cache, Llama-3.1 JSON", "160 MB → 0.46 MB"),
            ("Context-dependent tokens", "<1%"),
        ],
        lives_in=[
            ("Default backend", "python/sglang/srt/constrained/xgrammar_backend.py"),
            ("Jump forward", "python/sglang/srt/constrained/outlines_jump_forward.py"),
        ],
    ),
    "ch21-per-request-variation": dict(
        does=[
            "Explain why multi-adapter serving cannot use LoRA's merged-weight trick",
            "Describe the segmented gather that replaces it",
            "Say what an image costs once it becomes tokens",
            "Explain why position arithmetic for multimodal input moved into the scheduler",
        ],
        argument=[
            "Both features break the assumption that a batch is homogeneous, and both are solved by extending the batch rather than splitting it.",
            "A LoRA update is low-rank, so the base GEMM runs once and only a thin correction is per-request.",
            "That correction is ragged — different rows want different factors — so it becomes an index tensor, not control flow.",
            "A vision encoder's output is spliced into the token stream, so an image is tokens and carries every cost a token carries.",
            "This is the same move as the page table and the sorted expert order: irregularity expressed as data.",
        ],
        numbers=[
            ("A 336×336 image at patch size 14", "576 tokens"),
            ("LoRA trainable-parameter reduction, GPT-3", "10,000×"),
        ],
        lives_in=[
            ("Adapters", "python/sglang/srt/lora/lora_manager.py"),
            ("Multimodal", "python/sglang/srt/managers/mm_utils.py"),
        ],
    ),
    "ch22-observability": dict(
        does=[
            "Name the five metrics that tell you what the engine is doing",
            "Read a profile and decide whether a kernel is worth optimizing",
            "Explain why latency has an asymptote rather than a slope",
            "Tune in the order that relieves binding constraints",
        ],
        argument=[
            "Every metric exists because someone needed it to answer a question this book has already raised — the instrumentation is a map back to the design.",
            "Cache hit rate, pool utilization, retraction count, queue depth, and spec acceptance are the five that carry information.",
            "Waiting time goes as 1/(1−ρ), so the last few percent of utilization cost more latency than all the rest combined.",
            "In a profile, gap time means a CPU problem and no kernel work will help.",
            "Whether a kernel is worth optimizing depends on which side of the roofline it sits on — a memory-bound kernel near its bandwidth ceiling is finished.",
        ],
        numbers=[
            ("Target utilization for headroom", "70–80%"),
            ("Where p99 moves", "after the queue has already grown"),
        ],
        lives_in=[
            ("Metrics", "python/sglang/srt/metrics/collector.py"),
            ("Tracing", "python/sglang/srt/observability/trace.py"),
            ("Profiler", "python/sglang/profiler.py"),
        ],
    ),
    "ch23-extending": dict(
        does=[
            "Add a model, a kernel, or an attention backend, and know what will bite",
            "Recognize the registry-plus-capability pattern the codebase uses everywhere",
            "Explain why an inference engine's test pyramid is upside down",
            "Localize a wrong-output bug to a layer",
        ],
        argument=[
            "Every extension point is the same construction: an abstract contract, a registry, a selector, and capability predicates at the call sites.",
            "Predicates rather than identity checks are what make a new implementation addable without editing existing code.",
            "Correctness here has no exact oracle — a correct reimplementation differs in the last bits, and those bits amplify through depth and through argmax.",
            "So compare early rather than late, compare distributions rather than tokens, and trust accuracy evaluations over unit tests.",
            "A model that loads and produces fluent nonsense is almost always the weight mapping.",
        ],
        numbers=[
            ("Model files in the tree", "218"),
            ("The one test that catches a wrong model", "an accuracy eval, not a unit test"),
        ],
        lives_in=[
            ("Attention registry", "python/sglang/srt/layers/attention/attention_registry.py"),
            ("Kernel registry", "python/sglang/kernels/registry.py"),
            ("Divergence tool", "python/sglang/srt/debug_utils/comparator/"),
        ],
    ),
    "ch24-running-it": dict(
        does=[
            "Account for every second of a cold start and name the flag that governs it",
            "Choose an autoscaling signal, and say why GPU utilization is not one",
            "Explain what the health endpoint proves that a liveness probe does not",
            "Decide between replacing a replica and replacing its weights",
        ],
        argument=[
            "Every chapter until now described a steady state; production is mostly the other states.",
            "Cold start is dominated by weight load, which is bandwidth-bound, and graph capture, which is compute and tunable.",
            "A decode-bound engine always looks busy, so utilization carries no information — queue depth and pool utilization do.",
            "The health check runs a real generation because the failure modes that matter leave the front end responsive.",
            "Weights are the only part of a running engine that is cheap to replace; anything touching pools or graphs needs a restart.",
        ],
        numbers=[
            ("Graph capture", "10–120 s"),
            ("Status during startup", "503, not 200"),
        ],
        lives_in=[
            ("Health and warmup", "python/sglang/srt/entrypoints/http_server.py"),
            ("Watchdog", "python/sglang/srt/managers/scheduler_components/invariant_checker.py"),
        ],
    ),
}


def reading_time(body: str) -> int:
    words = len(re.sub(r"```.*?```", " ", body, flags=re.S).split())
    return max(3, round(words / WPM))


def render_objectives(spec: dict, minutes: int) -> str:
    items = "".join(f"<li>{d}</li>" for d in spec["does"])
    return (
        f"{OPEN_BEGIN}\n"
        f'<div class="bk-objectives">'
        f'<p class="bk-objectives-head">What this chapter gives you '
        f'<span class="bk-objectives-time">· about {minutes} min</span></p>'
        f"<ul>{items}</ul></div>\n"
        f"{OPEN_END}"
    )


def render_summary(slug: str, spec: dict) -> str:
    n = int(re.match(r"ch(\d+)", slug).group(1))
    arg = "".join(f"<li>{a}</li>" for a in spec["argument"])
    parts = [
        f'<div class="bk-card">',
        f'<p class="bk-card-head">Chapter {n} in one page</p>',
        f'<ol class="bk-card-arg">{arg}</ol>',
    ]
    if spec["numbers"]:
        rows = "".join(
            f"<tr><th scope='row'>{k}</th><td>{v}</td></tr>" for k, v in spec["numbers"]
        )
        parts.append(
            f'<p class="bk-card-sub">Numbers worth keeping</p>'
            f'<table class="bk-card-table"><tbody>{rows}</tbody></table>'
        )
    if spec["lives_in"]:
        rows = "".join(
            f"<tr><th scope='row'>{k}</th><td><code>{v}</code></td></tr>"
            for k, v in spec["lives_in"]
        )
        parts.append(
            f'<p class="bk-card-sub">Where it lives</p>'
            f'<table class="bk-card-table"><tbody>{rows}</tbody></table>'
        )
    parts.append("</div>")
    return f"{SUM_BEGIN}\n" + "".join(parts) + f"\n{SUM_END}"


def strip_block(text: str, begin: str, end: str) -> str:
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end) + r"\n?", re.S)
    return pattern.sub("", text)


def strip_trailing_rules(text: str) -> str:
    """Drop the rule that precedes the card, so regenerating stays idempotent.

    Without this the separator survives each strip while a fresh one is
    appended, and every run leaves one more `---` at the end of the file.
    """
    return re.sub(r"(?:\s*\n---)+\s*$", "", text)


def place(path: pathlib.Path, slug: str, spec: dict) -> str:
    text = path.read_text()
    text = strip_block(text, OPEN_BEGIN, OPEN_END)
    text = strip_block(text, SUM_BEGIN, SUM_END)
    body = text

    # Objectives go immediately before the first horizontal rule, i.e. after the
    # chapter's orientation opening and before the body starts.
    rule = re.search(r"^---$", text, re.M)
    if rule is None:
        raise SystemExit(f"{slug}: no opening rule to place objectives before")
    obj = render_objectives(spec, reading_time(body))
    # Normalize the gap rather than inserting into whatever the strip left, so
    # that regenerating an already-generated file is a no-op.
    head = text[: rule.start()].rstrip("\n")
    text = head + "\n\n" + obj + "\n\n" + text[rule.start():]

    # The card is the last thing on the page, after the closing handoff.
    text = strip_trailing_rules(text) + "\n\n---\n\n" + render_summary(slug, spec) + "\n"
    return text


def main() -> int:
    check = "--check" in sys.argv
    changed = []
    for slug, spec in CHAPTERS.items():
        path = SRC / f"{slug}.md"
        if not path.exists():
            print(f"  !! missing {path.name}")
            return 1
        new = place(path, slug, spec)
        if new != path.read_text():
            changed.append(slug)
            if not check:
                path.write_text(new)
    if check and changed:
        print("stale furniture in: " + ", ".join(changed))
        return 1
    print(f"{len(CHAPTERS)} chapters; {len(changed)} updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())

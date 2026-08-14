# SGLang Internals — Book Outline

**Working title:** *SGLang Internals: How a Production LLM Serving Engine Works*

A book that teaches LLM inference serving from first principles, then shows exactly
how each concept is realized in the SGLang codebase — with guided walkthroughs of the
real source files.

---

## About this outline

Every chapter follows the same three-part shape:

| Section | Purpose |
| --- | --- |
| **Concepts** | The idea, independent of SGLang. Why it exists, what problem it solves, the math/systems tradeoff. |
| **Code walkthrough** | The actual implementation, traced through named files, classes, and functions. |
| **Lab / Exercises** | Something the reader runs, instruments, or modifies to make the chapter stick. |

Code references use `path:line` anchors against the repository state at the time of
writing (commit `7562e74`). Line numbers drift; the class/function names are the stable
handle, and each chapter should re-verify anchors at publish time.

**Reading paths:**
- *Users / operators* — Parts I, II, VIII, and Appendix A.
- *Contributors adding a model* — Parts I, II, IV, IX.
- *Performance engineers* — Parts III, IV, V, VI, VIII.
- *Systems researchers* — read straight through.

**Prerequisites:** Python, PyTorch basics, familiarity with transformer architecture,
comfort reading C++/CUDA at a glance (not writing it). GPU access recommended but not
required for most chapters.

---

# Part I — Foundations

*Goal: by the end of Part I the reader can explain why an LLM serving engine exists at
all, and can navigate the repo without getting lost.*

## Chapter 1 — The LLM Inference Problem

**Concepts**
- Autoregressive generation: one token at a time, each conditioned on everything before it.
- The two phases: **prefill** (compute-bound, parallel over the prompt) vs **decode**
  (memory-bandwidth-bound, one token per step per sequence).
- Why naive `model.generate()` in a `for` loop wastes 90%+ of a GPU: arithmetic intensity,
  the roofline model, why decode is bandwidth-starved.
- The KV cache: what it stores, why it turns O(n²) recompute into O(n) memory.
- KV cache sizing math: `2 × layers × kv_heads × head_dim × dtype_bytes × seq_len` per request.
  Worked example on Llama-3-70B — how quickly memory, not FLOPs, becomes the wall.
- Serving metrics that matter: TTFT, ITL/TPOT, throughput, goodput, and how they trade off.

**Code walkthrough**
- A "from scratch" ~150-line reference decoder written for this book (`book/labs/ch01/`)
  that we will keep returning to as the mental baseline SGLang optimizes away from.
- `python/sglang/bench_one_batch.py` — the smallest end-to-end thing in the repo that
  runs a real forward pass, to show the shape of the problem in SGLang's own terms.

**Lab**
- Measure prefill vs decode throughput on a small model. Plot tokens/sec vs batch size.
  Observe where each phase saturates.

---

## Chapter 2 — What Makes Serving Fast: The Idea Inventory

A survey chapter. Each idea gets ~2 pages and a forward reference to its deep-dive.

**Concepts**
- **Continuous batching** — requests join and leave the batch every step instead of
  waiting for the slowest one. → Ch. 9
- **Paged / block KV memory** — stop reserving `max_len` per request; allocate in pages. → Ch. 12
- **Prefix caching (RadixAttention)** — shared prompt prefixes computed once. SGLang's
  signature contribution. → Ch. 14
- **Chunked prefill** — split long prompts so they don't stall decode. → Ch. 9
- **CUDA graphs** — amortize kernel launch overhead in decode. → Ch. 20
- **Overlap / zero-overhead scheduling** — hide CPU scheduling behind GPU compute. → Ch. 27
- **Speculative decoding** — draft cheap, verify in parallel. → Ch. 28
- **Quantization** — fewer bits per weight/activation/KV entry. → Ch. 19
- **Parallelism** — TP, PP, DP, EP, CP, and when each one is the right hammer. → Part V
- **PD disaggregation** — separate prefill and decode onto different machines. → Ch. 25
- **Hierarchical caching** — GPU → CPU → SSD/object store KV tiers. → Ch. 15

**Code walkthrough**
- `README.md:68` — the feature list, decoded feature-by-feature into "here's what that
  actually means and where it lives."
- `python/sglang/srt/server_args.py` — a tour of the flag surface as a map of the
  feature space (~9,900 lines; we read it as a table of contents, not line by line).

**Exercise**
- For each feature in the README bullet list, locate the directory that implements it.
  (Answer key in Appendix D.)

---

## Chapter 3 — A Tour of the Repository

**Concepts**
- SGLang is really several projects in one tree: the Python runtime (`srt`), a frontend
  DSL (`lang`), kernels (`kernels`, `sgl-kernel`), a Rust router/gateway
  (`sgl-model-gateway`), a diffusion stack (`multimodal_gen`), and docs.
- The `srt` ("SGLang RunTime") package layout as a dependency graph, not an alphabetical list.
- Naming conventions and the project's own code rules.

**Code walkthrough**
```
python/sglang/
├── srt/                    the serving runtime — 90% of this book
│   ├── entrypoints/        HTTP/gRPC servers, Engine API           → Ch. 4
│   ├── managers/           Scheduler, TokenizerManager, batches    → Ch. 5–9
│   ├── model_executor/     ModelRunner, ForwardBatch, CUDA graphs  → Ch. 10, 20
│   ├── mem_cache/          KV pools, allocators, radix cache       → Part III
│   ├── layers/             attention, MoE, quant, linear, sampler  → Part IV
│   ├── models/             218 model definitions                   → Ch. 17
│   ├── distributed/        process groups, communicators           → Ch. 21
│   ├── speculative/        EAGLE, MTP, ngram, DFlash               → Ch. 28
│   ├── disaggregation/     prefill/decode split                    → Ch. 25
│   ├── lora/, constrained/, function_call/, multimodal/            → Part VI
│   └── observability/      metrics, tracing, profiling             → Ch. 36
├── lang/                   the SGLang frontend DSL                 → Ch. 34
├── kernels/                JIT + AOT kernel layer                  → Ch. 41
└── test/                   test kits and harnesses                 → Ch. 38
```
- `.claude/rules/` — the repo's own enforced conventions: `no-dataclasses.md`,
  `no-getattr-defensive.md`, `general-code-style.md`,
  `schedule-batch-out-of-place-mutation.md`, `forward-batch-init-new-purity.md`.
  These are load-bearing; a chapter that explains *why* each rule exists teaches the
  architecture as much as the code does.
- `.claude/skills/` — maintainer playbooks (`large-class-style`, `sglang-runtime-context`,
  `speculative-naming`) that document design intent found nowhere else.

**Lab**
- Install from source, launch `sglang serve` with a 1B model, send a request, and read
  the startup log top to bottom — identifying which subsystem prints each line.

---

# Part II — The Life of a Request

*The spine of the book. One request, followed from HTTP socket to streamed token, with a
chapter per stage. Every later part is a detour off this path.*

## Chapter 4 — Entry Points: Server, Engine, and the API Surface

**Concepts**
- Three ways in: the HTTP server (OpenAI-compatible + native), the in-process `Engine`
  (offline batch inference, RL rollouts), and gRPC.
- Why the engine is multi-process, not multi-threaded: the GIL, fault isolation, and
  the tokenizer/scheduler/detokenizer split.
- The process topology:
  `TokenizerManager` (async, front) → `Scheduler` × TP ranks (sync, GPU) → `DetokenizerManager`,
  all connected by ZeroMQ.

**Code walkthrough**
- `python/sglang/launch_server.py` — dispatch across HTTP / gRPC / Ray / encoder-only modes.
- `python/sglang/srt/entrypoints/http_server.py:270` `lifespan` — server startup.
- `http_server.py:874` `generate_request` — the native `/generate` endpoint.
- `http_server.py:646` `health_generate` — how liveness is actually tested (a real forward pass).
- `python/sglang/srt/entrypoints/engine.py:199` `class Engine` — the embeddable API.
- `engine.py:1052` `_launch_subprocesses` — the fork/spawn that creates the topology;
  `engine.py:848` `_launch_scheduler_processes`, `engine.py:966` `_launch_detokenizer_subprocesses`.
- `engine.py:352` `Engine.generate` — the synchronous/offline path.
- `python/sglang/srt/entrypoints/openai/serving_chat.py` and `serving_completions.py` —
  OpenAI protocol adaptation; `openai/protocol.py` for the request/response models.

**Diagram**
- Process-and-socket topology diagram, referenced throughout the rest of the book.

**Lab**
- Run the same prompt through `/generate`, `/v1/chat/completions`, and `Engine.generate`.
  Diff the resulting `TokenizedGenerateReqInput` with a print statement.

---

## Chapter 5 — The Tokenizer Manager: Front Door of the Runtime

**Concepts**
- Why tokenization lives in its own process and its own async event loop.
- Request identity (`rid`), request state tracking, and the async future/queue pattern
  that turns a message-passing backend into an `await`-able API.
- Input validation as a security and stability boundary.
- Streaming: how partial outputs get routed back to the right caller.

**Code walkthrough**
- `python/sglang/srt/managers/tokenizer_manager.py:374` `class TokenizerManager`.
- `tokenizer_manager.py:755` `generate_request` — the async entry.
- `tokenizer_manager.py:985` `_tokenize_one_request` and `:1359` `_create_tokenized_object`.
- `tokenizer_manager.py:1185` `_validate_one_request` — length limits, vocab range,
  multimodal limits, logprob constraints.
- `tokenizer_manager.py:1722` `_wait_one_response` — the per-request async generator.
- `tokenizer_manager.py:2200` `handle_loop` / `:2215` `_handle_batch_output` — the return path.
- `python/sglang/srt/managers/io_struct.py:160` `GenerateReqInput`,
  `:941` `TokenizedGenerateReqInput` — the wire contract.
- `managers/async_dynamic_batch_tokenizer.py` — batching tokenization itself.

**Lab**
- Add a custom validation rule and observe the error surface end-to-end.

---

## Chapter 6 — Inter-Process Communication and the Message Protocol

**Concepts**
- ZeroMQ socket patterns used (PUSH/PULL, PUB/SUB) and why each was chosen.
- Serialization: why `pickle` here, `msgpack`/`msgspec` there, and the CUDA-IPC path for
  multimodal tensors that avoids copying pixels through the socket.
- Broadcast semantics under tensor parallelism: rank 0 receives, all ranks must agree.
- Failure modes: a dead scheduler, a full queue, a stale `rid`.

**Code walkthrough**
- `python/sglang/srt/managers/io_struct.py` — the full message catalog; the
  `BaseReq`/`BaseBatchReq` hierarchy, `:1404` `BatchTokenIDOutput`, `:1504` `BatchStrOutput`.
- `managers/scheduler_components/ipc_channels.py` and `scheduler.py:733` `init_ipc_channels`.
- `managers/scheduler_components/request_receiver.py` and
  `scheduler.py:1872` `process_input_requests`.
- `scheduler.py:1523` `init_request_dispatcher` — the request-type → handler table.
- `managers/communicator.py` — cross-rank coordination primitives.
- `scheduler.py:1906` `_materialize_cuda_vmm_inputs` — zero-copy multimodal transport.

**Exercise**
- Trace one `AbortReq` through every process it touches.

---

## Chapter 7 — The Scheduler Event Loop

**Concepts**
- The scheduler is the heart: a synchronous loop that owns the GPU and makes one decision
  per iteration — "what runs next?"
- Loop invariants: waiting queue, running batch, memory pool, cache tree.
- Normal loop vs overlap loop (the "zero-overhead batch scheduler"): why the CPU work of
  step *N+1* must happen while the GPU is still executing step *N*.
- Idle handling, watchdogs, health checks, and graceful shutdown.

**Code walkthrough**
- `python/sglang/srt/managers/scheduler.py:378` `class Scheduler` — note the 22-mixin
  composition, and read `.claude/skills/large-class-style/SKILL.md` on why `__init__`
  is a sequence of named `init_*` calls (`scheduler.py:388`).
- `scheduler.py:1658` `run_event_loop` — the dispatcher between loop variants.
- `scheduler.py:1714` `event_loop_normal` — read this first; it is the honest, simple version.
- `scheduler.py:1749` `event_loop_overlap` — then this, as a delta against the simple one.
- `scheduler.py:3012` `get_next_batch_to_run` — the central decision function.
- `scheduler.py:3623` `run_batch` and `:3917` `process_batch_result`.
- `scheduler.py:4036` `on_idle` / `:4078` `is_fully_idle`.
- `managers/overlap_utils.py`, `scheduler.py:1438` `init_overlap`.
- `managers/scheduler_components/` — the extracted collaborators:
  `batch_result_processor.py`, `output_streamer.py`, `new_token_ratio_tracker.py`,
  `invariant_checker.py`.

**Diagram**
- One iteration of `event_loop_overlap` as a timeline showing CPU and GPU lanes.

**Lab**
- Instrument the loop to log its per-iteration decision for 100 steps under load;
  reconstruct the schedule.

---

## Chapter 8 — Requests and Batches: `Req` and `ScheduleBatch`

**Concepts**
- The request state machine: waiting → running → finished/retracted/aborted.
- What a request must remember: token ids, output ids, prefix match, KV indices,
  sampling params, grammar state, logprob accumulators, multimodal payloads.
- `ScheduleBatch` (CPU-side scheduling view) vs `ModelWorkerBatch` (transport) vs
  `ForwardBatch` (GPU-side execution view) — three different objects for three different jobs.
- Retraction: what happens when the engine over-commits memory and must evict a *running*
  request.
- Why the repo bans in-place mutation of batches (`.claude/rules/schedule-batch-out-of-place-mutation.md`).

**Code walkthrough**
- `python/sglang/srt/managers/schedule_batch.py:811` `class Req`, `:814` `__init__` —
  a guided read of the fields, grouped by subsystem.
- `schedule_batch.py:1298` `init_next_round_input` — prefix matching per round.
- `schedule_batch.py:223`–`:283` — the `FinishReason` hierarchy.
- `schedule_batch.py` `class ScheduleBatch` — `prepare_for_extend`, `prepare_for_decode`,
  `retract_decode`, `filter_batch`, `merge_batch`.
- `schedule_batch.py:318` `MultimodalDataItem` / `:590` `MultimodalInputs`.
- `scheduler.py:2363` `handle_generate_request` — where a wire message becomes a `Req`.

**Exercise**
- Write out the full field-level diff of a `Req` before and after one decode step.

---

## Chapter 9 — Scheduling Policy: Continuous Batching in Practice

**Concepts**
- Continuous batching, precisely: the admission problem under a hard memory budget.
- Policies: FCFS, LPM (longest prefix match), DFS-weight, priority, random — and the
  cache-aware vs cache-agnostic split.
- The token budget: `max_total_tokens`, `max_prefill_tokens`, `max_running_requests`,
  and the "new token ratio" heuristic that predicts future decode demand.
- Chunked prefill: why long prompts must be split, and the latency it buys.
- Prefill/decode interleaving and the mixed-mode batch.
- Starvation, fairness, priority scheduling, and queue timeouts.

**Code walkthrough**
- `python/sglang/srt/managers/schedule_policy.py:216` `class SchedulePolicy`,
  `:237` `calc_priority`, `:314` `_compute_prefix_matches`, `:374` `_sort_by_longest_prefix`,
  `:387` `_sort_by_dfs_weight`.
- `schedule_policy.py:504` `class PrefillAdder` — the admission-control core;
  `:664` `rem_total_tokens`, `:857` `_update_prefill_budget`, `add_one_req`.
- `scheduler.py:3154` `get_new_batch_prefill` / `:3180` `_get_new_batch_prefill_raw`.
- `scheduler.py:3478` `update_running_batch` — decode-side admission and retraction.
- `scheduler.py:1153` `init_chunked_prefill`, `:1204` `init_schedule_policy`.
- `scheduler.py:2715` `_add_request_to_queue`, `:2739` `_set_or_validate_priority`,
  `:2813` `_abort_on_waiting_timeout`.

**Lab**
- Run a workload with shared prefixes under FCFS vs LPM; measure cache hit rate and TTFT.

---

## Chapter 10 — Model Execution: `TpModelWorker`, `ModelRunner`, `ForwardBatch`

**Concepts**
- The handoff from scheduling (CPU, Python objects) to execution (GPU, tensors).
- `ForwardMode`: `EXTEND`, `DECODE`, `MIXED`, `IDLE`, `TARGET_VERIFY`, `DRAFT_EXTEND`,
  `SPLIT_PREFILL` — the mode determines nearly everything downstream.
- Why `ForwardBatch.init_new` must be pure (`.claude/rules/forward-batch-init-new-purity.md`).
- The forward pass contract: what the model receives, what it must return.

**Code walkthrough**
- `python/sglang/srt/managers/tp_worker.py:74` `BaseTpWorker`, `:299` `TpModelWorker`.
- `python/sglang/srt/model_executor/model_runner.py:284` `class ModelRunner` —
  `:287` `__init__` as an ordered initialization script (weights → memory pool →
  attention backend → CUDA graphs).
- `model_runner.py:1057` `load_model`, `:807` `alloc_memory_pool`,
  `:927` `init_attention_backends`, `:992` `init_cuda_graphs`.
- `model_runner.py:1505` `forward` / `:1649` `_forward_raw` — mode dispatch.
- `model_runner.py:1766` `sample`.
- `python/sglang/srt/model_executor/forward_batch_info.py:98` `ForwardMode`,
  `:412` `class ForwardBatch`, `:739` `init_new`.
- `model_executor/forward_context.py` — the ambient per-forward context and why it exists.
- `model_executor/model_runner_components/` — extracted setup helpers.

**Diagram**
- `Req` → `ScheduleBatch` → `ModelWorkerBatch` → `ForwardBatch` field-flow diagram.

---

## Chapter 11 — Logits, Sampling, and Constrained Choice

**Concepts**
- From hidden states to logits: the LM head, and why only the *last* position matters in
  decode but every position may matter in prefill (logprobs, EAGLE, hidden-state capture).
- Sampling: temperature, top-k, top-p, min-p, repetition/frequency/presence penalties.
- Greedy vs stochastic, determinism, and seeded sampling.
- Logprobs: input logprobs, output logprobs, top-k logprobs, and their cost.
- Where grammar masking is applied (forward reference to Ch. 29).

**Code walkthrough**
- `python/sglang/srt/layers/logits_processor.py:282` `class LogitsProcessor`,
  `:332` `forward`, `:427` `_get_pruned_states`, `:646` `_get_logits`, `:693` `_compute_lm_head`.
- `logits_processor.py:96` `LogitsProcessorOutput`, `:149` `LogitsMetadata`.
- `python/sglang/srt/layers/sampler.py:70` `class Sampler`, `:97` `forward`,
  `:246` `_sample_from_probs`, `:563` `top_k_top_p_min_p_sampling_from_probs_torch`.
- `sampler.py:684` `multinomial_with_seed` — reproducible sampling.
- `python/sglang/srt/sampling/sampling_batch_info.py` — batched sampling parameters.
- `sampling/penaltylib/` — penalty implementations.
- `sampling/custom_logit_processor.py` — the user extension point.

**Lab**
- Implement a custom logit processor (e.g., banning a token list) and serve it.

---

## Chapter 12 — Detokenization and Streaming Output

**Concepts**
- Incremental detokenization is genuinely hard: BPE merges, multi-byte UTF-8, and why you
  cannot just `decode()` the new token.
- Stop conditions: EOS ids, stop strings, stop regex, max tokens — and why stop *strings*
  require lookback and trimming.
- Streaming chunk coalescing and the latency/overhead tradeoff.

**Code walkthrough**
- `python/sglang/srt/managers/detokenizer_manager.py:91` `class DetokenizerManager`,
  `:166` `event_loop`, `:290` `_decode_batch_token_id_output`, `:430` `handle_batch_token_id_out`.
- `detokenizer_manager.py:176` `trim_matched_stop`.
- `schedule_batch.py:1425` `init_incremental_detokenize`, `:1445` `_stop_match_tail_len`.
- `tokenizer_manager.py:1641` `_coalesce_streaming_chunks`.
- `entrypoints/openai/sse_utils.py` — SSE framing for OpenAI streaming.

**Exercise**
- Construct a prompt where naive per-token decoding produces mojibake; verify SGLang doesn't.

---

# Part III — Memory and Caching

*The part that explains SGLang's most distinctive engineering.*

## Chapter 13 — KV Cache Memory Management

**Concepts**
- The two-level indirection: request → token slots (`ReqToTokenPool`), token slot → KV
  storage (`TokenToKVPool`). Why two levels instead of one.
- Paged attention: pages, page size, fragmentation, and the allocator's job.
- Layout choices: layer-major vs page-major, and their effect on transfer and locality.
- KV dtype: FP16/BF16 vs FP8 vs FP4 caches.
- Memory budgeting at startup: how `--mem-fraction-static` becomes a concrete pool size.
- MLA (DeepSeek), sliding-window (SWA), and hybrid attention pools — one size does not fit all.

**Code walkthrough**
- `python/sglang/srt/mem_cache/memory_pool.py:256` `ReqToTokenPool`.
- `memory_pool.py:1624` `class KVCache` (ABC) → `:1755` `MHATokenToKVPool`,
  `:3932` `MLATokenToKVPool`, `:3577` `HybridLinearKVPool`, `:4348` `DSATokenToKVPool`,
  `:3135` `PageMajorMHATokenToKVPool`.
- `memory_pool.py:335` `MambaPool`, `:1153` `HybridReqToTokenPool`.
- `mem_cache/allocator/paged.py:105` `PagedTokenToKVPoolAllocator`,
  `:149` `alloc`, `:172` `alloc_extend`, `:222` `alloc_decode`, `:261` `free`.
- `mem_cache/allocator/token.py`, `swa.py`, `mamba.py`, `hisparse.py`.
- `mem_cache/kv_cache_configurator.py`, `allocation_sizing.py`, `kv_cache_dtype.py`.
- `mem_cache/layout/` — physical layout selection.

**Diagram**
- The full address translation: `rid` → `req_pool_idx` → `req_to_token[idx, :len]` →
  KV page indices → tensor offsets.

**Lab**
- Compute the theoretical max concurrency for a given model/GPU, then verify against
  `/get_server_info`.

---

## Chapter 14 — RadixAttention: Prefix Caching as a Tree

*The chapter the book exists for.*

**Concepts**
- The insight: in real workloads (chat history, few-shot prompts, agent loops, system
  prompts) prompts share long prefixes. Recomputing them is pure waste.
- The radix tree over token sequences: nodes hold token runs and the KV indices for them.
- Matching, splitting, insertion — and how page alignment constrains all three.
- Reference counting and locking: a node in use by a running request must not be evicted.
- LRU eviction over a tree, and why eviction must be leaf-first.
- Cache-aware scheduling: the tree feeds the scheduler's priority function (Ch. 9), closing
  the loop between memory and scheduling.
- Variants: chunk cache (no reuse), SWA radix cache, Mamba radix cache, session-aware cache.

**Code walkthrough**
- `python/sglang/srt/mem_cache/radix_cache.py:59` `class RadixKey` — the key abstraction
  (token ids + optional extra key for LoRA/session namespacing); `:181` `match`,
  `:217` `child_key`, `:150` `page_aligned`.
- `radix_cache.py:238` `class TreeNode` — children, `lock_ref`, `last_access_time`, host tier.
- `radix_cache.py:303` `class RadixCache` — the main event:
  - `:376` `match_prefix` and `:678` `_match_prefix_helper`
  - `:704` `_split_node`
  - `:436` `insert` / `:737` `_insert_helper`
  - `:458` `cache_finished_req`, `:515` `cache_unfinished_req`
  - `:592` `evict`, `:622` `inc_lock_ref`, `:637` `dec_lock_ref`
- `mem_cache/base_prefix_cache.py:230` `BasePrefixCache` and the params/results protocol
  (`MatchPrefixParams`, `MatchResult`, `EvictParams`) at `:49`–`:166`.
- `mem_cache/chunk_cache.py:35` `ChunkCache` — the no-reuse baseline, useful as contrast.
- `mem_cache/swa_radix_cache.py`, `mamba_radix_cache.py`, `radix_cache_cpp.py`
  (+ `cpp_radix_tree/` for the C++ implementation and why it exists).
- `layers/radix_attention.py:91` `class RadixAttention` — the model-facing layer that
  reads and writes the cache.

**Diagram**
- A worked example: three chat requests sharing a system prompt, drawn as the tree evolves
  across insert/match/split/evict.

**Lab**
- Build the same three-request scenario, dump the tree with `pretty_print`
  (`radix_cache.py:585`), and verify hit counts against `/get_server_info`.

---

## Chapter 15 — Hierarchical Cache (HiCache) and KV Offloading

**Concepts**
- Extending the cache hierarchy past GPU HBM: GPU → host DRAM → SSD / object store.
- Write-through vs write-back policies; when a prefetch pays for itself.
- Bandwidth math: at what prefix length does loading from host beat recomputing?
- Multi-tier prefix matching, and the bookkeeping to keep tiers coherent.
- Storage backends and pluggability (Mooncake, 3FS, NIXL, LMCache, file, mmap, shm).

**Code walkthrough**
- `python/sglang/srt/mem_cache/hiradix_cache.py:76` `class HiRadixCache` — subclassing
  `RadixCache` with a host tier; `:840` `write_backup`, `:369` `attach_storage_backend`,
  `:487` `detach_storage_backend`.
- `mem_cache/memory_pool_host.py` — the host-side pool.
- `mem_cache/cache_controller.py` (`managers/cache_controller.py`) — the transfer engine.
- `mem_cache/hicache_storage.py`, `mem_cache/storage/backend_factory.py` and the
  per-backend directories.
- `mem_cache/unified_cache/` — the newer unified tree core and its component registry.
- Docs cross-read: `docs/docs/advanced_features/hicache_design.mdx` and
  `hicache_best_practices.mdx`.

**Lab**
- Enable HiCache with a file backend; measure hit rate and TTFT on a long-prefix workload.

---

# Part IV — The Model Execution Layer

## Chapter 16 — Model Loading and Weights

**Concepts**
- The model registry: how `--model-path` becomes a Python class.
- HF config → SGLang config translation, and where architectures diverge.
- Weight formats: safetensors, GGUF, sharded checkpoints, remote/object-store loading.
- Sharded loading under TP: each rank loads only its slice; the `weight_loader` protocol
  on each parameter.
- Startup latency: parallel loading, `load_format` choices, and dummy weights for benchmarking.
- Hot weight updates for RL (forward reference to Ch. 39).

**Code walkthrough**
- `python/sglang/srt/model_loader/loader.py` and `auto_loader.py`.
- `model_loader/weight_utils.py` — the `default_weight_loader` and friends.
- `model_runner.py:1057` `load_model`.
- `models/llama.py:663` `load_weights` and `:743` `_load_weights_v2` — the two generations
  of the loading protocol, side by side.
- `python/sglang/srt/configs/` — per-model config classes (63 files) and `model_config.py`.
- `srt/connector/` — remote weight sources (S3, Azure, Redis, remote instance).

---

## Chapter 17 — Anatomy of a Model Implementation

**Concepts**
- What SGLang requires of a model: a `forward(input_ids, positions, forward_batch)` that
  returns logits, plus `load_weights`.
- Why models are rewritten rather than imported from HF: parallel layers, custom attention,
  cache integration, quantization hooks.
- The parallel layer vocabulary: `ColumnParallelLinear`, `RowParallelLinear`,
  `QKVParallelLinear`, `MergedColumnParallelLinear`, `VocabParallelEmbedding`, `ParallelLMHead`.
- Optional capabilities: `get_embed_and_head`, EAGLE layer capture, PP `start_layer`/`end_layer`,
  split prefill.

**Code walkthrough** — a full line-by-line read of `python/sglang/srt/models/llama.py`
- `:70` `LlamaMLP` — gate/up merged column-parallel, down row-parallel.
- `:138` `LlamaAttention` — QKV fusion, rotary embedding, `RadixAttention` instantiation.
- `:283` `LlamaDecoderLayer` — residual/normalization ordering and the fused-add-RMSNorm trick.
- `:372` `LlamaModel` — embedding, layer stack, PP boundaries.
- `:496` `LlamaForCausalLM` — `:563` `forward`, `:599` `forward_split_prefill`,
  `:891` `set_eagle3_layers_to_capture`.
- Supporting layers: `layers/linear.py:146`–`:1392` (the parallel linear family),
  `layers/vocab_parallel_embedding.py:188` / `:587`, `layers/layernorm.py`,
  `layers/activation.py`, `layers/rotary_embedding/`.

**Then, three contrast studies (short):**
- `models/deepseek_v2.py` — MLA and MoE.
- `models/qwen*_vl` (multimodal) — a vision tower bolted onto a decoder.
- `models/falcon_h1.py` or a Mamba hybrid — non-attention state.

**Lab**
- Port a small HF model to SGLang, guided by `docs/docs/supported-models/support_new_models.mdx`.

---

## Chapter 18 — Attention Backends

**Concepts**
- Why attention is pluggable at all: hardware, sequence shape, and kernel maturity all vary.
- The backend contract: metadata preparation (once per forward) vs kernel invocation
  (once per layer).
- The backend zoo: FlashInfer, FlashAttention-3, Triton, TRT-LLM, FlashMLA, CutlassMLA,
  AITER (ROCm), Ascend, Intel AMX, torch-native — and how to choose.
- MHA vs GQA vs MQA vs MLA, and why MLA needs its own pool and its own kernels.
- Sparse attention: NSA / DSA (DeepSeek), MiniMax sparse — indexer-based token selection.
- Linear attention and SSMs: Mamba2, GDN, KDA, short-conv — different state, different pool.
- Hybrid backends: different layers, different attention.

**Code walkthrough**
- `python/sglang/srt/layers/attention/base_attn_backend.py:33` `class AttentionBackend` —
  `:62` `init_forward_metadata`, `:160` `init_cuda_graph_state`, `:261` `forward_decode`,
  `:274` `forward_extend`.
- `layers/attention/attention_registry.py:34` `register_attention_backend` and the factory table.
- `layers/attention/flashinfer_backend.py` — the reference implementation, read in depth
  (metadata wrappers, page tables, CUDA-graph buffers).
- `layers/attention/triton_backend.py` — the portable fallback, easier to read whole.
- `layers/attention/flashinfer_mla_backend.py`, `flashmla_backend.py` — MLA paths.
- `layers/attention/nsa/nsa_indexer.py` + `nsa_backend.py` — sparse selection.
- `layers/attention/mamba/mamba.py`, `layers/attention/linear/gdn_backend.py`.
- `layers/attention/hybrid_attn_backend.py`, `hybrid_linear_attn_backend.py`.
- `layers/radix_attention.py:91` — where model code meets backend code.

**Lab**
- Benchmark two backends on the same workload with `--attention-backend`; explain the delta.

---

## Chapter 19 — Quantization

**Concepts**
- What gets quantized: weights, activations, KV cache — independently.
- Formats: FP8 (E4M3/E5M2), FP4/NVFP4/MXFP4, INT8, INT4, AWQ, GPTQ, blockwise schemes.
- Per-tensor vs per-channel vs per-block scales; static vs dynamic activation scaling.
- The `QuantizationConfig` → `QuantizeMethodBase` → per-layer `apply()` architecture.
- Accuracy: where quantization hurts, and how the repo measures it.
- MoE quantization as its own hard problem.

**Code walkthrough**
- `python/sglang/srt/layers/quantization/base_config.py` — `QuantizationConfig`,
  `QuantizeMethodBase`, `LinearMethodBase`.
- `layers/quantization/fp8.py` + `fp8_utils.py` — the most-used path, read end to end.
- `layers/quantization/modelopt_quant.py`, `mxfp4.py`, `w4afp8.py`, `compressed_tensors/`,
  `awq/`, `gptq/`.
- `layers/quantization/kv_cache.py` — quantized KV, and its interaction with Ch. 13 pools.
- `layers/parameter.py` — the parameter wrappers that make sharding + scales work together.
- Docs cross-read: `docs/docs/developer_guide/quantization_contribution_guide.mdx`.

---

## Chapter 20 — CUDA Graphs, `torch.compile`, and Launch Overhead

**Concepts**
- Why decode is launch-bound: thousands of tiny kernels, microseconds each.
- CUDA graphs: capture once, replay many; the constraints (static shapes, static pointers).
- Batch-size bucketing and padding; the capture matrix and its memory cost.
- Piecewise / breakable graphs: keeping graphs when part of the model can't be captured.
- `torch.compile` integration and the custom Inductor passes.
- Attention-backend cooperation: metadata buffers must be graph-safe.

**Code walkthrough**
- `python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py`,
  `decode_cuda_graph_runner.py`, `prefill_cuda_graph_runner.py`.
- `model_executor/runner_backend/full_cuda_graph_backend.py`,
  `breakable_cuda_graph_backend.py`, `tc_piecewise_cuda_graph_backend.py`.
- `model_executor/cuda_graph_config.py`, `cuda_graph_buffer_registry.py`,
  `graph_memory_usage.py`.
- `model_runner.py:992` `init_cuda_graphs`, `:1365` `init_decode_cuda_graph`,
  `:1380` `init_prefill_cuda_graph`.
- `srt/compilation/` — `backend.py`, `cuda_piecewise_backend.py`, `pass_manager.py`,
  `fix_functionalization.py`.
- Docs cross-read: `docs/docs/advanced_features/piecewise_cuda_graph.mdx`,
  `breakable_cuda_graph.mdx`.

**Lab**
- Compare decode latency with `--disable-cuda-graph` on and off; profile the launch gap.

---

# Part V — Parallelism and Distributed Execution

## Chapter 21 — Distributed Foundations

**Concepts**
- Process groups, ranks, and the four (five) axes: TP, PP, DP, EP, CP.
- Collectives that matter: all-reduce, all-gather, reduce-scatter, all-to-all — cost models
  for each on NVLink vs InfiniBand.
- Custom all-reduce, symmetric memory, and when the built-in NCCL path is not enough.
- Multi-node bootstrap: `--dist-init-addr`, node ranks, and what must match across nodes.

**Code walkthrough**
- `python/sglang/srt/distributed/parallel_state.py:237` `class GroupCoordinator`,
  `:2193` `init_distributed_environment`, `:2285` `initialize_model_parallel`.
- `distributed/device_communicators/` — custom all-reduce, PyNCCL, quick all-reduce,
  symmetric memory, and the per-vendor variants.
- `distributed/communication_op.py`, `distributed/bootstrap.py`.
- `model_runner.py:1037` `init_torch_distributed`.

---

## Chapter 22 — Tensor and Pipeline Parallelism

**Concepts**
- TP: splitting weights within a layer; the column/row pattern and its exactly-two
  communication points per block.
- Why TP needs fast interconnect and doesn't scale past a node cheaply.
- PP: splitting layers across devices; microbatching, bubbles, and schedule choice.
- Attention-TP vs MLP-TP asymmetry, and sequence-parallel norms.

**Code walkthrough**
- `layers/linear.py:293` `ColumnParallelLinear`, `:1392` `RowParallelLinear`,
  `:921` `QKVParallelLinear` — where the collectives actually live.
- `layers/communicator.py` — the layer-level communication strategy object.
- `managers/scheduler_pp_mixin.py` — pipeline scheduling in the scheduler loop.
- `models/llama.py:372` `LlamaModel` PP boundaries; `:640` `start_layer` / `:644` `end_layer`.
- `layers/model_parallel.py`, `model_runner.py:1404` `apply_torch_tp`.
- Docs cross-read: `docs/docs/advanced_features/pipeline_parallelism.mdx`.

---

## Chapter 23 — Data-Parallel Attention

**Concepts**
- The problem TP creates for MLA models: replicating a tiny KV cache across ranks wastes it.
- DP attention: each rank owns whole sequences for attention, then the batch is gathered
  for the (TP-sharded) MLP.
- The gather/scatter dance, padding modes, and the cross-rank synchronization that makes
  every rank agree on batch shape.
- Why DP attention forces "MLP sync" and idle batches.

**Code walkthrough**
- `python/sglang/srt/layers/dp_attention.py:338` `initialize_dp_attention`,
  `:412` `get_dp_local_info`, `:76` `DpPaddingMode`, `:450` `_dp_gather_via_all_reduce`,
  `:494` `_dp_gather_via_all_gather`.
- `model_executor/forward_batch_info.py:1305` `prepare_mlp_sync_batch`,
  `:1620` `post_forward_mlp_sync_batch`.
- `managers/scheduler_components/dp_attn.py`, `scheduler.py:2030` `init_dp_attn_adapter`.
- `managers/data_parallel_controller.py` — the other DP (whole-replica routing).
- `logits_processor.py:249` `compute_dp_attention_metadata`, `:757` `_gather_dp_attn_hidden_states`.

---

## Chapter 24 — Expert Parallelism and MoE

**Concepts**
- MoE inference: routing, top-k selection, expert capacity, and why MoE is
  all-to-all-bound rather than GEMM-bound.
- EP vs TP for experts; the DeepEP dispatch/combine pattern.
- Load imbalance: hot experts, and what EPLB (expert-parallel load balancing) does about it —
  rebalancing and redundant experts.
- Large-scale EP: the 96-GPU / GB200 deployments and what breaks at that scale.

**Code walkthrough**
- `python/sglang/srt/layers/moe/topk.py:392` `class TopK` and the `TopKOutput` variants (`:274`).
- `layers/moe/fused_moe_triton/` — the portable fused MoE kernel path.
- `layers/moe/ep_moe/` — expert-parallel layers and kernels.
- `layers/moe/token_dispatcher/` — DeepEP / all-to-all dispatch strategies.
- `layers/moe/moe_runner/` — the runner abstraction over MoE backends.
- `srt/eplb/expert_distribution.py`, `expert_location.py`, `eplb_manager.py`,
  `eplb_algorithms/`, `lplb_solver.py`.
- Docs cross-read: `docs/docs/advanced_features/expert_parallelism.mdx`.

**Lab**
- Record an expert distribution (`/start_expert_distribution_record`), plot imbalance,
  then enable EPLB and re-measure.

---

## Chapter 25 — Prefill/Decode Disaggregation

**Concepts**
- The core observation: prefill and decode want different hardware, different batch shapes,
  and different SLOs. Colocating them means one always compromises.
- The architecture: prefill instances, decode instances, and a KV transfer path between them.
- Bootstrap and handshake: how a decode instance learns where its KV lives.
- Transfer backends: Mooncake, NIXL, MoRI, Ascend — and the RDMA reality.
- Failure handling, and the extension to EPD (encode/prefill/decode) for multimodal.

**Code walkthrough**
- `python/sglang/srt/disaggregation/prefill.py:119` `PrefillBootstrapQueue`,
  `:485` `SchedulerDisaggregationPrefillMixin`, `:569` `event_loop_normal_disagg_prefill`.
- `disaggregation/decode.py` — the decode-side queues and `SchedulerDisaggregationDecodeMixin`.
- `disaggregation/base/conn.py` and `common/conn.py` — the KV-manager interface.
- `disaggregation/mooncake/`, `nixl/`, `mori/`, `fake/` (the test double).
- `managers/disagg_service.py`, `disaggregation/kv_events.py`.
- Docs cross-read: `docs/docs/advanced_features/pd_disaggregation.mdx`,
  `epd_disaggregation.mdx`.

---

## Chapter 26 — Routing: The Model Gateway

**Concepts**
- Why a router: cache-aware load balancing beats round-robin when prefix caching exists.
- Cache-aware routing policies and the approximate radix tree the router keeps.
- PD-aware routing, service discovery, and multi-model serving.
- Why it's written in Rust.

**Code walkthrough**
- `sgl-model-gateway/src/routers/` — the router implementations.
- `sgl-model-gateway/src/policies/` — cache-aware, round-robin, power-of-two policies.
- `sgl-model-gateway/src/core/`, `service_discovery.rs`, `server.rs`.
- `sgl-model-gateway/bindings/` — the Python bridge.
- Docs cross-read: `docs/docs/advanced_features/sgl_model_gateway.mdx`.

---

# Part VI — Advanced Runtime Features

## Chapter 27 — Overlap Scheduling and Batch Overlap

**Concepts**
- The zero-overhead batch scheduler: overlapping CPU scheduling with GPU execution, and
  the future-token trick that lets step *N+1* be prepared before step *N*'s tokens exist.
- Two-batch overlap (TBO): splitting a batch to overlap communication with computation.
- Single-batch overlap (SBO) and the operation-graph abstraction.
- Where overlap must be disabled, and why.

**Code walkthrough**
- `scheduler.py:1749` `event_loop_overlap`, `:1438` `init_overlap`,
  `:1823` `is_disable_overlap_for_batch`.
- `managers/overlap_utils.py` — future maps and resolution.
- `srt/batch_overlap/two_batch_overlap.py`, `single_batch_overlap.py`,
  `operations.py`, `operations_strategy.py`.
- `layers/attention/tbo_backend.py`.

---

## Chapter 28 — Speculative Decoding

**Concepts**
- The bandwidth argument: verifying *k* tokens costs almost the same as generating one.
- Draft-then-verify: acceptance rules, and why the output distribution is preserved.
- Algorithm families in SGLang: EAGLE / EAGLE-3 (feature-level drafting), MTP / frozen-KV MTP,
  n-gram / lookup drafting, standalone draft models, DFlash, DSpark.
- Tree drafting vs chain drafting; topk/depth/num-draft-token tuning.
- The scheduling cost: speculative steps change memory accounting, CUDA graph shapes, and
  grammar handling.
- Adaptive speculation: turning it off when acceptance drops.

**Code walkthrough** — read `.claude/skills/speculative-naming/SKILL.md` first.
- `python/sglang/srt/speculative/spec_info.py:30` `SpeculativeAlgorithm` — the capability
  predicates that gate everything else.
- `speculative/base_spec_worker.py`, `spec_registry.py`.
- `speculative/eagle_worker_v2.py:1008` `EAGLEWorkerV2` — `:1105` `forward_batch_generation`,
  `:1497` `verify`; and `:128` `EagleDraftWorker` — `:494` `draft`, `:557` `draft_forward`,
  `:726` `draft_extend`.
- `speculative/eagle_info.py` — the draft/verify batch metadata.
- `speculative/eagle_draft_cuda_graph_runner.py` — graphs for the draft model.
- `speculative/ngram_worker.py` + `cpp_ngram/`.
- `speculative/frozen_kv_mtp_worker_v2.py`, `dflash_worker_v2.py`, `standalone_worker_v2.py`.
- `speculative/adaptive_runtime_state.py`.
- Docs cross-read: `docs/docs/advanced_features/speculative_decoding.mdx`,
  `adaptive_speculative_decoding.mdx`.

**Lab**
- Measure acceptance length and end-to-end speedup across `--speculative-num-steps` /
  `--speculative-eagle-topk` sweeps.

---

## Chapter 29 — Structured Outputs and Constrained Decoding

**Concepts**
- Guaranteeing JSON/regex/EBNF-conforming output by masking logits.
- Compiling a grammar to a token-level mask: the FSM, the compressed FSM, and why
  tokenizer/grammar mismatch is the hard part.
- Jump-forward decoding: emitting deterministic spans without a forward pass.
- Backends: XGrammar, Outlines, LLGuidance — and their tradeoffs.
- Interaction with speculative decoding and with reasoning models
  (don't constrain the thinking block).

**Code walkthrough**
- `python/sglang/srt/constrained/base_grammar_backend.py:52` `BaseGrammarObject`,
  `:167` `BaseGrammarBackend`, `:311` `create_grammar_backend`.
- `constrained/xgrammar_backend.py`, `outlines_backend.py`, `llguidance_backend.py`.
- `constrained/outlines_jump_forward.py` — jump-forward implementation.
- `constrained/reasoner_grammar_backend.py`.
- `constrained/grammar_manager.py` and `scheduler.py:1962` `init_grammar_manager`,
  `:1861` `_advance_pending_grammar`.
- Mask application: `sampler.py:97` `forward` and the vocab-mask buffers
  (`base_grammar_backend.py:262` `register_vocab_mask_buffer`).

**Lab**
- Serve a strict JSON schema; measure the throughput cost of masking, then of jump-forward.

---

## Chapter 30 — LoRA and Multi-Adapter Serving

**Concepts**
- LoRA math recap, and the serving question: how do you batch requests that use
  *different* adapters?
- Adapter memory pool, slot assignment, and eviction.
- Batched LoRA kernels (SGEMM grouped / punica-style) vs merged weights.
- Dynamic adapter load/unload at runtime; radix-cache namespacing so adapters don't share
  a prefix cache incorrectly.
- LoRA on MoE layers, and on MLA models.

**Code walkthrough**
- `python/sglang/srt/lora/lora_manager.py:59` `LoRAManager` — `:221` `load_lora_adapter`,
  `:392` `fetch_new_loras`, `:428` `prepare_lora_batch`.
- `lora/mem_pool.py`, `lora/lora_registry.py`, `lora/eviction_policy.py`.
- `lora/layers.py` — the LoRA-aware layer wrappers.
- `lora/backend/` — kernel backends; `lora/lora_moe_runners.py`.
- `schedule_policy.py`/`scheduler.py:3450` `_can_schedule_lora_req` — admission with adapters.
- `RadixKey` extra-key namespacing (`radix_cache.py:64`) as it applies to LoRA.

---

## Chapter 31 — Multimodal Inputs

**Concepts**
- The pipeline: raw bytes → processor → embeddings → placeholder-token splicing → decoder.
- Where the vision/audio encoder runs: in-process, DP-parallel, or a separate encode server.
- Multimodal hashing and caching of encoder outputs.
- mrope / 3D positions for VLMs, and why position computation moves into the scheduler.
- CUDA IPC for image tensors, to avoid serializing pixels over ZMQ.

**Code walkthrough**
- `python/sglang/srt/multimodal/processors/` (53 processors) + `base_processor.py`.
- `managers/multimodal_processor.py`, `managers/mm_utils.py`, `managers/mm_schedule.py`.
- `schedule_batch.py:318` `MultimodalDataItem`, `:590` `MultimodalInputs`,
  `:569` `build_padded_input_ids`.
- `scheduler.py:2209` `_process_and_broadcast_mm_inputs`,
  `:2308` `_maybe_compute_mrope_positions`.
- `forward_batch_info.py:1163` `_compute_mrope_positions`.
- `layers/attention/vision.py` — the vision attention path.
- `mem_cache/multimodal_cache.py`.
- Docs cross-read: `docs/docs/advanced_features/vlm_query.mdx`,
  `cuda_graph_for_multi_modal_encoder.mdx`.

---

## Chapter 32 — Tool Calling, Reasoning, and Output Parsing

**Concepts**
- Model-specific function-call formats, and why every model family invented its own.
- Streaming-safe incremental parsing: you must decide "is this a tool call?" before the
  message is complete.
- Structural tags: constraining generation to a tool schema rather than parsing after.
- Reasoning separation (`<think>` blocks) and its interaction with structured output.

**Code walkthrough**
- `python/sglang/srt/function_call/function_call_parser.py`,
  `base_format_detector.py:` the detector protocol.
- A survey of detectors: `qwen*_detector.py`, `deepseekv3_detector.py`,
  `kimik2_detector.py`, `gpt_oss_detector.py`, `glm4_moe_detector.py` (39 files) —
  read two in full, skim the rest for the pattern.
- `srt/parser/reasoning_parser.py`, `srt/parser/harmony_parser.py`,
  `entrypoints/harmony_utils.py`.
- `entrypoints/openai/serving_chat.py` — where parsing plugs into the response path.
- Docs cross-read: `docs/docs/advanced_features/tool_parser.mdx`, `separate_reasoning.mdx`.

---

## Chapter 33 — Deterministic and Reproducible Inference

**Concepts**
- Why the same prompt can produce different tokens across batch sizes: non-deterministic
  reductions, split-K, and batch-variant kernels.
- Batch-invariant operators, and what they cost.
- Seeded sampling and per-request determinism.
- The prefill/decode logprob consistency (KL) test as the correctness oracle.

**Code walkthrough**
- `python/sglang/srt/batch_invariant_ops/` and
  `model_runner.py:764` `maybe_enable_batch_invariant_mode`.
- `scheduler.py:1506` `init_deterministic_inference_config`.
- `sampler.py:684` `multinomial_with_seed`.
- `.claude/skills/kl-consistency-test/SKILL.md` — the methodology, and the two independent
  conditions a zero KL requires.
- Docs cross-read: `docs/docs/advanced_features/deterministic_inference.mdx`.

---

# Part VII — The Frontend Language

## Chapter 34 — The SGLang DSL

**Concepts**
- The original research contribution: a language for *programs* over LLM calls, not just
  single completions.
- Primitives: `gen`, `select`, `fork`, `join`, system/user/assistant roles.
- Why co-designing frontend and runtime pays: `fork` becomes a radix-tree branch, and
  `select` becomes a scored comparison rather than *n* independent generations.
- Interpreter mode vs compiler/tracer mode.

**Code walkthrough**
- `python/sglang/lang/api.py` — the user-facing primitives.
- `python/sglang/lang/ir.py` — the program IR.
- `python/sglang/lang/interpreter.py:274` `StreamExecutor`, `:852` `ProgramState`,
  `:57` `run_program`, `:93` `run_program_batch`.
- `python/sglang/lang/tracer.py` — compilation to a graph.
- `python/sglang/lang/choices.py` — the `select` scoring methods.
- `python/sglang/lang/backend/` — runtime, OpenAI, Anthropic backends.
- Docs cross-read: `docs/docs/references/frontend/`.

**Lab**
- Write a branching agent program; show the radix cache hit rate that `fork` produces.

---

## Chapter 35 — API Surfaces in Practice

**Concepts**
- Native `/generate` vs OpenAI Chat/Completions vs Anthropic vs Ollama compatibility layers.
- Embeddings, reranking, scoring, and classification endpoints.
- Sampling parameter reference, and the semantics that differ from other engines.
- Sessions and the session-aware radix cache.

**Code walkthrough**
- `entrypoints/openai/serving_*.py` (chat, completions, embedding, rerank, score,
  classify, responses, transcription).
- `entrypoints/anthropic/`, `entrypoints/ollama/`.
- `srt/session/` and `docs/docs/advanced_features/session_radix_cache.mdx`.
- `sampling/sampling_params.py`.

---

# Part VIII — Running It in Production

## Chapter 36 — Observability

**Concepts**
- The metric taxonomy: throughput, queue depth, cache hit rate, memory utilization,
  spec-decode acceptance, per-request timing.
- Prometheus integration and the dashboards that matter.
- Distributed request tracing across the process topology.
- Structured logging, request dumping, and crash dumps.

**Code walkthrough**
- `python/sglang/srt/observability/metrics_collector.py`, `forward_pass_metrics.py`,
  `req_time_stats.py`, `request_metrics_exporter.py`.
- `observability/trace.py`, `trace_async.py`, `mooncake_trace.py`.
- `scheduler.py:718` `init_metrics_collector`, `:1191` `init_metrics_reporter`,
  `managers/scheduler_components/metrics_reporter.py`.
- `observability/startup_time.py` and `startup_func_log_and_timer.py` — startup breakdown.
- Docs cross-read: `docs/docs/references/production_metrics.mdx`,
  `production_request_trace.mdx`, `advanced_features/observability.mdx`.

---

## Chapter 37 — Benchmarking, Profiling, and Tuning

**Concepts**
- Benchmarking honestly: which knobs to hold fixed, what "throughput" means, and how to
  avoid measuring your client.
- The three benchmark harnesses and when to use each.
- Reading a torch profiler trace: kernel time vs gap time vs communication.
- A tuning playbook: `--mem-fraction-static`, `--chunked-prefill-size`,
  `--max-running-requests`, `--cuda-graph-max-bs`, attention backend, TP/DP layout.

**Code walkthrough**
- `python/sglang/bench_serving.py` — the online serving benchmark.
- `python/sglang/bench_offline_throughput.py`, `bench_one_batch.py`,
  `bench_one_batch_server.py`.
- `python/sglang/profiler.py`, `srt/managers/scheduler_components/profiler_manager.py`,
  the `/start_profile` endpoint (`http_server.py:1139`).
- `python/sglang/kernel_api_logging.py`.
- Skills cross-read: `.claude/skills/llm-torch-profiler-analysis/SKILL.md`,
  `generate-profile/SKILL.md`.
- Docs cross-read: `docs/docs/advanced_features/hyperparameter_tuning.mdx`,
  `developer_guide/benchmark_and_profiling.mdx`.

**Lab**
- Take an untuned deployment to a target SLO with a documented tuning trail.

---

## Chapter 38 — Testing and CI

**Concepts**
- The test pyramid for an inference engine: unit, kernel-correctness, accuracy (evals),
  performance regression, and multi-GPU integration.
- Why accuracy tests are the real safety net, and how thresholds are chosen.
- CI orchestration: stage ordering, fast-fail, gating, partitioning across runners.
- Debugging a CI-only failure.

**Code walkthrough**
- `test/run_suite.py`, `test/README.md`, `test/registered/`.
- `python/sglang/test/` — `CustomTestCase`, server fixtures, `test/kits/`.
- `.github/workflows/` — the pipeline definition.
- Skills cross-read: `.claude/skills/write-sglang-test/SKILL.md`,
  `ci-workflow-guide/SKILL.md`, `sglang-bisect-ci-regression/SKILL.md`.

---

## Chapter 39 — SGLang as an RL Rollout Backend

**Concepts**
- Why RL post-training needs an inference engine, and what it needs that serving doesn't:
  fast in-place weight updates, memory release/resume, deterministic replay.
- Weight update paths: from disk, from distributed group, from tensors, from IPC handles,
  and via the checkpoint engine.
- Memory occupation release/resume, so training and inference can share a GPU.
- Integrations: verl, slime, AReaL, Miles, Tunix.

**Code walkthrough**
- `engine.py:1365`–`:1478` — the weight-update API family.
- `srt/model_executor/model_runner_components/weight_updater.py`,
  `startup_weight_load.py`.
- `srt/checkpoint_engine/`, `srt/weight_sync/tensor_bucket.py`.
- `engine.py:1573` `release_memory_occupation` / `:1579` `resume_memory_occupation`.
- `scheduler.py:4576` `pause_generation` / `:4665` `continue_generation`.
- Docs cross-read: `docs/docs/advanced_features/sglang_for_rl.mdx`,
  `references/post_training_integration.mdx`.

---

# Part IX — Extending SGLang

*Each chapter here is a guided contribution, ending with a PR-shaped deliverable.*

## Chapter 40 — Adding a Model

- The checklist: config class, model class, weight mapping, registry entry, chat template,
  test, docs.
- Debugging a wrong-output model: layer-by-layer comparison against HF.
- Code: `srt/models/`, `srt/configs/`, `srt/debug_utils/comparator/`,
  `docs/docs/supported-models/support_new_models.mdx`.

## Chapter 41 — Adding a Kernel

- The two paths: lightweight JIT kernels (`python/sglang/kernels/jit/`) vs heavyweight AOT
  kernels (`python/sglang/kernels/aot/`, `sgl-kernel`).
- The registry/selector layer (`kernels/registry.py`, `kernels/selector.py`, `kernels/spec.py`).
- Benchmarks and correctness tests as non-optional deliverables.
- Skills cross-read: `.claude/skills/add-jit-kernel/SKILL.md`, `add-sgl-kernel/SKILL.md`.

## Chapter 42 — Adding an Attention Backend

- Implementing `AttentionBackend`; the metadata/CUDA-graph contract that trips everyone up.
- Registering via `attention_registry.py`; testing across forward modes.

## Chapter 43 — Hardware Backends and Platform Plugins

- The platform abstraction (`srt/platforms/`, `srt/hardware_backend/`, `srt/plugins/`).
- What porting to a new accelerator actually requires: device communicators, attention
  backend, memory pool, graph capture.
- Case studies: ROCm/AITER, Ascend NPU, Intel XPU/AMX, CPU, TPU (sglang-jax), Metal.
- Docs cross-read: `docs/docs/hardware-platforms/`.

---

# Appendices

- **A. Server Arguments Reference** — annotated tour of `srt/server_args.py`, grouped by
  subsystem, with "which chapter explains this" pointers.
- **B. Environment Variables** — `srt/environ.py` and the conventions in
  `.claude/skills/env-var-conventions/SKILL.md`.
- **C. Glossary** — TTFT, ITL, MLA, GQA, EAGLE, MTP, EP/EPLB, PD, TBO, SWA, NSA/DSA,
  radix cache, chunked prefill, and the rest.
- **D. Repository Map** — every top-level directory, one paragraph each, with chapter links.
- **E. Reading a Startup Log** — annotated line-by-line startup trace.
- **F. Debugging Playbooks** — condensed from the repo's own skills: CUDA crashes,
  distributed hangs, CI regressions, production incidents.
- **G. Further Reading** — the SGLang papers, LMSYS blog posts, and the related work
  (vLLM/PagedAttention, FlashAttention, Orca, FlashInfer, EAGLE, DeepSeek MLA).

---

## Proposed sequencing for writing

The chapters are not equally expensive. A suggested order that produces a useful artifact
early and de-risks the hard parts:

1. **Ch. 3, 4, 7, 8, 10** — the skeleton. Once the request path is written, everything else
   hangs off it.
2. **Ch. 13, 14** — the memory/radix core. This is the book's differentiator; write it while
   fresh.
3. **Ch. 1, 2, 9, 11, 12** — fill in Parts I–II to make a coherent standalone "Volume 1."
4. **Ch. 16–20** — the model execution layer.
5. **Ch. 21–26** — parallelism, which needs multi-GPU access to verify.
6. **Ch. 27–33** — advanced features, individually self-contained.
7. **Ch. 34–43 + appendices** — frontend, production, extension.

**Volume split option:** if the material is too large for one book, Parts I–IV form
*Volume 1: The Serving Engine* (~self-contained, single-GPU), and Parts V–IX form
*Volume 2: Scale, Features, and Extension*.

## Open questions for the author

1. **Target depth for kernels** — do we read CUDA/Triton source (Ch. 18, 41), or treat
   kernels as black boxes with a described contract? This changes the prerequisite bar
   significantly.
2. **Diffusion / `multimodal_gen`** — currently out of scope. It is a large parallel stack
   (~400 files) with its own pipelines, schedulers, and caching. Own book, appendix, or
   a Part X?
3. **Version pinning** — pin to a release tag so line anchors stay valid, or write against
   `main` and maintain an anchor-verification script in CI?
4. **Executable book** — should labs ship as runnable notebooks/scripts under `book/labs/`
   with a small-model default so readers without an H100 can follow along?

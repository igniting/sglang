# SGLang Internals — Book Outline

**Working title:** *SGLang Internals: Reading a Production LLM Serving Engine*

---

## The approach

Concepts and code are not separated. Each chapter is a single narrative in which an idea
is introduced and immediately grounded in the code that *is* that idea — the explanation
of paged KV memory is the walk through `PagedTokenToKVPoolAllocator`, not a preamble to it.
The reader should never encounter theory they cannot point at.

Each chapter below is given as its **thesis** plus the sequence of **beats** that carry it.
A beat is one idea welded to one piece of code.

**Format conventions**
- Code anchors are `path:line`, verified against commit `7562e74`. Names are the stable
  handle; line numbers get re-verified at publish.
- Long files are read selectively and the outline says which parts. Nothing pretends to
  read 9,000 lines.
- No exercises, no labs. Where a claim is empirical, the book states the measurement and
  its conditions rather than assigning it.

**Prerequisites:** Python, PyTorch, transformer architecture. CUDA/Triton is read at a
glance, never written.

**Scope:** the serving runtime (`python/sglang/srt`), plus the frontend DSL, kernels layer,
and Rust gateway where the request path crosses into them. The diffusion stack
(`python/sglang/multimodal_gen`, ~400 files) is out of scope — see Open Questions.

**Size:** 22 chapters in 7 parts, ~450–550 pages.

---

# Part I — Foundations

## Chapter 1 — Why Serving Engines Exist

*Thesis: the cost structure of autoregressive decoding, not model quality, is what forces
an engine to exist.*

1. **Two phases, two bottlenecks.** Prefill is compute-bound and parallel over the prompt;
   decode is memory-bandwidth-bound and produces one token per sequence per step. The
   roofline argument for why a naive generate-loop leaves 90% of a GPU idle.
2. **The KV cache, and the bill it creates.** Trading O(n²) recompute for O(n) memory;
   the sizing formula and a worked example on a 70B model showing memory — not FLOPs —
   as the binding constraint. This is the number that every later chapter is fighting.
3. **The problem in SGLang's own terms.** `python/sglang/bench_one_batch.py` is the
   smallest thing in the repo that runs a real forward pass; reading its prefill and
   decode timing paths turns the abstract argument into the engine's actual measurements.
4. **What the metrics mean.** TTFT, ITL/TPOT, throughput, goodput, and how optimizing one
   degrades another — the tradeoff space every subsequent design decision sits in.
5. **The idea inventory.** `README.md:68` lists SGLang's feature set in one dense
   paragraph. Decoding that list feature-by-feature into "what problem it solves and which
   chapter covers it" gives the reader a map before the descent.

## Chapter 2 — The Shape of SGLang

*Thesis: the engine's process topology is its architecture; understanding the boundaries
explains most design choices that follow.*

1. **Several projects in one tree.** The runtime (`srt`), the frontend DSL (`lang`),
   kernels (`kernels`, `sgl-kernel`), the Rust gateway (`sgl-model-gateway`) — and what
   each is for. The `srt` layout presented as a dependency graph, not an alphabetical list.
2. **Why multi-process, not multi-threaded.** The GIL, fault isolation, and the
   tokenizer/scheduler/detokenizer split. `python/sglang/srt/entrypoints/engine.py:1052`
   `_launch_subprocesses` is where the topology is literally constructed —
   `:848` spawns schedulers per TP rank, `:966` the detokenizers.
3. **Three front doors.** The HTTP server (`entrypoints/http_server.py:270` `lifespan`,
   `:874` `generate_request`), the embeddable `Engine` (`engine.py:199`, `:352` `generate`)
   used for offline batch and RL rollouts, and gRPC. `python/sglang/launch_server.py`
   dispatches between them and is short enough to read whole.
4. **The fourth front door: the DSL.** SGLang is named for its *Structured Generation
   Language*. `lang/api.py` gives the primitives (`gen`, `select`, `fork`),
   `lang/ir.py` the program IR, and `lang/interpreter.py:274` `StreamExecutor` the
   execution engine. The payoff — `fork` becoming a radix-tree branch rather than *n*
   independent generations — is the first hint of the frontend/runtime co-design that
   Chapter 9 completes.
5. **Reading the flag surface.** `srt/server_args.py` (~9,900 lines) is not read linearly;
   it is read as a table of contents for the feature space, with each argument group
   pointing at the subsystem that consumes it.
6. **The repo's own rules as design documents.** `.claude/rules/no-dataclasses.md`,
   `no-getattr-defensive.md`, `schedule-batch-out-of-place-mutation.md`, and
   `forward-batch-init-new-purity.md` encode invariants that exist nowhere else. Each one
   explains a hazard the architecture has already hit.

---

# Part II — The Request Path

*One request, followed from socket to streamed token. Every later part is a labeled detour
off this path.*

## Chapter 3 — From HTTP to Token IDs

*Thesis: the front of the engine is an async/sync boundary, and most of its complexity is
in making message passing look like `await`.*

1. **Tokenization gets its own process and its own event loop.**
   `managers/tokenizer_manager.py:374` `TokenizerManager` — `:755` `generate_request` is
   the async entry, `:985` `_tokenize_one_request` and `:1359` `_create_tokenized_object`
   the conversion.
2. **Validation as a stability boundary.** `tokenizer_manager.py:1185`
   `_validate_one_request` — length limits, vocab range, multimodal caps, logprob
   constraints. Each check exists because something downstream would otherwise crash a
   process shared by every other request.
3. **Turning messages into futures.** `:1722` `_wait_one_response` is the per-request async
   generator; `:2200` `handle_loop` and `:2215` `_handle_batch_output` are the return path
   that resolves it. This pair is the whole trick.
4. **The wire contract.** `managers/io_struct.py:160` `GenerateReqInput`,
   `:941` `TokenizedGenerateReqInput`, `:1404` `BatchTokenIDOutput`, `:1504` `BatchStrOutput`
   — the messages define the process boundaries more precisely than any diagram.
5. **ZeroMQ patterns and their failure modes.** Socket setup at `scheduler.py:733`
   `init_ipc_channels`; receipt at `scheduler.py:1872` `process_input_requests` and
   `:1523` `init_request_dispatcher` (the type → handler table). Broadcast semantics under
   TP: rank 0 receives, all ranks must agree. Serialization choices, and the CUDA-IPC path
   at `scheduler.py:1906` `_materialize_cuda_vmm_inputs` that keeps image tensors off the
   socket entirely.

## Chapter 4 — The Scheduler Loop

*Thesis: one synchronous loop owns the GPU and answers one question per iteration —
what runs next? Everything else in the engine is input to that question.*

1. **A request is a state machine with a lot of state.**
   `managers/schedule_batch.py:811` `Req`, `:814` `__init__` read field-group by
   field-group: tokens, prefix match, KV indices, sampling params, grammar state, logprob
   accumulators, multimodal payloads. `:1298` `init_next_round_input` is the per-round
   prefix match; `:223`–`:283` the finish-reason hierarchy.
2. **Three batch types for three jobs.** `ScheduleBatch` (CPU scheduling view),
   `ModelWorkerBatch` (transport), `ForwardBatch` (GPU execution view) — with
   `prepare_for_extend`, `prepare_for_decode`, `retract_decode`, `filter_batch`,
   `merge_batch` as the operations that move between them. Why the repo forbids in-place
   batch mutation, per `.claude/rules/schedule-batch-out-of-place-mutation.md`.
3. **The loop, honest version first.** `managers/scheduler.py:1714` `event_loop_normal` is
   short and does exactly what it says. Read it before anything else in the file.
4. **The loop, fast version, as a delta.** `:1749` `event_loop_overlap` — the zero-overhead
   batch scheduler. The future-token trick that lets step *N+1* be prepared while step *N*
   is still on the GPU, in `managers/overlap_utils.py` and `:1438` `init_overlap`; and
   `:1823` `is_disable_overlap_for_batch` for when it cannot be done.
5. **A 5,000-line class, and why it is shaped that way.** `scheduler.py:378` composes 22
   mixins and `:388` `__init__` is a sequence of named `init_*` calls.
   `.claude/skills/large-class-style/SKILL.md` documents this as deliberate; the extracted
   collaborators in `managers/scheduler_components/` (`batch_result_processor.py`,
   `output_streamer.py`, `invariant_checker.py`) show where the seams are.
6. **Liveness.** `:4036` `on_idle`, `:4078` `is_fully_idle`, watchdogs, and why
   `/health_generate` (`http_server.py:646`) runs a real forward pass instead of returning 200.

## Chapter 5 — Deciding What Runs Next

*Thesis: continuous batching is an admission-control problem under a hard memory budget,
and the memory system is what makes the decision interesting.*

1. **Admission, not scheduling.** `scheduler.py:3012` `get_next_batch_to_run` is the
   central decision function; `:3154` `get_new_batch_prefill` and `:3478`
   `update_running_batch` are its two halves.
2. **The token budget.** `managers/schedule_policy.py:504` `PrefillAdder` — `:664`
   `rem_total_tokens`, `:857` `_update_prefill_budget`, `add_one_req`. This is where
   `max_total_tokens`, `max_prefill_tokens`, and `max_running_requests` stop being flags
   and become arithmetic.
3. **Predicting the future.** The new-token-ratio heuristic
   (`scheduler_components/new_token_ratio_tracker.py`) estimates decode demand that has
   not happened yet; over-optimism here is what makes retraction necessary.
4. **Cache-aware ordering.** `schedule_policy.py:216` `SchedulePolicy`, `:237`
   `calc_priority`, `:314` `_compute_prefix_matches`, `:374` `_sort_by_longest_prefix`,
   `:387` `_sort_by_dfs_weight`. The scheduler consults the radix tree — this is the loop
   between Chapters 5 and 9 closing, and the reason cache-aware routing works at all.
5. **Chunked prefill.** `scheduler.py:1153` `init_chunked_prefill` — splitting long prompts
   so a 100k-token request cannot stall every decode in flight, and the TTFT/ITL trade
   that buys.
6. **Fairness and its absence.** `:2715` `_add_request_to_queue`, `:2739`
   `_set_or_validate_priority`, `:2813` `_abort_on_waiting_timeout` — starvation,
   priorities, and queue timeouts.

## Chapter 6 — Executing a Batch

*Thesis: the handoff from Python scheduling objects to GPU tensors is where the mode of the
batch starts determining everything downstream.*

1. **Mode determines the world.** `model_executor/forward_batch_info.py:98` `ForwardMode` —
   `EXTEND`, `DECODE`, `MIXED`, `IDLE`, `TARGET_VERIFY`, `DRAFT_EXTEND`, `SPLIT_PREFILL`.
   Attention backend, kernel choice, graph eligibility, and memory accounting all branch here.
2. **The execution view.** `:412` `ForwardBatch` and `:739` `init_new`, plus
   `.claude/rules/forward-batch-init-new-purity.md` on why construction must be pure.
   `model_executor/forward_context.py` for the ambient per-forward context and the problem
   it solves.
3. **The worker boundary.** `managers/tp_worker.py:74` `BaseTpWorker`, `:299` `TpModelWorker`
   — thin, and deliberately so.
4. **Initialization as an ordered script.** `model_executor/model_runner.py:284`
   `ModelRunner`, `:287` `__init__` — weights (`:1057` `load_model`), memory pool
   (`:807` `alloc_memory_pool`), attention backend (`:927`), CUDA graphs (`:992`). The
   order is a dependency chain, and reading it explains most startup failures.
5. **The forward call.** `:1505` `forward` and `:1649` `_forward_raw` — mode dispatch,
   graph replay vs eager, and the contract the model must satisfy.

## Chapter 7 — Sampling and the Return Path

*Thesis: turning hidden states into user-visible text is three separate hard problems that
happen to sit next to each other.*

1. **Not every position matters, except when it does.**
   `layers/logits_processor.py:282` `LogitsProcessor`, `:332` `forward`,
   `:427` `_get_pruned_states`, `:693` `_compute_lm_head`. Decode needs one position;
   prefill may need all of them for logprobs, EAGLE, or hidden-state capture — and
   `:96` `LogitsProcessorOutput` / `:149` `LogitsMetadata` encode which.
2. **Sampling as batched tensor work.** `layers/sampler.py:70` `Sampler`, `:97` `forward`,
   `:563` `top_k_top_p_min_p_sampling_from_probs_torch`. Temperature, top-k/p/min-p, and
   penalties (`sampling/penaltylib/`) applied to a batch whose requests all asked for
   different things — `sampling/sampling_batch_info.py` is how that is made possible.
   `sampling/custom_logit_processor.py` is the user extension point.
3. **Reproducibility, and why it is hard.** Non-deterministic reductions and batch-variant
   kernels mean identical prompts can diverge across batch sizes.
   `srt/batch_invariant_ops/`, `model_runner.py:764` `maybe_enable_batch_invariant_mode`,
   `scheduler.py:1506` `init_deterministic_inference_config`, and `sampler.py:684`
   `multinomial_with_seed`. `.claude/skills/kl-consistency-test/SKILL.md` states the two
   independent conditions a zero-KL result requires.
4. **Incremental detokenization is genuinely hard.** BPE merges and multi-byte UTF-8 mean
   you cannot simply decode the newest token.
   `managers/detokenizer_manager.py:91`, `:166` `event_loop`,
   `:290` `_decode_batch_token_id_output`; `schedule_batch.py:1425`
   `init_incremental_detokenize`.
5. **Stopping, and un-emitting.** Stop strings require lookback and retraction of already-
   produced text: `detokenizer_manager.py:176` `trim_matched_stop`,
   `schedule_batch.py:1445` `_stop_match_tail_len`.
6. **Streaming out.** `tokenizer_manager.py:1641` `_coalesce_streaming_chunks` and
   `entrypoints/openai/sse_utils.py` — the chunk-size/latency trade at the very last hop.

---

# Part III — Memory and Caching

## Chapter 8 — KV Cache Pools and Allocators

*Thesis: the engine's central data structure is a two-level indirection, and its shape
explains both paged attention and everything Chapter 9 builds on top.*

1. **Two levels, not one.** `mem_cache/memory_pool.py:256` `ReqToTokenPool` maps request →
   token slots; the `KVCache` hierarchy maps token slot → storage. Splitting them is what
   lets a prefix be shared by requests that disagree about everything else.
2. **Pages and the allocator's job.** `mem_cache/allocator/paged.py:105`
   `PagedTokenToKVPoolAllocator` — `:149` `alloc`, `:172` `alloc_extend`,
   `:222` `alloc_decode`, `:261` `free`. Page size as the knob that trades internal
   fragmentation against metadata cost.
3. **One abstraction, many pools.** `memory_pool.py:1624` `KVCache` and its implementations:
   `:1755` `MHATokenToKVPool` (the common case), `:3932` `MLATokenToKVPool` (DeepSeek's
   compressed cache — an order of magnitude smaller, which is why Chapter 15's DP attention
   exists), `:3577` `HybridLinearKVPool` and `:335` `MambaPool` (state, not keys and values),
   `:3135` `PageMajorMHATokenToKVPool` (layout as a transfer optimization).
4. **From a percentage to a number.** `mem_cache/kv_cache_configurator.py` and
   `allocation_sizing.py` turn `--mem-fraction-static` into a concrete pool size; this is
   the calculation behind every "out of memory at 90% utilization" report.
5. **Fewer bits per entry.** `mem_cache/kv_cache_dtype.py` and the FP8/FP4 pool variants,
   with the accuracy question deferred to Chapter 14.

## Chapter 9 — RadixAttention

*Thesis: real workloads share long prefixes, and a radix tree over token sequences turns
that redundancy into the engine's largest single win. This is SGLang's signature idea.*

1. **The observation.** Chat history, few-shot prompts, agent loops, and system prompts all
   mean the *n*-th request usually shares a long prefix with an earlier one. Recomputing it
   is pure waste.
2. **The key.** `mem_cache/radix_cache.py:59` `RadixKey` — token ids plus an extra key that
   namespaces LoRA adapters and sessions apart (`:181` `match`, `:217` `child_key`,
   `:150` `page_aligned`). Page alignment constrains every tree operation and is the source
   of most of the code's subtlety.
3. **The tree.** `:238` `TreeNode` (children, `lock_ref`, `last_access_time`, host tier)
   and `:303` `RadixCache`. The three operations, read in order:
   `:376` `match_prefix` / `:678` `_match_prefix_helper`, `:704` `_split_node`,
   `:436` `insert` / `:737` `_insert_helper`.
4. **Reference counting keeps live requests alive.** `:622` `inc_lock_ref`,
   `:637` `dec_lock_ref`, `:458` `cache_finished_req`, `:515` `cache_unfinished_req`.
   A node in use by a running request must survive eviction — this is the invariant that
   makes the whole thing safe.
5. **Eviction over a tree.** `:592` `evict` — LRU, but leaf-first, because interior nodes
   are prefixes of their children.
6. **Where the model touches it.** `layers/radix_attention.py:91` `RadixAttention` is the
   layer every model instantiates; it is the reason model code needs no cache logic of
   its own.
7. **The contrast case.** `mem_cache/chunk_cache.py:35` `ChunkCache` implements the same
   interface with no reuse at all, which makes the interface (`mem_cache/base_prefix_cache.py:230`
   `BasePrefixCache`, with `MatchPrefixParams`/`MatchResult` at `:49`–`:166`) legible.
8. **Variants.** `swa_radix_cache.py` (sliding window), `mamba_radix_cache.py` (state, not
   tokens), `radix_cache_cpp.py` + `cpp_radix_tree/` (the same algorithm in C++, and why
   Python became the bottleneck).

*A worked example — three chat requests sharing a system prompt — runs through beats 3–5
as the tree evolves.*

## Chapter 10 — Caching Beyond HBM

*Thesis: extending the cache hierarchy to host memory and disk is a bandwidth arbitrage,
and it only pays above a computable prefix length.*

1. **The arbitrage.** At what prefix length does loading KV from host DRAM beat recomputing
   it? The answer sets every policy in this chapter.
2. **A radix tree with tiers.** `mem_cache/hiradix_cache.py:76` `HiRadixCache` subclasses
   `RadixCache` from Chapter 9; `:840` `write_backup` is write-through/write-back, and the
   tier bookkeeping in `TreeNode` (`radix_cache.py:273` `backuped`, `:276` `protect_host`)
   is what keeps tiers coherent.
3. **Moving the bytes.** `mem_cache/memory_pool_host.py` and
   `managers/cache_controller.py` — the transfer engine, its queues, and its overlap with
   compute.
4. **Pluggable storage.** `mem_cache/hicache_storage.py`,
   `mem_cache/storage/backend_factory.py`, and the backends (Mooncake, 3FS, NIXL, LMCache,
   file, mmap, shm) — with `:369` `attach_storage_backend` / `:487` `detach_storage_backend`
   showing runtime attach as a first-class operation.
5. **Where it is heading.** `mem_cache/unified_cache/` — the unified tree core and
   component registry that generalizes the tiering.

*Cross-read: `docs/docs/advanced_features/hicache_design.mdx`.*

---

# Part IV — The Model Layer

## Chapter 11 — Loading and Updating Weights

*Thesis: weight loading is a distributed sharding problem disguised as file I/O, and the
same machinery serves both startup and reinforcement learning.*

1. **From a path to a class.** The model registry, HF config → SGLang config translation
   (`srt/configs/`, `model_config.py`), and where architectures diverge from their HF
   definitions.
2. **Each rank loads only its slice.** `model_loader/loader.py`, `auto_loader.py`, and the
   `weight_loader` protocol attached to every parameter in `model_loader/weight_utils.py`.
   `model_runner.py:1057` `load_model` orchestrates it.
3. **Two generations of the protocol, side by side.** `models/llama.py:663` `load_weights`
   and `:743` `_load_weights_v2` — a rare chance to see an interface migration mid-flight.
4. **Formats and sources.** Safetensors, GGUF, sharded checkpoints; `srt/connector/` for S3,
   Azure, Redis, and remote-instance loading; `load_format` as a startup-latency knob.
5. **Updating weights without restarting.** `engine.py:1365`–`:1478` — from disk, from a
   distributed group, from tensors, from IPC handles.
   `model_runner_components/weight_updater.py` and `srt/checkpoint_engine/` implement it;
   `engine.py:1573` `release_memory_occupation` / `:1579` `resume_memory_occupation` and
   `scheduler.py:4576` `pause_generation` let training and inference share a GPU. This is
   what makes SGLang an RL rollout backend rather than only a server.

## Chapter 12 — Anatomy of a Model

*Thesis: models are rewritten rather than imported because every layer must cooperate with
parallelism, quantization, and the KV cache — and llama.py shows exactly how.*

1. **The contract.** `forward(input_ids, positions, forward_batch) -> logits`, plus
   `load_weights`. Everything else is optional capability.
2. **A full read of `models/llama.py`.** `:70` `LlamaMLP` (merged gate/up column-parallel,
   row-parallel down), `:138` `LlamaAttention` (fused QKV, rotary, and the `RadixAttention`
   instantiation that connects Chapter 9), `:283` `LlamaDecoderLayer` (residual ordering and
   fused add-RMSNorm), `:372` `LlamaModel`, `:496` `LlamaForCausalLM` with `:563` `forward`.
3. **The parallel layer vocabulary.** `layers/linear.py:293` `ColumnParallelLinear`,
   `:1392` `RowParallelLinear`, `:921` `QKVParallelLinear`, `:492` `MergedColumnParallelLinear`;
   `layers/vocab_parallel_embedding.py:188` and `:587` `ParallelLMHead`. These are where
   tensor parallelism physically lives — Chapter 15 only explains what they already do.
4. **Optional capabilities as hooks.** `:640` `start_layer` / `:644` `end_layer` for
   pipeline parallelism, `:599` `forward_split_prefill`, `:891` `set_eagle3_layers_to_capture`
   for speculative decoding, `:854` `get_embed_and_head` for weight sync.
5. **Three short contrast studies.** `models/deepseek_v2.py` (MLA + MoE),
   a Qwen-VL variant (a vision tower on a decoder), `models/falcon_h1.py` (state instead of
   attention) — enough to show what varies and what never does.

## Chapter 13 — Attention Backends

*Thesis: attention is pluggable because hardware, sequence shape, and kernel maturity all
vary independently — and the plug is a two-phase metadata/kernel contract.*

1. **The contract.** `layers/attention/base_attn_backend.py:33` `AttentionBackend` —
   `:62` `init_forward_metadata` runs once per forward, `:261` `forward_decode` and
   `:274` `forward_extend` run once per layer. Nearly every backend bug is a violation of
   that split. `:160` `init_cuda_graph_state` is where Chapter 14's constraints intrude.
2. **The reference implementation.** `layers/attention/flashinfer_backend.py` read in
   depth — wrappers, page tables, and graph-safe buffers. Then
   `triton_backend.py` as the portable fallback, short enough to read whole.
3. **Registration and selection.** `attention_registry.py:34` `register_attention_backend`
   and the factory table; `model_runner.py:927` `init_attention_backends`.
4. **MLA needs its own everything.** `flashinfer_mla_backend.py`, `flashmla_backend.py`,
   and their dependence on the Chapter 8 pool — the clearest case of memory layout dictating
   kernel design.
5. **Sparse attention.** `nsa/nsa_indexer.py` + `nsa_backend.py` — selecting which tokens
   to attend to, and the indexer as a model component in its own right.
6. **When it isn't attention at all.** `mamba/mamba.py` and `linear/gdn_backend.py` — SSMs
   and linear attention carry state rather than a KV cache, which is why Chapter 8 needed
   `MambaPool`. `hybrid_linear_attn_backend.py` for models that do both.

## Chapter 14 — Making the Forward Pass Cheap

*Thesis: quantization attacks bytes moved, CUDA graphs attack launch overhead, and
compilation attacks kernel count — three independent taxes on the same forward pass.*

1. **Three independent targets.** Weights, activations, and KV cache can each be quantized
   separately, and the choice of scheme is per-target.
2. **The quantization architecture.** `layers/quantization/base_config.py` —
   `QuantizationConfig` → `QuantizeMethodBase` → per-layer `apply()`. Then
   `fp8.py` + `fp8_utils.py` read end to end as the most-used path, with
   `modelopt_quant.py`, `mxfp4.py`, `compressed_tensors/`, `awq/`, `gptq/` surveyed for
   what differs. `layers/parameter.py` is what makes sharding and scale tensors coexist.
3. **Quantized KV.** `layers/quantization/kv_cache.py` closing the loop with the Chapter 8
   pools, and where accuracy actually degrades.
4. **Decode is launch-bound.** Thousands of microsecond kernels mean CPU launch cost
   dominates. CUDA graphs capture once and replay — at the price of static shapes and
   static pointers.
5. **Capture and replay.** `model_executor/runner/base_cuda_graph_runner.py`,
   `decode_cuda_graph_runner.py`, `prefill_cuda_graph_runner.py`;
   `model_runner.py:992` `init_cuda_graphs`; batch-size bucketing, padding, and the memory
   cost of the capture matrix (`cuda_graph_config.py`, `graph_memory_usage.py`).
6. **When part of the model cannot be captured.**
   `runner_backend/breakable_cuda_graph_backend.py` and
   `tc_piecewise_cuda_graph_backend.py` — keeping most of the graph when one region must
   run eagerly.
7. **Compilation.** `srt/compilation/` — the Inductor backend, custom passes
   (`fix_functionalization.py`, `pass_manager.py`), and how it composes with graphs.

---

# Part V — Scaling Out

## Chapter 15 — Tensor, Pipeline, and Data Parallelism

*Thesis: the three classical parallelism axes differ in what they split and therefore in
which interconnect they stress; SGLang adds a fourth because MLA broke the assumptions.*

1. **Groups and collectives.** `distributed/parallel_state.py:237` `GroupCoordinator`,
   `:2193` `init_distributed_environment`, `:2285` `initialize_model_parallel`;
   `device_communicators/` for custom all-reduce, PyNCCL, and symmetric memory — with the
   cost model for each collective on NVLink vs InfiniBand.
2. **TP is already written.** The collectives live inside `ColumnParallelLinear` and
   `RowParallelLinear` from Chapter 12; this beat only explains the two communication
   points per block and why TP does not cross node boundaries cheaply.
   `layers/communicator.py` is the per-layer strategy object.
3. **PP splits layers.** `managers/scheduler_pp_mixin.py` for microbatch scheduling in the
   loop, `models/llama.py:640` for the layer-range boundary, and bubbles as the cost.
4. **MLA breaks TP.** A compressed KV cache replicated across ranks wastes the very thing
   that made it small — the setup for DP attention.
5. **Data-parallel attention.** `layers/dp_attention.py:338` `initialize_dp_attention`,
   `:412` `get_dp_local_info`, `:76` `DpPaddingMode`, `:450`/`:494` the two gather
   strategies. Each rank owns whole sequences for attention, then the batch is gathered for
   the TP-sharded MLP.
6. **The price: everyone must agree.** `forward_batch_info.py:1305` `prepare_mlp_sync_batch`
   and `:1620` `post_forward_mlp_sync_batch` force every rank to a common batch shape,
   which is why idle batches exist at all. `logits_processor.py:249`
   `compute_dp_attention_metadata` carries it to the end.
7. **The other DP.** `managers/data_parallel_controller.py` — whole-replica routing, an
   unrelated mechanism with a colliding name.

## Chapter 16 — Mixture-of-Experts and Expert Parallelism

*Thesis: MoE inference is all-to-all-bound rather than GEMM-bound, which makes routing,
placement, and communication overlap the whole game.*

1. **Routing.** `layers/moe/topk.py:392` `TopK` and the `TopKOutput` variants at `:274` —
   top-k selection, and why its output format matters to the kernel that follows.
2. **Computing the experts.** `layers/moe/fused_moe_triton/` as the portable path,
   `layers/moe/moe_runner/` as the abstraction over backends.
3. **Splitting experts across devices.** `layers/moe/ep_moe/` and
   `layers/moe/token_dispatcher/` — DeepEP dispatch/combine, and why all-to-all latency
   rather than FLOPs sets the step time.
4. **Hot experts.** `srt/eplb/expert_distribution.py` measures imbalance;
   `expert_location.py`, `eplb_manager.py`, `eplb_algorithms/`, and `lplb_solver.py`
   rebalance and replicate. Recording expert distribution is exposed as a server endpoint
   precisely because the imbalance is workload-dependent.
5. **Hiding the all-to-all.** `srt/batch_overlap/two_batch_overlap.py` and
   `single_batch_overlap.py` split a batch so communication overlaps computation, with
   `operations.py`/`operations_strategy.py` as the scheduling abstraction and
   `layers/attention/tbo_backend.py` on the attention side. This is the Chapter 4 overlap
   idea applied one level down.

## Chapter 17 — Disaggregation and Routing

*Thesis: prefill and decode want different hardware and different SLOs, so at scale the
right move is to stop running them on the same machine.*

1. **Why colocation compromises both.** Prefill wants large batches and compute; decode
   wants low latency and bandwidth. Chunked prefill (Chapter 5) mitigates the conflict;
   disaggregation removes it.
2. **The two sides.** `disaggregation/prefill.py:119` `PrefillBootstrapQueue`,
   `:485` `SchedulerDisaggregationPrefillMixin`, `:569` `event_loop_normal_disagg_prefill`;
   and `disaggregation/decode.py` for the mirror image. Note these are *variant event
   loops* — the Chapter 4 loop, respecialized.
3. **The handshake.** How a decode instance learns where its KV lives:
   `disaggregation/base/conn.py`, `common/conn.py`, and `managers/disagg_service.py`.
4. **Moving KV between machines.** `disaggregation/mooncake/`, `nixl/`, `mori/` — RDMA
   realities; `fake/` as the test double that makes the control flow readable.
5. **Routing in front of it all.** `sgl-model-gateway/src/policies/` — cache-aware load
   balancing beats round-robin precisely because Chapter 9 exists, and the router keeps an
   approximate radix tree to do it. `src/routers/`, `service_discovery.rs`, and why it is
   written in Rust.

---

# Part VI — Beyond Plain Decoding

## Chapter 18 — Speculative Decoding

*Thesis: verifying k tokens costs nearly what generating one costs, so the only question is
how good a draft you can produce cheaply.*

1. **The bandwidth argument.** Why the acceptance rule preserves the target model's output
   distribution, and why a memory-bound decode step has spare compute to give away.
2. **Capabilities before implementations.** `speculative/spec_info.py:30`
   `SpeculativeAlgorithm` — the predicate set (`is_eagle`, `has_draft_kv`,
   `supports_ragged_verify`, `supports_grammar_overlap`) that gates behavior across the
   entire engine. Read this before any worker.
   `.claude/skills/speculative-naming/SKILL.md` first for the vocabulary.
3. **EAGLE end to end.** `speculative/eagle_worker_v2.py:1008` `EAGLEWorkerV2` —
   `:1105` `forward_batch_generation` as the step, `:1497` `verify` as the accept rule;
   `:128` `EagleDraftWorker` with `:494` `draft`, `:557` `draft_forward`,
   `:726` `draft_extend`. `eagle_info.py` for the tree metadata.
4. **Trees, not chains.** Draft topology, and how `--speculative-eagle-topk` and
   `--speculative-num-steps` trade acceptance against wasted compute.
5. **What it costs the rest of the engine.** Extra `ForwardMode`s (Chapter 6), separate
   CUDA graphs (`eagle_draft_cuda_graph_runner.py`), changed memory accounting in the
   Chapter 5 budget, and grammar coordination with Chapter 19.
6. **The other algorithms, briefly.** `ngram_worker.py` + `cpp_ngram/` (no draft model at
   all), `frozen_kv_mtp_worker_v2.py`, `dflash_worker_v2.py`,
   `standalone_worker_v2.py` — and `adaptive_runtime_state.py`, which turns speculation off
   when acceptance drops.

## Chapter 19 — Shaping and Reading the Output

*Thesis: constraining generation and parsing generation are the same problem seen from two
sides, and both are made hard by streaming.*

1. **Guarantees by masking.** `constrained/base_grammar_backend.py:52` `BaseGrammarObject`,
   `:167` `BaseGrammarBackend`, `:311` `create_grammar_backend`; the mask lands in
   `sampler.py:97` via the vocab-mask buffers at `base_grammar_backend.py:262`.
2. **Compiling a grammar to token masks.** The FSM, the compressed FSM, and the real
   difficulty — a grammar is defined over characters, a mask over tokens.
   `xgrammar_backend.py` as the default, `outlines_backend.py` and `llguidance_backend.py`
   for contrast.
3. **Skipping the forward pass entirely.** `constrained/outlines_jump_forward.py` — when
   the grammar admits exactly one continuation, emit it without inference.
4. **Coordination costs.** `constrained/grammar_manager.py`, `scheduler.py:1962`
   `init_grammar_manager`, `:1861` `_advance_pending_grammar` — grammar state advances
   asynchronously, which is where it collides with Chapter 18.
5. **Parsing, the mirror image.** `function_call/function_call_parser.py` and
   `base_format_detector.py`; two detectors read in full and the other 37 skimmed for the
   pattern. Streaming forces a decision about whether text is a tool call before the text
   is complete.
6. **Structural tags** as the convergence of the two halves — constraining generation to a
   tool schema instead of parsing afterwards (`function_call/kimik3_structural_tag.py`).
7. **Reasoning blocks.** `srt/parser/reasoning_parser.py`, `harmony_parser.py`, and why
   `<think>` content must be excluded from grammar constraints.

## Chapter 20 — Per-Request Variation: LoRA and Multimodal

*Thesis: both features break the assumption that every request in a batch needs the same
weights and the same kind of input — and both are solved by extending the batch, not
splitting it.*

1. **Batching across adapters.** `lora/lora_manager.py:59` `LoRAManager` — `:428`
   `prepare_lora_batch` assembles per-request adapter indices so one kernel serves a batch
   using different adapters; `lora/backend/` for the grouped-GEMM kernels that make it work.
2. **Adapters as a memory pool.** `lora/mem_pool.py`, `lora/eviction_policy.py`, and
   `:221` `load_lora_adapter` for runtime load/unload. Admission gains a new constraint at
   `scheduler.py:3450` `_can_schedule_lora_req`.
3. **Adapters must not share a prefix cache.** The `RadixKey` extra key from Chapter 9
   (`radix_cache.py:64`) is what keeps two adapters' identical token sequences apart — a
   correctness bug waiting for anyone who skips it.
4. **Non-text input, spliced into text.** `multimodal/processors/` (53 of them) and
   `managers/mm_utils.py`; `schedule_batch.py:318` `MultimodalDataItem`, `:590`
   `MultimodalInputs`, `:569` `build_padded_input_ids` — encoder output replacing
   placeholder tokens.
5. **Position arithmetic moves into the scheduler.** `scheduler.py:2308`
   `_maybe_compute_mrope_positions` and `forward_batch_info.py:1163`
   `_compute_mrope_positions` — VLM 3D positions cannot be computed by the model alone.
6. **Not sending pixels through a socket.** `scheduler.py:2209`
   `_process_and_broadcast_mm_inputs` and the CUDA-IPC path from Chapter 3, plus
   `mem_cache/multimodal_cache.py` for reusing encoder output.

---

# Part VII — Operating and Extending

## Chapter 21 — Observability and Tuning

*Thesis: the instrumentation reveals the design — every metric the engine emits exists
because someone needed it to answer a question this book has already raised.*

1. **What the engine measures, and why each one.**
   `observability/metrics_collector.py`, `forward_pass_metrics.py`, `req_time_stats.py` —
   queue depth, cache hit rate, memory utilization, spec-decode acceptance, per-phase
   timing. Each maps back to a specific chapter's tradeoff.
2. **Tracing across processes.** `observability/trace.py`, `trace_async.py` — following one
   `rid` through the Chapter 2 topology; and `startup_time.py` for the startup breakdown
   that explains slow boots.
3. **Measuring honestly.** `bench_serving.py`, `bench_offline_throughput.py`,
   `bench_one_batch_server.py` — which to use when, what to hold fixed, and how to avoid
   measuring your own client.
4. **Reading a profile.** `python/sglang/profiler.py` and the `/start_profile` endpoint
   (`http_server.py:1139`); kernel time vs gap time vs communication, per
   `.claude/skills/llm-torch-profiler-analysis/SKILL.md`.
5. **A tuning order of operations.** `--mem-fraction-static`, `--chunked-prefill-size`,
   `--max-running-requests`, `--cuda-graph-max-bs`, attention backend, parallelism layout —
   sequenced by which chapter's constraint each one relieves.

## Chapter 22 — Extending SGLang

*Thesis: the extension points are the architecture's seams, and walking them is the final
check that the reader has understood where the boundaries are.*

1. **Adding a model.** Config class, model class, weight mapping, registry entry, chat
   template — and layer-by-layer comparison against HF via `srt/debug_utils/comparator/`
   when the output is wrong.
2. **Adding a kernel.** The two paths: JIT (`python/sglang/kernels/jit/`) and AOT
   (`kernels/aot/`, `sgl-kernel`), with `kernels/registry.py` and `selector.py` as the
   dispatch layer. Correctness tests and benchmarks are part of the deliverable, not
   follow-up work.
3. **Adding an attention backend.** Implementing the Chapter 13 contract, and the
   CUDA-graph obligations that catch every first attempt.
4. **Porting to new hardware.** `srt/platforms/`, `hardware_backend/`, `srt/plugins/` —
   what a new accelerator actually requires (communicators, attention backend, memory pool,
   graph capture), with ROCm/AITER, Ascend, and Intel XPU as case studies of how far the
   abstraction stretches.
5. **How the project keeps this safe.** `test/run_suite.py`, `python/sglang/test/kits/`,
   and why accuracy evals rather than unit tests are the real net. `.github/workflows/` for
   gating and partitioning.

---

# Appendices

- **A. Server Arguments** — `srt/server_args.py` grouped by subsystem, each group pointing
  at the chapter that explains it.
- **B. Environment Variables** — `srt/environ.py` and its conventions.
- **C. Glossary** — TTFT, ITL, MLA, GQA, EAGLE, MTP, EP/EPLB, PD, TBO, SWA, NSA/DSA,
  chunked prefill, radix cache.
- **D. Repository Map** — every top-level directory, one paragraph, with chapter links.
- **E. Annotated Startup Log** — one real startup trace, line by line, as a synthesis of
  Parts I–IV.
- **F. Further Reading** — SGLang papers, LMSYS blogs, and related work (PagedAttention,
  FlashAttention, Orca, FlashInfer, EAGLE, DeepSeek MLA).

---

## What changed from the first draft

- **43 chapters → 22.** Cut by merging rather than dropping: IPC folded into Chapter 3;
  `Req`/`ScheduleBatch` into the scheduler loop; logits/sampling/detokenization/determinism
  into one return-path chapter; distributed foundations + TP/PP + DP attention into
  Chapter 15; quantization + CUDA graphs + compilation into Chapter 14; grammars + tool
  parsers into Chapter 19; LoRA + multimodal into Chapter 20; the DSL into Chapter 2;
  RL weight sync into Chapter 11; testing/CI into Chapter 22.
- **Concepts fused with code.** No chapter has a theory section followed by a code section.
  Each beat is one idea welded to the code that implements it.
- **Labs removed** throughout.
- **Overlap scheduling** is no longer its own chapter — the request-path version lives in
  Chapter 4, and the MoE communication-overlap version in Chapter 16, where each is
  motivated.

## Writing order

1. **Ch. 2, 4, 6, 8, 9** — the topology, the loop, the executor, and the memory core.
   These carry the book; if they work, the rest is infill.
2. **Ch. 1, 3, 5, 7** — completing the request path into a coherent Part I–II.
3. **Ch. 11–14** — the model layer.
4. **Ch. 15–17** — scaling, which needs multi-GPU access to verify.
5. **Ch. 18–22 + appendices.**

## Open questions

1. **Kernel depth (Ch. 13, 14, 22).** Read CUDA/Triton source, or treat kernels as
   contracts with described semantics? This sets the prerequisite bar.
2. **Diffusion.** `python/sglang/multimodal_gen` is a large parallel stack with its own
   pipelines and caching. Out of scope entirely, or a single survey chapter?
3. **Version pinning.** Pin to a release tag so anchors stay valid, or track `main` with an
   anchor-verification script?

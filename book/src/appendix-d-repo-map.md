# Appendix D — Repository Map

Every top-level directory, and where the book explains it.

---

## Top level

**`python/`** — the Python package. Nearly everything this book covers.

**`sgl-model-gateway/`** — the Rust router: cache-aware load balancing, service discovery,
multi-model serving, and PD-aware routing. *Ch. 17*

**`rust/`** — additional Rust components (`sglang-server`, `sglang-grpc`, `sglang-mm`).

**`test/`** — the test suites, run_suite.py, CI registration, and `lm_eval_configs/` for
accuracy evaluations. *Ch. 22*

**`docs/`** — the Mintlify documentation site, including the cookbook.

**`benchmark/`** — model- and feature-specific benchmark scripts, distinct from the
harnesses in `python/sglang/benchmark/`.

**`docker/`, `scripts/`, `.github/`** — packaging, tooling, and CI. *Ch. 22*

**`.claude/`** — `rules/` (enforced conventions) and `skills/` (maintainer playbooks). The
most concentrated design documentation in the repository. *Ch. 2*

**`examples/`, `proto/`, `3rdparty/`, `experimental/`** — usage examples, gRPC definitions,
vendored dependencies, and staging for unstable work.

---

## `python/sglang/`

**`srt/`** — the SGLang RunTime. Detailed below.

**`lang/`** — the frontend DSL: api.py (primitives), ir.py (program IR),
interpreter.py (execution), tracer.py (compilation), `backend/`. *Ch. 2*

**`kernels/`** — the kernel layer. `ops/` holds Triton kernels by category, `jit/` the
just-in-time C++/CUDA path, `aot/` the ahead-of-time path (formerly the top-level
`sgl-kernel`), and registry.py / selector.py / spec.py the dispatch. *Ch. 13, 14, 16,
22*

**`benchmark/`** — serving.py, offline_throughput.py, one_batch.py,
one_batch_server.py. *Ch. 1, 21*

**`test/`** — the test harness: `CustomTestCase`, server fixtures, `kits/`. *Ch. 22*

**`multimodal_gen/`** — the diffusion stack. Out of scope for this book.

**launch_server.py, profiler.py, check_env.py, global_config.py** — entry points and
utilities. *Ch. 2, 21*

---

## `python/sglang/srt/` — the runtime

Ordered roughly by the path a request takes.

**`entrypoints/`** — HTTP server, embeddable `Engine`, gRPC, and the OpenAI / Anthropic /
Ollama compatibility layers under `openai/`, `anthropic/`, `ollama/`. *Ch. 2, 3*

**`managers/`** — the coordination layer, and the densest part of the runtime.
tokenizer_manager.py (front end), scheduler.py (the loop), schedule_batch.py
(`Req`, `ScheduleBatch`), schedule_policy.py (admission), detokenizer_manager.py,
io_struct.py (the wire protocol), tp_worker.py, `scheduler_components/` (extracted
collaborators), data_parallel_controller.py, cache_controller.py,
multimodal_processor.py. *Ch. 3–7, 20*

**`model_executor/`** — model_runner.py (owns the GPU), forward_batch_info.py
(`ForwardMode`, `ForwardBatch`), forward_context.py, `runner/` and `runner_backend/`
(CUDA graph runners), `model_runner_components/`. *Ch. 6, 14*

**`mem_cache/`** — memory_pool.py (the KV pools), `allocator/`, radix_cache.py,
hiradix_cache.py, chunk_cache.py, the SWA and Mamba variants, `storage/` (HiCache
backends), `unified_cache/`, `cpp_radix_tree/`. *Ch. 8–10*

**`models/`** — 218 model definitions. llama.py is the template. *Ch. 12*

**`configs/`** — 63 per-model config classes plus model_config.py. *Ch. 11*

**`model_loader/`** — loader.py, auto_loader.py, weight_utils.py. *Ch. 11*

**`layers/`** — the building blocks. linear.py (parallel linears),
vocab_parallel_embedding.py, radix_attention.py, logits_processor.py, sampler.py,
layernorm.py, activation.py, `rotary_embedding/`, parameter.py, dp_attention.py,
communicator.py, plus the subsystems `attention/`, `moe/`, and `quantization/`.
*Ch. 7, 12–16*

**`distributed/`** — parallel_state.py (process groups), `device_communicators/`,
communication_op.py. *Ch. 15*

**`speculative/`** — EAGLE, MTP, n-gram, DFlash, DSpark workers and their metadata.
*Ch. 18*

**`disaggregation/`** — prefill and decode mixins, the transfer backends (`mooncake/`,
`nixl/`, `mori/`, `ascend/`, `fake/`), and the encode servers. *Ch. 17*

**`eplb/`** — expert distribution measurement, placement, and rebalancing. *Ch. 16*

**`batch_overlap/`** — two-batch and single-batch overlap. *Ch. 16*

**`lora/`** — adapter manager, memory pool, kernels, and layer wrappers. *Ch. 20*

**`multimodal/`** — 53 input processors over a shared base. *Ch. 20*

**`constrained/`** — grammar backends (XGrammar, Outlines, LLGuidance), jump-forward
decoding, the grammar manager. *Ch. 19*

**`function_call/`** — 39 tool-call format detectors and the dispatching parser. *Ch. 19*

**`parser/`** — reasoning and Harmony parsers. *Ch. 19*

**`sampling/`** — sampling parameters, batch info, penalties, custom logit processors.
*Ch. 7*

**`observability/`** — metrics collectors, tracing, startup timing, profiling support.
*Ch. 21*

**`compilation/`** — the torch.compile integration and custom Inductor passes. *Ch. 14*

**`checkpoint_engine/`, `weight_sync/`, `connector/`** — weight updates and remote weight
sources. *Ch. 11*

**`platforms/`, `hardware_backend/`, `plugins/`** — the hardware abstraction and
out-of-tree extension mechanism. *Ch. 22*

**`debug_utils/`** — the layer-by-layer comparator used to localize wrong-output bugs.
*Ch. 22*

**server_args.py, environ.py, runtime_context.py** — configuration. *Appendix A, B*

**`batch_invariant_ops/`** — deterministic operator implementations. *Ch. 7*

**`session/`, `multiplex/`, `elastic_ep/`, `ray/`, `grpc/`, `dllm/`, `kv_canary/`,
`state_capturer/`, `weight_cache/`, `tokenizer/`, `arg_groups/`** — smaller subsystems this
book touches only in passing.

---

## Where to start reading

If you are opening the codebase for the first time, in this order:

1. `python/sglang/launch_server.py` — 70 lines, the whole dispatch.
2. `python/sglang/srt/managers/scheduler.py:1714` `event_loop_normal` — 30 lines, the whole
   engine.
3. `python/sglang/srt/models/llama.py` — what a model is.
4. `python/sglang/srt/mem_cache/radix_cache.py` — the idea the project is known for.
5. `.claude/rules/` — five short files, each a hazard already hit.

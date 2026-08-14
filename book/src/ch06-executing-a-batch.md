# 6. Executing a Batch

> *The handoff from Python scheduling objects to GPU tensors is where the mode of the batch starts determining everything downstream.*

> **Status: outline.** This chapter is planned but not yet written. The beats below are the writing plan — each one pairs an idea with the code that implements it.

## Beats

1. **Mode determines the world.**
   `python/sglang/srt/model_executor/forward_batch_info.py:98` `ForwardMode` — `EXTEND`,
   `DECODE`, `MIXED`, `IDLE`, `TARGET_VERIFY`, `DRAFT_EXTEND`, `SPLIT_PREFILL`. Attention
   backend, kernel choice, graph eligibility, and memory accounting all branch here.

2. **The execution view.** `python/sglang/srt/model_executor/forward_batch_info.py:412`
   `ForwardBatch` and `:739` `init_new`, plus `.claude/rules/forward-batch-init-new-
   purity.md` on why construction must be pure.
   `python/sglang/srt/model_executor/forward_context.py` for the ambient per-forward
   context and the problem it solves.

3. **The worker boundary.** `python/sglang/srt/managers/tp_worker.py:74` `BaseTpWorker`,
   `:299` `TpModelWorker` — thin, and deliberately so.

4. **Initialization as an ordered script.**
   `python/sglang/srt/model_executor/model_runner.py:284` `ModelRunner`, `:287` `__init__`
   — weights (`:1057` `load_model`), memory pool (`:807` `alloc_memory_pool`), attention
   backend (`:927` `init_attention_backends`), CUDA graphs (`:992` `init_cuda_graphs`).
   The order is a dependency chain, and reading it explains most startup failures.

5. **The forward call.** `python/sglang/srt/model_executor/model_runner.py:1505` `forward`
   and `:1649` `_forward_raw` — mode dispatch, graph replay vs eager, and the contract the
   model must satisfy.

# 18. Speculative Decoding

> *Verifying k tokens costs nearly what generating one costs, so the only question is how good a draft you can produce cheaply.*

> **Status: outline.** This chapter is planned but not yet written. The beats below are the writing plan — each one pairs an idea with the code that implements it.

## Beats

1. **The bandwidth argument.** Why the acceptance rule preserves the target model's output
   distribution, and why a memory-bound decode step has spare compute to give away.

2. **Capabilities before implementations.**
   `python/sglang/srt/speculative/spec_info.py:30` `SpeculativeAlgorithm` — the predicate
   set (`is_eagle`, `has_draft_kv`, `supports_ragged_verify`, `supports_grammar_overlap`)
   that gates behavior across the entire engine. Read this before any worker, and
   `.claude/skills/speculative-naming/SKILL.md` before that for the vocabulary.

3. **EAGLE end to end.** `python/sglang/srt/speculative/eagle_worker_v2.py:1008`
   `EAGLEWorkerV2` — `:1105` `forward_batch_generation` as the step, `:1497` `verify` as
   the accept rule; `:128` `EagleDraftWorker` with `:494` `draft`, `:557` `draft_forward`,
   `:726` `draft_extend`. `python/sglang/srt/speculative/eagle_info.py` for the tree
   metadata.

4. **Trees, not chains.** Draft topology, and how `--speculative-eagle-topk` and
   `--speculative-num-steps` trade acceptance against wasted compute.

5. **Down to the kernel.** `python/sglang/kernels/ops/speculative/` and the CUDA under
   `python/sglang/kernels/aot/csrc/speculative/` — tree mask construction and the
   verification kernel, where the accept rule becomes parallel tensor work.

6. **What it costs the rest of the engine.** Extra `ForwardMode`s (Chapter 6), separate
   CUDA graphs (`python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py`), changed
   memory accounting in the Chapter 5 budget, and grammar coordination with Chapter 19.

7. **The other algorithms, briefly.** `python/sglang/srt/speculative/ngram_worker.py` and
   `cpp_ngram/` (no draft model at all), `frozen_kv_mtp_worker_v2.py`,
   `dflash_worker_v2.py`, `standalone_worker_v2.py` — and `adaptive_runtime_state.py`,
   which turns speculation off when acceptance drops.

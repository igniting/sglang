# 7. Sampling and the Return Path

> *Turning hidden states into user-visible text is three separate hard problems that happen to sit next to each other.*

> **Status: outline.** This chapter is planned but not yet written. The beats below are the writing plan — each one pairs an idea with the code that implements it.

## Beats

1. **Not every position matters, except when it does.**
   `python/sglang/srt/layers/logits_processor.py:282` `LogitsProcessor`, `:332` `forward`,
   `:427` `_get_pruned_states`, `:693` `_compute_lm_head`. Decode needs one position;
   prefill may need all of them for logprobs, EAGLE, or hidden-state capture — and `:96`
   `LogitsProcessorOutput` / `:149` `LogitsMetadata` encode which.

2. **Sampling as batched tensor work.** `python/sglang/srt/layers/sampler.py:70`
   `Sampler`, `:97` `forward`, `:563` `top_k_top_p_min_p_sampling_from_probs_torch`.
   Temperature, top-k/p/min-p, and penalties (`python/sglang/srt/sampling/penaltylib/`)
   applied to a batch whose requests all asked for different things —
   `python/sglang/srt/sampling/sampling_batch_info.py` is how that is made possible.

3. **Reproducibility, and why it is hard.** Non-deterministic reductions and batch-variant
   kernels mean identical prompts can diverge across batch sizes.
   `python/sglang/srt/batch_invariant_ops/`,
   `python/sglang/srt/model_executor/model_runner.py:764`
   `maybe_enable_batch_invariant_mode`, and `python/sglang/srt/layers/sampler.py:684`
   `multinomial_with_seed`. `.claude/skills/kl-consistency-test/SKILL.md` states the two
   independent conditions a zero-KL result requires.

4. **Incremental detokenization is genuinely hard.** BPE merges and multi-byte UTF-8 mean
   you cannot simply decode the newest token.
   `python/sglang/srt/managers/detokenizer_manager.py:91` `DetokenizerManager`, `:166`
   `event_loop`, `:290` `_decode_batch_token_id_output`;
   `python/sglang/srt/managers/schedule_batch.py:1425` `init_incremental_detokenize`.

5. **Stopping, and un-emitting.** Stop strings require lookback and retraction of already-
   produced text: `python/sglang/srt/managers/detokenizer_manager.py:176`
   `trim_matched_stop`, `python/sglang/srt/managers/schedule_batch.py:1445`
   `_stop_match_tail_len`.

6. **Streaming out.** `python/sglang/srt/managers/tokenizer_manager.py:1641`
   `_coalesce_streaming_chunks` and `python/sglang/srt/entrypoints/openai/sse_utils.py` —
   the chunk-size/latency trade at the very last hop.

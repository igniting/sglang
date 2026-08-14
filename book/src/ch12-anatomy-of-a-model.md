# 12. Anatomy of a Model

> *Models are rewritten rather than imported because every layer must cooperate with parallelism, quantization, and the KV cache — and llama.py shows exactly how.*

> **Status: outline.** This chapter is planned but not yet written. The beats below are the writing plan — each one pairs an idea with the code that implements it.

## Beats

1. **The contract.** `forward(input_ids, positions, forward_batch) -> logits`, plus
   `load_weights`. Everything else is optional capability.

2. **A full read of `python/sglang/srt/models/llama.py`.** `:70` `LlamaMLP` (merged
   gate/up column-parallel, row-parallel down), `:138` `LlamaAttention` (fused QKV,
   rotary, and the `RadixAttention` instantiation that connects Chapter 9), `:283`
   `LlamaDecoderLayer` (residual ordering and fused add-RMSNorm), `:372` `LlamaModel`,
   `:496` `LlamaForCausalLM` with `:563` `forward`.

3. **The parallel layer vocabulary.** `python/sglang/srt/layers/linear.py:293`
   `ColumnParallelLinear`, `:1392` `RowParallelLinear`, `:921` `QKVParallelLinear`, `:492`
   `MergedColumnParallelLinear`;
   `python/sglang/srt/layers/vocab_parallel_embedding.py:188` `VocabParallelEmbedding` and
   `:587` `ParallelLMHead`. These are where tensor parallelism physically lives — Chapter
   15 only explains what they already do.

4. **Optional capabilities as hooks.** `python/sglang/srt/models/llama.py:640`
   `start_layer` / `:644` `end_layer` for pipeline parallelism, `:599`
   `forward_split_prefill`, `:891` `set_eagle3_layers_to_capture` for speculative
   decoding, `:854` `get_embed_and_head` for weight sync.

5. **Three short contrast studies.** `python/sglang/srt/models/deepseek_v2.py` (MLA +
   MoE), a Qwen-VL variant (a vision tower on a decoder),
   `python/sglang/srt/models/falcon_h1.py` (state instead of attention) — enough to show
   what varies and what never does.

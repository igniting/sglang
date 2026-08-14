# 13. Attention Backends

> *Attention is pluggable because hardware, sequence shape, and kernel maturity all vary independently — and the plug is a two-phase metadata/kernel contract.*

> **Status: outline.** This chapter is planned but not yet written. The beats below are the writing plan — each one pairs an idea with the code that implements it.

## Beats

1. **The contract.** `python/sglang/srt/layers/attention/base_attn_backend.py:33`
   `AttentionBackend` — `:62` `init_forward_metadata` runs once per forward, `:261`
   `forward_decode` and `:274` `forward_extend` run once per layer. Nearly every backend
   bug is a violation of that split. `:160` `init_cuda_graph_state` is where Chapter 14's
   constraints intrude.

2. **The reference implementation.**
   `python/sglang/srt/layers/attention/flashinfer_backend.py` read in depth — wrappers,
   page tables, and graph-safe buffers. Then
   `python/sglang/srt/layers/attention/triton_backend.py` as the portable fallback, short
   enough to read whole.

3. **Down to the kernel.** `python/sglang/kernels/ops/attention/decode_attention.py` and
   `extend_attention.py` are Triton — readable as Python, and the clearest place to see
   paging, masking, and the online-softmax accumulation actually happen. This is the
   chapter's first descent below the PyTorch line.

4. **Registration and selection.**
   `python/sglang/srt/layers/attention/attention_registry.py:34`
   `register_attention_backend` and the factory table;
   `python/sglang/srt/model_executor/model_runner.py:927` `init_attention_backends`.

5. **MLA needs its own everything.**
   `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`, `flashmla_backend.py`,
   and their dependence on the Chapter 8 pool — the clearest case of memory layout
   dictating kernel design.

6. **Sparse attention.** `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` and
   `nsa_backend.py` — selecting which tokens to attend to, and the indexer as a model
   component in its own right.

7. **When it isn't attention at all.** `python/sglang/srt/layers/attention/mamba/mamba.py`
   and `python/sglang/srt/layers/attention/linear/gdn_backend.py` — SSMs and linear
   attention carry state rather than a KV cache, which is why Chapter 8 needed
   `MambaPool`. `hybrid_linear_attn_backend.py` for models that do both.

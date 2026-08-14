# 14. Making the Forward Pass Cheap

> *Quantization attacks bytes moved, CUDA graphs attack launch overhead, and compilation attacks kernel count — three independent taxes on the same forward pass.*

> **Status: outline.** This chapter is planned but not yet written. The beats below are the writing plan — each one pairs an idea with the code that implements it.

## Beats

1. **Three independent targets.** Weights, activations, and KV cache can each be quantized
   separately, and the choice of scheme is per-target.

2. **The quantization architecture.**
   `python/sglang/srt/layers/quantization/base_config.py` — `QuantizationConfig` to
   `QuantizeMethodBase` to per-layer `apply()`. Then `fp8.py` and `fp8_utils.py` read end
   to end as the most-used path, with `modelopt_quant.py`, `mxfp4.py`,
   `compressed_tensors/`, `awq/`, `gptq/` surveyed for what differs.
   `python/sglang/srt/layers/parameter.py` is what makes sharding and scale tensors
   coexist.

3. **Down to the kernel.** `python/sglang/kernels/ops/quantization/` and the AOT CUDA
   under `python/sglang/kernels/aot/csrc/` — where a scale factor becomes a fused dequant-
   GEMM epilogue, and why per-block scales cost what they cost.

4. **Quantized KV.** `python/sglang/srt/layers/quantization/kv_cache.py` closing the loop
   with the Chapter 8 pools, and where accuracy actually degrades.

5. **Decode is launch-bound.** Thousands of microsecond kernels mean CPU launch cost
   dominates. CUDA graphs capture once and replay — at the price of static shapes and
   static pointers.

6. **Capture and replay.**
   `python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py`,
   `decode_cuda_graph_runner.py`, `prefill_cuda_graph_runner.py`;
   `python/sglang/srt/model_executor/model_runner.py:992` `init_cuda_graphs`; batch-size
   bucketing, padding, and the memory cost of the capture matrix.

7. **When part of the model cannot be captured.**
   `python/sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py` and
   `tc_piecewise_cuda_graph_backend.py` — keeping most of the graph when one region must
   run eagerly.

8. **Compilation.** `python/sglang/srt/compilation/` — the Inductor backend, custom passes
   (`fix_functionalization.py`, `pass_manager.py`), and how it composes with graphs.

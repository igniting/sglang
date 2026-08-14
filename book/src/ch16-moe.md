# 16. Mixture-of-Experts and Expert Parallelism

> *MoE inference is all-to-all-bound rather than GEMM-bound, which makes routing, placement, and communication overlap the whole game.*

> **Status: outline.** This chapter is planned but not yet written. The beats below are the writing plan — each one pairs an idea with the code that implements it.

## Beats

1. **Routing.** `python/sglang/srt/layers/moe/topk.py:392` `TopK` and the `TopKOutput`
   variants at `:274` — top-k selection, and why its output format matters to the kernel
   that follows.

2. **Computing the experts.** `python/sglang/srt/layers/moe/fused_moe_triton/` as the
   portable path, `python/sglang/srt/layers/moe/moe_runner/` as the abstraction over
   backends.

3. **Down to the kernel.** `python/sglang/kernels/ops/moe/` (Triton) and the CUDA under
   `python/sglang/kernels/aot/csrc/moe/` — the largest concentration of in-repo CUDA,
   including the CUTLASS w4a8 path. Grouped GEMM, and why expert-major layout is the whole
   trick.

4. **Splitting experts across devices.** `python/sglang/srt/layers/moe/ep_moe/` and
   `python/sglang/srt/layers/moe/token_dispatcher/` — DeepEP dispatch/combine, and why
   all-to-all latency rather than FLOPs sets the step time.

5. **Hot experts.** `python/sglang/srt/eplb/expert_distribution.py` measures imbalance;
   `expert_location.py`, `eplb_manager.py`, `eplb_algorithms/`, and `lplb_solver.py`
   rebalance and replicate. Recording expert distribution is exposed as a server endpoint
   precisely because the imbalance is workload-dependent.

6. **Hiding the all-to-all.** `python/sglang/srt/batch_overlap/two_batch_overlap.py` and
   `single_batch_overlap.py` split a batch so communication overlaps computation, with
   `operations.py` and `operations_strategy.py` as the scheduling abstraction and
   `python/sglang/srt/layers/attention/tbo_backend.py` on the attention side. This is the
   Chapter 4 overlap idea applied one level down.

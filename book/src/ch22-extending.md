# 22. Extending SGLang

> *The extension points are the architecture's seams, and walking them is the final check that the reader has understood where the boundaries are.*

> **Status: outline.** This chapter is planned but not yet written. The beats below are the writing plan — each one pairs an idea with the code that implements it.

## Beats

1. **Adding a model.** Config class, model class, weight mapping, registry entry, chat
   template — and layer-by-layer comparison against HF via
   `python/sglang/srt/debug_utils/comparator/` when the output is wrong.

2. **Adding a kernel.** The two paths: JIT (`python/sglang/kernels/jit/`, with its `csrc/`
   tree) and AOT (`python/sglang/kernels/aot/`, the former top-level `sgl-kernel`), with
   `python/sglang/kernels/registry.py` and `selector.py` as the dispatch layer.
   Correctness tests and benchmarks are part of the deliverable, not follow-up work.

3. **Adding an attention backend.** Implementing the Chapter 13 contract, and the CUDA-
   graph obligations that catch every first attempt.

4. **Porting to new hardware.** `python/sglang/srt/platforms/`,
   `python/sglang/srt/hardware_backend/`, `python/sglang/srt/plugins/` — what a new
   accelerator actually requires (communicators, attention backend, memory pool, graph
   capture), with ROCm/AITER, Ascend, and Intel XPU as case studies of how far the
   abstraction stretches.

5. **How the project keeps this safe.** `test/run_suite.py`, `python/sglang/test/kits/`,
   and why accuracy evals rather than unit tests are the real net. `.github/workflows/`
   for gating and partitioning.

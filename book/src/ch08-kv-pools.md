# 8. KV Cache Pools and Allocators

> *The engine's central data structure is a two-level indirection, and its shape explains both paged attention and everything Chapter 9 builds on top.*

> **Status: outline.** This chapter is planned but not yet written. The beats below are the writing plan — each one pairs an idea with the code that implements it.

## Beats

1. **Two levels, not one.** `python/sglang/srt/mem_cache/memory_pool.py:256`
   `ReqToTokenPool` maps request to token slots; the `KVCache` hierarchy maps token slot
   to storage. Splitting them is what lets a prefix be shared by requests that disagree
   about everything else.

2. **Pages and the allocator's job.** `python/sglang/srt/mem_cache/allocator/paged.py:105`
   `PagedTokenToKVPoolAllocator` — `:149` `alloc`, `:172` `alloc_extend`, `:222`
   `alloc_decode`, `:261` `free`. Page size as the knob that trades internal fragmentation
   against metadata cost.

3. **One abstraction, many pools.** `python/sglang/srt/mem_cache/memory_pool.py:1624`
   `KVCache` and its implementations: `:1755` `MHATokenToKVPool` (the common case),
   `:3932` `MLATokenToKVPool` (DeepSeek's compressed cache — an order of magnitude
   smaller, which is why Chapter 15's DP attention exists), `:3577` `HybridLinearKVPool`
   and `:335` `MambaPool` (state, not keys and values), `:3135`
   `PageMajorMHATokenToKVPool` (layout as a transfer optimization).

4. **From a percentage to a number.**
   `python/sglang/srt/mem_cache/kv_cache_configurator.py` and
   `python/sglang/srt/mem_cache/allocation_sizing.py` turn `--mem-fraction-static` into a
   concrete pool size; this is the calculation behind every "out of memory at 90%
   utilization" report.

5. **Fewer bits per entry.** `python/sglang/srt/mem_cache/kv_cache_dtype.py` and the
   FP8/FP4 pool variants, with the accuracy question deferred to Chapter 14.

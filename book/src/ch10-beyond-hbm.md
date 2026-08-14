# 10. Caching Beyond HBM

> *Extending the cache hierarchy to host memory and disk is a bandwidth arbitrage, and it only pays above a computable prefix length.*

> **Status: outline.** This chapter is planned but not yet written. The beats below are the writing plan — each one pairs an idea with the code that implements it.

## Beats

1. **The arbitrage.** At what prefix length does loading KV from host DRAM beat
   recomputing it? The answer sets every policy in this chapter.

2. **A radix tree with tiers.** `python/sglang/srt/mem_cache/hiradix_cache.py:76`
   `HiRadixCache` subclasses `RadixCache` from Chapter 9; `:840` `write_backup` is write-
   through/write-back, and the tier bookkeeping in
   `python/sglang/srt/mem_cache/radix_cache.py:273` `backuped` / `:276` `protect_host` is
   what keeps tiers coherent.

3. **Moving the bytes.** `python/sglang/srt/mem_cache/memory_pool_host.py` and
   `python/sglang/srt/managers/cache_controller.py` — the transfer engine, its queues, and
   its overlap with compute.

4. **Pluggable storage.** `python/sglang/srt/mem_cache/hicache_storage.py`,
   `python/sglang/srt/mem_cache/storage/backend_factory.py`, and the backends (Mooncake,
   3FS, NIXL, LMCache, file, mmap, shm) — with
   `python/sglang/srt/mem_cache/hiradix_cache.py:369` `attach_storage_backend` / `:487`
   `detach_storage_backend` showing runtime attach as a first-class operation.

5. **Where it is heading.** `python/sglang/srt/mem_cache/unified_cache/` — the unified
   tree core and component registry that generalizes the tiering.

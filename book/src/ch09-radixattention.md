# 9. RadixAttention

> *Real workloads share long prefixes, and a radix tree over token sequences turns that redundancy into the engine's largest single win. This is SGLang's signature idea.*

> **Status: outline.** This chapter is planned but not yet written. The beats below are the writing plan — each one pairs an idea with the code that implements it.

## Beats

1. **The observation.** Chat history, few-shot prompts, agent loops, and system prompts
   all mean the *n*-th request usually shares a long prefix with an earlier one.
   Recomputing it is pure waste.

2. **The key.** `python/sglang/srt/mem_cache/radix_cache.py:59` `RadixKey` — token ids
   plus an extra key that namespaces LoRA adapters and sessions apart (`:181` `match`,
   `:217` `child_key`, `:150` `page_aligned`). Page alignment constrains every tree
   operation and is the source of most of the code's subtlety.

3. **The tree.** `python/sglang/srt/mem_cache/radix_cache.py:238` `TreeNode` (children,
   `lock_ref`, `last_access_time`, host tier) and `:303` `RadixCache`. The three
   operations, read in order: `:376` `match_prefix` / `:678` `_match_prefix_helper`,
   `:704` `_split_node`, `:436` `insert` / `:737` `_insert_helper`.

4. **Reference counting keeps live requests alive.**
   `python/sglang/srt/mem_cache/radix_cache.py:622` `inc_lock_ref`, `:637` `dec_lock_ref`,
   `:458` `cache_finished_req`, `:515` `cache_unfinished_req`. A node in use by a running
   request must survive eviction — this is the invariant that makes the whole thing safe.

5. **Eviction over a tree.** `python/sglang/srt/mem_cache/radix_cache.py:592` `evict` —
   LRU, but leaf-first, because interior nodes are prefixes of their children.

6. **Where the model touches it.** `python/sglang/srt/layers/radix_attention.py:91`
   `RadixAttention` is the layer every model instantiates; it is the reason model code
   needs no cache logic of its own.

7. **The contrast case.** `python/sglang/srt/mem_cache/chunk_cache.py:35` `ChunkCache`
   implements the same interface with no reuse at all, which makes the interface
   (`python/sglang/srt/mem_cache/base_prefix_cache.py:230` `BasePrefixCache`) legible.

8. **Variants.** `python/sglang/srt/mem_cache/swa_radix_cache.py` (sliding window),
   `python/sglang/srt/mem_cache/mamba_radix_cache.py` (state, not tokens),
   `python/sglang/srt/mem_cache/radix_cache_cpp.py` plus
   `python/sglang/srt/mem_cache/cpp_radix_tree/` (the same algorithm in C++, and why
   Python became the bottleneck).

9. **A worked example.** Three chat requests sharing a system prompt, traced through match
   / split / insert / evict as the tree evolves.

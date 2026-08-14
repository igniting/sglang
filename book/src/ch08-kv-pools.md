# 8. KV Cache Pools and Allocators

> *The engine's central data structure is a two-level indirection, and its shape explains
> both paged attention and everything Chapter 9 builds on top.*

---

## Two levels, not one

Chapter 1 established the KV cache as the scarce resource. This chapter is how it is
actually stored.

The naive layout gives each request a contiguous buffer sized for its maximum length. It
fails on two counts. **Waste**: a request that could generate 4,096 tokens but stops at 100
has reserved 40× what it used, and that reservation cannot serve anyone else. And
**rigidity**: two requests sharing a prefix cannot share storage, because each owns a
private buffer.

SGLang splits the mapping in two:

```
  request  ──►  token slots  ──►  KV storage
           (1)               (2)

  (1)  ReqToTokenPool         req_to_token[req_idx, position] = kv_index
  (2)  KVCache + allocator    kv_index → offset into the big KV tensors
```

Level 1 answers "where are *my* tokens?" Level 2 answers "where does token index *i*
live?" The indirection is what makes sharing possible: two requests can hold the same
`kv_index` at different positions in their own rows, and the storage is written once. That
is the mechanical precondition for Chapter 9.

`python/sglang/srt/mem_cache/memory_pool.py:256` `ReqToTokenPool` is level one, and it is
refreshingly plain:

```python
class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""
    ...
        self.size = size
        # +1 padding row at index 0: cuda-graph padded batches default
        # req_pool_indices to 0, so dummy reads/writes land here harmlessly.
        self._alloc_size = size + 1
        self.max_context_len = max_context_len
        self.device = device
        with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token = torch.zeros(
                (self._alloc_size, max_context_len), dtype=torch.int32, device=device
            )
        self.free_slots = list(range(1, self._alloc_size))
```

A dense `[max_requests, max_context_len]` int32 matrix. For 4,096 requests and a 32k
context that is 512 MB — real memory, spent on indices rather than data, and the price of
the indirection.

The padding row is a nice piece of defensive design. CUDA-graph batches (Chapter 14) are
padded to a captured size, and padded slots default to `req_pool_idx = 0`. Rather than
branching to skip them, row 0 is a scratch row where dummy reads and writes land
harmlessly. Note `free_slots` starts at 1, so row 0 is never allocated.

`python/sglang/srt/mem_cache/memory_pool.py:290` `alloc` handles a case worth flagging:

```python
        # Indices of reqs that already have a req_pool_idx and will reuse
        # their existing slot (e.g. chunked prefill continuing across chunks).
        reusing = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]
```

A chunked-prefill request (Chapter 5) spans several scheduler iterations and must keep its
row across all of them. Allocation is therefore not "give me *n* free slots" but "give me
slots for the requests that lack one."

---

## Pages and the allocator's job

Level two is where paging lives.
`python/sglang/srt/mem_cache/allocator/paged.py:105` `PagedTokenToKVPoolAllocator` states
its own contract:

```python
class PagedTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """
    An allocator managing the indices to kv cache data.

    This class has the same interface as `TokenToKVPoolAllocator` but the output
    of one request is always page-aligned.
    """
```

The trade is the classic one. Larger pages mean less metadata and better locality, but more
internal fragmentation — a request needing 33 tokens with a 32-token page consumes 64. This
is the rounding Chapter 5's `ceil_paged_tokens` charges against the budget, and Chapter 9's
`page_aligned` truncates the cache key by.

`:149` `alloc` is the simple case:

```python
        num_pages = need_size // self.page_size
        if self.need_sort and num_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_pages > len(self.free_pages):
            return None

        out_pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]

        out_indices = (
            out_pages[:, None] * self.page_size
            + torch.arange(self.page_size, device=self.device)
        ).reshape(-1)
```

Free pages are a *tensor*, not a Python list, and expanding page ids into token indices is
one broadcast rather than a loop. At hundreds of allocations per second across thousands of
pages, allocator bookkeeping in Python would show up in profiles — a recurring theme in
this part of the codebase.

`merge_and_sort_free` is deferred defragmentation: coalescing runs only when an allocation
is about to fail. Returning `None` rather than raising is deliberate too — running out of
KV memory is an *expected* condition that Chapter 5 handles by retracting, not an error.

`:172` `alloc_extend` and `:222` `alloc_decode` are the specialized paths. Extend allocates
a variable number of tokens per request; decode allocates exactly one per sequence and can
be a much tighter kernel. `:273` `free_segment` takes a `start_pos`, which is what lets
Chapter 9's `cache_finished_req` free two disjoint ranges of a request's row while leaving
the tree-owned middle alone.

The most interesting comment in the file is not about allocation at all:

```python
        # Pre-warm the torch.unique HIP kernel used in free(). When a request
        # finishes with a prompt that already exists in the radix tree (e.g.
        # bench_serving sending the same warmup+measured prompt), the radix
        # cache's _insert_helper frees the duplicate KV indices via
        # token_to_kv_pool_allocator.free(value[start:prefix_len]). That call
        # path runs `torch.unique(free_index // self.page_size)` on a
        # ~prompt_len-sized int64 tensor. The first such call on AMD ROCm
        # JIT-compiles rocPRIM sort/unique kernels and costs ~200ms, which
        # shows up as a mysterious "second-request slow" (Run 1) for
        # repeated-prompt benchmarks.
```

A 200 ms JIT compile, triggered on the second request, only on ROCm, only when a prompt
repeats — surfacing as an unexplained latency spike in benchmarks. The fix is to run the
kernel once at startup. Worth reading in full as a specimen of what performance work on
this layer actually looks like.

Sibling allocators handle the other pool shapes: `python/sglang/srt/mem_cache/allocator/token.py`
(page size 1), `python/sglang/srt/mem_cache/allocator/swa.py` (sliding window),
`python/sglang/srt/mem_cache/allocator/mamba.py` (recurrent state), and
`python/sglang/srt/mem_cache/allocator/hisparse.py` (sparse).

---

## One abstraction, many pools

`python/sglang/srt/mem_cache/memory_pool.py:1624` `KVCache` is the abstract base, and its
`__init__` establishes what every pool shares:

```python
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.layer_num = layer_num
        self.start_layer = start_layer or 0
        self.end_layer = end_layer or layer_num - 1
```

The `dtype` / `store_dtype` split is a PyTorch workaround made permanent: FP8 tensors do
not support `index_put`, so the buffer is typed `uint8` and reinterpreted. Chapter 14
returns to quantized caches; note here that the pool, not the kernel, owns the storage type.

`start_layer` / `end_layer` are pipeline parallelism (Chapter 15) — a rank holding layers
20–39 allocates cache only for those.

The implementations:

**`:1755` `MHATokenToKVPool`** — the common case. Separate K and V buffers per layer,
indexed by token slot. Read this one first.

**`:3932` `MLATokenToKVPool`** — DeepSeek's multi-head latent attention. Instead of K and V
per head, a single compressed latent vector per token, decompressed during attention. An
order of magnitude smaller per token, which is why DeepSeek models serve long contexts on
modest hardware — and, as Chapter 15 explains, the reason data-parallel attention exists:
replicating a cache this small across tensor-parallel ranks wastes exactly what made it
valuable.

**`:3135` `PageMajorMHATokenToKVPool`** — same data, different memory layout. Layer-major
stores all tokens for layer 0, then layer 1; page-major groups all layers for a page
together. Page-major makes a page contiguous, which matters when you are transferring pages
to host memory (Chapter 10) or across machines (Chapter 17). Layout is a transfer
optimization, not a compute one.

**`:3577` `HybridLinearKVPool`** and **`:335` `MambaPool`** — for models where some layers
are not attention. An SSM layer carries a fixed-size recurrent state, not a growing KV run.
Fixed-size is a different allocation problem: it cannot be evicted incrementally, and it
cannot be reconstructed from a prefix. Chapter 5's separate `rem_mamba_slots` gate exists
for exactly this reason.

**`:4348` `DSATokenToKVPool`** and **`:4671` `MiniMaxSparseKVPool`** — sparse attention,
where only selected tokens are attended to (Chapter 13).

**`:2867` `NoOpMHATokenToKVPool`** — stores nothing. For disaggregated prefill (Chapter 17),
where KV is streamed to a decode node instead of retained.

`:1668` `_finalize_allocation_log` produces the `KV Cache is allocated. #tokens: ...` line
in the startup log, which is the single most useful number for capacity planning.

---

## From a percentage to a number

`--mem-fraction-static` is the flag people tune first and understand last. The chain is:

1. `python/sglang/srt/model_executor/model_runner.py:1057` `load_model` puts weights on the
   device.
2. `:1322` `configure_kv_cache_dtype` fixes bytes per entry (Chapter 14).
3. `:807` `alloc_memory_pool` computes what remains and converts it to a token count.
4. `python/sglang/srt/mem_cache/kv_cache_configurator.py` and
   `python/sglang/srt/mem_cache/allocation_sizing.py` do the arithmetic, and
   `python/sglang/srt/mem_cache/kv_cache_dtype.py` supplies the per-token size.

The result is `max_total_num_tokens` — the budget Chapter 5 spends. Not a request count:
**a token count, shared across all concurrent requests**, which is why concurrency depends
on context length rather than being a fixed number.

The residual — the gap between the fraction and reality — is activations, workspace,
fragmentation, and CUDA graph buffers. This is why raising `--mem-fraction-static` too far
produces an OOM not at startup but under load, when the first large batch demands
activation memory the pool has already taken. `:885` `post_capture_resize_kv_pool` recovers
some of that slack by returning unused capture memory after graphs are built.

---

## The full address translation

Putting both levels together, for token *t* of request *r*:

```
  r.req_pool_idx                    row in req_to_token       (ReqToTokenPool)
       │
       ▼
  req_to_token[req_pool_idx, t]  =  kv_index                  (level 1)
       │
       ▼
  kv_index // page_size          =  page id                   (allocator)
  kv_index %  page_size          =  offset within page
       │
       ▼
  k_buffer[layer][kv_index]                                   (level 2)
  v_buffer[layer][kv_index]
```

Two lookups per token per layer. The attention backends of Chapter 13 do not walk this
chain per token — they receive the `req_to_token` slice as a *page table* and index it
inside the kernel, which is what "paged attention" names.

Every other chapter in this book is spending the resource this chapter allocates. Chapter 5
budgets it, Chapter 9 shares it, Chapter 10 tiers it, Chapter 14 shrinks it, Chapter 15
avoids replicating it, and Chapter 17 moves it between machines.

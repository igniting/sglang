# 9. KV Cache Pools and Allocators

> *The engine's central data structure is a two-level indirection, and its shape explains
> both paged attention and everything Chapter 10 builds on top.*

Part II kept hitting the same wall. The scheduler could not admit requests because memory
was full; it evicted running requests when it guessed wrong; its priorities depended on a
cache. Chapter 1 predicted this: memory capacity, not arithmetic, is what limits how many
people a GPU serves.

This chapter is about how that memory is actually organized — and the organization is more
interesting than "a big array," because the obvious approach fails badly.

Give each request a contiguous buffer sized for its maximum length and you get two problems.
A request that could have generated 4,000 tokens but stopped at 100 has reserved forty times
what it used, and that reservation helps nobody. Worse, two requests with identical prompts
cannot share anything, because each owns a private buffer.

SGLang's answer is two levels of indirection, and that structure is what makes the rest of
Part III possible. It is also where `--mem-fraction-static` — the flag people tune first and
understand last — stops being a percentage and becomes a concrete number of tokens.

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
is the mechanical precondition for Chapter 10.

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

The padding row is a nice piece of defensive design. CUDA-graph batches (Chapter 15) are
padded to a captured size, and padded slots default to `req_pool_idx = 0`. Rather than
branching to skip them, row 0 is a scratch row where dummy reads and writes land
harmlessly. Note `free_slots` starts at 1, so row 0 is never allocated.

`python/sglang/srt/mem_cache/memory_pool.py:290` `alloc` handles a case worth flagging:

```python
        # Indices of reqs that already have a req_pool_idx and will reuse
        # their existing slot (e.g. chunked prefill continuing across chunks).
        reusing = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]
```

A chunked-prefill request (Chapter 6) spans several scheduler iterations and must keep its
row across all of them. Allocation is therefore not "give me *n* free slots" but "give me
slots for the requests that lack one."

---

## What paging is borrowed from, and what it fixes

Before reading the allocator, it is worth being precise about what problem paging solves,
because the naming is a deliberate analogy and the analogy is load-bearing.

The pre-2023 approach was to give each request a contiguous KV buffer sized to the maximum
it might reach. Kwon et al.'s vLLM paper (SOSP '23) measured what that costs and split the
waste into three kinds:

**Internal fragmentation.** A request allocated for 2,048 tokens that generates 100 wastes
the other 1,948 slots for its entire lifetime. Nobody knows the output length in advance
(Chapter 6's central difficulty), so the reservation is always sized for the worst case.

**Reservation waste.** Slots that the request *will* eventually use are unusable by anyone
else *now*. Even a perfectly-sized allocation holds memory for a future that has not arrived.

**External fragmentation.** Different requests reserve different sizes, so freed regions
leave holes that no subsequent request quite fits. Classic malloc pathology.

Together these left, in their measurements, only **20.4% to 38.2%** of KV memory holding
actual token state. Between three-fifths and four-fifths of the scarcest resource in the
system, doing nothing. Since Chapter 1 established that KV capacity is the direct limiter on
batch size and therefore on throughput, this is not a memory-efficiency footnote — it is
most of the available performance.

The fix is the one operating systems reached in the 1960s. Stop giving a process a
contiguous physical region; give it a contiguous *virtual* address space and a page table
mapping each virtual page to any physical frame. The correspondence is exact:

| Operating system | KV cache |
| --- | --- |
| Process | Request |
| Page | Block of *n* consecutive tokens' KV |
| Page table | `req_to_token` row |
| Physical frame | Slot in the flat `k_buffer` / `v_buffer` |
| Page fault → allocate | `alloc()` on demand as the sequence grows |
| `fork()` + copy-on-write | Prefix sharing, Chapter 10 |

Each of the three wastes disappears for the same reason it does in an OS. External
fragmentation cannot occur, because every page is the same size and therefore
interchangeable. Reservation waste cannot occur, because pages are allocated as the sequence
actually grows rather than in advance. Internal fragmentation is bounded to *at most one
partially-filled page per request* — a few tokens, not a few thousand.

What it costs is indirection: every KV access now needs a table lookup, and every attention
kernel has to be rewritten to do that lookup itself. That rewrite is what the phrase "paged
attention" names, and it is why Chapter 14's kernels take page tables as arguments.

The remaining design freedom is the page size, and it is the same tension an OS faces.
Larger pages mean fewer table entries, fewer lookups, and better locality inside a page;
smaller pages mean less internal fragmentation and finer-grained sharing. SGLang's default
of one token per page — page size 1 — is the extreme end: zero internal fragmentation and
maximally precise prefix sharing, paid for with the largest possible page table. Larger
sizes exist for backends whose kernels want them, and `PAGE_SIZE` being a compile-time
constant in the Triton kernels (Chapter 14) means the page arithmetic vanishes entirely when
it is 1.

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
is the rounding Chapter 6's `ceil_paged_tokens` charges against the budget, and Chapter 10's
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
KV memory is an *expected* condition that Chapter 6 handles by retracting, not an error.

`:172` `alloc_extend` and `:222` `alloc_decode` are the specialized paths. Extend allocates
a variable number of tokens per request; decode allocates exactly one per sequence and can
be a much tighter kernel. `:273` `free_segment` takes a `start_pos`, which is what lets
Chapter 10's `cache_finished_req` free two disjoint ranges of a request's row while leaving
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
not support `index_put`, so the buffer is typed `uint8` and reinterpreted. Chapter 15
returns to quantized caches; note here that the pool, not the kernel, owns the storage type.

`start_layer` / `end_layer` are pipeline parallelism (Chapter 16) — a rank holding layers
20–39 allocates cache only for those.

The implementations:

**`:1755` `MHATokenToKVPool`** — the common case. Separate K and V buffers per layer,
indexed by token slot. Read this one first.

**`:3932` `MLATokenToKVPool`** — DeepSeek's multi-head latent attention. Instead of K and V
per head, a single compressed latent vector per token, decompressed during attention. An
order of magnitude smaller per token, which is why DeepSeek models serve long contexts on
modest hardware — and, as Chapter 16 explains, the reason data-parallel attention exists:
replicating a cache this small across tensor-parallel ranks wastes exactly what made it
valuable.

**`:3135` `PageMajorMHATokenToKVPool`** — same data, different memory layout. Layer-major
stores all tokens for layer 0, then layer 1; page-major groups all layers for a page
together. Page-major makes a page contiguous, which matters when you are transferring pages
to host memory (Chapter 11) or across machines (Chapter 18). Layout is a transfer
optimization, not a compute one.

**`:3577` `HybridLinearKVPool`** and **`:335` `MambaPool`** — for models where some layers
are not attention. An SSM layer carries a fixed-size recurrent state, not a growing KV run.
Fixed-size is a different allocation problem: it cannot be evicted incrementally, and it
cannot be reconstructed from a prefix. Chapter 6's separate `rem_mamba_slots` gate exists
for exactly this reason.

**`:4348` `DSATokenToKVPool`** and **`:4671` `MiniMaxSparseKVPool`** — sparse attention,
where only selected tokens are attended to (Chapter 14).

**`:2867` `NoOpMHATokenToKVPool`** — stores nothing. For disaggregated prefill (Chapter 18),
where KV is streamed to a decode node instead of retained.

`:1668` `_finalize_allocation_log` produces the `KV Cache is allocated. #tokens: ...` line
in the startup log, which is the single most useful number for capacity planning.

---

## From a percentage to a number

`--mem-fraction-static` is the flag people tune first and understand last. The chain is:

1. `python/sglang/srt/model_executor/model_runner.py:1057` `load_model` puts weights on the
   device.
2. `:1322` `configure_kv_cache_dtype` fixes bytes per entry (Chapter 15).
3. `:807` `alloc_memory_pool` computes what remains and converts it to a token count.
4. `python/sglang/srt/mem_cache/kv_cache_configurator.py` and
   `python/sglang/srt/mem_cache/allocation_sizing.py` do the arithmetic, and
   `python/sglang/srt/mem_cache/kv_cache_dtype.py` supplies the per-token size.

The result is `max_total_num_tokens` — the budget Chapter 6 spends. Not a request count:
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

<figure>
<svg viewBox="0 0 700 330" role="img" aria-label="Two levels of indirection from a request to physical KV storage">
<title>Address translation, request to KV storage</title>
<rect class="dgm-box-accent" x="20" y="56" width="140" height="62" rx="6"/>
<text class="dgm-label" x="90.0" y="82.4" text-anchor="middle" font-weight="600" style="font-size:13.0px">request r</text>
<text class="dgm-small" x="90.0" y="100.4" text-anchor="middle" style="font-size:11.5px">req_pool_idx = 7</text>
<rect class="dgm-box" x="196" y="40" width="228" height="94" rx="6"/>
<text class="dgm-label" x="310.0" y="64.4" text-anchor="middle" font-weight="600" style="font-size:13.0px">ReqToTokenPool</text>
<text class="dgm-small" x="310.0" y="82.4" text-anchor="middle" style="font-size:11.5px">req_to_token[7, t] → kv_index</text>
<text class="dgm-small" x="310.0" y="100.4" text-anchor="middle" style="font-size:11.5px">one dense int32 row per request</text>
<text class="dgm-small" x="310.0" y="118.4" text-anchor="middle" style="font-size:11.5px">length = max_context_len</text>
<rect class="dgm-box" x="460" y="40" width="220" height="94" rx="6"/>
<text class="dgm-label" x="570.0" y="64.4" text-anchor="middle" font-weight="600" style="font-size:13.0px">allocator</text>
<text class="dgm-small" x="570.0" y="82.4" text-anchor="middle" style="font-size:11.5px">kv_index ÷ page_size → page</text>
<text class="dgm-small" x="570.0" y="100.4" text-anchor="middle" style="font-size:11.5px">kv_index mod page_size → slot</text>
<text class="dgm-small" x="570.0" y="118.4" text-anchor="middle" style="font-size:11.5px">pages need not be contiguous</text>
<rect class="dgm-box-accent" x="196" y="200" width="484" height="74" rx="6"/>
<text class="dgm-label" x="438.0" y="223.4" text-anchor="middle" font-weight="600" style="font-size:13.0px">KVCache</text>
<text class="dgm-small" x="438.0" y="241.4" text-anchor="middle" style="font-size:11.5px">k_buffer[layer][kv_index]   ·   v_buffer[layer][kv_index]</text>
<text class="dgm-small" x="438.0" y="259.4" text-anchor="middle" style="font-size:11.5px">one tensor pair per layer, shared by every request</text>
<text class="dgm-small" x="310.0" y="28" text-anchor="middle" style="font-size:11.5px">level 1 — where are my tokens?</text>
<text class="dgm-small" x="570.0" y="28" text-anchor="middle" style="font-size:11.5px">index → address</text>
<text class="dgm-small" x="438.0" y="188" text-anchor="middle" style="font-size:11.5px">level 2 — where does token index i actually live?</text>
<path class="dgm-line" d="M160 87.0 L190 87.0" marker-end="url(#arrow)"/>
<path class="dgm-line" d="M424 87.0 L454 87.0" marker-end="url(#arrow)"/>
<path class="dgm-line" d="M570.0 134 L570.0 194" marker-end="url(#arrow)"/>
<text class="dgm-small" x="350.0" y="300" text-anchor="middle" style="font-size:11.5px">Two requests can hold the same kv_index at different positions in their own rows.</text>
<text class="dgm-small" x="350.0" y="318" text-anchor="middle" style="font-size:11.5px">That is the mechanical precondition for prefix sharing.</text>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" style="fill:var(--dgm-rule)"/></marker><marker id="arrow-accent" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" style="fill:var(--dgm-accent)"/></marker></defs>
</svg>
<figcaption>Attention kernels do not walk this chain token by token: they receive the <code>req_to_token</code> row as a page table and index it inside the kernel. That is what “paged attention” names.</figcaption>
</figure>

Two lookups per token per layer. The attention backends of Chapter 14 do not walk this
chain per token — they receive the `req_to_token` slice as a *page table* and index it
inside the kernel, which is what "paged attention" names.

Every other chapter in this book is spending the resource this chapter allocates. Chapter 6
budgets it, Chapter 10 shares it, Chapter 11 tiers it, Chapter 15 shrinks it, Chapter 16
avoids replicating it, and Chapter 18 moves it between machines.

This chapter built a memory system with no memory of its own. Chapter 10 gives it one.

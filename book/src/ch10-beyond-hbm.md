# 10. Caching Beyond HBM

> *Extending the cache hierarchy to host memory and disk is a bandwidth arbitrage, and it
> only pays above a computable prefix length.*

Chapter 9's tree lives entirely in GPU memory, which means it competes for the same scarce
resource as everything else. A server with 50 GB of KV pool holds perhaps 150,000 cached
tokens; a busy deployment with long system prompts exhausts that in minutes, and after that
every eviction is future work being destroyed.

Meanwhile the machine has a terabyte or two of host memory sitting nearly idle, and possibly
an NVMe array or an object store behind that.

The obvious move is to spill the cold parts of the cache down the hierarchy. The
non-obvious part is that this is not free, and not always worth doing: loading a cached
prefix over PCIe competes with simply recomputing it on hardware that is very good at
prefill. Whether the trade pays is an arithmetic question with a computable answer, and it
depends on how long the prefix is.

This chapter covers that arbitrage, the tier bookkeeping it requires, and the invariant —
one rule about which nodes may be backed up — that makes the whole thing coherent. It also
covers the third tier, where SGLang stops implementing and starts integrating with shared
KV stores that let a whole cluster behave like one cache.

---

## The arbitrage

Chapter 9's radix tree lives entirely in GPU memory, which means it is bounded by the same
scarce resource everything else competes for. A server with 50 GB of KV pool holds perhaps
150,000 cached tokens. A busy deployment with long system prompts blows through that in
minutes, and then every eviction is future work re-created.

Meanwhile the machine has 1–2 TB of host DRAM sitting nearly idle, and possibly an NVMe
array or an object store behind that.

The question is not whether more capacity is available — it obviously is — but whether
using it beats simply recomputing. That is an arithmetic question:

- **Recompute** a prefix of *n* tokens: one prefill of *n* tokens. Compute-bound, and on a
  large model, meaningful GPU time.
- **Load** it from host: *n × bytes_per_token* over PCIe, at maybe 25 GB/s on Gen4 x16
  (Chapter 1's per-token figure was ~320 KB for a 70B model, so ~78k tokens/s).

The crossover depends on model size, batch size, PCIe generation, and whether the transfer
overlaps with compute. But its *shape* is fixed: **loading wins for long prefixes, loses for
short ones**, because transfer cost is linear in tokens while prefill has fixed overheads
and better hardware utilization. Every policy in this chapter is an approximation of that
threshold.

### The hierarchy, with numbers

Caching is only interesting when the levels differ by orders of magnitude, and here they do.
Approximate figures for a current server node:

| Tier | Capacity | Bandwidth | Latency to first byte |
| --- | --- | --- | --- |
| SRAM / registers | ~50 MB | ~20 TB/s | ~1 ns |
| HBM3 (GPU) | 80–192 GB | 3–8 TB/s | ~500 ns |
| Host DRAM over PCIe 5 | 1–2 TB | ~50 GB/s | ~2 µs |
| NVMe SSD | 10–100 TB | 2–14 GB/s | ~50 µs |
| Object store / network | unbounded | 1–25 GB/s | ~1 ms |

Each step down is roughly an order of magnitude more capacity and an order of magnitude less
bandwidth. That is precisely the shape that makes a cache hierarchy worth building — and it
is the same shape that made FlashAttention worth writing, one level up, between SRAM and
HBM. Chapter 13 is this chapter's argument applied to the top two rows.

Write the crossover down properly. Let *n* be the prefix length, *b* the bytes of KV per
token, *B* the tier's bandwidth, and `P(n)` the time to prefill *n* tokens. Fetching is
worthwhile when

```
n × b / B  <  P(n)
```

Prefill time is close to linear in *n* once past the roofline's ridge point (Chapter 1), so
both sides scale the same way and the comparison reduces to a ratio of rates: the tier's
bandwidth against the model's prefill throughput measured in bytes of KV produced per second.
For a 70B model at ~320 KB per token, host DRAM at 50 GB/s supplies about 160k tokens/s of
KV — comfortably faster than the model can generate it, so host tier hits are almost always
worth taking. NVMe at 5 GB/s supplies ~16k tokens/s, which is competitive only for large
models and long prefixes. A remote store at 1 GB/s usually is not, unless the model is very
large or the alternative is a cold prefill of tens of thousands of tokens.

Two corrections make the real decision less favourable than that arithmetic suggests.

**Fixed costs dominate at small *n*.** Every fetch pays for a lookup, an allocation, a
transfer setup, and a synchronization. Below a few hundred tokens these swamp the transfer
itself, which is why every tier in this chapter carries a minimum-size threshold rather than
fetching whatever it finds.

**Bandwidth is shared, and the thing it is shared with is the model.** PCIe is also carrying
weight updates, multimodal inputs, and logits; on a disaggregated deployment (Chapter 17) it
is carrying whole KV caches between machines. A fetch that is free in isolation may not be
free at load.

The mitigation for both is the same and it is the reason this chapter's write-back is
asynchronous: if the transfer overlaps with compute the engine would be doing anyway, its
cost is hidden exactly as Chapter 4's scheduling overhead is hidden. A prefetch issued when a
request is admitted, landing before its prefill is scheduled, costs nothing at all. A fetch
issued synchronously at the moment of need costs its full latency. Most of the engineering
in `HiRadixCache` is about staying in the first case.

---

## A radix tree with tiers

`python/sglang/srt/mem_cache/hiradix_cache.py:76` `HiRadixCache` extends `RadixCache`
directly:

```python
class HiRadixCache(RadixCache):

    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        ...
        self.kv_cache = params.token_to_kv_pool_allocator.get_kvcache()

        allocator_type = get_allocator_type(server_args)

        if isinstance(self.kv_cache, MHATokenToKVPool):
            self.token_to_kv_pool_host = get_mha_host_pool_cls(self.kv_cache)(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=allocator_type,
            )
        ...
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            ...
        else:
            raise ValueError("HiRadixCache only supports MHA, MLA, DSA, and MSA models")
```

Subclassing rather than wrapping is the key decision. The tree algorithm — match, split,
insert, reference-count, evict — is unchanged; what changes is that a node's data may live
in one of two places. Everything Chapter 9 established still holds.

The host pool is chosen by device pool type, and the `raise ValueError` is honest about the
limit: pool layouts differ enough (Chapter 8) that each needs its own host counterpart.
Note the MLA branch reads `attn_dcp_size` / `attn_dcp_rank` — with decode context
parallelism the cache is already split across ranks, and the host pool must mirror that
split.

`hicache_ratio` and `hicache_size` size the host pool; `hicache_mem_layout` picks the layout
Chapter 8 introduced, and page-major matters more here than on the device because a
contiguous page is a single DMA rather than a scatter.

### The tier state on a node

Chapter 9 flagged four `TreeNode` fields as belonging here:

- `python/sglang/srt/mem_cache/radix_cache.py:268` `evicted` — `value is None`. The node
  exists, its device data does not.
- `python/sglang/srt/mem_cache/radix_cache.py:272` `backuped` — `host_value is not None`.
  There is a host copy.
- `python/sglang/srt/mem_cache/radix_cache.py:276` `protect_host` / `:280` `release_host` —
  a second reference count, for the host tier.

The two flags are independent, and their four combinations are the tier states:

| `evicted` | `backuped` | Meaning |
| --- | --- | --- |
| no | no | device only — a plain Chapter 9 node |
| no | yes | on device *and* backed up — free to evict from device at no loss |
| yes | yes | host only — must be loaded before use |
| yes | no | nothing left; the node is structure only |

The second row is what the whole design is for. A node that has been written through to
host costs nothing to evict from the device, because the data is still reachable. Eviction
stops being destructive and becomes a demotion.

That is also why `protect_host` is a *separate* counter from `lock_ref`. A node may have no
running request holding it (so `lock_ref == 0`, device-evictable) while an in-flight
transfer is reading its host copy. Reusing one counter would either block eviction that
should proceed or free host memory mid-DMA.

---

## Writing back

`python/sglang/srt/mem_cache/hiradix_cache.py:840` `write_backup` promotes a node's data to the host tier:

```python
    def write_backup(self, node: TreeNode, write_back=False) -> int:
        # Backup invariant (for write-through mode): backed-up nodes must form a
        # contiguous prefix from root — no gaps.  Skip if parent isn't backed
        # up yet;
        if not write_back and (
            node.parent != self.root_node and not node.parent.backuped
        ):
            return 0

        host_indices = self.cache_controller.write(
            device_indices=node.value,
            node_id=node.id,
            **self._get_extra_pools(),
        )
        if host_indices is None:
            self.evict_host(len(node.value))
            host_indices = self.cache_controller.write(...)
        if host_indices is not None:
            node.host_value = host_indices.clone()
            assert len(node.host_value) > 0
            self._track_write_through_node(node, len(node.key))
            if not write_back:
                self.inc_lock_ref(node)
        else:
            return 0
```

**The backup invariant is the load-bearing rule.** Backed-up nodes must form a contiguous
prefix from the root, with no gaps. The reason is what a cache hit means: to serve a prefix
from the host tier you need *every* node along the path, because a prefix is only valid if
all of it is present. A node whose parent is not backed up is useless — you could never
assemble a complete prefix through it. So the write is simply skipped.

The retry structure is a standard allocate-or-evict, and the `inc_lock_ref` at the end is
subtle: while a write-through is in flight, the *device* node must not be evicted, because
the DMA is reading from it. Chapter 9's device-side lock is reused for a transfer that has
nothing to do with a request.

`write_back=True` is the opposite direction — a device eviction pushing data down rather
than a background promotion — and it skips both the invariant check and the lock, because
the node is already on its way out.

`python/sglang/srt/mem_cache/hiradix_cache.py:872` `_track_write_through_node` and
`python/sglang/srt/mem_cache/hiradix_cache.py:904` `_finish_write_through_ack` handle completion.
Transfers are asynchronous, so the tree must track outstanding writes and only mark a node
`backuped` once the data has actually landed. `python/sglang/srt/mem_cache/hiradix_cache.py:876`
`_replace_pending_write_through_node` covers the case where the tree changes shape — a
split (Chapter 9) — while a write against the old node is still in flight.

---

## Moving the bytes

`python/sglang/srt/mem_cache/memory_pool_host.py` holds the host-side pools, mirroring the
device pools of Chapter 8 with pinned memory so DMA can proceed without staging.

`python/sglang/srt/managers/cache_controller.py` is the transfer engine — queues,
worker threads, and the overlap with compute that makes the arbitrage viable. A transfer
that stalls the forward pass would defeat the purpose; the controller's job is to keep
copies in flight alongside GPU work.

The synchronization requirements are heavier than they look, because every tensor-parallel
rank holds a shard of the same logical cache. `python/sglang/srt/mem_cache/hiradix_cache.py:217` `_all_reduce_attn_groups`,
`python/sglang/srt/mem_cache/hiradix_cache.py:226` `_barrier_attn_groups`,
`python/sglang/srt/mem_cache/hiradix_cache.py:247` `_all_reduce`, and `python/sglang/srt/mem_cache/hiradix_cache.py:260` `_pp_sync` exist so that ranks
agree on which nodes are backed up. If rank 0 believed a prefix was on the host and rank 1
did not, the two would take different paths through the scheduler — Chapter 3's
divergence hazard, one layer down.

---

## Pluggable storage

Beyond host DRAM is a third tier, and here SGLang stops implementing and starts
integrating.

`python/sglang/srt/mem_cache/hicache_storage.py` defines the interface;
`python/sglang/srt/mem_cache/storage/backend_factory.py` constructs backends. The
implementations under `python/sglang/srt/mem_cache/storage/` span local options (`file/`,
`mmap/`, `shm/`) and distributed KV stores (`mooncake_store/`, `hf3fs/`, `nixl/`,
`lmcache/`, `eic/`, `aibrix_kvcache/`).

The distributed backends change what caching *means*. A shared KV store makes a prefix
computed on node A available to node B — so a system prompt is prefilled once per cluster
rather than once per replica. Combined with Chapter 17's cache-aware routing, the cluster
starts behaving like one large cache rather than *n* independent ones.

Attachment is a runtime operation, not a startup flag:

- `python/sglang/srt/mem_cache/hiradix_cache.py:369` `attach_storage_backend`
- `python/sglang/srt/mem_cache/hiradix_cache.py:487` `detach_storage_backend`
- `python/sglang/srt/mem_cache/hiradix_cache.py:516` `_force_release_pending_storage_ops`
- `python/sglang/srt/mem_cache/hiradix_cache.py:818` `clear_storage_backend`

with HTTP endpoints at `python/sglang/srt/entrypoints/http_server.py:1060`
`attach_hicache_storage_backend` and `:1094` `detach_hicache_storage_backend`. Storage
backends fail — a network store goes away, a mount hangs — and an engine that had to
restart to recover would turn a degraded cache into an outage. Detach must therefore drain
in-flight operations safely, which is what `_force_release_pending_storage_ops` is for.

`python/sglang/srt/mem_cache/hiradix_cache.py:684` `_parse_storage_backend_extra_config` handles
per-backend configuration, and `python/sglang/srt/mem_cache/hiradix_cache.py:317`
`_apply_storage_runtime_config` applies it live.

---

## Where it is heading

`python/sglang/srt/mem_cache/unified_cache/` is a newer structure generalizing all of this:
`python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py` with a component
registry (`python/sglang/srt/mem_cache/unified_cache/component_type.py`,
`python/sglang/srt/mem_cache/unified_cache/tree_core_registry.py`) and explicit cache
actions (`python/sglang/srt/mem_cache/unified_cache/cache_action.py`).

The motivation is visible in this chapter. `HiRadixCache` hardcodes two tiers with a third
bolted on, and the pool-type dispatch in `__init__` grows a branch per pool variant. The
unified core makes tiers and pool types compositional instead. Chapter 9's variants —
`python/sglang/srt/mem_cache/swa_radix_cache.py` and
`python/sglang/srt/mem_cache/mamba_radix_cache.py` — are the same pressure
from the other direction.

Of everything in Part III, this is the area most likely to look different by the time you
read the code.

---

## What it costs

Honest accounting, since HiCache is not free:

- **Host memory** — `hicache_ratio` is a real reservation, pinned so it cannot be paged.
- **PCIe bandwidth** — shared with weight loading, multimodal inputs, and Chapter 17's KV
  transfers.
- **Complexity** — every invariant in Chapter 9 now has a tier dimension, and every
  transfer is asynchronous.
- **Cross-rank synchronization** — the barriers above are on the critical path.

The gain is a cache measured in millions of tokens rather than hundreds of thousands. For a
workload with long shared prefixes and a working set larger than HBM — the case HiCache is
designed for — that is the difference between an 80% hit rate and a 20% one. For a workload
of short, unique prompts, it is overhead with no benefit.

`docs/docs/advanced_features/hicache_design.mdx` and
`docs/docs/advanced_features/hicache_best_practices.mdx` carry the project's own tuning
guidance.

---

Part III ends here. Part IV goes into what actually consumes this memory: the model.

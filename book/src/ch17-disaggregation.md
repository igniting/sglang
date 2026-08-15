# 17. Disaggregation and Routing

> *Prefill and decode want different hardware and different SLOs, so at scale the right move
> is to stop running them on the same machine.*

Chapter 1 showed that prefill and decode are opposite workloads. Chapter 5 showed the
scheduler negotiating between them every single iteration, and chunked prefill softening the
conflict without removing it.

This chapter is about giving up on the negotiation.

Disaggregation refuses the compromise: separate fleets of prefill and decode instances, each
configured for its own job, with the KV cache shipped between them over the network. The
cost is that the shipping is now on the critical path — a 4,000-token prefill produces over
a gigabyte of cache that has to arrive before the first output token.

The chapter ends one level further out, with the router that decides which instance gets
each request. That decision turns out to depend on Chapter 9, and the dependence is strong
enough that routing policy can matter more than any kernel in this book.

---

## Why colocation compromises both

Chapter 1's two phases have opposite characteristics, and Chapter 5 showed the scheduler
negotiating between them every iteration. Chunked prefill mitigates the conflict; it does
not remove it.

Look at what each phase actually wants:

| | Prefill | Decode |
| --- | --- | --- |
| Bottleneck | compute | memory bandwidth |
| Wants | large batches, high FLOPs | many concurrent sequences, large KV pool |
| Metric | TTFT | ITL |
| Best parallelism | TP for compute | DP attention for cache capacity |
| Batch shape | few requests, many tokens | many requests, one token each |

Every row is a conflict. On one machine you must pick a middle setting for each and be
mediocre at both. A GPU sized for decode's memory needs is over-provisioned in memory for
prefill; a parallelism layout tuned for prefill's compute replicates the cache that decode
depends on.

Disaggregation resolves it by refusing the compromise: **prefill instances and decode
instances, each configured for its own job**, with KV cache transferred between them.

The cost is that the transfer is now on the critical path. A 4,000-token prefill produces
1.25 GB of KV (Chapter 1's figure for a 70B model) that must reach the decode node before
the first output token. That is why this chapter is mostly about the transfer.

### The interference, measured

Zhong et al.'s DistServe (OSDI '24) is the paper that made this case, and its central
measurement is worth restating because it is sharper than the table above.

Add **one** prefill request to a batch of decodes, and both get worse. The decodes wait for
the prefill's much longer forward pass, so ITL spikes for every sequence in the batch — one
long prompt taxes every user in the batch, whether or not they asked for anything. And the
prefill itself is slowed by the decode rows sharing its kernels. Neither phase gets what it
wants, and the damage is proportional to how heterogeneous the batch is — which continuous
batching, by design, maximizes.

The reason this is worse than it sounds is that the two phases are graded on different
scales. Prefill is measured by **TTFT**, decode by **TPOT/ITL**, and an SLO is normally
stated as both: "95% of requests under 500 ms to first token and under 50 ms between
tokens." **Goodput** — the request rate at which both targets are met — is the honest metric,
and it is not a function of average throughput. A colocated deployment can post excellent
aggregate tokens per second while missing one SLO or the other on most requests, because the
interference lands unevenly.

Once you accept two pools, a second freedom appears that a single pool cannot have: **the two
phases can use different parallelism.** Prefill is compute-bound, so tensor parallelism buys
it real latency reduction. Decode is bandwidth-bound and cache-hungry, so replicating the KV
cache across TP ranks is precisely wrong for it, and Chapter 15's data-parallel attention is
right. On one machine you must choose one layout for both. DistServe's placement algorithm
searches the two configurations independently and reports **7.4× the request rate**, or 12.6×
tighter SLOs, at 90% attainment.

The obvious objection is the transfer, and the paper's answer is arithmetic. The KV for a
request is `bytes_per_token × prompt_length`, and it is transferred exactly once, against a
prefill that took tens of milliseconds of GPU time to produce. For OPT-175B they measure the
transfer at **under 0.1%** of total request latency — provided it crosses a fast link. Which
is the real constraint: the placement algorithm is bandwidth-aware precisely because the
conclusion inverts on a slow one. Given NVLink between the pools, disaggregation is nearly
free; given commodity Ethernet, the transfer becomes the bottleneck it appears to be.

This is the counter-argument to Chapter 5's chunked prefill, and the two papers genuinely
disagree. Sarathi says: interleave them, and the decodes ride along on compute the prefill
was buying anyway. DistServe says: interleaving is what causes the interference, chunking
only bounds it, and chunking makes attention re-read the prefix once per chunk — a cost that
grows quadratically with context length. Both are correct in their own regime. Short prompts
and modest SLOs favour chunking, which needs one pool and no network. Long prompts, strict
TTFT targets, and a fast interconnect favour disaggregation. SGLang implements both, and the
choice is a deployment decision rather than an architectural one.

---

## The two sides

`python/sglang/srt/disaggregation/prefill.py:119` `PrefillBootstrapQueue` holds requests
that have arrived but whose transfer path is not yet established:

```python
class PrefillBootstrapQueue:
    """
    Store the requests in bootstrapping
    """

    def __init__(
        self,
        token_to_kv_pool: KVCache,
        draft_token_to_kv_pool: Optional[KVCache],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        ...
        transfer_backend: TransferBackend,
    ):
```

The constructor names the moving parts. `token_to_kv_pool` is Chapter 8's pool — the source
of the bytes. `metadata_buffers` carry the small out-of-band information (sampled token,
finish state) that must accompany the KV. `draft_token_to_kv_pool` is there because
speculative decoding (Chapter 18) has a second cache that must also cross.

`:299` `create_sender`, `:326` `ensure_metadata_buffer`, and `:336` `finalize_bootstrap`
are the handshake steps, and `:383` `pop_bootstrapped` yields requests ready to run.
`:367` `_check_if_req_exceed_kv_capacity` rejects early: a request too large for the decode
node's pool must fail before prefill spends compute on it.

`:485` `SchedulerDisaggregationPrefillMixin` is how this attaches to Chapter 4, and the
important detail is that it provides *variant event loops*:

```
:569  event_loop_normal_disagg_prefill
:607  event_loop_overlap_disagg_prefill
:543  get_next_disagg_prefill_batch_to_run
:501  resolve_waiting_queue_bootstrap
```

Same structure as Chapter 4 — receive, decide, run, process — with two changes. Requests
must be bootstrapped before they are schedulable, and a request is *finished* when its KV
has been sent rather than when it produced a token. A prefill instance never decodes.

`python/sglang/srt/disaggregation/decode.py` is the mirror. Its requests arrive with KV
already computed, which is Chapter 6's `PREBUILT` forward mode:

> Used in disaggregated decode worker. Represent a batch of requests having their KV cache
> ready to start decoding.

The decode side has its own queues — waiting for transfer, transfer complete, ready to run —
and `python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py`,
`python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py`, and
`python/sglang/srt/disaggregation/decode_hicache_mixin.py` connect it to Chapters 8 and 10.

---

## The handshake

Before a byte moves, the two sides must agree on where it goes. The decode instance
allocates KV pages from *its own* pool (Chapter 8) and tells the prefill instance the
addresses; the prefill instance writes directly into them.

`python/sglang/srt/disaggregation/base/conn.py` defines the interface and
`python/sglang/srt/disaggregation/common/conn.py` holds the shared implementation —
`CommonKVManager`, plus sender and receiver roles.
`python/sglang/srt/managers/disagg_service.py` runs the bootstrap service that puts the two
sides in touch, and `python/sglang/srt/disaggregation/common/staging_buffer.py` and
`python/sglang/srt/disaggregation/common/staging_handler.py` handle cases where a direct write is not possible.

Two details make this harder than it sounds. **Layout must match**: Chapter 8's pool layouts
differ, and a page-major sender writing into a layer-major receiver produces garbage. And
**parallelism may differ**: a prefill instance running TP 8 and a decode instance running TP
4 with DP attention hold their caches in different shard shapes, so the transfer is a
redistribution, not a copy. That is a large part of why the connection layer is as
substantial as it is.

---

## Moving KV between machines

The transfer backends map onto the RDMA landscape:

- `python/sglang/srt/disaggregation/mooncake/` — Mooncake's transfer engine, the most
  commonly deployed.
- `python/sglang/srt/disaggregation/nixl/` — NVIDIA's NIXL.
- `python/sglang/srt/disaggregation/mori/` — AMD.
- `python/sglang/srt/disaggregation/ascend/` — Ascend NPUs.
- `python/sglang/srt/disaggregation/fake/` — the test double.

The `fake` backend deserves attention. It implements the interface with no network at all,
which makes the control flow — bootstrap, handshake, transfer, completion — testable
without a cluster. When reading this subsystem, start there: it is the version where you
can see the state machine without RDMA obscuring it.

`python/sglang/srt/disaggregation/kv_events.py` publishes cache events, which is what lets
an external system (the router below) know what is cached where.

---

## EPD: a third stage

Multimodal models (Chapter 20) add another asymmetric phase. Running a vision encoder is
compute-heavy, bursty, and needs no KV cache at all — a third workload with a third profile.

`python/sglang/srt/disaggregation/encode_server.py` and
`python/sglang/srt/disaggregation/encode_grpc_server.py` run encoding as its own service,
with `python/sglang/srt/disaggregation/encode_receiver.py` on the consuming side. The
`--encoder-only` flag in `python/sglang/launch_server.py` (Chapter 2) selects it.

`docs/docs/advanced_features/epd_disaggregation.mdx` covers the three-stage arrangement.

---

## Routing in front of it all

Disaggregation multiplies instances, and something must decide which one gets each request.
`sgl-model-gateway/` is that something, in Rust.

Round-robin is the obvious policy and the wrong one, **because Chapter 9 exists**. If
request *B* shares a prefix with request *A*, sending it to the replica that already served
*A* turns a full prefill into a cache hit. Sending it elsewhere throws that away. At an 80%
prefix-sharing rate, routing policy is worth more than any kernel optimization in this book.

`sgl-model-gateway/src/policies/` holds the implementations:

```
cache_aware.rs        route by prefix locality — the interesting one
tree.rs               the router's own approximate radix tree
prefix_hash.rs        hash-based prefix affinity
consistent_hashing.rs stable assignment under membership change
power_of_two.rs       sample two, pick the less loaded
round_robin.rs        the baseline
random.rs             the other baseline
bucket.rs             bucketed assignment
```

`sgl-model-gateway/src/policies/tree.rs` is the key file: the router maintains its *own* radix tree, approximating what
each replica has cached. It cannot be exact — it does not see evictions — but it does not
need to be. It only needs to rank replicas well enough that requests land near their
prefixes.

This is Chapter 9's feedback loop lifted one level. Within an instance, the tree tells the
scheduler which request is cheapest to run next. Across instances, an approximate copy of
the same tree tells the router which replica makes a request cheapest. Same idea, two
scales.

`sgl-model-gateway/src/policies/power_of_two.rs` is the counterweight: pure cache affinity would send every request with a
popular system prompt to one replica and overload it. Sampling two candidates and choosing
the less loaded blends locality with balance.

`sgl-model-gateway/src/routers/` holds the request handling — `http/`, `grpc/`, `openai/`,
plus `sgl-model-gateway/src/routers/router_manager.rs` for multi-model serving and
`sgl-model-gateway/src/routers/mesh/` for distributed topologies.
`sgl-model-gateway/src/service_discovery.rs` tracks which instances exist, which matters
because in a disaggregated deployment they are not interchangeable: prefill and decode
instances are different pools with different roles.

### Why Rust

The router sits in front of every request and does non-trivial work per request — prefix
hashing, tree lookup, load comparison. In Python that would be a bottleneck at the request
rates this is built for, and it would inherit the GIL problems Chapter 2 described. It also
has no reason to touch a GPU, so nothing pulls it toward the Python ecosystem.

`sgl-model-gateway/bindings/` exposes it to Python for embedded use, and
`docs/docs/advanced_features/sgl_model_gateway.mdx` covers deployment.

---

## What a full deployment looks like

<figure>
<svg viewBox="0 0 700 356" role="img" aria-label="A cache-aware router in front of separate prefill and decode pools">
<title>A disaggregated deployment</title>
<rect class="dgm-box-accent" x="160" y="52" width="380" height="68" rx="6"/>
<text class="dgm-label" x="350.0" y="72.4" text-anchor="middle" font-weight="600" style="font-size:13.0px">sgl-model-gateway</text>
<text class="dgm-small" x="350.0" y="90.4" text-anchor="middle" style="font-size:11.5px">Rust · cache-aware routing over an approximate</text>
<text class="dgm-small" x="350.0" y="108.4" text-anchor="middle" style="font-size:11.5px">radix tree · service discovery · PD-aware placement</text>
<rect class="dgm-box" x="20" y="180" width="290" height="96" rx="6"/>
<text class="dgm-label" x="165.0" y="205.4" text-anchor="middle" font-weight="600" style="font-size:13.0px">Prefill pool</text>
<text class="dgm-small" x="165.0" y="223.4" text-anchor="middle" style="font-size:11.5px">compute-bound · large batches</text>
<text class="dgm-small" x="165.0" y="241.4" text-anchor="middle" style="font-size:11.5px">TP for arithmetic throughput</text>
<text class="dgm-small" x="165.0" y="259.4" text-anchor="middle" style="font-size:11.5px">never runs a decode step</text>
<rect class="dgm-box" x="390" y="180" width="290" height="96" rx="6"/>
<text class="dgm-label" x="535.0" y="205.4" text-anchor="middle" font-weight="600" style="font-size:13.0px">Decode pool</text>
<text class="dgm-small" x="535.0" y="223.4" text-anchor="middle" style="font-size:11.5px">bandwidth-bound · many sequences</text>
<text class="dgm-small" x="535.0" y="241.4" text-anchor="middle" style="font-size:11.5px">DP attention · large KV pool</text>
<text class="dgm-small" x="535.0" y="259.4" text-anchor="middle" style="font-size:11.5px">receives cache, never prefills</text>
<text class="dgm-label" x="350.0" y="30" text-anchor="middle" font-weight="600" style="font-size:13px">clients</text>
<path class="dgm-line" d="M350.0 36 L350.0 46" marker-end="url(#arrow)"/>
<path class="dgm-line" d="M260.0 120 L260.0 152 L165.0 152 L165.0 174" marker-end="url(#arrow)"/>
<path class="dgm-line" d="M440.0 120 L440.0 152 L535.0 152 L535.0 174" marker-end="url(#arrow)"/>
<path class="dgm-line-accent" d="M310 228.0 L384 228.0" marker-end="url(#arrow-accent)"/>
<text class="dgm-small" x="350" y="216.0" text-anchor="middle" style="font-size:11.5px">KV cache</text>
<text class="dgm-small" x="350" y="250.0" text-anchor="middle" style="font-size:11.5px">RDMA</text>
<text class="dgm-small" x="350.0" y="306" text-anchor="middle" style="font-size:11.5px">The transfer sits on the critical path: a 4,000-token prefill produces over a gigabyte</text>
<text class="dgm-small" x="350.0" y="324" text-anchor="middle" style="font-size:11.5px">of cache that must land before the decode pool can emit a first token.</text>
<text class="dgm-small" x="350.0" y="348" text-anchor="middle" style="font-size:11.5px">Every box here is a chapter — pools (8), routing tree (9), parallelism (15, 16).</text>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" style="fill:var(--dgm-rule)"/></marker><marker id="arrow-accent" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" style="fill:var(--dgm-accent)"/></marker></defs>
</svg>
<figcaption>Prefill and decode want opposite hardware and opposite parallelism. Disaggregation stops asking one machine to be good at both.</figcaption>
</figure>

Every box is a chapter. The pools are Chapter 8, the routing is Chapter 9, the parallelism
choices are Chapters 15 and 16, and the transfer is this one.

Whether it is worth it is a scale question. Disaggregation adds a network hop to every
request's critical path and a great deal of operational complexity. Below a few dozen GPUs
that is a bad trade. At the scale the LMSYS blog posts describe — 96 H100s, GB200 racks —
the prefill/decode conflict is the binding constraint, and removing it is worth more than
anything else available.

Part V ends here, and with it the story of making the engine bigger. Part VI is about making
it do more.

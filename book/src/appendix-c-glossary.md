# Appendix C — Glossary

Chapter references point at where the term is explained, not merely mentioned.

---

**Arithmetic intensity** — FLOPs performed per byte moved from memory. The measure that
makes decode memory-bound and prefill compute-bound. *Ch. 1*

**Batch invariance** — the property that an operator produces bit-identical results
regardless of batch shape. Required for deterministic inference. *Ch. 8*

**Chunked prefill** — splitting a long prompt across scheduler iterations so it does not
stall active decodes. Trades the long request's TTFT for everyone else's ITL. *Ch. 6*

**Continuous batching** — admitting and retiring requests every step rather than running a
batch to completion. Really an admission-control problem under a memory budget. *Ch. 6*

**CP (context parallelism)** — splitting a sequence across ranks. *DCP* is the
decode-time variant that splits the KV cache within a TP group. *Ch. 16*

**CUDA graph** — a recorded sequence of GPU operations replayed with one launch, removing
per-kernel CPU overhead. Requires static shapes and stable buffer addresses. *Ch. 15*

**DP attention (data-parallel attention)** — splitting attention by *sequence* rather than
by head, so each rank stores only its own KV. Exists because MLA's compressed cache would
otherwise be replicated across TP ranks. Not to be confused with the DP controller, which
routes across whole replicas. *Ch. 16*

**EAGLE** — a speculative decoding algorithm whose draft head consumes the target model's
hidden states rather than running an independent small model. *Ch. 19*

**EP (expert parallelism)** — splitting MoE experts across ranks. Makes the step
all-to-all-bound. *Ch. 17*

**EPLB (expert-parallel load balancing)** — rebalancing or replicating experts to even out
skewed routing. *Ch. 17*

**Extend** — SGLang's name for prefill, reflecting that the KV cache of a prefix may already
exist. The `ForwardMode` is `EXTEND`. *Ch. 7*

**Forward mode** — the enum that classifies a batch (`EXTEND`, `DECODE`, `MIXED`, `IDLE`,
`TARGET_VERIFY`, `PREBUILT`, …) and drives nearly every downstream branch. *Ch. 7*

**GQA (grouped-query attention)** — several query heads sharing one KV head, reducing cache
size. Constrains how far TP can shard before KV heads must be replicated. *Ch. 13*

**Goodput** — throughput counting only requests that met their latency targets. *Ch. 1*

**HiCache** — the hierarchical cache: GPU → host DRAM → storage tiers beneath the radix
tree. *Ch. 11*

**ITL / TPOT (inter-token latency / time per output token)** — the gap between successive
streamed tokens. *Ch. 1*

**Jump-forward decoding** — emitting tokens without a forward pass when the grammar admits
exactly one continuation. *Ch. 20*

**KV cache** — stored keys and values for previous tokens, turning O(n²) recompute into O(n)
memory. The scarce resource the whole engine is organized around. *Ch. 1, 8*

**LoRA** — a low-rank weight update served as a per-request correction, so many adapters
share one batch. *Ch. 21*

**LPM (longest prefix match)** — a cache-aware scheduling policy that runs the request with
the largest cached prefix first. *Ch. 6*

**MLA (multi-head latent attention)** — DeepSeek's compressed KV representation: one latent
vector per token instead of per-head keys and values. *Ch. 9, 13*

**MoE (mixture of experts)** — many expert MLPs with a router activating a few per token.
*Ch. 17*

**mrope** — multi-dimensional rotary embedding, giving vision-language tokens
(temporal, height, width) positions. *Ch. 21*

**MTP (multi-token prediction)** — draft heads shipped with a model, used for speculation.
*Ch. 19*

**NSA / DSA** — sparse attention with a learned indexer selecting which tokens to attend to.
*Ch. 14*

**Overlap scheduler** — the "zero-overhead batch scheduler": preparing step *N+1* on the CPU
while step *N* runs on the GPU. *Ch. 5*

**Page / page size** — the KV allocation unit. Larger pages mean less metadata and more
internal fragmentation. *Ch. 9*

**PD disaggregation** — running prefill and decode on separate instances with KV transferred
between them. *Ch. 18*

**PP (pipeline parallelism)** — splitting layers across ranks. Cheap communication, at the
cost of pipeline bubbles. *Ch. 16*

**Prefill / decode** — the two phases of generation. Prefill processes the prompt
(compute-bound); decode produces tokens one at a time (memory-bandwidth-bound). *Ch. 1*

**RadixAttention** — SGLang's prefix caching: a radix tree over token sequences whose nodes
hold KV cache indices. *Ch. 10*

**Retraction** — evicting a *running* request when decode memory runs out, returning it to
the waiting queue and discarding its computed KV. *Ch. 6*

**SWA (sliding-window attention)** — attending only to a recent window, bounding cache
growth. Needs its own pool and cache variants. *Ch. 9, 9*

**TBO (two-batch overlap)** — splitting a batch so one half's communication overlaps the
other half's computation. Used to hide MoE all-to-all. *Ch. 17*

**TP (tensor parallelism)** — splitting weights within each layer. Two collectives per
transformer block; needs fast interconnect. *Ch. 16*

**TTFT (time to first token)** — arrival to first streamed token: queueing plus prefill.
*Ch. 1*

**Structural tag** — constraining generation directly to a tool-call schema, converging
constrained decoding and tool parsing. *Ch. 20*

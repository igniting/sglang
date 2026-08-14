# 17. Disaggregation and Routing

> *Prefill and decode want different hardware and different SLOs, so at scale the right move is to stop running them on the same machine.*

> **Status: outline.** This chapter is planned but not yet written. The beats below are the writing plan — each one pairs an idea with the code that implements it.

## Beats

1. **Why colocation compromises both.** Prefill wants large batches and compute; decode
   wants low latency and bandwidth. Chunked prefill (Chapter 5) mitigates the conflict;
   disaggregation removes it.

2. **The two sides.** `python/sglang/srt/disaggregation/prefill.py:119`
   `PrefillBootstrapQueue`, `:485` `SchedulerDisaggregationPrefillMixin`, `:569`
   `event_loop_normal_disagg_prefill`; and `python/sglang/srt/disaggregation/decode.py`
   for the mirror image. Note these are *variant event loops* — the Chapter 4 loop,
   respecialized.

3. **The handshake.** How a decode instance learns where its KV lives:
   `python/sglang/srt/disaggregation/base/conn.py`,
   `python/sglang/srt/disaggregation/common/conn.py`, and
   `python/sglang/srt/managers/disagg_service.py`.

4. **Moving KV between machines.** `python/sglang/srt/disaggregation/mooncake/`, `nixl/`,
   `mori/` — RDMA realities; `fake/` as the test double that makes the control flow
   readable.

5. **Routing in front of it all.** `sgl-model-gateway/src/policies/` — cache-aware load
   balancing beats round-robin precisely because Chapter 9 exists, and the router keeps an
   approximate radix tree to do it. `sgl-model-gateway/src/routers/`,
   `service_discovery.rs`, and why it is written in Rust.

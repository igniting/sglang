# 1. Why Serving Engines Exist

> *The cost structure of autoregressive decoding, not model quality, is what forces an engine to exist.*

> **Status: outline.** This chapter is planned but not yet written. The beats below are the writing plan — each one pairs an idea with the code that implements it.

## Beats

1. **Two phases, two bottlenecks.** Prefill is compute-bound and parallel over the prompt;
   decode is memory-bandwidth-bound and produces one token per sequence per step. The
   roofline argument for why a naive generate-loop leaves most of a GPU idle.

2. **The KV cache, and the bill it creates.** Trading O(n^2) recompute for O(n) memory;
   the sizing formula and a worked example on a 70B model showing memory — not FLOPs — as
   the binding constraint. This is the number every later chapter is fighting.

3. **The problem in SGLang's own terms.** `python/sglang/bench_one_batch.py` is the
   smallest thing in the repo that runs a real forward pass; reading its prefill and
   decode timing paths turns the abstract argument into the engine's actual measurements.

4. **What the metrics mean.** TTFT, ITL/TPOT, throughput, goodput, and how optimizing one
   degrades another — the tradeoff space every subsequent design decision sits in.

5. **The idea inventory.** `README.md:68` lists SGLang's feature set in one dense
   paragraph. Decoding it feature-by-feature into "what problem it solves and which
   chapter covers it" gives the reader a map before the descent.

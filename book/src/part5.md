# Part V — When One GPU Is Not Enough

Everything so far has assumed one graphics card.

That assumption fails early. A 70-billion-parameter model in half precision is 140 GB of
weights; the largest single GPU holds 192 GB, and that leaves nothing for the KV cache
Chapter 8 spent a chapter on. The frontier models are several times larger again.

So the work has to be split — and "split" turns out to admit several different answers that
are not interchangeable. Each cuts the problem along a different axis, and each pays a
different communication bill on a network whose speed varies by an order of magnitude
depending on whether you stay inside one machine.

**Chapter 15** covers the classical axes: splitting within each layer, splitting between
layers, and replicating whole copies. Then it covers the one SGLang added, because
DeepSeek's compressed cache broke the assumptions the classical answers were built on.

**Chapter 16** is about mixture-of-experts models, where the cost structure changes shape
entirely. The bottleneck stops being matrix multiplication and becomes *communication*: with
experts scattered across dozens of GPUs, every token has to travel to its experts and back,
twice per layer. Routing, placement, and hiding that communication become the whole game.

**Chapter 17** takes the final step. Chapter 1 showed that prefill and decode are opposite
workloads — one compute-bound, one memory-bound. Chapter 5 showed the scheduler negotiating
between them every iteration. At sufficient scale the right move is to stop negotiating: run
them on separate machines, each configured for its own job, and ship the cache between them.
The chapter ends with the routing layer that sits in front of the whole arrangement, and
which turns out to depend on Chapter 9.

These chapters describe the configurations behind the published benchmark results — the
96-GPU and rack-scale deployments. Nothing here is optional at that scale, and each piece
was added because the previous combination hit a wall.

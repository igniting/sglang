# 24. Running It

> *Everything so far has assumed the engine is already up and the traffic already arrived.
> Both assumptions cost money.*

A reader who has got this far understands the engine better than most people who operate
one. That is not the same as being able to operate one.

The gap is that every chapter until now has described a *steady state*: a process that is
running, holding weights, with requests arriving. Production is mostly the other states.
Starting up. Scaling out because traffic doubled at 9 a.m. Scaling in because it is 3 a.m.
Replacing weights without dropping requests. Noticing that a rank has wedged. Deciding how
many replicas to pay for at all.

None of that is inside the engine, and this book is about the engine — so this chapter takes
a narrow slice of a large subject. It covers the places where the engine's internals *decide*
an operational outcome: where a startup second goes and which flag governs it, which of
Chapter 22's metrics is the right autoscaling signal and why the obvious one is wrong, what
the health endpoint actually proves, and how the engine reports that it is stuck. Everything
that is genuinely about Kubernetes is left to books about Kubernetes.

<!-- objectives:begin -->
<div class="bk-objectives"><p class="bk-objectives-head">What this chapter gives you <span class="bk-objectives-time">· about 11 min</span></p><ul><li>Account for every second of a cold start and name the flag that governs it</li><li>Choose an autoscaling signal, and say why GPU utilization is not one</li><li>Explain what the health endpoint proves that a liveness probe does not</li><li>Decide between replacing a replica and replacing its weights</li></ul></div>
<!-- objectives:end -->

---

## Where a cold start goes

The single most consequential operational number is how long it takes a fresh replica to
serve its first token, because it bounds how fast you can scale and therefore how much
headroom you must pay for continuously. It is also the number people most often try to
improve by guessing.

The startup sequence is Chapter 3's launch path, and every phase in it has appeared in this
book already:

| Phase | Roughly | Governed by | Chapter |
| --- | --- | --- | --- |
| Container image pull | 0–300 s | image size, registry locality | — |
| Process launch, port allocation, ZMQ setup | ~1 s | — | 3 |
| Distributed init (NCCL rendezvous) | 2–20 s | parallel size, network | 16 |
| Weight load | 10–300 s | checkpoint bytes ÷ storage bandwidth | 12 |
| Memory pool allocation | 1–5 s | `--mem-fraction-static` | 9 |
| CUDA graph capture | 10–120 s | `--cuda-graph-max-bs`, bucket count | 15 |
| Warmup request | 1–10 s | — | — |

Two of those dominate, and they are dominated for different reasons.

**Weight load is bandwidth-bound and mostly not the engine's fault.** Chapter 12 established
the floor: bytes over the slowest link on the path. A 140 GB checkpoint pulled from object
storage at 1 GB/s is 140 seconds no matter how good the loader is. The fixes are all about
the bytes and the path — a quantized checkpoint is a smaller checkpoint, a local or
same-region cache beats a remote bucket, and safetensors' zero-copy layout means the format
is not adding a second pass over the data. The one engine-side lever is that ranks load in
parallel, so the wall-clock cost is a shard rather than the whole model.

**Graph capture is pure compute and entirely the engine's fault.** Chapter 15 records a
separate graph per batch-size bucket, and each capture runs a full forward pass. Forty
buckets is forty forward passes plus allocation, which on a large model is a minute or two of
a replica doing nothing useful. `--cuda-graph-max-bs` and the bucket list are the direct
levers, and they are a real trade: fewer buckets means faster starts and more padding waste
at serve time.

The order matters too, and Chapter 3's `_launch_subprocesses` fixes it — the pool cannot be
sized before the weights are resident, because the pool is what is left over, and graphs
cannot be captured before the pool exists, because they capture kernels that address it. This
is why startup failures are usually reported at a phase *later* than the one that caused
them: an over-optimistic `--mem-fraction-static` does not fail at pool allocation, it fails
during capture, or on the first large batch.

**Scale-to-zero is a function of this table.** If a cold start is 30 seconds, scaling to zero
means a 30-second first request, which is fine for a batch job and unacceptable for a chat
product. The honest way to decide is to add the rows for your model rather than to reason
about it abstractly.

---

## What "ready" means, and what it proves

A replica that has finished starting is not necessarily a replica that works, and the
distinction shows up in what the health endpoint does.

`python/sglang/srt/entrypoints/http_server.py:648` `health_generate` is unusual for a health
check — it does not return 200 after checking that a flag is set. It runs a real generation:

```python
async def health_generate(request: Request) -> Response:
    """
    Check the health of the inference server by sending a special request to generate one token.

    If the server is running something, this request will be ignored, so it creates zero overhead.
    If the server is not running anything, this request will be run, so we know whether the server is healthy.
    """
```

The design is worth understanding because it is the correct answer to a genuinely hard
problem. In Chapter 3's topology a request crosses four processes and several sockets. A
liveness check on the HTTP process proves that the HTTP process is alive, which is the *least*
likely thing to be wrong. The failure modes that matter — a scheduler wedged on a collective,
a rank that diverged, a GPU that fell off the bus — all leave the front end perfectly
responsive.

So the check exercises the whole path, and the docstring explains the trick that makes that
affordable: **under load it is a no-op.** A busy server ignores the probe because there is
already traffic proving the path works; an idle server runs it, which is exactly when you have
no other evidence. Zero overhead when you do not need it, real coverage when you do.

The status gate above it is the other half:

```python
    if _global_state.tokenizer_manager.server_status == ServerStatus.Starting:
        return Response(status_code=503)
```

503 during startup, not 200. That is what keeps a load balancer from routing to a replica
that is still capturing graphs — and it is the mechanism that makes the cold-start table
above operationally safe rather than merely informative.

The warmup at `python/sglang/srt/entrypoints/http_server.py:2163` `_execute_server_warmup`
completes the picture: before announcing readiness the server sends itself a real request, so
the first *user* request is not the one that pays for lazy initialization. Note what
`python/sglang/srt/managers/scheduler.py:1658` `run_event_loop` says about what is left:

```python
        # Engine init (graph capture, warmups) is done; from here on any
        # Triton kernel device-load is a lazy first-use at serving time.
```

Some cost is irreducibly deferred — a Triton kernel for a shape nobody warmed up compiles on
first use, at serving time, in front of a user. Chapter 9's ROCm `torch.unique` pre-warm was
the same class of problem caught in the same way, and the general lesson is that "warm" is
per-code-path, not per-process.

---

## The autoscaling signal

> [!warning] GPU utilization is not a capacity signal
> `nvidia-smi` reports near 100% for a replica serving one user and for the same replica
> serving two hundred, because a decode-bound engine saturates the memory bus either way.
> Autoscaling on it will scale at the wrong time in both directions.

The instinct is to scale on GPU utilization. It is the wrong signal, and Chapter 1 explains
why in one line: a decode-bound engine is *always* busy from the GPU's point of view, because
the memory bus is saturated even while the arithmetic units idle. `nvidia-smi` reports near
100% utilization for a replica serving one user and for the same replica serving two hundred.
It measures whether kernels are resident, not whether there is room for more work.

The signals that carry information are the ones Chapter 22 already named, and they rank in a
specific order:

**Queue depth and waiting time first.** These are the leading indicators, and Chapter 22's
utilization curve is why: `W ∝ 1/(1 − ρ)` has a vertical asymptote, so a queue that has begun
to grow is minutes ahead of the latency that will eventually reflect it. Scale on the
derivative, not the level.

**Token pool utilization second.** Chapter 9's number. Consistently near 100% means admission
control is throttling and more replicas would convert directly into throughput. Consistently
low means the replicas are over-provisioned or the traffic is not what you sized for.

**Retraction count as an alarm, never a target.** Chapter 6 retracts when admission was too
optimistic. A nonzero rate under steady traffic means the replica is past its real capacity
and is destroying work to stay alive — scale out, or lower `--max-running-requests`.

**Latency percentiles last**, because by the time p99 moves you are already behind.

The concurrency setting deserves its own note, because it interacts with everything in Part
II. `--max-running-requests` caps concurrency *independently of memory*, and the reason to use
it is that the memory limit and the latency limit are different limits. Chapter 1's capacity
arithmetic tells you how many sequences fit; it says nothing about whether serving that many
leaves each of them within its ITL budget. When the answer is no — and for interactive
workloads it usually is — the right configuration admits fewer requests than memory allows and
lets the queue absorb the rest, where Chapter 6's policies can order it.

---

## Replacing what is running

There are two ways to change a deployment, and this book has already described the mechanism
for both.

**Replace the replica.** The ordinary path: start new replicas, shift traffic, retire the old
ones. Cost is one full cold start per replica, from the table above, paid at whatever
parallelism your rollout allows.

**Replace the weights.** Chapter 12's `update_weights_from_*` family swaps the weights inside
a live process, keeping the memory pool, the captured graphs, and the caches. It exists for
reinforcement learning, where a rollout engine takes new weights every few minutes and a cold
start per update would dominate the training loop. But the same machinery is a deployment
strategy: for a change that is *only* weights — a new fine-tune of the same architecture at
the same precision — it turns a multi-minute rollout into seconds, because everything the cold
start was rebuilding is still valid.

The limits are exactly the things the fast path is reusing. A change to parallelism, precision,
memory fraction, or graph configuration invalidates the pool or the graphs, so it needs a
restart. The distinction to hold onto is that the weights are the *only* part of a running
engine that is cheap to replace, and Chapter 12 is the reason why.

> [!warning] A hot swap under a populated cache serves the old model's KV
> The radix tree is keyed on token ids, so cached entries survive a weight update while no
> longer corresponding to the weights that produced them. The update paths quiesce
> generation and drop the cache for this reason; a hand-rolled swap that skips either step
> produces plausible, wrong output.

One caveat worth stating plainly: Chapter 10's prefix cache is keyed on token ids, not on
weights. Swapping weights under a populated radix tree leaves cached KV that was computed by
the *previous* model — which is why the update paths quiesce generation first
(`python/sglang/srt/managers/scheduler.py:4576` `pause_generation`) and why the cache is
dropped as part of the swap rather than after it.

---

## When a rank wedges

The characteristic distributed failure is not a crash. Chapter 3 explained that every rank
runs the same scheduler over broadcast data and must reach the same conclusions; when one
does not, the ranks launch mismatched collectives and every one of them blocks forever. No
exception is raised, no process exits, and the HTTP front end keeps answering.

The engine's answer is a watchdog, built at
`python/sglang/srt/managers/scheduler_components/invariant_checker.py:462`
`create_scheduler_watchdog`, and its shape is instructive:

```python
    return WatchdogRaw(
        debug_name="Scheduler",
        get_counter=lambda: scheduler.forward_ct,
        is_active=lambda: (
            scheduler.is_initializing or scheduler.cur_batch_for_debug is not None
        ),
        watchdog_timeout=watchdog_timeout,
        soft=soft,
        dump_info=dump_info,
    )
```

Progress is defined as **the forward counter increasing**, and liveness is only demanded when
the scheduler claims to be doing something — either still initializing, or holding a current
batch. An idle server with an empty queue is not making progress and must not be killed for
it; a server holding a batch and not advancing it is wedged. That two-part condition is the
whole design, and getting it wrong in either direction gives you either a watchdog that kills
healthy idle replicas or one that never fires.

The `dump_info` callback is the part that makes an incident tractable. On timeout it prints
the current batch, its requests, and the results of the pool invariant checks — so the log
entry that accompanies a hang is a description of *what the ranks disagreed about*, not just
the fact that they did. `python/sglang/srt/managers/scheduler.py:1249` `init_soft_watchdog`
adds a shorter-fused variant that reports without killing, for the case where you would rather
have the diagnostic than the restart.

The operational posture that follows is the one Chapter 23 argued for testing: expect
hardware and distributed failure as routine rather than exceptional. A replica that trips its
watchdog should be replaced, not debugged in place; the dump is what you debug afterwards.

---

## What this book cannot tell you

Three questions come up immediately after this chapter and are genuinely outside its scope,
which is worth saying explicitly rather than leaving as a gap.

**How many replicas.** It is Chapter 1's Little's Law with your own numbers — peak arrival
rate, average residence time, and the per-replica concurrency the capacity calculator gives —
plus headroom for the utilization curve. The arithmetic is here; the traffic is yours.

**Which cloud, and reserved versus spot.** A procurement question with no engine content.

**Whether to run this at all.** For low or spiky volume, a shared endpoint is cheaper than a
GPU you are paying for while it idles, and everything in this book is then somebody else's
problem. The case for a dedicated deployment is scale, specialization — a fine-tune, a custom
speculator, a workload whose prefix structure you know — or a requirement that the engine be
yours to tune. If none of those applies, the most useful thing this book gives you is the
ability to tell when one starts to.

---

That is the engine, and now the machine around it.

Chapter 1 made a claim: that generating text is memory-bound, and that memory capacity — not
arithmetic — decides how many people a GPU can serve. Every chapter since has been a response
to one half of that sentence or the other. Paged pools and prefix trees and hierarchical
caching are about the memory. Batching, graphs, quantization, speculation, and expert
parallelism are about wasting less of the compute that the memory wall leaves idle.
Disaggregation is about refusing to compromise between them. And this chapter is about the
fact that all of it has to start up, scale, and fail somewhere.

None of it is arbitrary, and none of it is finished. The appendices that follow are reference
material; the code is still moving; and the argument, once you can see it, is the part that
will still be true when the implementations have changed.

---

<!-- summary:begin -->
<div class="bk-card"><p class="bk-card-head">Chapter 24 in one page</p><ol class="bk-card-arg"><li>Every chapter until now described a steady state; production is mostly the other states.</li><li>Cold start is dominated by weight load, which is bandwidth-bound, and graph capture, which is compute and tunable.</li><li>A decode-bound engine always looks busy, so utilization carries no information — queue depth and pool utilization do.</li><li>The health check runs a real generation because the failure modes that matter leave the front end responsive.</li><li>Weights are the only part of a running engine that is cheap to replace; anything touching pools or graphs needs a restart.</li></ol><p class="bk-card-sub">Numbers worth keeping</p><table class="bk-card-table"><tbody><tr><th scope='row'>Graph capture</th><td>10–120 s</td></tr><tr><th scope='row'>Status during startup</th><td>503, not 200</td></tr></tbody></table><p class="bk-card-sub">Where it lives</p><table class="bk-card-table"><tbody><tr><th scope='row'>Health and warmup</th><td><code>python/sglang/srt/entrypoints/http_server.py</code></td></tr><tr><th scope='row'>Watchdog</th><td><code>python/sglang/srt/managers/scheduler_components/invariant_checker.py</code></td></tr></tbody></table></div>
<!-- summary:end -->

# 21. Observability and Tuning

> *The instrumentation reveals the design — every metric the engine emits exists because
> someone needed it to answer a question this book has already raised.*

You now understand what the engine does. This chapter is about seeing what it *is* doing —
on a particular machine, under a particular workload, right now.

There is a pleasant symmetry here. Every metric SGLang emits exists because someone needed
it to answer a question, and by this point in the book you have asked most of those
questions yourself. Queue depth is Chapter 5's admission problem. Cache hit rate is Chapter
9. Retraction count is what happens when Chapter 5 guesses wrong. Acceptance length is
Chapter 18 telling you whether it is earning its keep. The instrumentation is a map back to
the design.

The chapter has three parts. First, reading the metrics — including the five numbers that
actually tell you what is happening, out of the hundreds available. Then measuring honestly,
which is mostly a catalogue of ways benchmarks lie. Then a tuning procedure, ordered not by
how the flags appear in `--help` but by which constraint each one relieves.

That ordering is the chapter's main practical claim: most tuning effort is spent optimizing
something that was not the bottleneck.

---

## Reading the metrics as a map

`python/sglang/srt/observability/metrics_collector.py` is nearly 2,200 lines, and the
fastest way to understand it is to notice that its collectors partition along the same lines
as this book's chapters:

```
:238   SchedulerMetricsCollector      the loop and its queues       (Ch. 4, 5)
:1480  TokenizerMetricsCollector      request-level latency          (Ch. 3, 7)
:1962  RadixCacheMetricsCollector     prefix cache behavior          (Ch. 9)
:1849  StorageMetricsCollector        HiCache tiers                  (Ch. 10)
:1947  ExpertDispatchCollector        MoE routing balance            (Ch. 16)
:2160  EncoderMetricsCollector        multimodal encoding            (Ch. 20)
```

`:65` `SchedulerStats` is the core snapshot, and `:45` `QueueCount` breaks the queue down by
state — because "queue depth" is not one number when requests can be waiting for admission,
waiting for a grammar to compile (Chapter 19), or waiting for a KV transfer (Chapter 17).

`:172` `DPCooperationInfo` measures Chapter 15's imbalance, which matters because a
data-parallel rank with nothing to do still runs an `IDLE` forward.

`:215` `_StatLoggerDIMixin` is dependency injection for the logging backend, so the same
collector serves Prometheus, logs, or a test double.

### The five numbers that matter

Everything else is diagnostic. These five tell you what the engine is doing:

**Cache hit rate** (Chapter 9). The single highest-leverage number. If it is low on a
workload with shared prefixes, something is wrong — the cache is too small, requests are
being routed badly (Chapter 17), or the `extra_key` namespace is fragmenting (Chapter 20).

**Token pool utilization** (Chapter 8). Consistently near 100% means you are memory-bound
and admission is throttling. Consistently low means `--mem-fraction-static` is leaving
memory unused.

**Retraction count** (Chapter 5). Should be near zero. Anything else means admission is
over-optimistic and work is being destroyed.

**Queue depth** with **waiting time**. Growing queues mean you are past capacity; the
question is whether to shed load or add replicas.

**Spec acceptance length** (Chapter 18). Must be comfortably above 1 or speculation is
costing you. `python/sglang/srt/managers/tokenizer_manager.py:2766`
`_calculate_spec_decoding_metrics` computes it.

`python/sglang/srt/observability/forward_pass_metrics.py` covers per-forward timing, and
`python/sglang/srt/observability/req_time_stats.py` the per-request breakdown that turns a
TTFT number into "queueing versus prefill."

### Why latency falls off a cliff rather than a slope

Two of those five numbers — queue depth and waiting time — behave in a way that surprises
people the first time they watch it, and the surprise is worth pre-empting because it changes
what a dashboard is telling you.

Chapter 1 introduced Little's Law, `L = λW`. Queueing theory adds the shape of the
relationship between utilization and delay. For a simple queue at utilization ρ (arrival rate
over service rate), average waiting time behaves like

```
W  ∝  1 / (1 − ρ)
```

Read the denominator. At ρ = 0.5, wait time is proportional to 2. At ρ = 0.9, to 10. At
ρ = 0.99, to 100. **The curve has a vertical asymptote at full utilization**, and the last few
percent of capacity cost more latency than all the preceding ones combined.

This is why a serving deployment that looks healthy at 85% utilization becomes unusable at
95% with no change in code, no bad request, and no obvious event in the logs. The dashboard
shows GPU utilization climbing smoothly and p99 latency going vertical, and the two look
unrelated. They are the same curve.

Three practical consequences:

**Target utilization well below 1.** A serving pool run at 70–80% has headroom for the
variance that real traffic has; one run at 95% is a queue waiting for an excuse. The cost of
the idle 20% is far less than the cost of the latency it prevents.

**Watch the derivative, not the level.** Queue depth rising steadily is the signal, and it is
visible well before the latency cliff. By the time p99 has moved, the queue has been growing
for a while.

**Variance is as expensive as the mean.** The formula above understates the damage when
service times vary — and LLM service times vary enormously, since a request generating 2,000
tokens occupies a slot a hundred times longer than one generating 20. That variance is why
Chapter 5's admission control and its priority policies exist, and why shedding load at the
queue boundary beats accepting work you will time out later.

None of this is visible from throughput. An engine at 99% utilization posts the best tokens
per second it will ever post, right up to the point where every individual request misses its
SLO — which is exactly the failure mode goodput was defined to catch.

Wiring: `python/sglang/srt/managers/scheduler.py:718` `init_metrics_collector`, `:1191`
`init_metrics_reporter`, and
`python/sglang/srt/managers/scheduler_components/metrics_reporter.py`. `:1095`
`emit_metrics_constants` publishes static facts — model name, parallelism sizes, pool
capacity — so dashboards can label series without separate configuration.
`docs/docs/references/production_metrics.mdx` is the reference.

---

## Tracing across processes

Metrics tell you the aggregate. When one request is slow, you need its path.

That path crosses Chapter 2's process boundaries, so a stack trace is useless — the request
exists in the tokenizer process, then the scheduler, then the detokenizer, and no single
call stack spans them.

`python/sglang/srt/observability/trace.py` and
`python/sglang/srt/observability/trace_async.py` propagate a trace context by `rid` across
those hops. `python/sglang/srt/observability/mooncake_trace.py` extends it into Chapter 17's
KV transfers, so a disaggregated request can be followed across machines.
`docs/docs/references/production_request_trace.mdx` covers it, and
`python/sglang/srt/entrypoints/http_server.py:1160` `set_trace_level` adjusts verbosity live.

Startup gets its own instrumentation, because "why does the server take four minutes to
start" is a real and frequent question. `python/sglang/srt/observability/startup_time.py`
and
`python/sglang/srt/observability/startup_func_log_and_timer.py` break it down;
`python/sglang/srt/managers/scheduler.py:656` `init_startup_timing_begin` and `:659`
`init_startup_timing_summary` produce the summary. The answer is usually weight loading
(Chapter 11) or CUDA graph capture (Chapter 14), and the breakdown tells you which.

`python/sglang/srt/observability/cpu_monitor.py` watches for the case where the CPU is the
bottleneck — the condition Chapter 4's overlap scheduler and Chapter 14's graphs both exist
to prevent.

---

## Measuring honestly

Three harnesses, for three questions.

**`python/sglang/benchmark/serving.py`** — online serving. Sends requests at a configured
rate against a running server and reports TTFT, ITL, and throughput distributions. This is
the one that answers "how will this behave in production," and the one most easily misused.

**`python/sglang/benchmark/offline_throughput.py`** — maximum throughput with no latency
constraint. Answers "what is this hardware capable of."

**`python/sglang/benchmark/one_batch.py`** — one batch, no server, no scheduler. Chapter 1
used it to demonstrate the prefill/decode gap. This is what you use when you have changed a
kernel and want to know whether it is faster, without the scheduler in the way.

`python/sglang/benchmark/one_batch_server.py` sits between the last two.

The ways to get this wrong are consistent enough to list:

- **Measuring your client.** At high request rates, a Python client can become the
  bottleneck and you end up benchmarking `asyncio`.
- **Not warming up.** First requests pay CUDA graph capture, JIT compilation, and allocator
  warm-up — including the ROCm `torch.unique` case Chapter 8 quoted, which shows up
  precisely as a slow *second* request.
- **Accidental cache hits.** Sending the same prompt repeatedly measures Chapter 9's cache,
  not the model. Sometimes that is the point; it should be deliberate.
- **Reporting the mean.** Latency distributions are skewed. P50 and P99 differ by an order
  of magnitude under load, and only one of them is your SLO.
- **Comparing across configurations.** Changing `--mem-fraction-static` changes concurrency,
  which changes batch size, which changes everything.

`docs/docs/developer_guide/bench_serving.mdx` documents the harness.

---

## Reading a profile

When benchmarks say "slow" and metrics do not say why, profile.

`python/sglang/profiler.py` wraps the PyTorch profiler, and
`python/sglang/srt/managers/scheduler_components/profiler_manager.py` runs it inside the
scheduler process. The endpoints are
`python/sglang/srt/entrypoints/http_server.py:1139` `start_profile_async` and `:1150`
`stop_profile_async`, so you can profile a live server under real load rather than a
synthetic reproduction.

`.claude/skills/llm-torch-profiler-analysis/SKILL.md` is the project's own triage procedure,
and `.claude/skills/generate-profile/SKILL.md` drives capture. What to look for, in order:

**Gap time.** Space between kernels means the GPU is starved — a CPU-side problem. Chapter
4's overlap loop and Chapter 14's graphs are the fixes. If gaps dominate, nothing you do to
kernels will help.

**Kernel time distribution.** Which kernels actually cost. Usually attention (Chapter 13) and
the large GEMMs, but for MoE models often the all-to-all (Chapter 16).

**Communication time.** Collectives on the critical path. If all-to-all dominates, Chapter
16's two-batch overlap is the answer; if all-reduce dominates, the parallelism layout
(Chapter 15) is wrong.

**Fusion opportunities.** Adjacent elementwise kernels that could be one — Chapter 14's
compilation.

Two more tools: `python/sglang/kernel_api_logging.py` logs kernel API calls, which
`.claude/skills/debug-cuda-crash/SKILL.md` uses to find the call that crashed; and
`python/sglang/srt/debug_utils/comparator/` compares tensors layer-by-layer against a
reference, which is how a model producing wrong output gets localized (Chapter 22).

---

## A tuning order of operations

Tune in the order that relieves binding constraints, not in the order the flags appear in
`--help`.

**1. `--mem-fraction-static`** (Chapter 8). Sets the KV pool, which sets concurrency, which
sets everything. Raise until you see OOM under load, then back off. Too high fails not at
startup but under the first large batch, when activation memory is demanded from a pool that
already took it.

**2. Parallelism layout** (Chapters 15, 16). TP to fit the model, PP across nodes, DP
attention for MLA, EP for MoE. Getting this wrong cannot be compensated for by anything
below it.

**3. `--chunked-prefill-size`** (Chapter 5). Smaller chunks lower ITL for active decodes and
raise TTFT for long prompts. Set it by which latency you are being measured on.

**4. `--max-running-requests`** (Chapter 5). Caps concurrency independently of memory. Useful
when memory allows more than latency does.

**5. Attention backend** (Chapter 13). Worth benchmarking; the default is not always best for
your shapes, and prefill and decode can be set separately.

**6. `--cuda-graph-max-bs`** (Chapter 14). Higher covers more batch sizes with graphs, at
capture memory that comes out of the KV pool. Interacts with step 1.

**7. Speculative decoding** (Chapter 18). Large wins at low batch size, negative at high.
Check accepted length before keeping it.

**8. HiCache** (Chapter 10) and **routing** (Chapter 17). Only pay off with real prefix
sharing — check the cache hit rate first.

`docs/docs/advanced_features/hyperparameter_tuning.mdx` carries the project's guidance, and
`.claude/skills/sglang-prod-incident-triage/SKILL.md` is the replay-first procedure for when
a live deployment is misbehaving rather than merely slow.

The general rule: **measure which of Chapter 1's two phases you are bound by, and which
resource within it, before changing anything.** Most tuning effort is spent optimizing a
constraint that was not binding.

One chapter left. Chapter 22 is about changing the engine rather than watching it — and it
is the last check on whether everything before it landed.

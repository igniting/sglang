# 21. Observability and Tuning

> *The instrumentation reveals the design — every metric the engine emits exists because someone needed it to answer a question this book has already raised.*

> **Status: outline.** This chapter is planned but not yet written. The beats below are the writing plan — each one pairs an idea with the code that implements it.

## Beats

1. **What the engine measures, and why each one.**
   `python/sglang/srt/observability/metrics_collector.py`, `forward_pass_metrics.py`,
   `req_time_stats.py` — queue depth, cache hit rate, memory utilization, spec-decode
   acceptance, per-phase timing. Each maps back to a specific chapter's tradeoff.

2. **Tracing across processes.** `python/sglang/srt/observability/trace.py`,
   `trace_async.py` — following one `rid` through the Chapter 2 topology; and
   `startup_time.py` for the startup breakdown that explains slow boots.

3. **Measuring honestly.** `python/sglang/bench_serving.py`,
   `python/sglang/bench_offline_throughput.py`, `python/sglang/bench_one_batch_server.py`
   — which to use when, what to hold fixed, and how to avoid measuring your own client.

4. **Reading a profile.** `python/sglang/profiler.py` and the `/start_profile` endpoint
   (`python/sglang/srt/entrypoints/http_server.py:1139`); kernel time vs gap time vs
   communication, per `.claude/skills/llm-torch-profiler-analysis/SKILL.md`.

5. **A tuning order of operations.** `--mem-fraction-static`, `--chunked-prefill-size`,
   `--max-running-requests`, `--cuda-graph-max-bs`, attention backend, parallelism layout
   — sequenced by which chapter's constraint each one relieves.

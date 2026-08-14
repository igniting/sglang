# Appendix E — Reading a Startup Log

A server's startup log is the engine narrating its own construction. Read in order, it is a
compressed tour of Parts I–IV, and it is the fastest way to diagnose a server that starts
slowly, allocates the wrong amount of memory, or fails before serving anything.

This appendix walks the phases and says which chapter explains each, what the numbers mean,
and what it looks like when something is wrong. Exact wording changes between releases;
the sequence does not.

---

## Phase 1 — Argument resolution

```
server_args=ServerArgs(model_path='...', tokenizer_path='...', tp_size=8, ...)
```

`python/sglang/srt/entrypoints/engine.py:1052` `_launch_subprocesses` logs the fully
resolved `ServerArgs` before anything else happens.

**Read this first when debugging anything.** These are the *resolved* values, after defaults,
validation, and inter-flag adjustment — not what you typed. A flag you set may have been
overridden by `check_server_args` because of an incompatible combination (Appendix A), and
this line is where you find out.

## Phase 2 — Process topology

Subprocess spawn, one scheduler per (pipeline rank, tensor rank), plus detokenizers.
Chapter 2 covers the topology; `_launch_scheduler_processes`
(`python/sglang/srt/entrypoints/engine.py:848`) is the code.

From here, log lines carry rank identifiers. Under tensor parallelism you are reading *n*
interleaved narratives — and when only some ranks report a phase, that asymmetry is itself
the finding.

## Phase 3 — Distributed init

NCCL initialization and process group construction.
`python/sglang/srt/model_executor/model_runner.py:1037` `init_torch_distributed`, explained
in Chapter 15.

**A hang here is a connectivity problem**, not a model problem: wrong `--dist-init-addr`,
a blocked port, or a mismatched world size. It is also the most common place for a
multi-node deployment to stall silently, which is why the watchdogs of Chapter 4 exist.

## Phase 4 — Weight loading

```
Load weight begin. avail mem=78.32 GB
...
Load weight end. type=LlamaForCausalLM, dtype=torch.bfloat16, avail mem=61.05 GB, mem usage=17.27 GB
```

`python/sglang/srt/model_executor/model_runner.py:1057` `load_model` (Chapter 11) brackets
this with memory readings, and the pair is the most useful arithmetic in the log.

- **`mem usage`** is what the weights cost *this rank*. Under TP 8, roughly one eighth of
  the model — if it is not, your parallelism is not what you think it is.
- **`avail mem` at the end** is what remains for everything downstream.

Slow loading is usually I/O; Chapter 11's `set_num_threads(1)` and load-format notes are the
levers, and Phase 10's summary will confirm.

## Phase 5 — KV cache allocation

```
KV Cache is allocated. dtype: torch.bfloat16, #tokens: 458752, K size: 28.00 GB, V size: 28.00 GB
Memory pool end. avail mem=4.21 GB
```

`python/sglang/srt/mem_cache/memory_pool.py:1668` `_finalize_allocation_log`, from Chapter 8.

**`#tokens` is the single most important number in the log.** It is
`max_total_num_tokens` — the budget Chapter 5 spends, shared across all concurrent requests.
Divide by your expected context length for a rough concurrency ceiling: 458,752 tokens at
4,000 tokens per conversation is about 114 concurrent requests.

If it is lower than expected, the cause is one of: `--mem-fraction-static` too low, weights
larger than expected (check Phase 4), a KV dtype that did not take effect (Chapter 14), or
KV head replication under high TP (Chapter 12).

The trailing `avail mem` is deliberately small — the pool takes what the fraction allows.
Too small and Phase 7 will fail.

## Phase 6 — Attention backend

```
Attention backend not set. Use flashinfer backend by default.
```

`python/sglang/srt/model_executor/model_runner.py:927` `init_attention_backends`
(Chapter 13). Worth reading even when you did not set the flag, because the *resolved*
backend may differ from the default for your model — MLA models take a different path — and
prefill and decode may resolve differently.

## Phase 7 — CUDA graph capture

```
Capture cuda graph begin. This can take up to several minutes. avail mem=4.21 GB
Capture cuda graph bs [1, 2, 4, 8, 16, 24, 32, ...]
Capture cuda graph end. Time elapsed: 42.31 s. mem usage=1.83 GB. avail mem=2.38 GB
```

Chapter 14. Three things to read here.

**Time elapsed** is often the largest single component of startup. `--cuda-graph-max-bs`
controls it directly: fewer captured shapes, faster start, less graph coverage.

**mem usage** is the capture cost — the static buffers per shape.

**The batch-size list** is what Chapter 14's bucketing captured. A batch outside this list
runs eager, so if your production batch sizes are larger than the largest bucket, you are
not getting graphs where it matters.

An OOM here means Phase 5 took too much: lower `--mem-fraction-static`, or reduce
`--cuda-graph-max-bs`. Note the ordering — Chapter 6 explained that graph capture runs
*after* pool allocation, which is why the failure appears here rather than earlier.
`post_capture_resize_kv_pool` returns unused capture memory to the pool afterwards.

## Phase 8 — Warmup

`python/sglang/srt/entrypoints/warmup.py` runs dummy requests through the full pipeline.
This pays first-call costs — kernel JIT, allocator warm-up, and the ROCm `torch.unique`
case Chapter 8 quoted — before real traffic arrives. It is also the first end-to-end proof
that the engine works.

## Phase 9 — Ready

```
The server is fired up and ready to roll!
```

At this point `/health_generate` (Chapter 2) will run a real forward pass, which is a
stronger claim than the log line.

## Phase 10 — Startup timing summary

`python/sglang/srt/managers/scheduler.py:659` `init_startup_timing_summary`, backed by
`python/sglang/srt/observability/startup_time.py` (Chapter 21), prints the per-phase
breakdown.

For "why does startup take four minutes," this is the answer, and it is almost always weight
loading (Phase 4) or graph capture (Phase 7).

---

## The three numbers to extract

From any startup log:

1. **Weight `mem usage`** (Phase 4) — is the model sharded as intended?
2. **KV `#tokens`** (Phase 5) — what concurrency does this support?
3. **Graph capture time and memory** (Phase 7) — what is startup costing, and how much
   memory went to graphs rather than cache?

Those three place a deployment in the design space this book describes. Everything in
Chapter 21's tuning order is an adjustment to one of them.

---

## Common failure signatures

| Symptom | Phase | Likely cause |
| --- | --- | --- |
| Hang with no error | 3 | Distributed init: address, ports, world size (Ch. 15) |
| OOM during capture | 7 | `--mem-fraction-static` too high (Ch. 8, 14) |
| `#tokens` far lower than expected | 5 | Larger weights, KV dtype not applied, or KV head replication (Ch. 12, 14) |
| Startup takes many minutes | 4 or 7 | I/O-bound loading, or a large capture matrix |
| Only some ranks log a phase | any | Rank divergence — see `.claude/skills/debug-distributed-hang/SKILL.md` |
| Second request unexpectedly slow | 8 | Warmup did not cover a JIT path (Ch. 8) |

`.claude/skills/clean-startup-log/SKILL.md` is the project's own guidance on keeping this
log readable — which is itself evidence that the log is treated as an interface.

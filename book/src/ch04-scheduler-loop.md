# 4. The Scheduler Loop

> *One synchronous loop owns the GPU and answers one question per iteration — what runs
> next? Everything else in the engine is input to that question.*

---

## The center of the engine

`python/sglang/srt/managers/scheduler.py` is about 5,000 lines, and almost every other
subsystem in this book exists to inform, constrain, or serve the loop inside it. The loop
is not complicated. What surrounds it is.

This chapter builds up to that loop in four moves: the request object it schedules, the
batch objects it moves them between, the simple loop, and then the fast loop as a delta
against the simple one.

---

## A request is a state machine with a lot of state

`python/sglang/srt/managers/schedule_batch.py:811` `Req` is what a
`TokenizedGenerateReqInput` becomes once it crosses into the scheduler. Its `__init__`
(`:814`) runs to nearly 400 lines, which is itself informative: a request in flight must
remember something for almost every feature in the engine.

Read the fields in groups rather than in order.

**Identity and input** — `rid`, `origin_input_ids`, `output_ids`, `sampling_params`. The
irreducible core. Note that input and output ids are kept separate and concatenated on
demand; `:1274` `get_fill_ids` and `:1277` `_refresh_fill_ids` manage that view, because
the *fill* sequence (what has been written to the KV cache) is not always the same as
input plus output.

**Cache position** — `prefix_indices`, `last_node`, `cache_protected_len`, `req_pool_idx`.
This is the request's stake in Chapter 9's radix tree and Chapter 8's pools:
which prefix it matched, which tree node it holds a lock on, how much of its KV the tree
owns, and which row of `req_to_token` is its own.

**Progress** — `:1207` `seqlen`, `:1234` `effective_kv_committed_len`, extend ranges via
`:1271` `set_extend_range`. How far along it is, in several senses that do not always
agree.

**Termination** — `finished_reason`, plus the small class hierarchy at `:223`:
`FINISH_MATCHED_TOKEN` (`:228`), `FINISH_MATCHED_STR` (`:240`),
`FINISHED_MATCHED_REGEX` (`:252`), `FINISH_LENGTH` (`:264`), `FINISH_ABORT` (`:276`). Each
serializes itself through `to_json`, so the reason survives the trip back to the client.
Chapter 7 shows why stop *strings* need their own class rather than a flag.

**Feature state** — grammar objects (Chapter 19), LoRA ids (Chapter 20), multimodal inputs
(`:318` `MultimodalDataItem`, `:590` `MultimodalInputs`), logprob accumulators, and
speculative-decoding histograms such as `:1241` `update_spec_correct_drafts_histogram`.

The single most useful method for understanding the loop is `:1298`
`init_next_round_input`, which recomputes the request's prefix match against the tree
before each scheduling round. A request's cached prefix is not fixed at arrival — other
requests may have inserted an overlapping prefix in the meantime, which is exactly the
concurrency Chapter 9's `cache_unfinished_req` re-match protects against.

---

## Three batch types for three jobs

A request passes through three different batch representations, and conflating them is the
most common source of confusion when reading this code.

**`ScheduleBatch`** (`python/sglang/srt/managers/schedule_batch.py`) is the scheduler's
working view: a list of `Req` objects plus CPU-side bookkeeping. It is what the loop
manipulates. Its operations are the verbs of scheduling:

- `prepare_for_extend` — allocate KV for prompt tokens, build the extend metadata.
- `prepare_for_decode` — allocate one slot per sequence, advance positions.
- `retract_decode` — evict running requests when memory ran out. Chapter 5's failure path.
- `filter_batch` — drop finished requests.
- `merge_batch` — fold a newly-prefilled batch into the running one.
- `copy` — snapshot, which the overlap loop below depends on.

**`ModelWorkerBatch`** is the transport form — the subset the worker needs, without the
scheduler's private state.

**`ForwardBatch`** (`python/sglang/srt/model_executor/forward_batch_info.py:412`) is the
GPU-side view: tensors, positions, page tables, attention metadata. Chapter 6 covers it.

The separation is enforced by rule.
`.claude/rules/schedule-batch-out-of-place-mutation.md` forbids in-place mutation of
`ScheduleBatch` fields:

```python
# Bad
self.reqs.extend(other.reqs)
self.seq_lens.add_(1)

# Good
self.reqs = self.reqs + other.reqs
self.seq_lens = self.seq_lens + 1
```

The stated reason is the overlap loop: `copy()` snapshots and queued references "rely on
old objects staying frozen." Once you have read the overlap loop below, this stops being a
style preference and becomes an obvious safety requirement.

---

## The loop, honest version first

`python/sglang/srt/managers/scheduler.py:1714` `event_loop_normal` is the whole scheduler in
thirty lines. Read it before anything else in the file:

```python
    def event_loop_normal(self):
        """A normal scheduler loop."""
        while True:
            if self.gracefully_exit:
                break

            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue

            # Get the next batch to run
            plan = self.get_next_batch_to_run(
                running_batch=self.running_batch, last_batch=self.last_batch
            )
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            self.cur_batch_for_debug = batch

            # Launch the current batch
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # When the server is idle, do self-check and re-init some states.
                self._sched_idled = True
                self.on_idle()

            # Update last_batch
            self.last_batch = batch
```

Receive, decide, run, process results, repeat. Four steps.

- **Receive** — `:1872` `process_input_requests` drains the socket from Chapter 3 and
  dispatches each message through the type table at `:1523` `init_request_dispatcher`. A
  generate request lands in `:2363` `handle_generate_request`, which builds a `Req` and
  queues it via `:2715` `_add_request_to_queue`.
- **Decide** — `:3012` `get_next_batch_to_run` is the whole of Chapter 5.
- **Run** — `:3623` `run_batch` hands the batch to the worker (Chapter 6).
- **Process** — `:3917` `process_batch_result` appends sampled tokens, checks stop
  conditions, frees KV for finished requests, and sends output onward.

Everything else in this 5,000-line file is in service of those four calls.

---

## The loop, fast version, as a delta

The problem with `event_loop_normal` is a gap. `run_batch` launches GPU work and
`process_batch_result` needs its output, so the CPU sits idle while the GPU runs — and then
the GPU sits idle while the CPU prepares the next batch. On a small model with a large
batch, that Python-side gap can be a substantial share of each decode step. This is what
the "zero-overhead batch scheduler" removes.

`:1749` `event_loop_overlap` is the same four steps rearranged so the CPU is always one
step ahead:

```python
        def pop_and_process():
            # Process the results of the last batch
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)
```

```python
            # Launch the current batch
            if batch:
                batch_result = self.run_batch(batch)
                # Fence result processing behind this forward's shared reads.
                self._apply_war_barrier()
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None
                self._sched_idled = True

            # Process the last batch
            if self.last_batch:
                if not disable_overlap_for_batch:
                    pop_and_process()
```

The reordering is the entire idea: launch batch *N*, then process the results of batch
*N−1* while *N* is on the GPU. `result_queue` holds one step of lag.

Three details make it work.

**`batch.copy()`** is why the out-of-place mutation rule exists. The queued batch must
describe the state *as launched*; if the live batch object were mutated while the GPU was
still reading it, the result processing would attribute the wrong tokens to the wrong
requests.

**`_apply_war_barrier`** (`:1696`) — write-after-read. Result processing writes to buffers
the in-flight forward may still be reading, so the barrier fences those writes behind the
forward's reads. This is the class of hazard overlap introduces: correctness now depends on
GPU-side ordering, not just Python-side ordering.

**Sampling comes last:**

```python
            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            if self.is_generation:
                self.launch_batch_sample_if_needed(batch_result, batch)
```

Sampling for batch *N* may depend on state that processing batch *N−1* updates — a grammar
advanced by the token *N−1* just produced (Chapter 19). So the forward is overlapped but
the sample is not.

Overlap cannot always be done. `:1823` `is_disable_overlap_for_batch` decides per batch,
and when it returns true the loop calls `pop_and_process()` *before* launching, collapsing
back to the synchronous shape. There is even an opportunistic memory flush at that
boundary:

```python
            if disable_overlap_for_batch:
                pop_and_process()
                # Opportunistic flush at the disable_overlap sync boundary:
                # forward_stream is idle (prev forward drained, next not launched),
                # so `_flush`'s non-urgent guard compacts freely. Sync-free, best-effort.
                if self.enable_unified_memory:
                    try:
                        self.token_to_kv_pool_allocator.flush_opportunistic()
                    except Exception:
                        pass
```

A synchronization point the loop was forced into gets reused as the one safe moment to
compact memory. The `except Exception: pass` is honest about it being best-effort.

`:1438` `init_overlap` sets the machinery up, and
`python/sglang/srt/managers/overlap_utils.py` holds the "future token" bookkeeping — token
ids that do not exist yet, referenced by placeholder so batch *N+1* can be built before
batch *N* has sampled.

---

## A 5,000-line class, and why it is shaped that way

`python/sglang/srt/managers/scheduler.py:378` declares `Scheduler` as a composition of
twenty-two mixins — for disaggregation, pipeline parallelism, multiplexing, profiling, and
more. `.claude/rules/general-code-style.md` says "avoid mixins." The tension is
acknowledged rather than hidden: `.claude/skills/large-class-style/SKILL.md` documents the
conventions this class follows *because* it is exempt.

`:388` `__init__` is a sequence of named steps:

```
init_model_config          init_tokenizer            init_memory_pools
init_ipc_channels          init_tp_model_worker      init_all_attention_backends
init_running_status        init_chunked_prefill      init_all_cuda_graphs
init_schedule_policy       init_disaggregation       init_overlap
init_grammar_manager       init_request_dispatcher   init_metrics_collector
```

Roughly fifty of them, and the order is a dependency chain: memory pools need the model
config; attention backends need the pools; CUDA graphs need the backends. Reading
`__init__` top to bottom is the best available map of what a scheduler *is*, and startup
failures are usually explained by the step that precedes the one that threw.

The class is also being decomposed. `python/sglang/srt/managers/scheduler_components/`
holds collaborators extracted out of it —
`python/sglang/srt/managers/scheduler_components/batch_result_processor.py`,
`python/sglang/srt/managers/scheduler_components/output_streamer.py`,
`python/sglang/srt/managers/scheduler_components/request_receiver.py`,
`python/sglang/srt/managers/scheduler_components/new_token_ratio_tracker.py`,
`python/sglang/srt/managers/scheduler_components/invariant_checker.py`. Each was once inline. Reading them shows where the seams are, and the corresponding `init_*` method
(`:2147` `init_batch_result_processor`, `:2134` `init_output_streamer`, `:2071`
`init_invariant_checker`) shows how each is wired back in.

---

## Liveness

An idle engine still has work to do. `:4036` `on_idle` runs when there is no batch, and
`:4078` `is_fully_idle` decides whether the engine is genuinely quiet — a question
complicated by pipeline parallelism, where microbatches may still be in flight
(`:4136` `_pp_microbatches_drained`).

Idleness is also when invariants get checked. `:2071` `init_invariant_checker` builds a
component that verifies memory accounting adds up — that every KV page is either allocated
to a request, held by the tree, or free. Under `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY`
the same check runs every iteration, at a cost you would not pay in production but want
when hunting a leak.

Watchdogs are set up in `:1249` `init_soft_watchdog` and `:1255`
`init_watch_dog_memory_saver_input_blocker`. Their job is to notice that the loop has
stopped making progress and to fail loudly, because a scheduler wedged inside a collective
operation produces no error at all — it simply stops, and every request behind it times
out.

This is the same reasoning behind `/health_generate` from Chapter 2 running a real forward
pass. In a system where the failure mode is *silence*, liveness has to be proven by doing
work, not by answering a question.

---

## One iteration, drawn

Two decode steps of the overlap loop, with time running down:

```
   CPU (scheduler process)                GPU
   ─────────────────────────────────      ─────────────────────────
t0 recv + process_input_requests
   get_next_batch_to_run  ──► batch N
   run_batch(N)  ─────────────────────►   forward N launched
   result_queue.append(N.copy())          │
   pop_and_process(N-1)                   │  (running N)
     ├ append sampled tokens              │
     ├ check stop conditions              │
     ├ free KV of finished reqs           │
     └ stream output                      │
   launch_batch_sample_if_needed(N)  ─►   │  sample N
                                          ▼
t1 recv + process_input_requests          forward N complete
   get_next_batch_to_run  ──► batch N+1
   run_batch(N+1)  ───────────────────►   forward N+1 launched
   pop_and_process(N)                     │  (running N+1)
```

The CPU work in the shaded middle — result processing, stop checking, KV freeing, output
streaming — is entirely hidden behind GPU compute. In `event_loop_normal` all of it sits
between two forwards, and the GPU waits.

That is the zero-overhead scheduler. Everything else in this chapter is what it costs:
snapshot copies, a write-after-read barrier, a rule against in-place mutation, and a
deferred sample.

---

Chapter 5 opens up the one call this chapter skipped — `get_next_batch_to_run`, where the
actual scheduling decision is made.

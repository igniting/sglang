# 4. The Scheduler Loop

> *One synchronous loop owns the GPU and answers one question per iteration — what runs next? Everything else in the engine is input to that question.*

> **Status: outline.** This chapter is planned but not yet written. The beats below are the writing plan — each one pairs an idea with the code that implements it.

## Beats

1. **A request is a state machine with a lot of state.**
   `python/sglang/srt/managers/schedule_batch.py:811` `Req`, `:814` `__init__` read field-
   group by field-group: tokens, prefix match, KV indices, sampling params, grammar state,
   logprob accumulators, multimodal payloads. `:1298` `init_next_round_input` is the per-
   round prefix match; `:223`–`:283` the finish-reason hierarchy.

2. **Three batch types for three jobs.** `ScheduleBatch` (CPU scheduling view),
   `ModelWorkerBatch` (transport), `ForwardBatch` (GPU execution view) — with
   `prepare_for_extend`, `prepare_for_decode`, `retract_decode`, `filter_batch`,
   `merge_batch` as the operations that move between them. Why the repo forbids in-place
   batch mutation, per `.claude/rules/schedule-batch-out-of-place-mutation.md`.

3. **The loop, honest version first.** `python/sglang/srt/managers/scheduler.py:1714`
   `event_loop_normal` is short and does exactly what it says. Read it before anything
   else in the file.

4. **The loop, fast version, as a delta.** `python/sglang/srt/managers/scheduler.py:1749`
   `event_loop_overlap` — the zero-overhead batch scheduler. The future-token trick that
   lets step *N+1* be prepared while step *N* is still on the GPU, in
   `python/sglang/srt/managers/overlap_utils.py` and `:1438` `init_overlap`; and `:1823`
   `is_disable_overlap_for_batch` for when it cannot be done.

5. **A 5,000-line class, and why it is shaped that way.**
   `python/sglang/srt/managers/scheduler.py:378` composes 22 mixins and `:388` `__init__`
   is a sequence of named `init_*` calls. `.claude/skills/large-class-style/SKILL.md`
   documents this as deliberate; the extracted collaborators in
   `python/sglang/srt/managers/scheduler_components/` show where the seams are.

6. **Liveness.** `python/sglang/srt/managers/scheduler.py:4036` `on_idle`, `:4078`
   `is_fully_idle`, watchdogs, and why `/health_generate`
   (`python/sglang/srt/entrypoints/http_server.py:646`) runs a real forward pass instead
   of returning 200.

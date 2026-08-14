# 5. Deciding What Runs Next

> *Continuous batching is an admission-control problem under a hard memory budget, and the memory system is what makes the decision interesting.*

> **Status: outline.** This chapter is planned but not yet written. The beats below are the writing plan — each one pairs an idea with the code that implements it.

## Beats

1. **Admission, not scheduling.** `python/sglang/srt/managers/scheduler.py:3012`
   `get_next_batch_to_run` is the central decision function; `:3154`
   `get_new_batch_prefill` and `:3478` `update_running_batch` are its two halves.

2. **The token budget.** `python/sglang/srt/managers/schedule_policy.py:504`
   `PrefillAdder` — `:664` `rem_total_tokens`, `:857` `_update_prefill_budget`. This is
   where `max_total_tokens`, `max_prefill_tokens`, and `max_running_requests` stop being
   flags and become arithmetic.

3. **Predicting the future.** The new-token-ratio heuristic
   (`python/sglang/srt/managers/scheduler_components/new_token_ratio_tracker.py`)
   estimates decode demand that has not happened yet; over-optimism here is what makes
   retraction necessary.

4. **Cache-aware ordering.** `python/sglang/srt/managers/schedule_policy.py:216`
   `SchedulePolicy`, `:237` `calc_priority`, `:314` `_compute_prefix_matches`, `:374`
   `_sort_by_longest_prefix`, `:387` `_sort_by_dfs_weight`. The scheduler consults the
   radix tree — this is the loop between Chapters 5 and 9 closing, and the reason cache-
   aware routing works at all.

5. **Chunked prefill.** `python/sglang/srt/managers/scheduler.py:1153`
   `init_chunked_prefill` — splitting long prompts so a 100k-token request cannot stall
   every decode in flight, and the TTFT/ITL trade that buys.

6. **Fairness and its absence.** `python/sglang/srt/managers/scheduler.py:2715`
   `_add_request_to_queue`, `:2739` `_set_or_validate_priority`, `:2813`
   `_abort_on_waiting_timeout` — starvation, priorities, and queue timeouts.

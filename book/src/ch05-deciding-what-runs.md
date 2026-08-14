# 5. Deciding What Runs Next

> *Continuous batching is an admission-control problem under a hard memory budget, and the
> memory system is what makes the decision interesting.*

---

## The decision, stated properly

"Continuous batching" is usually described as: requests join and leave the batch every
step instead of waiting for the slowest one to finish. True, and not the hard part.

The hard part is that admitting a request commits KV cache memory you do not yet know the
size of. A request's prompt length is known; its *output* length is not. Admit too few and
the GPU is underutilized. Admit too many and you run out of memory mid-generation, with
requests already half-decoded and no way to make room except by destroying work.

So the scheduler is making a decision under uncertainty, every iteration, with a hard
constraint. That is what this chapter is about.

---

## Where the decision lives

`python/sglang/srt/managers/scheduler.py:3012` `get_next_batch_to_run` is the entry point
Chapter 4 skipped. Its opening is not decision-making at all — it is cleanup:

```python
    def get_next_batch_to_run(
        self, running_batch: ScheduleBatch, last_batch: Optional[ScheduleBatch]
    ) -> NextBatchPlan:
        self.process_pending_chunked_abort()

        if self.enable_fpm:
            self._fpm_batch_t0 = time.monotonic()
        self._abort_on_waiting_timeout()
        self._abort_on_running_timeout(running_batch)
```

Timeouts are enforced here rather than on a timer, because this is the one place that runs
every iteration and holds the queue. Then the previous prefill batch is merged into the
running batch, with the chunked request carefully excluded:

```python
        if self.chunked_req is not None:
            # Move the chunked request out of the batch so that we can merge
            # only finished requests to running_batch.
            chunked_req_to_exclude.add(self.chunked_req)
```

A partially-prefilled request must not join the decode batch — it has no complete KV yet.
This is the first appearance of chunked prefill, which we return to below.

After the merge, the method makes the actual choice, and it is a priority: **prefill if
possible, otherwise decode.** `:3154` `get_new_batch_prefill` is tried first; if it
produces nothing, `:3478` `update_running_batch` advances the decode batch.

That ordering is deliberate. Prefill is what reduces queue depth and TTFT; decode is what
serves requests already admitted. Preferring prefill keeps the queue short at the cost of
occasionally delaying a decode step by one iteration — the trade chunked prefill exists to
bound.

---

## The token budget

`python/sglang/srt/managers/schedule_policy.py:504` `PrefillAdder` is the admission
controller. It is constructed fresh each time a prefill batch is considered, seeded with
the current memory state, and then asked to accept or reject requests one at a time.

Its central question is `:664` `rem_total_tokens`:

```python
    def rem_total_tokens(self):
        if self.is_all_swa:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.swa_available_size()
                + self.tree_cache.swa_evictable_size()
            )
        ...
        else:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.available_size()
                + self.tree_cache.evictable_size()
            )
        return available_and_evictable - self.rem_total_token_offset
```

Read the arithmetic. Available memory is *free pages plus evictable cache*. The radix tree
from Chapter 9 is not a competitor for memory — it is a reserve. Cached prefixes that no
running request holds a lock on can be dropped the moment the space is needed, so from the
scheduler's point of view they are available. This is precisely the `evictable_size_` /
`protected_size_` split that Chapter 9's `inc_lock_ref` maintains: **locked cache is
memory; unlocked cache is free space that happens to be useful.**

The branches are for the pool variants of Chapter 8 — sliding-window attention, hybrid SWA,
hybrid SSM — each of which has its own notion of what is available, because their caches do
not grow or shrink the same way.

`:834` `budget_state` turns the numbers into a verdict:

```python
    def budget_state(self):
        no_token = self.rem_total_tokens <= 0 or self.cur_rem_tokens <= 0
        if not no_token and self.is_hybrid_swa:
            no_token = self.rem_swa_tokens <= 0
        # Gate new mamba slots separately: rem_total_tokens' full_evictable can't
        # cover a mamba slot, which needs mamba-recoverable bytes (see __init__).
        if not no_token and self.rem_mamba_slots is not None:
            no_token = self.rem_mamba_slots <= 0
        if no_token:
            return AddReqResult.NO_TOKEN

        if self.rem_input_tokens <= 0:
            return AddReqResult.OTHER
        ...
        return AddReqResult.CONTINUE
```

Three distinct outcomes (`:498` `AddReqResult`): `CONTINUE` means keep admitting,
`NO_TOKEN` means memory is exhausted, `OTHER` means a different limit bound — the prefill
token budget, or the chunk size. The caller treats them differently: `NO_TOKEN` sets
`batch_is_full` and stops the whole round, while `OTHER` may simply mean this batch is
big enough.

The Mamba comment is a good example of why this code is not simpler than it is. A
fixed-size recurrent state cannot be carved out of evictable token cache, so it needs its
own gate. Generic budget arithmetic would silently over-admit.

`:857` `_update_prefill_budget` charges an accepted request against the budget, and opens
with an admission of imprecision:

```python
        # TODO(lsyin): check this workaround logic, which only ensures the prefill will not out of memory, and may be too conservative
        extend_input_len = self.ceil_paged_tokens(extend_input_len)
```

`ceil_paged_tokens` (`:831`) rounds up to a page. Chapter 8's allocator hands out whole
pages, so a 33-token request with a 32-token page consumes 64 tokens of budget. Budgeting
in tokens while allocating in pages would over-admit by up to a page per request — which,
at 200 concurrent requests, is a substantial and entirely invisible overdraft.

---

## Predicting the future

The budget above accounts for prompt tokens. It cannot account for output tokens, because
nobody knows how long the output will be.

SGLang's answer is the **new token ratio** — a running estimate of how much KV each admitted
request will additionally consume. It lives in
`python/sglang/srt/managers/scheduler_components/new_token_ratio_tracker.py` and is set up
at `python/sglang/srt/managers/scheduler.py:1131` `init_running_status`.

The ratio is adaptive, and the feedback signal is failure. When the scheduler runs out of
memory and has to retract, the ratio moves toward pessimism; when things run smoothly it
decays back toward optimism. It is a controller whose error term is "did we over-admit?"

This is the honest engineering position: the estimate cannot be right, so the system is
built to survive being wrong.

---

## Being wrong: retraction

`python/sglang/srt/managers/scheduler.py:3478` `update_running_batch` handles the decode
side, and its second act is the recovery path:

```python
        # Check if decode out of memory
        if (kv_full_retract_flag := not batch.check_decode_mem()) or (
            TEST_RETRACT and self.forward_ct % TEST_RETRACT_INTERVAL == 0
        ):
            ...
            retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode(
                self.server_args
            )
```

`TEST_RETRACT` is worth noticing: retraction is rare in normal operation and catastrophic
if broken, so there is a mode that forces it periodically. A path that only executes under
pressure is a path that only fails in production.

`python/sglang/srt/managers/schedule_batch.py:2811` `retract_decode` does the work:

```python
        sorted_indices = self._get_decode_retraction_order(self.reqs, server_args)

        retracted_reqs = []
        first_iter = True
        while first_iter or (
            not self.check_decode_mem(selected_indices=sorted_indices)
        ):
            if len(sorted_indices) == 1:
                # Always keep at least one request
                break

            first_iter = False
            idx = sorted_indices.pop()
            req = self.reqs[idx]
            retracted_reqs.append(req)
            # release memory and don't insert into the tree because we need the space instantly
            self.release_req(idx, len(sorted_indices), server_args)
```

Evict running requests until memory is sufficient. Three things stand out.

**"Always keep at least one request."** Retracting everything would free memory and
guarantee no forward progress — the system would thrash. One request always survives, even
if that means it is the only thing running.

**"don't insert into the tree because we need the space instantly."** Normally a finished
request's KV goes into the radix tree for reuse (Chapter 9). Here it is discarded outright.
Inserting would make the memory *evictable* rather than *free*, and the scheduler needs it
now. The retracted request's computed tokens are thrown away.

**Retraction is not abortion.** The retracted request goes back to the waiting queue and
will be re-prefilled later — from its prompt, but now with whatever prefix the tree still
holds. Chapter 9's cache softens the blow: the work is repeated, but often not all of it.

If even retracting to one request is not enough, `reqs_to_abort` is populated and those
requests are genuinely failed, with an `AbortReq` sent back through the Chapter 3 return
path. That is the terminal case: a single request whose context does not fit in the GPU at
all.

The metrics around this block — `num_retracted_reqs`, retracted input and output tokens —
exist because retraction is the clearest signal that admission is mistuned. Chapter 21
covers reading them.

---

## Cache-aware ordering

The budget decides *how many*. The policy decides *which*.

`python/sglang/srt/managers/schedule_policy.py:216` `SchedulePolicy` sorts the waiting
queue, and `:237` `calc_priority` is the entry point. The policies split into two families
(`:200` `CacheAwarePolicy`, `:207` `CacheAgnosticPolicy`):

**Cache-agnostic** — FCFS, longest-output-first (`:410` `_sort_by_longest_output`), random
(`:427` `_sort_randomly`), priority (`:432` `_sort_by_priority_and_fcfs`). These need
nothing from the cache.

**Cache-aware** — LPM (`:374` `_sort_by_longest_prefix`) and DFS-weight (`:387`
`_sort_by_dfs_weight`). These consult the radix tree.

`:314` `_compute_prefix_matches` is the bridge: it runs `match_prefix` against the Chapter 9
tree for queued requests, so the sort can rank by how much each would hit.

**LPM** sorts by longest match. Run the biggest cache hit first: it needs the least new
computation, so it finishes fastest and frees its memory soonest.

**DFS-weight** (`:477` `_calc_weight`, `:483` `_get_dfs_priority`) is subtler. It orders the
queue as a depth-first traversal of the tree, so requests sharing a subtree run
consecutively. LPM optimizes each request in isolation; DFS-weight optimizes the *sequence*,
keeping a shared prefix hot and locked for the whole group rather than letting it be
evicted between two requests that both needed it.

`:296` `_validate_and_adjust_policy` can downgrade a cache-aware policy when the tree is
disabled or the queue is too long to bother matching — the sort itself has a cost, and at
sufficient queue depth it stops paying for itself.

This closes the loop Chapter 9 opened. The cache determines what is cheap; the scheduler
runs what is cheap; running it extends the cache along the same branch; the next similar
request is cheaper still. Chapter 17 lifts the same loop one level up, into routing.

---

## Chunked prefill

A 100,000-token prompt is a problem. Prefill is compute-bound (Chapter 1), so a prompt that
large occupies the GPU for a long time — during which every request in the decode batch
produces nothing. One user's long prompt becomes every other user's latency spike.

Chunked prefill splits it. `python/sglang/srt/managers/scheduler.py:1153`
`init_chunked_prefill` sets the chunk size, and `self.chunked_req` tracks the request
currently being fed through in pieces. Each iteration takes one chunk, so decode steps
interleave between chunks.

The bookkeeping shows up everywhere in this chapter: `get_next_batch_to_run` excludes the
chunked request from the merge; `:2922` `stash_chunked_request` parks it between rounds;
`:2925` `process_pending_chunked_abort` handles the case where it is cancelled mid-prefill.
`PrefillAdder`'s `rem_chunk_tokens` is the third budget in `budget_state`.

The trade is exactly the one Chapter 1 described: TTFT for the long request gets worse,
ITL for everyone else gets better. Nothing is created; latency is moved between customers.

Chapter 17's disaggregation is the alternative answer — instead of interleaving prefill and
decode on one GPU, run them on different machines entirely.

---

## Fairness, and its absence

`:2715` `_add_request_to_queue` is where a new request joins:

```
_set_or_validate_priority   (:2739)   assign or check the request's priority
_abort_on_queued_limit      (:2764)   shed load if the queue is over its cap
_abort_on_waiting_timeout   (:2813)   drop requests that have waited too long
```

Load shedding at the queue boundary is a deliberate choice: better to reject a request
immediately than to accept it, hold it for thirty seconds, and then time it out having
consumed memory and produced nothing.

Note also that **cache-aware policies are not fair.** LPM systematically prefers requests
with long cached prefixes, which in a chat workload means it prefers long-running
conversations over new ones. A brand-new session with no cache hit can be repeatedly
overtaken. `_abort_on_waiting_timeout` is the backstop, and explicit priorities are the
mechanism for saying that some requests matter more than their cache locality suggests.

---

## The whole decision, in order

Each iteration:

1. **Clean up** — process aborts, enforce waiting and running timeouts.
2. **Merge** — fold the last prefill batch into the running batch, excluding any chunked
   request.
3. **Try prefill** — sort the queue by policy (consulting the radix tree if cache-aware),
   then admit requests one at a time through `PrefillAdder` until the budget says
   `NO_TOKEN` or `OTHER`.
4. **Otherwise decode** — filter finished requests, check decode memory, and if it is
   short, retract from the tail until it is not.
5. **Return a plan** — which batch to run, and what the running batch now is.

Every step is bounded by the same resource. Chapter 8 is where that resource is actually
managed.

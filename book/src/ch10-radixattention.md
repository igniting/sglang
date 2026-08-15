# 10. RadixAttention

> *Real workloads share long prefixes, and a radix tree over token sequences turns that
> redundancy into the engine's largest single win. This is SGLang's signature idea.*

Chapter 9 built a memory system that can hand out KV cache a page at a time and take it
back. It has no memory of its own: two identical prompts arriving a second apart are
computed twice, stored twice, and freed twice.

That is a strange thing to accept once you look at real traffic. Production requests are not
independent — they overlap, usually heavily. A chat turn re-sends the entire conversation. A
system prompt of several thousand tokens is identical across every request an application
makes. An agent loop re-sends its whole trajectory on each step, appending one observation.
In all of these the shared part is a *prefix*, and a prefix is exactly the thing a language
model computes left to right.

If the first 4,000 tokens of two requests agree, their first 4,000 KV entries are identical,
bit for bit. Computing them twice is not an approximation being refined. It is arithmetic
being repeated.

RadixAttention is the mechanism that stops the repetition, and the name describes the whole
design: attention that reads its cache out of a radix tree keyed by token sequence. This
chapter builds that tree from the key up, then shows the invariant that keeps it safe while
hundreds of requests read and write it concurrently.

---

## The key

The obvious key for "a cached prefix" is a list of token ids. SGLang's key is a little
more than that, and each addition is load-bearing.

`python/sglang/srt/mem_cache/radix_cache.py:59` `RadixKey`:

```python
class RadixKey:
    __slots__ = ("token_ids", "extra_key", "cache_salt", "is_bigram", "limit")

    def __init__(
        self,
        token_ids: array[int],
        extra_key: Optional[str] = None,
        is_bigram: bool = False,
        limit: Optional[int] = None,
        cache_salt: Optional[str] = None,
    ):
```

The `__slots__` declaration is the first tell. `RadixKey` is constructed on every match,
every insert, and every slice taken during a tree walk — this is the hottest allocation
site in the cache — so it refuses a per-instance `__dict__`.

**`token_ids`** is the sequence itself, held as a Python `array`, not a list. An `array`
of machine integers compares with a single C-level `memcmp` rather than a per-element
Python loop, which the matcher below depends on.

**`extra_key`** is the namespace. Two requests whose token ids agree do *not* share cache
if their `extra_key` differs. The docstring on `match_prefix`
(`python/sglang/srt/mem_cache/radix_cache.py:376`) names the cases:

> * Isolate KV cache lines for different LoRA / adapter IDs.
> * Separate requests that intentionally should not share state (e.g., different sampling
>   salt, cache version, or retrieval augmentation context) by supplying a distinct
>   `extra_key`.

This is a correctness boundary, not a tuning knob. Two requests using different LoRA
adapters produce genuinely different KV entries for the same tokens, because the adapter
changes the projections that produce K and V. Sharing them would silently return another
adapter's activations. Chapter 21 returns to this from the LoRA side.

**`cache_salt`** is deliberately a separate field rather than a convention on `extra_key`,
so that a caller-supplied salt cannot collide with an internally-assigned namespace. The
comment states the scope precisely: it namespaces the in-process tree and the external KV
events, while remote storage keys stay token-only.

**`is_bigram`** exists for EAGLE speculative decoding, where the cached unit is a pair of
adjacent tokens rather than a single one. Rather than materializing a list of tuples,
`maybe_to_bigram_view` (`:156`) flips a flag:

```python
    def maybe_to_bigram_view(
        self,
        is_eagle: bool,
        value: Optional[torch.Tensor] = None,
    ) -> Tuple[RadixKey, Optional[torch.Tensor]]:
        # O(1): flip the bigram flag instead of materializing a tuple list.
        if is_eagle and not self.is_bigram:
            self.is_bigram = True
            if value is not None:
                value = value[: len(self)]
        return self, value
```

`__len__`, `__iter__`, and `__getitem__` (`:99`, `:106`, `:118`) then interpret
`token_ids` through that flag. A bigram key of *n* raw tokens has *n−1* logical units, and
slicing one keeps the boundary token that the next bigram needs — the comment at `:131`
spells out the off-by-one, which is exactly the kind of thing that would otherwise be
rediscovered by a debugger.

**`limit`** caps the key without copying. `_raw_len` (`:87`) returns `min(limit, len)`, and
every other method is written against `_raw_len` rather than `len(token_ids)`. The comment
says why: "behave as if token_ids were sliced to `token_ids[:limit]`, without the O(n)
copy."

### Matching two keys

`RadixKey.match` (`:181`) answers "how long a prefix do these two share?" It is called
once per node on every tree walk, so its cost matters more than its length suggests:

```python
        matched_tokens = n
        lo = 0
        step = 1
        while lo < n:
            hi = lo + step if lo + step < n else n
            if t0[lo:hi] != t1[lo:hi]:
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if t0[lo:mid] == t1[lo:mid]:
                        lo = mid
                    else:
                        hi = mid
                matched_tokens = lo
                break
            lo = hi
            step *= 2
```

This is exponential search — gallop forward in doubling windows, then binary-search the
window that contained the divergence. The comment at `:188` explains the motive: each
window comparison is one C-level slice compare, so a 4,000-token shared prefix costs about
a dozen `memcmp` calls rather than 4,000 iterations of a Python loop. The naive version
would put an O(prefix length) Python loop on the hot path of every request, which for the
workloads in the opening section is precisely the case you cannot afford to be slow in.

Note the first line of the method: `self._check_compatible(other)` (`:169`) raises if
`extra_key` or `cache_salt` differ. The namespace boundary is enforced by an exception,
not by returning zero — a mismatched comparison is a programming error, not a cache miss.

### Page alignment

Chapter 9 established that KV memory is allocated in pages. The tree inherits that
constraint, and `page_aligned` (`:150`) is where it enters:

```python
    def page_aligned(self, page_size: int) -> RadixKey:
        if page_size == 1:
            return self
        aligned_len = len(self) // page_size * page_size
        return self[:aligned_len]
```

Every public entry point calls this before touching the tree. The consequence is worth
stating plainly: **with `page_size > 1` the tree can only remember prefixes that are whole
multiples of the page size.** A 4,001-token prompt with a 32-token page contributes 3,968
tokens to the cache and drops the remaining 33 on the floor. Most of the subtlety in the
rest of this chapter — and the entire existence of `req.cache_protected_len`, which we
reach below — is bookkeeping for that discarded tail.

`child_key` (`:217`) produces the dictionary key used to index a node's children. It
returns the first `page_size` logical units, wrapped in the namespace:

```python
        if self.cache_salt is not None:
            return ((self.extra_key, self.cache_salt), plain)
        return plain if self.extra_key is None else (self.extra_key, plain)
```

For `page_size == 1` with no namespace this degenerates to a bare integer, which makes the
common case a plain dict lookup on a small int.

---

## The node

`python/sglang/srt/mem_cache/radix_cache.py:238` `TreeNode`. The fields divide cleanly by
purpose:

```python
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()
```

**Structure** — `children`, `parent`, `key`. A node holds a *run* of tokens, not a single
token; that is what makes this a radix tree (a compressed trie) rather than a trie. A
4,000-token system prompt shared by every request is one node with a 4,000-element key,
not 4,000 nodes.

**Payload** — `value` is a tensor of KV cache indices, one per logical unit of `key`. This
is the entire content of the cache: the tree does not store keys and values, it stores
*where in the Chapter 9 pool* they live. `evicted` (`:268`) is defined as `value is None`,
which makes "this node's data is gone but its structure remains" a representable state —
the state HiCache needs in Chapter 11.

**Lifetime** — `lock_ref` is the reference count that keeps a node alive while a request
is using it. `last_access_time` feeds eviction.

**Tiering** — `host_value`, `host_ref_counter`, `protect_host` / `release_host` (`:272`–
`:285`) belong to Chapter 11. They are declared here because `HiRadixCache` subclasses
this node rather than wrapping it.

The final method is the one that makes eviction possible:

```python
    def __lt__(self, other: TreeNode):
        return self.last_access_time < other.last_access_time
```

`TreeNode` is orderable by access time, so a heap of nodes is an LRU queue for free.

---

## Matching

`match_prefix` (`:376`) is the read path. Its body, stripped of the docstring, is short:

```python
        key, _ = key.maybe_to_bigram_view(self.is_eagle)

        if self.disable or len(key) == 0:
            return self._empty_match_result

        key = key.page_aligned(self.page_size)

        if len(key) == 0:
            return self._empty_match_result

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
```

Two guards for the empty case, because page alignment can *produce* an empty key from a
non-empty one — a 20-token prompt with a 32-token page matches nothing. `disable` short-
circuits the entire cache, which is how `--disable-radix-cache` is implemented; note it
returns the same pre-built `_empty_match_result` object (`:364`) rather than allocating,
since this path runs per request.

The walk itself, `_match_prefix_helper` (`:678`):

```python
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = access_time
            prefix_len = child.key.match(key, page_size=self.page_size)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = key.child_key(self.page_size)
```

Descend while a child starts with the right token. At each child there are two cases. If
the query agrees with the *whole* of the child's key, consume that much of the query and
continue deeper. If the query diverges partway through the child's key, the tree does not
have a node boundary where we need one — so we make one, and stop.

Two details are easy to read past. `child.last_access_time` is updated during the walk,
which is what makes eviction LRU rather than random: matching against a node counts as
using it, even if the request that matched never gets scheduled. And the accumulated
`value` list is concatenated only once at the end, in `match_prefix`, rather than
repeatedly during descent.

---

## Splitting

`_split_node` (`:704`) is the operation that keeps the tree a *radix* tree. When a query
diverges inside a node's run, that node must become two: a shared parent and a divergent
child.

```python
    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # new_node -> child
        new_node = TreeNode(priority=child.priority)
        new_node.hit_count = child.hit_count
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len].clone()
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:].clone()
        new_node.parent.children[key.child_key(self.page_size)] = new_node
```

Read it as surgery on a linked structure. The new node takes the first `split_len` units
of the child's key and value; the child keeps the rest. The new node is spliced in as the
child's parent and registered under the old child's key in the grandparent.

The line that matters most is `new_node.lock_ref = child.lock_ref`. The child may be in
use by a running request right now. After the split, the prefix that request holds is
spread across two nodes — so the reference count must be *copied*, not moved, or the
shared half becomes evictable while someone is reading it.

The `.clone()` calls are equally deliberate. Slicing a tensor produces a view sharing
storage with the original; if the two halves shared storage, freeing one would corrupt the
other. The clone costs a small copy of an index tensor — never of KV data — to buy
independent lifetimes.

Splitting is not a failure mode. It is how the tree learns where the branch points in the
workload actually are. A system prompt followed by two different user messages starts as
one node and becomes three the first time the second message arrives, and stays that shape
forever after. The docstring at `:408` frames it exactly this way: "this structural
refinement improves subsequent match efficiency and does not duplicate data."

---

## Inserting

`insert` (`:436`) is the write path, and `_insert_helper` (`:737`) is a near-mirror of the
match walk with one extra case at the end:

```python
        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            self._inc_hit_count(new_node, chunked)
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)
            self._record_store_event(new_node)
            node = new_node
        return total_prefix_length, node
```

Walk down matching as far as the tree already knows, splitting where the new key diverges,
then hang whatever is left as a fresh node. The return value is the length that was
*already* present — which the callers use to decide what to free.

One line deserves attention. `insert` at `:447` does:

```python
        if value is not None:
            value = value[: len(key)]
        else:
            # Debug/test fallback: use token ids themselves as values.
            value = torch.tensor(key.token_ids[: len(key)], dtype=torch.int64)
```

That fallback is why `RadixCache.create_simulated` (`:333`) can build a working tree with
no memory pools at all. The tree's logic is independent of what the values mean, so it can
be tested and simulated without a GPU — a property worth noticing when you come to write
tests against it.

`_inc_hit_count` (`:729`) carries a subtle correction:

```python
    def _inc_hit_count(self, node: TreeNode, chunked: bool = False):
        # Skip the hit count update for chunked requests to avoid self-referencing
        # inflation where a chunked request increments hit_count on nodes it created
        # in previous chunks.
        if chunked:
            return
```

A chunked prefill (Chapter 6) inserts its own prefix repeatedly as each chunk lands.
Counting those as hits would make a single long request look like a stream of cache-
friendly ones and distort every metric built on hit count.

---

## The invariant: reference counting

Everything so far describes a cache. What makes it safe is `lock_ref`.

A running request holds KV indices that live in tree nodes. If eviction frees those pages
while the request is still decoding, the request reads memory that now belongs to someone
else. The rule that prevents this is: **a node on the path to any live request's prefix
must not be evicted.**

`inc_lock_ref` (`:622`) walks from a node to the root, incrementing:

```python
        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)
            node = node.parent
        return IncLockRefResult(delta=delta)
```

The walk is to the *root*, not just the node, because holding a prefix means holding every
ancestor that composes it. The accounting moves tokens between two counters —
`evictable_size_` and `protected_size_` — and only on the 0→1 transition, so a node held
by three requests is counted once. Those two numbers are what the scheduler reads in
Chapter 6 when it asks how much memory it can actually reclaim; `delta` is returned so the
caller can update its own budget without recomputing.

`dec_lock_ref` (`:637`) is the mirror, with a guard worth quoting:

```python
            if node.parent is None:
                assert (
                    node is self.root_node
                ), "This request holds the node from another tree"
```

Reaching a parentless node that is not the root means the node was detached — the request
is holding a pointer into a tree that no longer contains it. The assertion converts a
silent memory-corruption bug into a loud one.

The root itself is initialized with `lock_ref = 1` in `reset` (`:353`) so it is never a
candidate for eviction, and with `priority=-sys.maxsize` so any real priority overrides it.

---

## Eviction

`evict` (`:592`) reclaims memory when the allocator runs dry:

```python
        leaves = list(self.evictable_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            self.token_to_kv_pool_allocator.free_segment(x.value, start_pos=0)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))
```

**Eviction is leaf-first, and it must be.** An interior node is a prefix of its children;
freeing it would leave descendants whose cached KV depends on pages that no longer exist.
So only leaves are candidates — and when a leaf is removed, its parent may have just
*become* a leaf, which is why the parent is pushed back onto the heap mid-loop. A single
`evict` call can therefore peel a whole branch from the tip inward.

The candidate set is maintained incrementally rather than recomputed. `evictable_leaves`
is a set kept current by `_update_leaf_status` (`:820`):

```python
    def _update_leaf_status(self, node: TreeNode):
        if node.evicted or node.lock_ref > 0:
            if node in self.evictable_leaves:
                self.evictable_leaves.remove(node)
            return

        for child in node.children.values():
            if not child.evicted:
                if node in self.evictable_leaves:
                    self.evictable_leaves.remove(node)
                return

        if node not in self.evictable_leaves:
            self.evictable_leaves.add(node)
```

A node is evictable exactly when it is unlocked, still resident, and has no resident
children. Every operation that could change one of those three — insert, split, lock,
unlock, delete — calls this. The payoff is that `evict` heapifies a set proportional to the
number of *leaves*, not the number of nodes.

Ordering is delegated to `self.eviction_strategy` (`:328`), selected by name from
`params.eviction_policy`. LRU falls out of `TreeNode.__lt__`; other strategies weigh
priority, which is why `_insert_helper` propagates `max(node.priority, priority)` down the
path it walks.

---

## Where requests meet the tree

Two methods connect the data structure to the request lifecycle, and both are more subtle
than their names suggest.

`cache_unfinished_req` (`:515`) runs when a request has produced KV worth publishing but
is not done — after each prefill chunk, and between decode steps:

```python
        result = self.insert(InsertParams(key=radix_key, value=values, ...))
        new_prefix_len = result.prefix_len

        self.token_to_kv_pool_allocator.free_segment(
            kv_indices[req.cache_protected_len : new_prefix_len],
            start_pos=req.cache_protected_len,
        )

        # The prefix indices could be updated, reuse it
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices, new_last_node = (...)
        ...
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )
        ...
        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)
```

Insert, free the duplicate range the insert revealed, then **match again**. The re-match is
not redundant: if another request inserted an overlapping prefix concurrently, this
request's tokens may now live at *different* pool indices than the ones it just computed
into. The request adopts the tree's indices, writes them back into its `req_to_token` row
(the Chapter 9 first-level mapping), and only then swaps its lock from the old node to the
new one — increment before decrement in effect, so the prefix is never momentarily
unprotected.

`cache_protected_len` is the page-alignment tail from earlier, and the comment at `:564`
is one of the more valuable in the file:

> The `cache_protected_len` is not always equal to `len(req.prefix_indices)` since for
> `page_size > 1`, the partial part is added to `req.prefix_indices`, but that part of kv
> indices is not added to the tree. It should be freed in the next `cache_unfinished_req`
> and final `cache_finished_req` to avoid memory leak.

The request holds more KV than the tree knows about — the partial trailing page. That tail
belongs to the request alone, and the request must free it itself.

`cache_finished_req` (`:458`) runs at completion and frees in two segments (`:501`):

```python
        self.token_to_kv_pool_allocator.free_segments(
            [
                (
                    kv_indices[req.cache_protected_len : freed_end],
                    req.cache_protected_len,
                ),
                (kv_indices[key_len:], key_len),
            ]
        )
```

The first segment is the range the tree already had — a duplicate this request computed
before the insert told it so. The second is the unaligned tail past `key_len`, which was
never eligible for the tree. What is *not* freed is the middle: the range this request
contributed, now owned by the tree and protected by the `lock_ref` it took during insert.

---

## Where the model touches it

Nothing in this chapter has been visible to model code, and that is the point.
`python/sglang/srt/layers/radix_attention.py:91` `RadixAttention` is the layer a model
instantiates in place of a plain attention module. Chapter 13 shows `LlamaAttention`
constructing one and Chapter 14 covers the backends it dispatches to; here it is enough to
note the division of labor. The model declares *that* it attends. The tree decides *what
is already computed*, the Chapter 9 allocator decides *where new entries go*, and the
attention backend reads whatever page table it is handed. A model file contains no cache
logic at all, which is why adding a model (Chapter 23) does not require understanding this
chapter.

---

## The scheduler feedback loop

The tree's most consequential consumer is not the model — it is the scheduler.

`python/sglang/srt/managers/schedule_policy.py:314` `_compute_prefix_matches` runs
`match_prefix` for queued requests, and `:374` `_sort_by_longest_prefix` orders the queue
by how much each one would hit. Chapter 6 covers the policy in full; what matters here is
the shape of the loop:

1. The cache determines which request is cheapest to run.
2. The scheduler runs that request next.
3. Running it extends the cache along the branch it already shared.
4. Which makes the next similar request cheaper still.

Cache-aware scheduling and prefix caching are not two features that happen to compose;
each makes the other worth more. The same loop, one level up, is what makes cache-aware
*routing* work in Chapter 18 — the gateway keeps its own approximate copy of this tree so
it can send a request to the replica most likely to hold its prefix.

---

## A worked example

Three requests, `page_size = 1`, sharing a 6-token system prompt `S`. Write `S = s1..s6`.

**Request A** arrives with `S + "what is 2+2"`. `match_prefix` finds nothing: the root has
no child under `s1`. `_match_prefix_helper` returns immediately with an empty value list
and the root as `last_node`. A runs a full prefill. On completion `cache_finished_req`
inserts the whole sequence; `_insert_helper` finds no existing child, so the entire
sequence becomes one node hanging off the root:

```
root
 └── [s1..s6, "what is 2+2"]          (A's node)
```

**Request B** arrives with `S + "what is 3+3"`. The walk enters A's node and calls
`RadixKey.match`, which galloping-compares and returns 6 + the shared part of "what is "
before diverging at `2` vs `3`. Since `prefix_len < len(child.key)`, `_split_node` fires:

```
root
 └── [s1..s6, "what is "]             (shared, lock_ref copied from A's node)
      └── ["2+2"]                     (A's tail)
```

B's prefill starts from the split point — the shared node's `value` is returned as
`device_indices`, and B computes only its own suffix. On completion B inserts, walks to the
shared node, finds no child under `3`, and hangs a second tail:

```
root
 └── [s1..s6, "what is "]
      ├── ["2+2"]
      └── ["3+3"]
```

**Request C** arrives with just `S`. The walk enters the shared node and matches 6 units
before running out of query. `prefix_len < len(child.key)` again — the query is shorter
than the node — so the node splits once more:

```
root
 └── [s1..s6]                          (the system prompt, now its own node)
      └── ["what is "]
           ├── ["2+2"]
           └── ["3+3"]
```

The system prompt is now a node in its own right, and every subsequent request that starts
with it matches in a single hop. Nobody planned that node; the third request's shape
carved it. That is the property to take away: **the tree's structure converges on the
branch points of the workload, without being told what they are.**

Under memory pressure, `evict` pops from `evictable_leaves` — which contains `["2+2"]` and
`["3+3"]` but *not* the two interior nodes. If A is still decoding, its `lock_ref` also
keeps `["2+2"]` out of the candidate set entirely. Only after both tails are freed does
`["what is "]` become a leaf and join the heap, and only after that the system prompt
itself — exactly the reverse of the order in which the nodes were created, and exactly the
order that preserves the most reuse per byte freed.

<figure>
<svg viewBox="0 0 700 364" role="img" aria-label="A radix tree evolving as three chat requests share a system prompt">
<title>The radix tree after each of three requests</title>
<rect class="dgm-box" x="28.0" y="92" width="180" height="28" rx="6"/>
<text class="dgm-small" x="118.0" y="110.4" text-anchor="middle" style="font-size:11.0px">S + “what is 2+2”</text>
<rect class="dgm-box-accent" x="274.0" y="92" width="152" height="28" rx="6"/>
<text class="dgm-small" x="350.0" y="110.4" text-anchor="middle" style="font-size:11.0px">S + “what is ”</text>
<rect class="dgm-box" x="252.0" y="148" width="76" height="28" rx="6"/>
<text class="dgm-small" x="290.0" y="166.4" text-anchor="middle" style="font-size:11.0px">“2+2”</text>
<rect class="dgm-box" x="346.0" y="148" width="76" height="28" rx="6"/>
<text class="dgm-small" x="384.0" y="166.4" text-anchor="middle" style="font-size:11.0px">“3+3”</text>
<rect class="dgm-box-accent" x="544.0" y="92" width="76" height="28" rx="6"/>
<text class="dgm-small" x="582.0" y="110.4" text-anchor="middle" style="font-size:11.0px">S</text>
<rect class="dgm-box-accent" x="521.0" y="148" width="122" height="28" rx="6"/>
<text class="dgm-small" x="582.0" y="166.4" text-anchor="middle" style="font-size:11.0px">“what is ”</text>
<rect class="dgm-box" x="500.0" y="204" width="76" height="28" rx="6"/>
<text class="dgm-small" x="538.0" y="222.4" text-anchor="middle" style="font-size:11.0px">“2+2”</text>
<rect class="dgm-box" x="594.0" y="204" width="76" height="28" rx="6"/>
<text class="dgm-small" x="632.0" y="222.4" text-anchor="middle" style="font-size:11.0px">“3+3”</text>
<text class="dgm-label" x="118" y="34" text-anchor="middle" font-weight="600" style="font-size:11.5px">after request A</text>
<circle class="dgm-box" cx="118" cy="56" r="9"/>
<text class="dgm-small" x="134" y="60" text-anchor="start" style="font-size:10.5px">root</text>
<path class="dgm-line" d="M118 65 L118 92"/>
<text class="dgm-label" x="350" y="34" text-anchor="middle" font-weight="600" style="font-size:11.5px">after request B — split</text>
<circle class="dgm-box" cx="350" cy="56" r="9"/>
<path class="dgm-line" d="M350 65 L350 92"/>
<path class="dgm-line" d="M340 120 L290 148"/>
<path class="dgm-line" d="M360 120 L384 148"/>
<text class="dgm-label" x="582" y="34" text-anchor="middle" font-weight="600" style="font-size:11.5px">after request C</text>
<circle class="dgm-box" cx="582" cy="56" r="9"/>
<path class="dgm-line" d="M582 65 L582 92"/>
<path class="dgm-line" d="M582 120 L582 148"/>
<path class="dgm-line" d="M572 176 L538 204"/>
<path class="dgm-line" d="M592 176 L632 204"/>
<path class="dgm-dash" d="M234 20 L234 246"/>
<path class="dgm-dash" d="M466 20 L466 246"/>
<path class="dgm-dash" d="M20 264 L680 264"/>
<text class="dgm-small" x="350.0" y="290" text-anchor="middle" style="font-size:11.5px">Nobody planned the node holding the system prompt S — the third request's shape carved it.</text>
<text class="dgm-small" x="350.0" y="308" text-anchor="middle" style="font-size:11.5px">The tree finds the workload's branch points without being told what they are.</text>
<text class="dgm-small" x="350.0" y="334" text-anchor="middle" style="font-size:11.5px">Eviction runs the other way, leaves first: the tails go before “what is ”, and S —</text>
<text class="dgm-small" x="350.0" y="352" text-anchor="middle" style="font-size:11.5px">shared by everything — goes last.</text>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" style="fill:var(--dgm-rule)"/></marker><marker id="arrow-accent" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" style="fill:var(--dgm-accent)"/></marker></defs>
</svg>
<figcaption>Splitting is not a failure mode; it is how the tree learns where the workload actually branches.</figcaption>
</figure>

`pretty_print` (`:585`) will dump this structure from a live server, which is the fastest
way to check that a workload is sharing what you think it is.

---

## Variants

`RadixCache` is one implementation of an interface, and the alternatives are the clearest
way to see what the interface actually promises.

`python/sglang/srt/mem_cache/base_prefix_cache.py:230` `BasePrefixCache` defines it, with
the parameter and result objects at `:49`–`:166` (`MatchPrefixParams`, `MatchResult`,
`EvictParams`, `IncLockRefResult`). Passing structs rather than positional arguments is
what lets `HiRadixCache` return host-tier fields from the same call.

`python/sglang/srt/mem_cache/chunk_cache.py:35` `ChunkCache` implements that interface with
no reuse whatsoever — `match_prefix` always misses, `insert` does nothing durable. It is
the honest baseline, and reading it is the fastest way to see exactly which parts of the
scheduler depend on prefix caching and which do not.

The remaining variants specialize the same algorithm for a different unit of storage:

- `python/sglang/srt/mem_cache/swa_radix_cache.py` — sliding-window attention, where only a
  suffix of the KV is retained, so an ancestor's data may be gone while the node remains.
- `python/sglang/srt/mem_cache/mamba_radix_cache.py` — state-space models, where the cached
  object is a fixed-size recurrent state rather than a growing KV run. Chapter 9's
  `MambaPool` is its backing store.
- `python/sglang/srt/mem_cache/radix_cache_cpp.py` with
  `python/sglang/srt/mem_cache/cpp_radix_tree/` — the same algorithm in C++. Its existence
  is the strongest evidence for the claims made about `RadixKey.match` above: at high
  request rates with long prefixes, tree maintenance became measurable against the Python
  interpreter itself.

Chapter 11 takes the last variant, `HiRadixCache`, which keeps the algorithm and adds a
memory tier beneath it.

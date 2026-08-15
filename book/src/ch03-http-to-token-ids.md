# 3. From HTTP to Token IDs

> *The front of the engine is an async/sync boundary, and most of its complexity is in
> making message passing look like `await`.*

A request has arrived. Over the next five chapters we follow it all the way to a token
appearing on someone's screen, and this chapter covers the first leg: from an HTTP socket
to a message sitting in the scheduler's queue.

Chapter 2 established the topology — a tokenizer process at the front, schedulers owning the
GPUs, detokenizers at the back, ZeroMQ between them. What that diagram does not convey is
the awkwardness of the boundary it creates. The web server is asynchronous, with thousands
of coroutines in flight. The scheduler is a synchronous loop that would not know what an
`await` was. Between them is a socket that has no concept of a request at all, only bytes.

Something has to reconcile those three worlds, and that something is `TokenizerManager`. It
converts text to token ids, which is the job its name advertises. Its larger and more
interesting job is making a fire-and-forget message protocol behave like an ordinary
awaitable function call.

That trick — and the validation, batching, and zero-copy transport that surround it — is
what this chapter is about.

---

## The manager and its event loop

`python/sglang/srt/managers/tokenizer_manager.py:374` `TokenizerManager` runs in the main
process, on the same asyncio event loop as the HTTP server. Its `__init__` (`:387`) is a
sequence of named setup steps in the style Chapter 4 discusses:
`:449` `init_model_config`, `:466` `init_tokenizer_and_processor`,
`:534` `init_ipc_channels`, `:567` `init_running_status`,
`:732` `init_request_dispatcher`.

The public entry point is `:755` `generate_request`, and its first line is worth pausing
on:

```python
    async def generate_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
```

`auto_create_handle_loop` (`:2175`) lazily starts the background task that drains results
from the scheduler. It is called per request rather than at construction because the
manager may be built before an event loop exists — in the embedded `Engine` case
(Chapter 2), construction happens in a context where `asyncio.get_running_loop()` would
fail. Doing it on first use sidesteps the ordering problem entirely.

---

## Text to token ids

`:985` `_tokenize_one_request` handles a single request. It begins by working out what kind
of input it actually received:

```python
        if obj.input_embeds is not None:
            if not self.server_args.disable_radix_cache:
                raise ValueError(
                    "input_embeds is provided while disable_radix_cache is False. "
                    "Please add `--disable-radix-cache` when you launch the server "
                    "if you want to use input_embeds as inputs."
                )
            input_embeds = obj.input_embeds
            input_ids = obj.input_ids
        elif obj.input_ids is not None:
            input_ids = obj.input_ids
        else:
            if self.tokenizer is None:
                raise ValueError(...)
```

Three input formats — raw embeddings, pre-tokenized ids, or text — and the first branch is
a genuine cross-subsystem constraint stated as a runtime error. Chapter 9 explains why:
the radix cache is keyed on *token ids*. Supplying embeddings directly means two requests
with identical ids may have entirely different activations, so prefix sharing would return
the wrong data. The engine cannot detect this, so it refuses the combination.

`:823` `_detect_input_format` and `:361` `InputFormat` classify the request;
`:881` `_tokenize_texts` does batch tokenization; and `:1359` `_create_tokenized_object`
assembles the message that crosses to the scheduler.

Tokenization is CPU work, and at high request rates it is not free.
`python/sglang/srt/managers/tokenizer_manager.py:1570` `_should_use_batch_tokenization`
decides when to batch tokenizer calls, and
`python/sglang/srt/managers/async_dynamic_batch_tokenizer.py` implements the batching so
the fast Rust tokenizer amortizes across requests.

---

## Validation as a stability boundary

`python/sglang/srt/managers/tokenizer_manager.py:1185` `_validate_one_request` is a wall of
checks. In a single-process server such checks
are a nicety; here they are structural. Every request that gets past this point enters a
process shared by every other in-flight request — one that holds tens of gigabytes of model
weights and cannot be cheaply restarted. **A malformed request that crashes the scheduler
is an outage for everyone.**

The dedicated validators name the hazards:

- `:1278` `_validate_mm_limits` — image and video counts, before decoders allocate for them.
- `:1321` `_validate_token_ids_logprob` — logprob parameters against the request's shape.
- `:1341` `_validate_input_ids_in_vocab` — every id inside the embedding table. An
  out-of-range id here would be an illegal memory access inside a CUDA kernel, which does
  not raise a Python exception; it corrupts or kills the process.
- `:1293` `_validate_for_matryoshka_dim` — embedding dimension requests.

The pattern to take away: validation lives on the *safe* side of the process boundary, and
the checks that look most paranoid are the ones guarding operations that fail
uncatchably on the far side.

---

## Turning messages into futures

This is the mechanism the chapter exists for.

Each in-flight request gets a `ReqState` (`:201`):

```python
class ReqState:
    """Store the state a request."""

    out_list: List[Dict[Any, Any]]
    finished: bool
    event: asyncio.Event
    obj: Union[GenerateReqInput, EmbeddingReqInput]

    # For performance metrics
    time_stats: APIServerReqTimeStats
    last_completion_tokens: int = 1
    ttft_observed: bool = False

    # For streaming output
    last_output_offset: int = 0
```

Three fields carry the mechanism. `out_list` accumulates outputs as they arrive. `finished`
marks completion. `event` is an `asyncio.Event` — the bridge. States live in
`self.rid_to_state`, keyed by request id.

The awaiting side is `:1722` `_wait_one_response`:

```python
        state = self.rid_to_state[obj.rid]
        is_stream = getattr(obj, "stream", False)
        while True:
            try:
                await asyncio.wait_for(
                    state.event.wait(), timeout=_REQUEST_STATE_WAIT_TIMEOUT
                )
            except asyncio.TimeoutError:
                if (
                    request is not None
                    and not obj.background
                    and await request.is_disconnected()
                ):
                    # Abort the request for disconnected requests (non-streaming, waiting queue)
                    self.abort_request(obj.rid)
                    raise ValueError(...)
                continue

            # Drain all pending outputs atomically.
            out_list = state.out_list
            state.out_list = []
            finished = state.finished
            state.event.clear()
```

Wait on the event; when it fires, take everything queued and clear. The timeout is not a
failure path — it is a *liveness* path. Without it, a client that disconnects mid-generation
would leave the coroutine parked forever and the request occupying GPU memory until it hit
its token limit. The periodic wake lets the manager poll `request.is_disconnected()` and
abort, which is how a cancelled HTTP request actually frees a KV cache allocation.

Note also `out_list = state.out_list; state.out_list = []` rather than draining in place.
The producer runs on the same event loop, so no lock is needed — but rebinding rather than
mutating means a concurrently-appending producer cannot lose an item between the read and
the clear.

The producing side is `:2200` `handle_loop`, the background task, and `:2215`
`_handle_batch_output`, which looks up `rid_to_state`, appends to `out_list`, sets
`finished`, and calls `event.set()`. That is the whole trick: **a socket receive becomes a
coroutine wake-up.**

`ReqState.append_text` / `get_text` (`:222`, `:226`) are a small but instructive
optimization:

```python
    def append_text(self, chunk: str):
        if chunk:
            self.text_chunks.append(chunk)

    def get_text(self) -> str:
        if self.text_chunks:
            self.text += "".join(self.text_chunks)
            self.text_chunks.clear()
        return self.text
```

Text accumulates as a list and is joined only when read. Concatenating strings per token
would be O(n) per step and O(n²) over a generation — a real cost at 4,000 output tokens.
The comment at `:1739` notes the same concern from the other direction: intermediate
streaming chunks carry `"text": None` so the full string is never rebuilt per step.

---

## The wire contract

`python/sglang/srt/managers/io_struct.py` defines every message that crosses a process
boundary, and reading its type list is the most precise description of the architecture
available.

`python/sglang/srt/managers/io_struct.py:160` `GenerateReqInput` is what arrives from HTTP — text or ids, sampling parameters,
streaming flags, multimodal payloads. `python/sglang/srt/managers/io_struct.py:941` `TokenizedGenerateReqInput` is what leaves for
the scheduler: text resolved to ids, defaults filled, validation passed. Two types rather
than one because the boundary is real — everything past it may assume validity.

Coming back, `python/sglang/srt/managers/io_struct.py:1404` `BatchTokenIDOutput` carries token ids from the scheduler to the
detokenizer, and `python/sglang/srt/managers/io_struct.py:1504` `BatchStrOutput` carries text from the detokenizer to the
tokenizer manager. Both are *batch* types: the scheduler produces results for a whole batch
per step, and sending them individually would multiply syscalls by batch size.

Per `.claude/rules/no-dataclasses.md` these are `msgspec.Struct` rather than `@dataclass` —
faster to serialize, strictly typed, and translatable if the boundary is ever
reimplemented in another language.

---

## ZeroMQ, and who is allowed to talk

Sockets are created in `python/sglang/srt/managers/scheduler.py:733` `init_ipc_channels`
(with `python/sglang/srt/managers/scheduler_components/ipc_channels.py` holding the
details) using the names from `PortArgs` that Chapter 2 introduced. The transport is Unix
domain sockets for same-node communication — the common case, and considerably cheaper
than TCP loopback.

On the scheduler side, `python/sglang/srt/managers/scheduler.py:2007` `init_request_receiver` and
`python/sglang/srt/managers/scheduler_components/request_receiver.py` own the receive path,
and `python/sglang/srt/managers/scheduler.py:1872` `process_input_requests` drains it each iteration.

Dispatch is table-driven. `python/sglang/srt/managers/scheduler.py:1523` `init_request_dispatcher` maps message type to
handler, and the tokenizer manager has a mirror at `python/sglang/srt/managers/tokenizer_manager.py:732`:

```python
        self._result_dispatcher = TypeBasedDispatcher(
            [
                (AbortReq, self._handle_abort_req),
                (OpenSessionReqOutput, self._handle_open_session_req_output),
                (
                    UpdateWeightFromDiskReqOutput,
                    self._handle_update_weights_from_disk_req_output,
                ),
                (FreezeGCReq, lambda x: None),
                # For handling case when scheduler skips detokenizer and forwards back to the tokenizer manager, we ignore it.
                (HealthCheckOutput, lambda x: None),
                ...
            ]
        )
```

The `lambda x: None` entries are informative. A health check bypasses detokenization —
there is no text anyone wants — so it arrives at the tokenizer manager as a message with no
handler. Rather than special-casing the sender, the receiver registers an explicit no-op.
Silence is expressed in the table rather than in a comment somewhere else.

### The rank-0 asymmetry

Under tensor parallelism, `PortArgs` says the tokenizer sends to "scheduler (rank 0)". The
other ranks receive nothing from the front end.

They cannot. Tensor parallelism requires every rank to execute the *same* collective
operations in the *same* order (Chapter 15). If ranks independently received from a socket,
network timing alone could give them different batches, and the first `all_reduce` would
deadlock — or worse, silently mismatch. So rank 0 receives and broadcasts, and every rank
proceeds from an identical view.

The cost is a broadcast on every scheduler iteration, and a structural rule that follows
from it: **any state that affects control flow must be broadcast, never recomputed
per-rank.** A surprising share of distributed hangs are violations of that rule; the
`.claude/skills/debug-distributed-hang/SKILL.md` playbook is organized around finding the
first rank whose state diverged.

---

## Not sending pixels through a socket

Multimodal requests strain the design. An image is megabytes; serializing it, copying it
through a Unix socket, and deserializing it per request would dwarf the cost of the model
forward pass.

`python/sglang/srt/managers/scheduler.py:1906` `_materialize_cuda_vmm_inputs` is the escape
hatch. When the processor has already produced tensors on the GPU, what crosses the socket
is a CUDA IPC handle — a reference other processes can map directly — rather than the data.
`python/sglang/srt/managers/schedule_batch.py:430` `has_cuda_ipc_proxy` and `:440`
`reconstruct` are the receiving side, and
`python/sglang/srt/managers/tokenizer_manager.py:521`
`_validate_cuda_vmm_feature_transport_support` refuses the path where the platform cannot
support it.

Chapter 20 covers the multimodal pipeline in full. The point here is that the process
boundary, which buys isolation and parallelism, charges for every byte crossing it — and
the engine pays that toll only where it is cheap.

---

## Tracing an abort

`AbortReq` is worth following end to end, because it touches every mechanism above.

1. A client disconnects. `_wait_one_response` wakes on its timeout, sees
   `request.is_disconnected()`, and calls `python/sglang/srt/managers/tokenizer_manager.py:1980` `abort_request`.
2. That sends an `AbortReq` over `scheduler_input_ipc_name`.
3. Rank 0 receives it in `process_input_requests`, dispatches via the type table to
   `python/sglang/srt/managers/scheduler.py:4437` `abort_request`, and broadcasts so all
   ranks agree.
4. The scheduler removes the request from the waiting queue, or marks a running request
   finished with `FINISH_ABORT` (`python/sglang/srt/managers/schedule_batch.py:276`).
5. Its KV cache is released through the Chapter 9 path — `cache_finished_req`, which frees
   the request's pages and decrements the tree's lock references.
6. A final output flows back through the detokenizer, `_handle_batch_output` sets the
   state's event, and the awaiting coroutine — if any is still there — unblocks and cleans
   up `rid_to_state`.

Six hops for a cancellation. That is the price of the topology, and it is why abort is a
first-class message type rather than an exception: an exception cannot cross a process
boundary, but a `msgspec.Struct` can.

---

Chapter 4 picks the request up on the other side of the socket, where a synchronous loop is
deciding what to do with it.

# 1. Why Serving Engines Exist

> *The cost structure of autoregressive decoding, not model quality, is what forces an
> engine to exist.*

---

## Two phases, two bottlenecks

A transformer generates text one token at a time. Given a prompt, it produces a
distribution over the next token, samples one, appends it, and repeats. That loop is the
whole of inference, and it splits into two phases with almost nothing in common.

**Prefill** processes the prompt. Every token in the prompt is known up front, so all of
them go through the network at once. A 2,000-token prompt means matrix multiplications
with a 2,000-row activation matrix against the model's weights. Each weight is loaded from
memory once and used 2,000 times.

**Decode** produces the output, one token per step. Only one token per sequence is new, so
the activation matrix has *one* row. Each weight is still loaded from memory once — and
used once.

That difference is not a detail. It is the entire economics of LLM serving.

The relevant measure is **arithmetic intensity**: floating-point operations performed per
byte moved from memory. An H100 delivers roughly 1,000 TFLOP/s of BF16 compute against
roughly 3.35 TB/s of memory bandwidth. The ratio is near 300 FLOP per byte — meaning the
hardware needs to do about 300 operations on every byte it loads to keep its arithmetic
units busy.

A matrix-vector product — which is what decode does — performs two operations per weight
byte loaded. Two, against a requirement of three hundred. The GPU spends more than 99% of
its cycles waiting for memory. You can verify this without any profiler: decode a single
sequence, measure tokens per second, multiply by the model's parameter count in bytes, and
you will land within a few percent of the GPU's rated memory bandwidth. The arithmetic
units are idle; the memory bus is saturated.

Prefill sits on the other side of the line. With 2,000 rows instead of one, arithmetic
intensity rises by three orders of magnitude and the same hardware becomes compute-bound.

The consequence is the founding observation of this book:

> **Decode is memory-bound, and its cost is dominated by re-reading weights that have not
> changed.** The fix is not faster arithmetic. It is to make each weight read serve more
> tokens — which means batching.

If you process 64 sequences together, one weight load serves 64 token computations.
Arithmetic intensity rises 64×, and throughput rises nearly as much for nearly free. This
is why *every* serving engine is, at its core, a batching machine — and why the hard part
is not the model, but deciding what to batch and finding the memory to batch it in.

---

## The KV cache, and the bill it creates

Naively, generating token *n* means re-running attention over all *n−1* previous tokens,
which means recomputing their keys and values every step. Total work grows with the square
of sequence length.

But those keys and values do not change. Token 5's key vector is the same at step 6 and at
step 600. So they are computed once and kept: the **KV cache**. Generation becomes linear
in sequence length, and decode becomes the memory-bound matrix-vector problem described
above.

The trade is compute for memory, and the memory bill is large. Per token:

```
bytes_per_token = 2 × num_layers × num_kv_heads × head_dim × dtype_bytes
```

The leading 2 is for K and V. Work it through for Llama-3-70B — 80 layers, 8 KV heads
(grouped-query attention), 128 head dimension, BF16:

```
2 × 80 × 8 × 128 × 2 = 327,680 bytes ≈ 320 KB per token
```

A 4,000-token conversation therefore holds **1.25 GB** of KV cache. On an 80 GB H100 with
140 GB of weights already spread across two GPUs, perhaps 50 GB per GPU remains. That is
about 40 concurrent 4,000-token conversations — and that number, not FLOPs, is what
determines how many users a GPU serves.

Now recall the previous section: throughput demands large batches, and batch size is
capped by KV cache memory. **Memory capacity is the direct limiter on throughput.** This
is the tension every remaining chapter is a response to:

| Chapter | Response to the memory wall |
| --- | --- |
| 5 | Admit only as many requests as the budget allows; retract when wrong |
| 8 | Allocate in pages so nothing is reserved-but-unused |
| 9 | Store each shared prefix once instead of once per request |
| 10 | Spill cold cache to host memory and disk |
| 14 | Store fewer bytes per entry via quantization |
| 15 | Avoid replicating the cache across parallel ranks |
| 17 | Move prefill somewhere else entirely |

Once you see the KV cache as the scarce resource, the architecture of the engine stops
looking like a collection of features and starts looking like a single sustained argument.

---

## The problem in SGLang's own terms

The smallest thing in the repository that runs a real forward pass is
`python/sglang/benchmark/one_batch.py`. It has no server, no scheduler, and no HTTP — it
loads a model, builds one batch, and times prefill and decode separately. (The top-level
`python/sglang/bench_one_batch.py` is a deprecated shim that re-exports it.)

Its two core functions are exactly the two phases. `python/sglang/benchmark/one_batch.py:487` `extend`:

```python
    batch.prepare_for_extend()
    ...
    forward_batch = ForwardBatch.init_new(
        batch,
        model_runner,
        return_hidden_states_before_norm=False,
    )
    logits_output = model_runner.forward(forward_batch).logits_output
    next_token_ids = model_runner.sample(logits_output, forward_batch)
```

And `python/sglang/benchmark/one_batch.py:526` `decode`:

```python
    batch.input_ids = input_token_ids.to(torch.int64)
    batch.prepare_for_decode()
    ...
    logits_output = model_runner.forward(forward_batch).logits_output
    next_token_ids = model_runner.sample(logits_output, forward_batch)
```

The bodies are nearly identical. The difference is one method call —
`prepare_for_extend` versus `prepare_for_decode` — and everything downstream branches on
what that produces. Chapter 6 shows exactly how far that branch propagates.

Two things in `extend` are worth noticing now. The `dummy_tree_cache` built at `:488` is a
`TreeCacheNamespace` (`:469`) that carries only a page size, a device, and an allocator:
the benchmark deliberately runs *without* prefix caching, so its numbers reflect raw
forward-pass cost rather than cache hits. And both functions call `model_runner.forward`
followed by `model_runner.sample` — the same two-step contract Chapter 7 examines.

The measurement code (`python/sglang/benchmark/one_batch.py:734` `latency_test_run_once`) reports the
two phases in different units, which is itself the lesson:

```python
    prefill_latency = time.perf_counter() - tic
    ...
    throughput = input_len * batch_size / prefill_latency
```

```python
        latency = time.perf_counter() - tic
        ...
        throughput = batch_size / latency
```

Prefill throughput counts `input_len × batch_size` tokens; decode counts `batch_size`
tokens. One step of prefill on a batch of 8 with 2,000-token prompts moves 16,000 tokens.
One step of decode on the same batch moves 8. Run it and the printed numbers differ by
roughly three orders of magnitude — the arithmetic-intensity argument, made concrete on
your own hardware in about ninety seconds.

---

## What the metrics mean

Serving performance is not one number, and the common ones pull against each other.

**TTFT (time to first token)** — request arrival to first streamed token. Dominated by
queueing plus prefill. This is what a user perceives as responsiveness.

**ITL / TPOT (inter-token latency / time per output token)** — the gap between successive
tokens once generation starts. This is perceived as reading speed; below roughly 30 ms it
stops mattering to a human reader.

**Throughput** — total tokens per second across all requests. The operator's metric, since
it sets cost per token.

**Goodput** — throughput counting only requests that met their latency targets. The honest
metric, and the one that exposes the others as incomplete: an engine can post excellent
throughput while every individual request violates its SLO.

They conflict in specific, structural ways:

- **Batch size** raises throughput and raises ITL. Every sequence in a batch waits for the
  slowest operation in that step.
- **Chunked prefill** (Chapter 5) lowers ITL for requests already decoding, and raises TTFT
  for the request being chunked. It moves latency between customers rather than removing
  it.
- **Prefix caching** (Chapter 9) is the rare case that improves everything at once — it
  removes work rather than relocating it. This is why it gets a chapter of its own.
- **Speculative decoding** (Chapter 18) lowers ITL while *raising* total compute. It spends
  the idle arithmetic units that the first section identified.

Whenever a later chapter says a technique "helps," the question to ask is which of these
four it helps, and which it charges.

---

## The idea inventory

`README.md:68` compresses SGLang's answer into a single paragraph:

> **Fast Runtime**: Provides efficient serving with RadixAttention for prefix caching, a
> zero-overhead CPU scheduler, prefill-decode disaggregation, speculative decoding,
> continuous batching, paged attention, tensor/pipeline/expert/data parallelism, structured
> outputs, chunked prefill, quantization (FP4/FP8/INT4/AWQ/GPTQ), and multi-LoRA batching.

Unpacked, with the problem each one solves and where this book covers it:

| Feature | The problem it attacks | Chapter |
| --- | --- | --- |
| RadixAttention | Shared prompt prefixes recomputed per request | 9 |
| Zero-overhead scheduler | CPU scheduling stalling the GPU between steps | 4 |
| PD disaggregation | Prefill and decode contending on one machine | 17 |
| Speculative decoding | Decode's idle arithmetic units | 18 |
| Continuous batching | Requests waiting for a batch to drain | 5 |
| Paged attention | Memory reserved for growth that never happens | 8 |
| TP / PP / EP / DP | A model too large, or a cache too replicated, for one GPU | 15, 16 |
| Structured outputs | Generation that must satisfy a schema | 19 |
| Chunked prefill | One long prompt stalling every active decode | 5 |
| Quantization | Too many bytes per weight and per cache entry | 14 |
| Multi-LoRA batching | Requests in one batch needing different weights | 20 |

Read that table as a map of the memory-and-latency argument above. Nothing in it is a
feature for its own sake; each is a response to a specific line in the cost model. The rest
of the book is that table, expanded, with the code that implements each row.

Chapter 2 gets us oriented in the repository before we start walking a request through it.

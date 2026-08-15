# 1. Why Serving Engines Exist

> *The cost structure of autoregressive decoding, not model quality, is what forces an
> engine to exist.*

Ask an engineer why running a large language model is expensive and you will usually hear
that the models are enormous and matrix multiplication is costly. That answer is correct
about training and almost entirely wrong about serving — and the difference is not a
detail. It determines what an inference engine is for.

This chapter works out what generating a token actually costs. The answer is strange enough
to be worth stating up front: a graphics card producing one token for one user is idle
more than 99% of the time, and it is idle not because the work is easy but because the data
cannot reach the arithmetic units fast enough.

Getting from there to a useful machine takes two steps, and the second creates the problem
the rest of this book is about. By the end you will be able to compute, for a given model
and GPU, roughly how many people it can serve at once — and see why that number, rather than
any measure of speed, is the one everything else in the engine is organized around.

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

## The roofline, and the batch size it demands

The argument above deserves to be made precisely, because the precise version tells you
*how much* batching is enough — a number the engine will spend the rest of the book trying
to reach.

The tool is the **roofline model** (Williams, Waterman, and Patterson, 2009). Any kernel has
an arithmetic intensity *I*, in FLOP per byte of memory traffic. The hardware has a peak
compute rate π (FLOP/s) and a peak bandwidth β (byte/s). Achievable performance is capped by
whichever runs out first:

```
performance ≤ min(π, I × β)
```

Plotted against *I*, that is a rising line that flattens into a ceiling. The corner is the
**ridge point**, *I*\* = π / β — the intensity at which a kernel stops being starved by
memory and starts being limited by arithmetic. For an H100 SXM at BF16, π ≈ 990 TFLOP/s
and β ≈ 3.35 TB/s, so

```
I* = 990e12 / 3.35e12 ≈ 296 FLOP per byte
```

Now compute *I* for the thing decode actually does. One linear layer with weight matrix
*W* ∈ R^(*K*×*N*), applied to a batch of *B* token rows, costs 2*BKN* FLOP and reads *KN*·*s*
bytes of weights, where *s* is bytes per element. Ignoring activation traffic, which is
small when *B* is small:

```
I(B) = 2BKN / (KN × s) = 2B / s
```

The weight dimensions cancel. Arithmetic intensity for a weight-bound GEMM depends on
**nothing but the batch size and the element width** — not on the model, not on the layer,
not on how big the matrix is. That single fact explains most of what a serving engine does.

Setting *I*(*B*) = *I*\* and solving gives the **critical batch size**, the point at which the
GPU stops idling:

```
B* = s × π / (2β) = 2 × 296 / 2 ≈ 296 rows at BF16
```

Under 300 rows, adding work to a batch is nearly free: you are paying for bandwidth you have
already spent. Past it, arithmetic starts to cost real time. The number moves with the dtype
in the direction you would expect — an FP8 weight halves *s* but roughly doubles π on
Hopper, so *B*\* lands in the same neighbourhood — and it moves with the hardware, which is
why the engine measures rather than assumes.

<figure>
<svg viewBox="0 0 700 348" role="img" aria-label="A log-log roofline plot showing decode far to the left of the ridge point and prefill at the compute ceiling">
<title>The roofline, and where decode sits on it</title>
<text class="dgm-label" x="376" y="22" text-anchor="middle" font-weight="600" style="font-size:12.5px">H100 SXM, BF16 — attainable performance vs arithmetic intensity</text>
<path class="dgm-line" d="M96 44 L96 244 L664 244"/>
<text class="dgm-small" x="88" y="248" text-anchor="end" style="font-size:10.5px">1</text>
<text class="dgm-small" x="88" y="184.0" text-anchor="end" style="font-size:10.5px">10</text>
<text class="dgm-small" x="88" y="120.0" text-anchor="end" style="font-size:10.5px">100</text>
<text class="dgm-small" x="88" y="56.0" text-anchor="end" style="font-size:10.5px">1000</text>
<text class="dgm-small" x="38" y="150" text-anchor="middle" style="font-size:11px">TFLOP/s</text>
<text class="dgm-small" x="96" y="264" text-anchor="middle" style="font-size:10.5px">1</text>
<text class="dgm-small" x="236.0" y="264" text-anchor="middle" style="font-size:10.5px">10</text>
<text class="dgm-small" x="376.0" y="264" text-anchor="middle" style="font-size:10.5px">100</text>
<text class="dgm-small" x="516.0" y="264" text-anchor="middle" style="font-size:10.5px">1000</text>
<text class="dgm-small" x="656" y="264" text-anchor="middle" style="font-size:10.5px">10⁴</text>
<text class="dgm-small" x="376.0" y="284" text-anchor="middle" style="font-size:11px">arithmetic intensity — FLOP per byte</text>
<path class="dgm-line-accent" d="M96.0 210.4 L441.9 52.3 L656.0 52.3"/>
<text class="dgm-small" x="124.6" y="224.7" text-anchor="start" style="font-size:11px">memory-bound</text>
<text class="dgm-small" x="124.6" y="233.7" text-anchor="start" style="font-size:11px">slope = 3.35 TB/s</text>
<text class="dgm-small" x="536.5" y="40.3" text-anchor="middle" style="font-size:11px">compute-bound — 990 TFLOP/s</text>
<path class="dgm-dash" d="M441.9 244 L441.9 52.3"/>
<text class="dgm-small" x="450.9" y="236" text-anchor="start" font-weight="600" style="font-size:11px">I* = 296</text>
<circle class="dgm-fill-accent" cx="138.1" cy="191.1" r="4.5"/>
<text class="dgm-small" x="148.1" y="195.1" text-anchor="start" style="font-size:11px">decode, batch 1</text>
<circle class="dgm-fill-accent" cx="348.9" cy="94.8" r="4.5"/>
<text class="dgm-small" x="358.9" y="98.8" text-anchor="start" style="font-size:11px">decode, batch 64</text>
<circle class="dgm-fill-accent" cx="558.1" cy="52.3" r="4.5"/>
<text class="dgm-small" x="558.1" y="70.3" text-anchor="middle" style="font-size:11px">prefill</text>
<text class="dgm-small" x="350.0" y="318" text-anchor="middle" style="font-size:11.5px">Batching moves a workload right along the slope. The whole point is to reach the corner.</text>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" style="fill:var(--dgm-rule)"/></marker><marker id="arrow-accent" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" style="fill:var(--dgm-accent)"/></marker></defs>
</svg>
<figcaption>Arithmetic intensity for a weight-bound GEMM is <code>2B/s</code> — batch size over element width, and nothing else. Everything the scheduler does is an attempt to move right.</figcaption>
</figure>

So "batch more" has a target: **a few hundred token rows per forward pass**. Below it the
GPU is a very expensive memory controller. That target is the reason chunked prefill
(Chapter 5) mixes prefill chunks with decode rows rather than running them separately, the
reason speculative decoding (Chapter 18) is profitable at all, and the reason the scheduler
would rather wait a few milliseconds than launch a batch of four.

### Where the roofline stops applying

One part of the forward pass refuses to follow this argument, and it is worth naming now
because three later chapters exist because of it.

Batching amortizes a weight read across the whole batch *because every sequence multiplies
against the same weights*. Attention has no such matrix. Each sequence attends over **its
own** KV cache, which no other sequence in the batch shares. Reading it is
`2 × L × H_kv × d × s × T` bytes for a sequence of *T* tokens, and it serves exactly one
query row. Its arithmetic intensity is around 2 FLOP per byte at any batch size.

Attention is therefore memory-bound *no matter how large the batch gets*. Batching fixes the
GEMMs and leaves attention exactly where it was. That is why:

- attention gets its own pluggable backend layer and its own hand-written kernels
  (Chapter 13) while the linear layers are ordinary PyTorch;
- shrinking the KV cache per token is an architectural priority worth redesigning attention
  around — GQA, and then MLA (Chapter 12);
- *sharing* KV across requests, so one read serves many, is the single highest-leverage
  optimization in the system (Chapter 9).

Two regimes, then, inside one forward pass: weight traffic that batching amortizes, and KV
traffic that it does not. Almost every design decision in SGLang is aimed at one or the
other.

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

### Little's Law closes the loop

There is one more relation worth writing down, because it turns "how much memory do I have"
into "how many requests per second can I serve" without any reference to the model.

Little's Law, from queueing theory, says that for any stable system the average number of
items resident inside it equals arrival rate times average residence time:

```
L = λ × W
```

Here *L* is the number of requests concurrently in flight, λ the arrival rate, and *W* the
average end-to-end latency. The engine does not get to choose *L* freely: it is bounded
above by how many KV caches fit in memory. Rearranged,

```
λ_max = L_max / W
```

Take the Llama-3-70B numbers above — about 40 concurrent 4,000-token conversations — and
suppose an average request takes 20 seconds end to end. Then the ceiling is 2 requests per
second, and no amount of kernel tuning moves it. Only three things do: fit more caches into
memory (raise `L_max`), finish requests faster (lower *W*), or stop storing the same prefix
forty times (raise `L_max` again, and by the largest factor available).

This is also why an overloaded engine degrades so sharply rather than gracefully. Push λ
above λ_max and *W* does not rise a little — queueing delay grows without bound until
something sheds load. Chapter 5's admission control is that something, and Chapter 5's
retraction machinery is what happens when the estimate that admitted a request turns out to
have been optimistic.

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

# 20. Per-Request Variation: LoRA and Multimodal

> *Both features break the assumption that every request in a batch needs the same weights
> and the same kind of input — and both are solved by extending the batch, not splitting it.*

Every chapter so far has assumed a batch is homogeneous — same weights, same kind of input,
one kernel over all of it. That assumption is precisely what makes batching profitable, and
Chapter 1 established that batching is everything.

Two features break it.

Some requests want **different weights**: one is using a summarization adapter, another a
code adapter, a third the base model. Some requests carry **different input types**: text,
or three images, or audio.

The obvious response in both cases is to split the batch — group by adapter, group by
modality, run them separately. That would destroy the thing being protected. With twenty
adapters you get twenty tiny batches instead of one large one, and the GPU goes back to
being idle.

So both are solved the same way instead: keep one batch, and turn the per-request variation
into a tensor the kernel indexes into. If that sounds familiar, it should — it is the same
move as the page table in Chapter 8, the sorted expert order in Chapter 16, and the page
directory passed to the attention kernel in Chapter 13. This chapter is where the pattern
becomes impossible to miss.

---

## Batching across adapters

### The low-rank hypothesis

LoRA (Hu et al., 2021) starts from an empirical claim about fine-tuning: the *change* a
fine-tune makes to a weight matrix has far lower rank than the matrix itself. If that is
true, the update can be factored:

```
W = W₀ + ΔW = W₀ + (α/r) · B A
```

with *A* ∈ R^(*r*×*k*), *B* ∈ R^(*d*×*r*), and *r* ≪ min(*d*, *k*). *W*₀ is frozen; only *A*
and *B* train. The α/*r* scaling exists so that changing *r* does not force you to re-tune the
learning rate.

The evidence is that it works at absurd ratios. For GPT-3 175B the paper reports a
**10,000×** reduction in trainable parameters and 3× less optimizer memory, at quality
matching full fine-tuning — and finds that on some tasks *r* = 1 suffices against a full rank
of 12,288. They adapt only `W_q` and `W_v`, leaving the MLP untouched.

For training, the win is memory. For *serving*, the win is a different one, and it is the
reason this chapter exists.

An adapter is a few megabytes against a base model of a hundred gigabytes. So a single loaded
base model can serve dozens of fine-tunes at once, if — and this is the whole engineering
problem — you can apply *different* adapters to *different rows of the same batch*.

The paper's own deployment advice points the wrong way for this. It observes there is no
inference latency penalty because you can fold the update in: compute *W*₀ + *BA* once and use
it as an ordinary weight matrix. That is right for one adapter and useless for many, since a
merged weight serves exactly one fine-tune and re-merging per request would cost more than
the forward pass.

So a multi-adapter server must keep the factors separate and compute

```
y = x W₀  +  (α/r) (x A) B
```

where *W*₀ is shared by the whole batch and *A*, *B* vary per row. The base GEMM runs once,
exactly as before. The correction is a pair of very thin matrix multiplies — rank 16 against
a 4,096-dimensional weight is well under 1% of the work — but it is a *ragged* one, because
each row wants a different *A* and *B*, and possibly a different *r*.

That is the same shape as Chapter 16's grouped GEMM and Chapter 8's page table: irregularity
expressed as an index tensor rather than as control flow. Punica and S-LoRA named the
resulting primitive **segmented gather matrix-vector multiply** — group the batch's rows by
adapter, and have the kernel look up each row's factors from a shared pool. That is precisely
what the arrays below construct.

`python/sglang/srt/lora/lora_manager.py:59` `LoRAManager` orchestrates it, and `:428`
`prepare_lora_batch` is where the batch is assembled:

```python
    def prepare_lora_batch(self, forward_batch: ForwardBatch):
        # set up batch info shared by all lora modules
        bs = forward_batch.batch_size
        ...
        weight_indices = [0] * len(forward_batch.lora_ids)
        lora_ranks = [0] * self.max_loras_per_batch
        scalings = [0] * self.max_loras_per_batch
        for i, uid in enumerate(forward_batch.lora_ids):
            if uid not in self.memory_pool.uid_to_buffer_id:
                continue
            weight_indices[i] = self.memory_pool.get_buffer_id(uid)
            if uid is not None:
                lora = self.loras[uid]
                lora_ranks[weight_indices[i]] = lora.config.r
                scalings[weight_indices[i]] = lora.scaling
```

Three parallel arrays, and they are the segmented-gather primitive in concrete form.
`weight_indices` says which adapter buffer each *request* uses; `lora_ranks` and `scalings`
carry per-adapter *r* and α/*r*. These go to the kernel, which looks up each row's adapter and
applies the right correction. **One kernel launch, many adapters.**

Note `lora_ranks` is sized `max_loras_per_batch`, not batch size: it is indexed by *buffer
slot*, not by request. Several requests sharing an adapter share a slot.

The CUDA-graph interaction shows in the same method:

```python
        use_cuda_graph = (
            hasattr(self, "max_bs_in_cuda_graph")
            and bs <= self.max_bs_in_cuda_graph
            and forward_batch.forward_mode.is_cuda_graph()
        )
        # Eligible extend batches refresh the static prefill batch info in
        # place so captured kernels read current values at replay.
        use_prefill_cuda_graph = not use_cuda_graph and self.can_use_prefill_cuda_graph(
            forward_batch
        )
```

Chapter 14's graphs need static buffers, so the index arrays must live at fixed addresses
and be *updated in place* rather than reallocated. `:121` `init_cuda_graph_batch_info` and
`:146` `init_prefill_cuda_graph_batch_info` allocate them once.

`python/sglang/srt/lora/backend/` holds the kernels — grouped GEMMs of the punica family,
which compute many small rank-*r* products in one launch.
`python/sglang/srt/lora/layers.py` wraps the Chapter 12 layers,
`python/sglang/srt/lora/lora_moe_runners.py` handles MoE layers (Chapter 16), and
`python/sglang/srt/lora/deepseek_mla_correction.py` handles MLA, where the compressed KV
representation means the correction cannot be applied the usual way.

---

## Adapters as a memory pool

Adapters live in a fixed pool of buffer slots, exactly as KV does in Chapter 8.
`python/sglang/srt/lora/mem_pool.py` manages it, `max_loras_per_batch` caps how many can be
active at once, and `python/sglang/srt/lora/eviction_policy.py` decides what leaves when a
new one arrives.

Loading is a runtime operation: `:221` `load_lora_adapter` and `:315` `unload_lora_adapter`,
reachable through `python/sglang/srt/entrypoints/engine.py:1523` `load_lora_adapter` and
`:1536` `unload_lora_adapter`. `python/sglang/srt/lora/lora_registry.py` tracks what exists,
`python/sglang/srt/lora/lora_drainer.py` waits for in-flight requests before an unload, and
`python/sglang/srt/lora/lora_overlap_loader.py` overlaps loading with compute.

Admission gains a constraint. `python/sglang/srt/managers/scheduler.py:3450`
`_can_schedule_lora_req` is consulted during Chapter 5's prefill admission: a request whose
adapter is not resident, in a batch already using all slots, cannot be scheduled — a
*non-memory* reason to reject a request, which the token budget alone would never catch.
`python/sglang/srt/lora/lora_manager.py:361` `validate_lora_batch` enforces it.

### Adapters must not share a prefix cache

Here is the correctness trap.

Two requests with identical token sequences but different adapters produce **different KV
entries**, because the adapter changes the projections that compute K and V. If Chapter 9's
radix cache matched them by tokens alone, the second request would receive the first
adapter's activations. No error, no crash — just a model producing subtly wrong output.

Chapter 9's `RadixKey` extra key
(`python/sglang/srt/mem_cache/radix_cache.py:64`) is the guard. Setting `extra_key` to the
adapter id makes the namespaces disjoint, and `:169` `_check_compatible` raises rather than
silently comparing across namespaces.

The cost is that adapters do not share cache with each other or with the base model — *n*
adapters serving the same system prompt cache it *n* times. That is unavoidable: the entries
genuinely differ.

---

## Non-text input, spliced into text

Now the other half.

A vision-language model runs images through an encoder into embeddings, then feeds those
embeddings to the decoder *in place of* text token embeddings. The prompt contains
placeholder tokens; the pipeline replaces their embeddings with encoder output.

### Why that specific architecture won

The design above is not the only way to build a multimodal model, and the alternatives lost
for reasons that matter to a serving engine.

The transformer takes a sequence of `d`-dimensional vectors. It has no opinion about where
they came from. So *any* modality that can be turned into a sequence of `d`-dimensional
vectors can be fed to a language model — the question is only how, and what has to be
retrained.

**Cross-attention** was the first widely used answer: keep the two streams separate and let
the decoder attend to the encoder's output through added cross-attention layers, as Flamingo
did. It works, but it changes the decoder's architecture, so the language model must be
modified and retrained.

**Projection into the token stream** is what almost everything since LLaVA does instead.
Encode the image with a vision transformer, pass its patch embeddings through a small learned
projector — often a two-layer MLP — into the language model's embedding dimension, and splice
the results into the sequence as if they were tokens. The decoder is *unmodified*. Only the
projector is trained, on a comparatively small dataset.

Three consequences follow, and all three are visible in the code below.

**The image becomes tokens, with a token's costs.** A 336×336 image at patch size 14 is 576
patches, so 576 positions of context, 576 rows of KV cache, and 576 tokens of Chapter 5's
budget. High-resolution schemes that tile an image into sub-images multiply that — a single
photograph can cost more context than a page of text. Every accounting question in this book
applies to it unchanged, which is exactly why the engine treats image embeddings as tokens
rather than as a special case.

**The splice must happen after tokenization and before the decoder.** The prompt is
tokenized with *placeholders* — real token ids that stand in for image positions — and the
placeholders' embeddings are overwritten with projector output. Hence
`build_padded_input_ids` below, and hence the care about placeholder ids not colliding with
real ones.

**The encoder is a separate model with separate economics.** It runs once per image, is
compute-bound, and its output depends only on the image. That combination is the definition
of something worth caching, and — in Chapter 17's terms — worth running somewhere else
entirely.

`python/sglang/srt/multimodal/processors/` holds 53 processors — one per model family,
because each has its own preprocessing, patch layout, and placeholder convention — over a
shared `python/sglang/srt/multimodal/processors/base_processor.py`.
`python/sglang/srt/managers/multimodal_processor.py` and
`python/sglang/srt/managers/mm_utils.py` are the runtime glue.

`python/sglang/srt/managers/schedule_batch.py:318` `MultimodalDataItem` is one piece of
non-text input and `:590` `MultimodalInputs` the per-request collection.
`:569` `build_padded_input_ids` does the splicing:

```python
    def build_padded_input_ids(input_ids, mm_items: List[MultimodalDataItem]):
```

with `:374` `set_pad_value` and `:218` `_compute_pad_value` deriving placeholder values from
a hash. Hashing matters for two reasons: the placeholder must not collide with a real token
id (hence `:194` `sanity_check_mm_pad_shift_value`), and the hash identifies the item for
caching. `python/sglang/srt/mem_cache/multimodal_cache.py` caches encoder outputs, so the
same image in a follow-up turn is not re-encoded — Chapter 9's idea applied to a different
resource.

---

## Position arithmetic moves into the scheduler

Text positions are `0, 1, 2, …`. Images are two-dimensional, and models like Qwen-VL use
**mrope** — multi-dimensional rotary embeddings — where a token's position is a
(temporal, height, width) triple.

Those positions cannot be computed by the model alone, because they depend on the image
grid, the placeholder layout, and where the request is in its generation. So the computation
moves up the stack: `python/sglang/srt/managers/scheduler.py:2308`
`_maybe_compute_mrope_positions` in the scheduler, and
`python/sglang/srt/model_executor/forward_batch_info.py:1163` `_compute_mrope_positions` in
the forward batch, with separate decode and extend paths (`:1181`
`_compute_mrope_positions_decode`, `:1247` `_compute_mrope_positions_extend`) and a
speculative variant (`:1104` `compute_spec_mrope_positions`) for Chapter 18.

This is a genuine abstraction leak — position computation belongs to the model — and it is
accepted because the scheduler is the only component that knows the batch layout.

---

## Not sending pixels through a socket

Chapter 3 introduced the problem: images are megabytes, and serializing them through ZeroMQ
per request would dwarf the forward pass.

`python/sglang/srt/managers/scheduler.py:2209` `_process_and_broadcast_mm_inputs` is the
receiving side, and the CUDA IPC path avoids the copy entirely:
`python/sglang/srt/managers/schedule_batch.py:430` `has_cuda_ipc_proxy`, `:440`
`reconstruct`, `:465` `can_defer_cuda_ipc_feature_reconstruction`, and `:486`
`acknowledge_deferred_cuda_ipc_feature`.

The deferred reconstruction is worth noting: under tensor parallelism several ranks need the
same tensor, so the handle is mapped by each and released only once all have acknowledged —
a reference count across processes.

`:624` `release_features` frees encoder output once it is spliced, because embeddings for a
high-resolution image are large and holding them for the request's lifetime would be
expensive.

---

## Where the encoder runs

The encoder is a third workload with a third profile — compute-heavy, bursty, no KV cache.
There are three placements:

**In-process.** Simplest; the encoder competes with the decoder for the same GPU.

**Data-parallel.** Several encoder replicas feeding one decoder, for image-heavy workloads.
`docs/docs/advanced_features/dp_for_multi_modal_encoder.mdx`.

**Separate service.** Chapter 17's EPD disaggregation —
`python/sglang/srt/disaggregation/encode_server.py`.

`python/sglang/srt/managers/mm_schedule.py` handles the scheduling interaction, and
`docs/docs/advanced_features/cuda_graph_for_multi_modal_encoder.mdx` covers capturing the
encoder under Chapter 14's graphs, which is harder than for the decoder because image sizes
vary more than batch sizes do.

---

## The shared shape

Strip away the specifics and both features have the same structure:

| | LoRA | Multimodal |
| --- | --- | --- |
| Varies per request | weights | input type |
| Naive fix | separate batch per adapter | separate batch per modality |
| Actual fix | per-request index into a buffer pool | splice encoder output into the token stream |
| Pool | adapter slots (`python/sglang/srt/lora/mem_pool.py`) | encoder cache (`python/sglang/srt/mem_cache/multimodal_cache.py`) |
| Admission constraint | adapter must be resident | encoder capacity |
| Cache interaction | namespaced by `extra_key` | hashed and cached separately |

Both convert what looks like control flow into a data structure the kernel indexes. That is
the same move as Chapter 8's page table, Chapter 13's `kv_indptr`, and Chapter 16's
expert-major sort — probably the single most repeated idea in this codebase.

Part VI ends here. You now have the whole engine. Part VII is about living with it: seeing
what it is doing, and changing what it does.

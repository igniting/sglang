# 13. Attention Backends

> *Attention is pluggable because hardware, sequence shape, and kernel maturity all vary
> independently — and the plug is a two-phase metadata/kernel contract.*

Chapter 12 ended with a single line in a model file: `self.attn = RadixAttention(...)`.
Behind that line is the most heavily optimized operation in machine learning, and there is
no single best implementation of it.

The variation runs along four independent axes. Hardware — NVIDIA, AMD, Intel, Ascend, TPU,
CPU. Sequence shape — a 100,000-token prefill and a batch-256 single-token decode are
different problems wearing the same name. Attention variant — standard, grouped-query,
DeepSeek's compressed latent form, sparse, sliding-window, or not attention at all. And
maturity, because a new kernel library ships a fast path for one case long before it covers
the rest.

The product of those axes is a matrix no single kernel fills. So SGLang defines a contract
and lets implementations compete for each cell.

This is also where the book first goes below the PyTorch line. The Triton kernels are
Python, and reading one is the clearest way to see what "paged attention" actually means:
not a metaphor, but an address computation performed inside the inner loop.

---

## The contract

`python/sglang/srt/layers/attention/base_attn_backend.py:33` `AttentionBackend` documents
itself unusually well:

```python
class AttentionBackend(ABC):
    """The base class of attention backends.

    Forward-data init contract (3 methods):

      - ``init_forward_metadata(fb)`` — eager entry point. Default is a wrapper
        that calls ``_out_graph(fb)`` then ``_in_graph(fb)``. Backends may
        override to keep an independent eager body.
      - ``init_forward_metadata_out_graph(fb, in_capture=False)`` — per-iter
        metadata prep, runs outside ``with graph.capture():``. Capture
        sites pass ``in_capture=True``; replay/eager use the default
        ``False``.
      - ``init_forward_metadata_in_graph(fb)`` — graph-recordable static-shape
        GPU op, runs inside ``with graph.capture():`` at capture time and
        is auto-replayed by ``graph.replay()``. Default is no-op.
    """
```

The essential split is **metadata once per forward, kernel once per layer**. A batch's page
tables, sequence-length arrays, and workspace do not change between layer 3 and layer 40,
so computing them 80 times would be pure waste. `:62` `init_forward_metadata` prepares;
`:261` `forward_decode` and `:274` `forward_extend` consume. Nearly every backend bug is a
violation of that split — metadata rebuilt per layer (slow) or metadata stale across layers
(wrong).

The three-way split of metadata prep is Chapter 14 intruding. CUDA graph capture records
GPU operations for replay, but only *some* work can be recorded:

```python
    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        """Graph-recordable static-shape GPU op.
        ...
        Lint contract for overrides: body must NOT call ``.item()`` /
        ``.cpu()`` / ``.tolist()`` / dynamic-shape ``torch.empty()``.
        Such ops belong in :py:meth:`init_forward_metadata_out_graph`; they
        cannot be recorded into a cuda graph.
        """
```

A "lint contract" in a docstring: `.item()` synchronizes with the host and cannot be
recorded; a dynamically-shaped `torch.empty()` allocates at a different address each replay.
Work that *can* be recorded goes in-graph and is replayed for free; work that cannot goes
out-of-graph and runs every iteration. Splitting them correctly is most of what makes a
backend graph-compatible.

The class also documents its own deprecations — the older
`init_forward_metadata_capture_cuda_graph` / `..._replay_cuda_graph` overrides are gone,
with a migration note for out-of-tree backends. `:160` `init_cuda_graph_state` preallocates
the static buffers, and `:187` `get_cuda_graph_seq_len_fill_value` supplies the padding
value for unused slots.

---

## The reference implementation

`python/sglang/srt/layers/attention/flashinfer_backend.py` is the default on NVIDIA and the
one to read for how the contract is met in earnest — 2,400 lines, mostly metadata
management.

FlashInfer's own API is built around *wrappers*: objects that plan a decode or prefill given
page-table layout and then execute against it. The plan step is expensive and shape-
dependent, which maps precisely onto the out-of-graph phase, so most of the backend is
about creating wrappers, keeping them alive across iterations, and pointing them at the
right buffers.

Three responsibilities dominate:

**Page tables.** Chapter 8's `req_to_token` rows must become the `kv_indptr` / `kv_indices`
format FlashInfer expects — a CSR-like structure where `indptr[i]` marks where sequence
*i*'s page list begins. Building this per forward, on the GPU, without host
synchronization, is the bulk of the metadata code.

**Workspace.** FlashInfer needs scratch memory, allocated once and reused. Under CUDA
graphs it must be at a *stable address*, since the recorded kernel holds the pointer.

**Mode dispatch.** Decode, extend, mixed, and speculative verification each use different
wrappers with different plans.

`python/sglang/srt/layers/attention/triton_backend.py` is the portable fallback and is far
easier to read whole. It has no external dependency, runs anywhere Triton runs, and is
where a new feature usually lands first.

---

## Down to the kernel

The Triton backend's kernels are in `python/sglang/kernels/ops/attention/`, and they are
Python — the best place in the repository to watch paged attention actually happen.

`python/sglang/kernels/ops/attention/decode_attention.py:97` `_fwd_kernel_stage1` is the
decode kernel. Its parameter list is the chapter in miniature:

```python
def _fwd_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale_withk,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    ...
    # Page-aware strides (used when PAGE_SIZE > 1). For
    # PAGE_SIZE == 1 the address math degenerates and these are unused
    # (Triton specializes the dead branch away at compile time).
    stride_buf_kpage,
    stride_buf_ktok,
    ...
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ...
    PAGE_SIZE: tl.constexpr,
```

`K_Buffer` and `V_Buffer` are Chapter 8's pools, passed whole. `kv_indptr` and `kv_indices`
are the page table: the kernel does not receive a contiguous per-sequence tensor, it
receives the pool plus a directory and does the indirection itself. **That is what "paged
attention" means at the kernel level** — the address computation Chapter 8 described,
performed per block inside the kernel.

`kv_group_num` is the GQA ratio from Chapter 12 — how many query heads share one KV head.
`PAGE_SIZE` is a `tl.constexpr`, so Triton compiles a separate specialization per page size
and, as the comment notes, deletes the page arithmetic entirely when it is 1.

The `Att_Lse` output and `num_kv_splits` parameter are the flash-attention structure. A
sequence's KV is split across thread blocks, each computing a partial softmax over its
chunk and emitting a log-sum-exp alongside it; a second stage
(`python/sglang/kernels/ops/attention/decode_attention.py:732` `_fwd_kernel_stage2`)
combines the partials using those LSEs. This is what makes attention memory-bounded rather
than memory-quadratic: no `[seq_len, seq_len]` matrix is ever materialized.

The first comment in the body is a good specimen of production kernel code:

```python
    # int64 to avoid overflow of flat offsets into Mid_O when
    # batch * num_head * max_kv_splits * head_dim exceeds 2**31.
```

At batch 256, 64 heads, 32 splits, and 128 head dim the flat offset exceeds a 32-bit
integer. The default index type silently wraps. This is the class of bug that appears only
at scale and corrupts rather than crashes.

`python/sglang/kernels/ops/attention/extend_attention.py` is the prefill counterpart, and
its extra difficulty is that an extend attends over *both* cached prefix KV and the new
tokens — two different memory layouts in one kernel.

`python/sglang/kernels/ops/attention/decode_attention.py:384` `_fwd_grouped_kernel_stage1`
is a GQA-specialized variant: when many query heads share one KV head, loading that KV once
for the whole group is a large saving, and it is worth a separate kernel to express.

---

## Registration and selection

`python/sglang/srt/layers/attention/attention_registry.py:34` `register_attention_backend`
is a decorator-based registry, with factories per backend (`:43`
`create_flashinfer_backend`, `:70` `create_trtllm_mla_backend`, `:111`
`create_aiter_backend`, `:125` `create_ascend_backend`, `:134` `create_dsa_backend`, and
more).

`python/sglang/srt/model_executor/model_runner.py:927` `init_attention_backends` resolves
`--attention-backend` — or separate `--prefill-attention-backend` and
`--decode-attention-backend`, since the best choice differs by phase. `:266`
`resolve_draft_attention_backend` picks one for the draft model in Chapter 18, which may
differ again.

---

## MLA needs its own everything

DeepSeek's multi-head latent attention compresses KV into a single latent vector per token
rather than per-head keys and values. Chapter 8 showed the pool
(`python/sglang/srt/mem_cache/memory_pool.py:3932` `MLATokenToKVPool`); the consequence
here is that the *stored* form is not the form attention consumes. The latent must be
decompressed, and whether that happens before or inside the kernel is a real design choice
with real performance consequences.

Hence a family of MLA backends: `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`,
`python/sglang/srt/layers/attention/flashmla_backend.py`,
`python/sglang/srt/layers/attention/cutlass_mla_backend.py`,
`python/sglang/srt/layers/attention/cutedsl_mla_backend.py`, and
`python/sglang/srt/layers/attention/trtllm_mla_backend.py`. Five
implementations of one attention variant is the clearest evidence for the argument at the
top of this chapter — and for the claim that memory layout dictates kernel design.

---

## Sparse attention

If attention cost is linear in sequence length, the way to serve very long contexts is to
attend to fewer tokens.

`python/sglang/srt/layers/attention/nsa/nsa_indexer.py` implements the selection: a learned
component that scores which tokens matter for this query, so attention runs over hundreds
rather than hundreds of thousands. `python/sglang/srt/layers/attention/nsa_backend.py` is
the backend, and `python/sglang/srt/layers/attention/nsa/transform_index.py`,
`python/sglang/srt/layers/attention/nsa/quant_k_cache.py`, and
`python/sglang/srt/layers/attention/nsa/dequant_k_cache.py` handle the surrounding machinery.

The indexer is a *model component* — it has weights and must be loaded — which makes it the
one place where the attention backend boundary genuinely blurs into the model.
`python/sglang/srt/mem_cache/memory_pool.py:4348` `DSATokenToKVPool` is its pool, and
`python/sglang/srt/layers/attention/minimax_sparse_backend.py` is an independent take on
the same idea.

---

## When it isn't attention at all

Some models replace attention with recurrence. A state-space layer carries a fixed-size
state and updates it per token — O(1) memory per sequence rather than O(n), and no KV cache
at all.

`python/sglang/srt/layers/attention/mamba/mamba.py` and
`python/sglang/srt/layers/attention/mamba/causal_conv1d.py` implement Mamba;
`python/sglang/srt/layers/attention/linear/gdn_backend.py`,
`python/sglang/srt/layers/attention/linear/kda_backend.py`,
`python/sglang/srt/layers/attention/linear/lightning_backend.py`, and
`python/sglang/srt/layers/attention/linear/short_conv_backend.py` cover the linear-attention family.

They live under `layers/attention/` because they occupy the same slot in the model and
implement the same backend interface — but almost nothing else transfers. Chapter 8's
`MambaPool` exists for them, and Chapter 5's separate `rem_mamba_slots` budget gate exists
because a fixed-size state cannot be carved out of evictable token cache.

Most current models are **hybrids**: some layers attention, some linear.
`python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` dispatches per layer, and
`python/sglang/srt/layers/attention/hybrid_attn_backend.py` does the same for models mixing
full and sliding-window attention. `python/sglang/srt/mem_cache/memory_pool.py:3577`
`HybridLinearKVPool` and `python/sglang/srt/mem_cache/memory_pool.py:1153`
`HybridReqToTokenPool` are the storage side.

The `.claude/skills/compute-mamba-ratio/SKILL.md` playbook exists because sizing the two
pools against each other is genuinely hard: too much state pool starves the KV pool and
caps context; too little caps concurrency.

---

## Choosing one

There is no universal answer, but the shape of the decision is stable:

- **NVIDIA, standard attention** — FlashInfer by default; FlashAttention-3 competitive on
  long prefills.
- **DeepSeek / MLA** — one of the MLA backends, matched to your GPU generation.
- **AMD** — AITER (`python/sglang/srt/layers/attention/aiter_backend.py`).
- **Debugging or a new feature** — Triton. Readable, hackable, and the reference behavior.
- **Non-NVIDIA accelerators** — the vendor backend (`python/sglang/srt/layers/attention/xpu_backend.py`,
  `python/sglang/srt/layers/attention/intel_amx_backend.py`,
  `python/sglang/srt/layers/attention/wave_backend.py`).

`docs/docs/advanced_features/attention_backend.mdx` carries the project's current
recommendations, which move faster than a book can.

Chapter 14 covers what constrains all of them equally: making the forward pass cheap.

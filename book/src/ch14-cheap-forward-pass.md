# 14. Making the Forward Pass Cheap

> *Quantization attacks bytes moved, CUDA graphs attack launch overhead, and compilation
> attacks kernel count — three independent taxes on the same forward pass.*

---

Chapter 1 established that decode is memory-bound: the GPU spends its time moving weights,
not multiplying them. Three techniques attack that from different directions, and they
compose because they attack different things.

- **Quantization** — fewer bytes per weight, so each read is cheaper.
- **CUDA graphs** — fewer CPU launches, so the GPU is not waiting on Python.
- **Compilation** — fewer kernels, so less is moved between them.

---

## Three independent targets

Quantization applies to three things, and the choices are independent:

**Weights.** The largest and most valuable target. A 70B model in BF16 is 140 GB; in FP8 it
is 70 GB; in 4-bit it is 35 GB. Since decode reads every weight every step, halving weight
bytes nearly halves decode time.

**Activations.** Only pays if the *kernel* consumes the smaller type — an FP8 GEMM with FP8
inputs uses hardware FP8 tensor cores. Quantizing activations only to dequantize before the
GEMM is pure loss.

**KV cache.** Attacks Chapter 8's capacity rather than speed. Halving KV bytes doubles
concurrency at the same memory, which is a throughput win by a different route.

The failure modes differ too: weight quantization degrades the model uniformly, while KV
quantization degrades *long contexts specifically*, because error accumulates over the
sequence.

---

## The quantization architecture

`python/sglang/srt/layers/quantization/base_config.py` defines a three-level structure.

`:126` `QuantizationConfig` describes a scheme — what formats it supports, which layers to
skip, what the minimum GPU capability is. It is what `--quantization` selects and what
Chapter 12's layers receive as `quant_config`.

`:20` `QuantizeMethodBase` is the per-layer strategy:

```python
class QuantizeMethodBase(ABC):

    def create_weights(
        self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs
    ):
        ...

    @abstractmethod
    def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
```

Two methods carry the design. `create_weights` runs at construction and creates parameters
*of the quantized shapes* — a 4-bit scheme allocates packed int32 storage plus scale
tensors, not a float matrix. `apply` runs at inference and dispatches to the right kernel.

`:46` `LinearMethodBase` specializes it for linear layers, which is what
`ColumnParallelLinear` and friends hold.

The consequence is worth stating plainly: **quantization is not a transformation applied to
a model — it is a different implementation of every layer, chosen at construction.** That is
why `quant_config` threads through every constructor in Chapter 12, and why a layer that
forgets to pass it silently runs unquantized.

A third method appears in the implementations:
`python/sglang/srt/layers/quantization/fp8.py:845` `process_weights_after_loading`. Weights
arrive from the checkpoint in whatever layout the exporter chose; the kernel wants a
specific one. Rather than transposing per forward, the layout is fixed once after loading.
This is also where online quantization happens — loading a BF16 checkpoint and quantizing
at startup, via `python/sglang/srt/layers/quantization/online_quantization.py`.

### FP8 in full

`python/sglang/srt/layers/quantization/fp8.py` is the most-used path and worth reading end
to end. Its structure:

- `:225` `Fp8Config` — the scheme.
- `:432` `Fp8LinearMethod` — linear layers, with `:628` `create_weights`, `:845`
  `process_weights_after_loading`, `:957` `apply`.
- `:1064` `Fp8MoEMethod` — mixture-of-experts (Chapter 16), which needs its own everything.
- `:2710` `Fp8KVCacheMethod` — quantized KV, closing the loop with Chapter 8.

Note `:655` `process_weights_after_loading_block_quant` as a separate path from the
per-tensor one. **Scale granularity** is the central design axis of any quantization scheme:

- **Per-tensor** — one scale for a whole matrix. Cheapest, least accurate.
- **Per-channel** — one per output channel. A good default.
- **Per-block** — one per tile (say 128×128). Most accurate, and needs the *kernel* to apply
  scales during accumulation, which is why it has its own code path rather than a different
  constant.

The other schemes follow the same shape: `python/sglang/srt/layers/quantization/modelopt_quant.py`
(NVIDIA ModelOpt, FP8 and NVFP4), `python/sglang/srt/layers/quantization/mxfp4.py` (microscaling
FP4), `python/sglang/srt/layers/quantization/w4afp8.py` (4-bit weights, FP8 activations), and the
directories `python/sglang/srt/layers/quantization/compressed_tensors/`, `python/sglang/srt/layers/quantization/awq/`,
`python/sglang/srt/layers/quantization/gptq/`, `python/sglang/srt/layers/quantization/quark/` (AMD).

`python/sglang/srt/layers/parameter.py` is the piece that makes it all composable: parameter
wrappers that know both their *shard* geometry (Chapter 12) and their *scale* geometry, so
that loading a sharded quantized checkpoint slices the weights and the scales consistently.
Getting that wrong yields a model that loads cleanly and produces nonsense.

---

## Down to the kernel

`python/sglang/kernels/ops/quantization/` holds the Triton side —
`python/sglang/kernels/ops/quantization/fp8_kernel.py`, `python/sglang/kernels/ops/quantization/fp8_quantize.py`,
`python/sglang/kernels/ops/quantization/per_token_group_quant.py`,
`python/sglang/kernels/ops/quantization/per_tensor_quant_fp8.py`,
`python/sglang/kernels/ops/quantization/int8_kernel.py`, `python/sglang/kernels/ops/quantization/awq_triton.py`,
`python/sglang/kernels/ops/quantization/gptq_marlin.py` — and reading
`python/sglang/kernels/ops/quantization/per_token_group_quant.py` alongside
`python/sglang/kernels/ops/quantization/per_tensor_quant_fp8.py` shows the granularity axis above as two
concrete kernels.

The heavier machinery is AOT C++/CUDA under `python/sglang/kernels/aot/csrc/`, with
`python/sglang/kernels/aot/csrc/gemm/` for quantized GEMMs. The essential trick in all of
them is the **fused epilogue**: a quantized GEMM accumulates in higher precision and applies
scales as it writes out, rather than producing an intermediate tensor and scaling it in a
second pass. A separate scaling kernel would re-read the entire output from memory —
precisely the cost quantization exists to avoid.

This is also where per-block scaling earns its complexity: the kernel must fetch the right
scale for each accumulation tile mid-loop, which is a change to the inner loop rather than
to the epilogue.

---

## Quantized KV

`python/sglang/srt/layers/quantization/kv_cache.py` connects to Chapter 8. Recall that
`KVCache.__init__` sets `store_dtype = torch.uint8` for FP8 types because `index_put` is not
implemented for FP8 — the pool holds bytes and reinterprets them.

`python/sglang/srt/models/llama.py:888` `load_kv_cache_scales` loads calibrated scales from
a checkpoint. Static scales need calibration; dynamic scales cost a pass over the tensor.

The accuracy story is specific: KV quantization error compounds along the sequence, because
early tokens' quantized keys are attended to by every later query. A model that looks fine
at 2k context can degrade visibly at 100k.
`docs/docs/advanced_features/quantized_kv_cache.mdx` covers the current guidance.

---

## Decode is launch-bound

Now the second tax.

A 70B model has 80 layers, each running perhaps a dozen kernels — attention, several GEMMs,
norms, activations. That is roughly a thousand kernel launches per forward pass. Each costs
5–10 µs of CPU time to prepare and submit.

In prefill, kernels run for milliseconds and launch overhead vanishes. In decode, a kernel
may run for 20 µs — so a thousand launches at 5 µs each is 5 ms of CPU time against maybe
20 ms of GPU work. **The CPU cannot keep the GPU fed.**

Chapter 4's overlap scheduler removed the *scheduling* gap. This is the *launch* gap, one
level down, and CUDA graphs are the fix: record the entire sequence of kernels once, then
replay it with a single call.

The price is rigidity. A recorded graph has fixed shapes, fixed buffer addresses, and no
data-dependent control flow. Everything below is a consequence.

---

## Capture and replay

`python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py:102`
`BaseCudaGraphRunner` documents the split:

```python
class BaseCudaGraphRunner(BaseRunner):
    """
    A subclass (DecodeCudaGraphRunner / PrefillCudaGraphRunner) owns one
    ...
      - buffers and backend are populated by the subclass before
    """
```

with `:151` `capture_prepare`, `:154` `capture`, and `:157` `capture_one_shape` as the
template methods. `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` and
`python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` are the two subclasses.

**Bucketing** is how fixed shapes meet variable batch sizes.
`python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py:61` `get_batch_sizes_to_capture` chooses which sizes to record —
typically 1, 2, 4, 8, 16, 24, 32, … up to `--cuda-graph-max-bs` — and `python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py:134` `_pad_to_bucket`
rounds a real batch up to the nearest one. A batch of 37 replays the 40-graph with three
padded slots, which is where Chapter 8's row-0 padding trick earns its place: padded
requests point at the scratch row and their work is harmlessly discarded.

The cost is memory. Each captured graph holds its own static input and output buffers, and
capturing forty shapes is forty sets. `python/sglang/srt/model_executor/cuda_graph_config.py`
(`python/sglang/srt/model_executor/cuda_graph_config.py:123` `CudaGraphConfig`,
`python/sglang/srt/model_executor/cuda_graph_config.py:88` `PhaseConfig`) is the configuration surface, and
`python/sglang/srt/model_executor/graph_memory_usage.py` reports what it actually cost.
`python/sglang/srt/model_executor/model_runner.py:885` `post_capture_resize_kv_pool` returns
the slack to the KV pool afterwards.

`python/sglang/srt/model_executor/cuda_graph_buffer_registry.py` tracks the static buffers,
which is what the attention backends' `init_cuda_graph_state` (Chapter 13) populates.

`python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py:43` `freeze_gc` is a small, telling detail: Python garbage
collection during capture can free a tensor whose address the graph has already recorded.

---

## When part of the model cannot be captured

Full capture requires the *whole* forward to be graph-safe. One host synchronization, one
data-dependent branch, one dynamically-shaped allocation, and capture fails.

That happens often enough to need an answer. Chapter 13's `.item()` prohibition is one
example; MoE routing (Chapter 16) is another, since expert dispatch is inherently
data-dependent.

`python/sglang/srt/model_executor/runner_backend/` holds the graduated responses:

- `python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py` — capture everything.
  Fastest when possible.
- `python/sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py` — capture around an
  eager "break" region. Chapter 12's
  `RadixAttention` cooperates via `python/sglang/srt/layers/radix_attention.py:47`
  `force_eager_attention`, whose comment describes a caller wrapping norm + attention +
  short-conv in one eager region.
- `python/sglang/srt/model_executor/runner_backend/tc_piecewise_cuda_graph_backend.py` — many small
  graphs with eager gaps, the torch.compile-integrated form.
- `python/sglang/srt/model_executor/runner_backend/cuda_graph_dedup_mixin.py` — deduplicating identical
  captured regions to reclaim memory.

`docs/docs/advanced_features/piecewise_cuda_graph.mdx` and
`docs/docs/advanced_features/breakable_cuda_graph.mdx` carry the operational guidance.

---

## Compilation

The third tax is kernel *count*. A norm followed by an activation followed by a scale is
three kernels, each reading and writing the full activation tensor. Fusing them into one
reads and writes once.

`python/sglang/srt/compilation/` integrates `torch.compile`:

- `python/sglang/srt/compilation/backend.py` and `python/sglang/srt/compilation/compiler_interface.py` —
  the Inductor integration.
- `python/sglang/srt/compilation/cuda_piecewise_backend.py` (with NPU and XPU variants) —
  splitting the graph at points that cannot be compiled, the same problem as above from
  the compiler's side.
- `python/sglang/srt/compilation/pass_manager.py`, `python/sglang/srt/compilation/inductor_pass.py`,
  `python/sglang/srt/compilation/fix_functionalization.py` — custom passes.
  `fix_functionalization` deals with PyTorch's functionalization pass introducing copies
  around in-place operations, which for a memory-bound workload is exactly wrong.
- `python/sglang/srt/compilation/compilation_config.py` with `register_split_op`, which Chapter 12's `RadixAttention`
  calls to mark itself as a split point.

Compilation and CUDA graphs compose: compile to fuse, then capture the compiled result.
`python/sglang/srt/compilation/torch_compile_decoration.py` marks compilable regions, and
`docs/docs/references/torch_compile_cache.mdx` covers caching compiled artifacts, since
compilation is slow and paying it on every server start is not viable.

---

## What each one buys

| Technique | Attacks | Typical gain | Cost |
| --- | --- | --- | --- |
| Weight quantization (FP8) | bytes read per step | ~1.5–2× decode | small accuracy loss |
| KV quantization | cache capacity | ~2× concurrency | long-context accuracy |
| CUDA graphs | CPU launch overhead | large at small batch | memory; shape rigidity |
| torch.compile | kernel count | modest, model-dependent | compile time |

The gains are not additive, because they attack different bottlenecks: once graphs remove
the launch gap, further launch reduction buys nothing. The right order is to find which tax
you are actually paying — Chapter 21's profiling — before spending effort on the others.

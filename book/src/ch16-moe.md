# 16. Mixture-of-Experts and Expert Parallelism

> *MoE inference is all-to-all-bound rather than GEMM-bound, which makes routing, placement,
> and communication overlap the whole game.*

---

## A different cost structure

A mixture-of-experts model replaces the dense MLP with *n* expert MLPs and a router that
sends each token to the top *k* of them. DeepSeek-V3 has 256 experts and activates 8 —
roughly 37B active parameters out of 671B total.

For inference this changes the arithmetic from Chapter 1 in a specific way. **Parameter
count grows; active parameters per token do not.** The FLOPs per token stay near a dense
37B model's, while memory holds 671B.

Two consequences follow, and the second is the one that shapes everything here.

First, MoE models need more GPUs to hold their weights — but not proportionally more
compute. Second, and less obviously: **tokens in a batch go to different experts.** In a
dense model every token takes the same path. In an MoE model, a batch of 256 tokens
scatters across 256 experts, and if those experts live on different GPUs, the tokens must
be *sent there and the results brought back*. That is an all-to-all, twice per MoE layer,
and it is the dominant cost.

---

## Routing

`python/sglang/srt/layers/moe/topk.py:392` `TopK` is the router, and its docstring is worth
reading for the vocabulary alone:

```python
class TopK(BaseFusedOp):
    """
    Parameters:
    --top_k: The all number of top experts selected per token, including the fused shared expert(s).
    --num_fused_shared_experts: num of shared experts, can be activate both in TP or EP mode.
    --routed_scaling_factor: the scaling factor for routed experts in topk_weights.
    --fused_shared_experts_scaling_factor: scaling factor for fused shared experts on AMD-platform.
    """
```

**Shared experts** are the first idea: experts every token goes through, in addition to its
routed ones. They capture what is common across all inputs, letting the routed experts
specialize. They are also *cheap* — no routing decision, no all-to-all, since every rank
needs them anyway — which is why they get fused into the top-k path rather than treated
separately.

The constructor exposes the rest of the design space:

```python
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        renormalize: bool = True,
        ...
        scoring_func: str = "softmax",
        correction_bias: Optional[torch.Tensor] = None,
```

**Grouped top-k** is the routing constraint that makes large-scale EP tractable. Rather than
letting a token pick any *k* of 256 experts — which could mean talking to 8 different GPUs —
experts are partitioned into groups, and a token first picks groups, then experts within
them. That bounds how many ranks each token communicates with, converting an unbounded
all-to-all into a bounded one. It is a *systems* constraint imposed on the model
architecture, and DeepSeek's models are trained with it.

**Correction bias** is load balancing at the routing level: a learned per-expert bias
nudging traffic away from over-subscribed experts, without changing the loss.

`:274` `TopKOutput` and its variants (`:283` `StandardTopKOutput`, `:300`
`StandardTopKOutputPacked`, `:314` `TritonKernelTopKOutput`, `:354` `PackedTopKOutput`,
`:328` `BypassedTopKOutput`) exist because the *format* of the routing result determines
which downstream kernel can consume it. A fused Triton MoE wants a different layout than a
CUTLASS grouped GEMM. Rather than converting between them — which at these sizes is a real
cost — the router emits the format its consumer wants, and `:238` `TopKOutputChecker`
validates the pairing.

---

## Computing the experts

Once tokens are routed, the computation is a **grouped GEMM**: many small matrix multiplies,
one per expert, with different numbers of rows each.

`python/sglang/srt/layers/moe/fused_moe_triton/` is the portable implementation, and
`python/sglang/srt/layers/moe/moe_runner/` is the abstraction over backends —
mirroring Chapter 13's attention backend registry for the same reason.

The specialized paths: `python/sglang/srt/layers/moe/cutlass_moe.py` and
`python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` (CUTLASS grouped GEMM, including 4-bit
weights with FP8 activations), `python/sglang/srt/layers/moe/flashinfer_cutedsl_moe.py`,
`python/sglang/srt/layers/moe/flashinfer_trtllm_moe.py`, `python/sglang/srt/layers/moe/mega_moe.py`, and
`python/sglang/srt/layers/moe/fused_moe_native.py` as the readable reference.

---

## Down to the kernel

MoE is where in-repo CUDA is densest, which makes it the best place to see production
kernel work at scale.

`python/sglang/kernels/ops/moe/` holds the Triton side.
`python/sglang/kernels/aot/csrc/moe/` holds the CUDA:

```
moe_align_kernel.cu           sort tokens by expert
moe_topk_softmax_kernels.cu   fused routing
moe_topk_sigmoid_kernels.cu   routing, sigmoid scoring
moe_sum.cu, moe_sum_reduce.cu combine expert outputs
prepare_moe_input.cu          gather tokens into expert-major order
fp8_blockwise_moe_kernel.cu   quantized expert GEMM (Chapter 14)
cutlass_moe/                  grouped GEMM
```

`python/sglang/kernels/aot/csrc/moe/moe_align_kernel.cu` and
`python/sglang/kernels/aot/csrc/moe/prepare_moe_input.cu` are the ones to understand, because they
implement the trick the whole subsystem rests on.

Tokens arrive in arbitrary order, each tagged with its experts. A GEMM wants contiguous rows
per expert. So the tokens are **sorted into expert-major order**: all of expert 0's tokens
together, then expert 1's, and so on, with an offset array marking the boundaries. Now one
grouped GEMM handles every expert in a single launch, with each group reading a contiguous
slice.

Without that reordering you would launch one GEMM per expert — 256 launches of tiny
matrices, which is Chapter 14's launch-overhead problem in its worst form.
`python/sglang/kernels/aot/csrc/moe/moe_sum.cu` and `python/sglang/kernels/aot/csrc/moe/moe_sum_reduce.cu`
then scatter the results back and combine each token's *k* expert
outputs weighted by its routing scores.

The pattern — **sort, group, one big operation, scatter back** — is the same one that makes
paged attention work in Chapter 13, applied to a different irregularity.

---

## Splitting experts across devices

With 256 experts, no single GPU holds them all. Expert parallelism assigns each rank a
subset.

`python/sglang/srt/layers/moe/ep_moe/layer.py` is the EP layer, and
`python/sglang/srt/layers/moe/token_dispatcher/` is where the communication lives. The
backends there map exactly onto the deployment landscape:
`python/sglang/srt/layers/moe/token_dispatcher/deepep.py` (DeepEP, the DeepSeek kernels),
`python/sglang/srt/layers/moe/token_dispatcher/pplx.py`, `python/sglang/srt/layers/moe/token_dispatcher/mooncake.py`,
`python/sglang/srt/layers/moe/token_dispatcher/nixl.py`, `python/sglang/srt/layers/moe/token_dispatcher/moriep.py` (AMD),
`python/sglang/srt/layers/moe/token_dispatcher/ascend_tp.py`, and `python/sglang/srt/layers/moe/token_dispatcher/standard.py`, with
`python/sglang/srt/layers/moe/token_dispatcher/base.py` defining the interface.

The pattern is **dispatch and combine**. Dispatch sends each token to the ranks holding its
experts; combine brings the results back. Both are all-to-all, and both are on the critical
path of every MoE layer.

Why latency rather than bandwidth dominates: in decode, a batch might be 256 tokens across
8 ranks — each rank sends a few kilobytes to each other rank. That is a *small-message*
all-to-all, where fixed per-message costs and synchronization dominate and raw bandwidth is
irrelevant. It is also why these kernels are hand-written rather than left to NCCL, and why
`DpPaddingMode`'s deadlock comment in Chapter 15 exists: a symmetric collective where one
rank contributes nothing hangs.

---

## Hot experts

Routing is learned, not uniform. Some experts are far more popular than others, and in a
real workload the distribution is skewed and *workload-dependent* — a coding workload
activates different experts than a chat workload.

This is a problem because the slowest rank sets the step time. If rank 3 holds two popular
experts and rank 5 holds two rare ones, rank 3 does several times the work and everyone
waits.

`python/sglang/srt/eplb/` is the response:

- `python/sglang/srt/eplb/expert_distribution.py` — measurement. Records which experts fire, exposed through the
  `/start_expert_distribution_record` and `/dump_expert_distribution_record` endpoints
  (`python/sglang/srt/entrypoints/engine.py:1334` and `:1344`) precisely *because* the
  answer depends on your traffic.
- `python/sglang/srt/eplb/expert_location.py` — the placement map: which expert lives
  on which rank.
- `python/sglang/srt/eplb/eplb_algorithms/` — the rebalancing algorithms.
- `python/sglang/srt/eplb/eplb_manager.py` — applying a new placement at runtime.
- `python/sglang/srt/eplb/expert_location_updater.py` — moving expert weights between
  ranks without a restart.
- `python/sglang/srt/eplb/lplb_solver.py` — local placement balancing.

Two mechanisms are available. **Rebalancing** moves experts between ranks so popular ones
are spread out. **Redundant experts** replicate a hot expert onto several ranks so its
traffic splits — spending memory to buy balance, which is worthwhile when one expert takes
10% of all tokens.

`python/sglang/srt/eplb/eplb_simulator/` lets you evaluate a placement against a recorded distribution without
deploying it.

---

## Hiding the all-to-all

If all-to-all dominates the step, the highest-value optimization is to make it happen
*during* computation rather than between computations.

`python/sglang/srt/batch_overlap/two_batch_overlap.py` splits a batch in two and staggers
them: while half A is in its all-to-all, half B is computing, and vice versa. The
communication disappears behind compute rather than adding to it.

This is Chapter 4's overlap idea one level down. There it was CPU scheduling hidden behind
GPU compute; here it is GPU communication hidden behind GPU compute. Same shape, different
resources.

`python/sglang/srt/batch_overlap/single_batch_overlap.py` achieves the same within one batch
by splitting along a different axis. Both need the model expressed as a sequence of
schedulable stages rather than a monolithic forward, which is what
`python/sglang/srt/batch_overlap/operations.py` and `python/sglang/srt/batch_overlap/operations_strategy.py` provide, and
`python/sglang/srt/layers/attention/tbo_backend.py` is the attention side.

The cost is halved effective batch size per stream, so TBO pays when communication is a
large fraction of step time — large-scale EP — and loses when it is not.

---

## What large-scale EP looks like

The DeepSeek deployments described in the LMSYS blog posts — 96 H100s, GB200 NVL72 racks —
combine everything in this chapter and the last:

- **EP** across dozens of ranks, so each holds a handful of experts.
- **DP attention** (Chapter 15), because DeepSeek is MLA and TP would replicate the cache.
- **EPLB**, because at that scale imbalance is the difference between good and unusable.
- **TBO**, because all-to-all across 96 ranks must overlap or it dominates.
- **PD disaggregation** (Chapter 17), because prefill and decode want different layouts
  entirely.

None of these is optional at that scale, and each was added because the previous
combination hit a wall. `docs/docs/advanced_features/expert_parallelism.mdx` covers the
configuration; the blog posts linked from `README.md` cover the results.

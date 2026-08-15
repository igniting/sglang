# 16. Mixture-of-Experts and Expert Parallelism

> *MoE inference is all-to-all-bound rather than GEMM-bound, which makes routing, placement,
> and communication overlap the whole game.*

A mixture-of-experts model replaces the dense feed-forward block with many expert blocks and
a router that sends each token to a few of them. DeepSeek-V3 has 256 experts and activates 8
per token: 671 billion parameters total, roughly 37 billion active.

For serving, that changes the cost structure in a way worth being precise about. Parameter
count grows enormously; arithmetic per token barely moves. So MoE models need many more GPUs
to hold their weights without needing proportionally more compute — and that is the setup
for the real consequence.

In a dense model every token in a batch takes the same path through the same weights. In an
MoE model, a batch of 256 tokens scatters across 256 experts. If those experts live on
different GPUs, the tokens must be *sent there and the results brought back* — twice per
layer, for every layer.

That communication, not matrix multiplication, is what dominates. Which means the
interesting engineering is not in the expert computation at all. It is in routing, in
deciding which expert lives where, and in hiding the communication behind other work.

This chapter is also where the in-repo CUDA is densest, and where one trick that has already
appeared twice in this book appears again in its clearest form.

---

## A different cost structure

### The lineage, in four steps

Conditional computation is an old idea, but the version this code implements arrived through
a specific sequence, and each step left a parameter in the constructor below.

**GShard** (Lepikhin et al., 2020) established the shape still in use: replace the MLP in
every other transformer layer with *n* experts and a learned router, send each token to its
top-2, and shard the experts across devices. It also named the two problems that dominate
everything since. Routing is *learned*, so nothing stops it collapsing onto a few popular
experts — so GShard added an **auxiliary loss** penalizing imbalance. And a device can only
hold so many tokens, so GShard added **capacity factors**: each expert accepts a fixed
number of tokens per batch and *drops* the overflow.

**Switch Transformer** (Fedus et al., 2021) simplified top-2 to top-1, showing the routing
quality survived it, and made the engineering case that sparse models are worth the trouble
at scale.

**DeepSeekMoE** (2024) changed the granularity. Instead of a few large experts, use many
small ones — 256 rather than 8 — so that a token's top-8 selection is a far more expressive
combination. And carve out **shared experts** that every token passes through
unconditionally, which frees the routed experts from re-learning what is common to all
inputs. Both choices are visible in the constructor below.

**DeepSeek-V3** (2024) removed the auxiliary loss. Its argument is that a balance penalty is
a gradient fighting the quality objective — you get balance by making routing slightly worse.
Instead, keep a per-expert **bias** added to the affinity scores only for the purpose of
choosing the top-*k*, and adjust it outside the gradient: raise an underloaded expert's bias,
lower an overloaded one's. Balance is achieved by moving the selection threshold rather than
by punishing the router. That bias is the `correction_bias` parameter below, and at inference
it is a loaded constant.

V3 also constrained routing spatially — **node-limited routing**, where a token may reach at
most *M* nodes (M = 4 for 256 experts across 8 nodes). This is the grouped top-k described
below, and it is a systems constraint written into the model architecture: the network
topology reached back into the training recipe.

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

<figure>
<svg viewBox="0 0 700 388" role="img" aria-label="Tokens dispatched to expert-holding ranks and combined back, with one rank carrying a hot expert">
<title>One MoE layer across four ranks</title>
<rect class="dgm-box" x="20" y="44" width="150" height="56" rx="6"/>
<text class="dgm-label" x="95.0" y="67.4" text-anchor="middle" font-weight="600" style="font-size:13.0px">rank 0</text>
<text class="dgm-small" x="95.0" y="85.4" text-anchor="middle" style="font-size:10.5px">its share of the batch</text>
<rect class="dgm-box" x="193" y="44" width="150" height="56" rx="6"/>
<text class="dgm-label" x="268.0" y="67.4" text-anchor="middle" font-weight="600" style="font-size:13.0px">rank 1</text>
<text class="dgm-small" x="268.0" y="85.4" text-anchor="middle" style="font-size:10.5px">its share of the batch</text>
<rect class="dgm-box" x="366" y="44" width="150" height="56" rx="6"/>
<text class="dgm-label" x="441.0" y="67.4" text-anchor="middle" font-weight="600" style="font-size:13.0px">rank 2</text>
<text class="dgm-small" x="441.0" y="85.4" text-anchor="middle" style="font-size:10.5px">its share of the batch</text>
<rect class="dgm-box" x="539" y="44" width="150" height="56" rx="6"/>
<text class="dgm-label" x="614.0" y="67.4" text-anchor="middle" font-weight="600" style="font-size:13.0px">rank 3</text>
<text class="dgm-small" x="614.0" y="85.4" text-anchor="middle" style="font-size:10.5px">its share of the batch</text>
<rect class="dgm-box-accent" x="20" y="122" width="669" height="40" rx="6"/>
<text class="dgm-small" x="354.5" y="146.4" text-anchor="middle" style="font-size:11.5px">all-to-all dispatch — every token to the ranks holding its top-k experts</text>
<rect class="dgm-box" x="20" y="186" width="150" height="62" rx="6"/>
<text class="dgm-label" x="95.0" y="203.4" text-anchor="middle" font-weight="600" style="font-size:13.0px">experts 0–63</text>
<text class="dgm-small" x="95.0" y="221.4" text-anchor="middle" style="font-size:10.5px">grouped GEMM</text>
<text class="dgm-small" x="95.0" y="239.4" text-anchor="middle" style="font-size:10.5px">ordinary load</text>
<rect class="dgm-box" x="193" y="186" width="150" height="62" rx="6"/>
<text class="dgm-label" x="268.0" y="203.4" text-anchor="middle" font-weight="600" style="font-size:13.0px">experts 64–127</text>
<text class="dgm-small" x="268.0" y="221.4" text-anchor="middle" style="font-size:10.5px">grouped GEMM</text>
<text class="dgm-small" x="268.0" y="239.4" text-anchor="middle" style="font-size:10.5px">ordinary load</text>
<rect class="dgm-box-accent" x="366" y="186" width="150" height="62" rx="6"/>
<text class="dgm-label" x="441.0" y="203.4" text-anchor="middle" font-weight="600" style="font-size:13.0px">experts 128–191</text>
<text class="dgm-small" x="441.0" y="221.4" text-anchor="middle" style="font-size:10.5px">grouped GEMM</text>
<text class="dgm-small" x="441.0" y="239.4" text-anchor="middle" style="font-size:10.5px">hot</text>
<rect class="dgm-box" x="539" y="186" width="150" height="62" rx="6"/>
<text class="dgm-label" x="614.0" y="203.4" text-anchor="middle" font-weight="600" style="font-size:13.0px">experts 192–255</text>
<text class="dgm-small" x="614.0" y="221.4" text-anchor="middle" style="font-size:10.5px">grouped GEMM</text>
<text class="dgm-small" x="614.0" y="239.4" text-anchor="middle" style="font-size:10.5px">ordinary load</text>
<rect class="dgm-box-accent" x="20" y="272" width="669" height="40" rx="6"/>
<text class="dgm-small" x="354.5" y="296.4" text-anchor="middle" style="font-size:11.5px">all-to-all combine — partial outputs returned and weighted</text>
<path class="dgm-line" d="M95.0 100 L95.0 117" marker-end="url(#arrow)"/>
<path class="dgm-line" d="M95.0 162 L95.0 181" marker-end="url(#arrow)"/>
<path class="dgm-line" d="M95.0 248 L95.0 267" marker-end="url(#arrow)"/>
<path class="dgm-line" d="M268.0 100 L268.0 117" marker-end="url(#arrow)"/>
<path class="dgm-line" d="M268.0 162 L268.0 181" marker-end="url(#arrow)"/>
<path class="dgm-line" d="M268.0 248 L268.0 267" marker-end="url(#arrow)"/>
<path class="dgm-line" d="M441.0 100 L441.0 117" marker-end="url(#arrow)"/>
<path class="dgm-line" d="M441.0 162 L441.0 181" marker-end="url(#arrow)"/>
<path class="dgm-line" d="M441.0 248 L441.0 267" marker-end="url(#arrow)"/>
<path class="dgm-line" d="M614.0 100 L614.0 117" marker-end="url(#arrow)"/>
<path class="dgm-line" d="M614.0 162 L614.0 181" marker-end="url(#arrow)"/>
<path class="dgm-line" d="M614.0 248 L614.0 267" marker-end="url(#arrow)"/>
<text class="dgm-label" x="350.0" y="34" text-anchor="middle" font-weight="600" style="font-size:12.5px">one MoE layer, four ranks</text>
<text class="dgm-small" x="350.0" y="336" text-anchor="middle" style="font-size:11.5px">Two collectives per layer, sixty layers: the cost is the count of synchronizations,</text>
<text class="dgm-small" x="350.0" y="354" text-anchor="middle" style="font-size:11.5px">not the bytes. And the slowest rank sets the step — which is what makes one hot</text>
<text class="dgm-small" x="350.0" y="372" text-anchor="middle" style="font-size:11.5px">expert everyone's problem, and why EPLB moves or replicates it.</text>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" style="fill:var(--dgm-rule)"/></marker><marker id="arrow-accent" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" style="fill:var(--dgm-accent)"/></marker></defs>
</svg>
<figcaption>Every token crosses the network twice per MoE layer. Expert parallelism turns a memory problem into a communication one.</figcaption>
</figure>

Why latency rather than bandwidth dominates: in decode, a batch might be 256 tokens across
8 ranks — each rank sends a few kilobytes to each other rank. That is a *small-message*
all-to-all, where fixed per-message costs and synchronization dominate and raw bandwidth is
irrelevant. It is also why these kernels are hand-written rather than left to NCCL, and why
`DpPaddingMode`'s deadlock comment in Chapter 15 exists: a symmetric collective where one
rank contributes nothing hangs.

### The arithmetic that makes it hurt

Put numbers on it. A rank holding *T* tokens, each routed to *k* experts, sends `T × k`
hidden vectors of *H* elements out and receives roughly as many back — twice, since combine
mirrors dispatch. For DeepSeek-V3's shape (*H* = 7168, *k* = 8, BF16) that is about 14 KB per
token per direction, and around 115 KB per token dispatched.

Against a *decode* batch that is a small number of tokens per rank, so the transfer is a few
megabytes — trivially within NVLink's capacity, and entirely dominated by the fixed cost of
initiating it. Two all-to-alls per layer across 60 MoE layers is 120 synchronization points
per forward pass. At even 20 µs each, that is 2.4 ms of pure latency in a step whose compute
might be 10 ms.

So the optimizations that matter are not bandwidth optimizations. They are:

**Reduce the number of peers.** Node-limited routing caps how many nodes a token reaches, so
the all-to-all is over a bounded set rather than all-to-everyone. This is the systems
constraint that shaped the model.

**Exploit the bandwidth asymmetry.** Inside a node, NVLink runs at roughly 160 GB/s; between
nodes, InfiniBand at roughly 50 GB/s. DeepEP's kernels route a token to one GPU per
destination node over IB, then fan it out to that node's other GPUs over NVLink — so a token
crosses the slow link once rather than once per destination rank. Warp specialization lets
the two transfers proceed concurrently.

**Overlap it with something.** Which is the next section, and the largest of the three.

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

Chapters 15 and 16 split the model. Chapter 17 splits the *work* — and stops asking one
machine to be good at two opposite things.

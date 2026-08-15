# 2. The Machine Underneath

> *Chapter 1's argument is parameterized by two numbers on a spec sheet. Everything that
> follows depends on knowing which two, and what happens when they change.*

Chapter 1 reached its conclusions from a ratio: peak arithmetic over peak bandwidth, about
296 FLOP per byte on an H100. Every later chapter inherits that number. The batch size the
scheduler aims for, the choice between chunking a prefill and shipping it to another
machine, whether quantization buys anything, whether speculation pays — all of them are
answers whose sign can flip if the ratio moves.

So it is worth an hour to know where those numbers come from, how they differ across the
hardware you might actually be handed, and — the part most treatments skip — which numbers
on a vendor's slide are real and which are marketing.

This chapter is the shortest in the book and the one with the least SGLang in it. That is
deliberate: it is about the machine, not the software. But it ends where the rest of the
book begins, with the handful of hardware facts SGLang actually reads at startup and the
decisions it makes from them.

<!-- objectives:begin -->
<div class="bk-objectives"><p class="bk-objectives-head">What this chapter gives you <span class="bk-objectives-time">· about 11 min</span></p><ul><li>Read a GPU spec sheet and say which number binds your workload</li><li>Compute the ridge point for any accelerator, and explain why it barely moves across generations</li><li>Place a collective on the interconnect ladder and predict what it costs</li><li>Find the handful of hardware facts SGLang reads at startup</li></ul></div>
<!-- objectives:end -->

---

## Inside one accelerator

A GPU is not a fast CPU. It is a throughput machine, organized so that thousands of simple
operations proceed at once, and its structure explains both what it is good at and the very
specific way it fails.

**Streaming multiprocessors.** The chip is divided into SMs — 132 of them on an H100 SXM,
148 on a B200. Each SM is an independent scheduler with its own register file and its own
slice of on-chip memory. Work is dispatched to SMs in units of *thread blocks*, and a kernel
that cannot fill the SMs leaves silicon idle no matter how fast the silicon is. This is the
occupancy problem, and at decode's tiny batch sizes it is a real one: a kernel with four
thread blocks uses 3% of an H100.

**Three kinds of core, and only one of them matters here.** Each SM holds CUDA cores
(scalar arithmetic), tensor cores (matrix multiply-accumulate), and special function units
(transcendentals). For inference, essentially all the arithmetic that matters runs on tensor
cores. When a spec sheet says "FLOPS," ask which cores it means; the number that governs
Chapter 1's roofline is tensor-core FLOPS.

**A memory hierarchy with a cliff in it.** Registers are effectively free. Each SM has
around 256 KB of L1 / shared memory. The whole chip shares 50 MB of L2. Then HBM — 80 GB on
an H100, 141 GB on an H200, 180 GB on a B200 — at a bandwidth two orders of magnitude below
on-chip.

That cliff is not a footnote; it is the reason Chapter 14 exists. FlashAttention is
precisely the observation that an algorithm which keeps its intermediates in the 256 KB
rather than spilling them to the 80 GB wins by the ratio between those two speeds.

Notice that this chapter's hierarchy and Chapter 11's are the same picture at different
scales: SRAM to HBM here, HBM to host DRAM to SSD there. Both are worth exploiting for the
same reason and both are exploited the same way, by keeping the hot working set one level up.

---

## The three numbers, and what each one binds

For serving, an accelerator is almost entirely described by three quantities.

**Memory capacity** decides whether the model fits at all, and then — after weights — how
much room is left for the KV cache. Chapter 1 showed that leftover capacity is the direct
limiter on concurrency, so this is the number that sets how many users the GPU serves.

**Memory bandwidth** decides decode speed. Single-stream tokens per second is very nearly
bandwidth divided by model bytes, and no amount of arithmetic capability changes it.

**Tensor-core FLOPS** decides prefill speed, and therefore time to first token on long
prompts.

Their ratio is Chapter 1's ridge point. Here is what it looks like across the hardware you
are likely to meet, at the precision each part is normally served in:

| Accelerator | Memory | Bandwidth | Dense FLOPS | Ridge *I*\* |
| --- | --- | --- | --- | --- |
| A100 80 GB SXM | 80 GB HBM2e | 2.0 TB/s | 312 TF (BF16) | ~156 |
| L40S | 48 GB GDDR6 | 0.86 TB/s | 362 TF (FP8) | ~421 |
| H100 SXM | 80 GB HBM3 | 3.35 TB/s | 990 TF (BF16) | ~296 |
| H100 SXM (FP8) | 80 GB HBM3 | 3.35 TB/s | 1,979 TF (FP8) | ~591 |
| H200 SXM | 141 GB HBM3e | 4.8 TB/s | 1,979 TF (FP8) | ~412 |
| B200 | 180 GB HBM3e | 8.0 TB/s | 4,500 TF (FP8) | ~563 |
| MI300X | 192 GB HBM3 | 5.3 TB/s | 1,307 TF (FP8) | ~247 |

Read the last column first, because it is the surprise. Across seven parts, six vendors'
worth of marketing, and four years of releases, the ridge point stays inside a single order
of magnitude — roughly 150 to 600 FLOP per byte. **Nobody is fixing the memory wall.**
Bandwidth and arithmetic have grown together, so the batch size at which a GEMM stops being
bandwidth-starved is a few hundred rows on essentially every accelerator in production.

Chapter 1's critical-batch-size calculator takes the middle two columns of that table as its
inputs, so any row of it can be run through the formula directly.

That is the fact that makes this book's structure stable. If the ridge point were falling,
batching would matter less each generation and half the engine could be deleted. It is not,
so it does not.

The second thing to read off the table is the H100 rows. Same chip, two precisions, and the
ridge point doubles — because FP8 doubles arithmetic while leaving bandwidth alone. Chapter
15's quantization chapter is, from this angle, a way of moving *left* on the roofline by
shrinking the bytes, on hardware that simultaneously moved the ridge *right*. Both effects
are real and they partly cancel; which one wins depends on whether you quantized the weights
only or the arithmetic too.

Third, notice the outliers. The L40S has a very high ridge point for a bad reason — GDDR6
instead of HBM, so its bandwidth is a quarter of a datacenter part's while its FLOPS are not.
It is a compute-rich, bandwidth-poor accelerator: fine for prefill-heavy or small-model work,
poor for high-concurrency decode. The MI300X is the reverse: 192 GB at 5.3 TB/s gives it the
lowest ridge point of the modern parts and the most room for KV cache, which is exactly the
profile a decode pool wants.

---

## The interconnect ladder

One more hierarchy governs Chapters 16 through 18 entirely, and it is the one people
underestimate. When a model does not fit on a single accelerator, every collective in the
forward pass crosses one of these:

| Link | Bandwidth | Scope |
| --- | --- | --- |
| On-chip SRAM | ~20 TB/s | within one SM |
| HBM | 3–8 TB/s | within one GPU |
| NVLink 4/5 + NVSwitch | 0.9–1.8 TB/s | GPUs within a node |
| PCIe Gen5 x16 | ~64 GB/s | GPU to host |
| InfiniBand NDR / 400G RoCE | ~50 GB/s | node to node |
| Datacenter Ethernet | 1–25 GB/s | rack to rack |

Each step down is roughly an order of magnitude, and two steps are worth memorizing because
they are where designs break.

**NVLink to InfiniBand is a 20× cliff.** This single fact is why Chapter 16 says tensor
parallelism stays inside a node: TP performs two all-reduces per transformer layer, 160 per
forward pass on an 80-layer model, and 160 collectives at InfiniBand latency is not a
forward pass, it is a stall. Pipeline parallelism crosses nodes because it moves one
activation tensor per stage boundary instead.

**HBM to PCIe is a 100× cliff.** That is the wall Chapter 11's hierarchical cache is trading
against, and the reason its fetch-versus-recompute arithmetic comes out the way it does.

The asymmetry between the middle two rows is also what shapes Chapter 17's expert
parallelism. DeepEP's dispatch kernels route a token across InfiniBand *once* per destination
node and then fan out over NVLink inside it, precisely because the two links differ by 20×
and a naive all-to-all would pay the slow one repeatedly.

---

## Reading a spec sheet without being fooled

Vendor numbers are not lies, but they are chosen. Five things to check.

**Dense or sparse?** Tensor cores can skip multiplications against zero in a 2:4 structured
sparsity pattern, which doubles the headline number. Inference is dense. If a slide says
"4 petaFLOPS" with an asterisk, halve it.

**Which precision?** FLOPS roughly double for each halving of element width, so FP4 numbers
are 4× the BF16 ones on the same silicon. Comparing a Blackwell FP4 figure to a Hopper BF16
figure compares two different quantities.

**Peak or achieved?** Peak assumes every tensor core is busy every cycle with perfectly
staged operands. Real kernels reach 60–80% of peak bandwidth and, for well-tuned GEMMs at
good shapes, perhaps 70% of peak FLOPS. Chapter 22 calls the ratio MFU and MBU; the useful
habit is to compute both from a profile and see which one is close to its ceiling.

**Which SKU?** "H100" is at least three parts. The SXM module has 3.35 TB/s and NVLink; the
PCIe card has 2.0 TB/s and no NVLink between arbitrary pairs; the NVL variant differs again.
The gap between SXM and PCIe is larger than the gap between some *generations*.

**Is the memory usable?** Weights are not the only resident. Activations, CUDA graph buffers
(Chapter 15), communication buffers for expert parallelism (Chapter 17), and the framework's
own allocator all take a cut before the KV cache gets what is left. A 180 GB part does not
give you 180 GB of cache, and the difference is large enough that SGLang computes it rather
than assuming it — which is the next section.

---

## What SGLang actually reads

Almost nothing in this chapter appears in the code as a number. SGLang does not carry a table
of GPUs. It carries a small set of *questions* it asks the device at startup, and everything
hardware-specific follows from the answers.

The abstraction is `python/sglang/srt/platforms/interface.py:26` `SRTPlatform`, with
`python/sglang/srt/platforms/cuda.py`, `python/sglang/srt/platforms/rocm.py`,
`python/sglang/srt/platforms/xpu.py`, and `python/sglang/srt/platforms/cpu.py` as the
implementations, and out-of-tree vendors registering through setuptools entry points:

```python
class SRTPlatform(DeviceMixin):
    """
    Base class for SRT hardware platform backends.

    Inherits device identity queries and operations from DeviceMixin.
    Adds SRT-specific factory methods, capability flags, and lifecycle hooks.

    OOT platforms should subclass SRTPlatform and override the methods
    relevant to their hardware.
    """
```

The first question is **what generation is this**. On NVIDIA that is the compute capability,
a `(major, minor)` pair, and
`python/sglang/srt/platforms/device_mixin.py:65` `DeviceCapability` is careful about how it
is compared:

```python
class DeviceCapability(NamedTuple):
    """Device compute capability (major, minor).

    Uses NamedTuple for built-in comparison support:
    ``DeviceCapability(9, 0) >= DeviceCapability(8, 9)`` works naturally.
    """
```

A `NamedTuple` so that `>=` does the right thing lexicographically — because nearly every use
is a threshold test. `python/sglang/srt/utils/common.py:610` `get_device_sm` flattens it to
the two-digit form the ecosystem uses (`90` for Hopper, `100` for datacenter Blackwell), and
a family of cached predicates sits on top:

```python
is_sm100_supported = lru_cache(maxsize=1)(
    partial(
        _check_cuda_device_version, device_capability_majors=[10], cuda_version=(12, 8)
    )
)
```

Note that each predicate pairs a *hardware* requirement with a *toolkit* requirement. An
SM100 device under CUDA 12.4 is not "Blackwell supported" in any useful sense, because the
kernels that exploit it will not compile. Hardware capability alone is the wrong question and
this is the code admitting it.

The comment above one of these is a good demonstration of how fine the distinctions get:

```python
# Datacenter Blackwell (SM100) plus SM110; excludes consumer Blackwell (SM120).
# This is the arch set flash_attn.cute accepts for the absorbed-MLA qv argument.
```

Consumer and datacenter parts of the same generation share a major version and not a kernel.

The second question is **what precision can this thing do**, and it is asked through
`python/sglang/srt/layers/quantization/base_config.py:149` `get_min_capability`:

```python
    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int:
        """Minimum GPU capability to support the quantization method.

        E.g., 70 for Volta, 75 for Turing, 80 for Ampere.
        This requirement is due to the custom CUDA kernels used by the
        quantization method.
        """
```

Every quantization scheme declares its own floor, and Chapter 15's config layer refuses the
combination rather than producing wrong numbers. The docstring's last sentence is the honest
reason: the limit is not the format, it is the kernels somebody wrote.

The third question is **how much memory is there**, and this one is answered with real
arithmetic rather than a lookup. `python/sglang/srt/server_args.py:4872` computes a default
`mem_fraction_static` by subtracting everything that is *not* KV cache:

```python
                # Constant meta data (e.g., from attention backend) + activation slack.
                reserved_mem = 512
                reserved_mem += activation_tokens * 1.5
                # Some adjustments for large parallel size
                reserved_mem += self.tp_size * self.pp_size / 8 * 1024
                reserved_mem += self.reserve_for_graph_mb()
                if gpu_mem is not None and gpu_mem > 60 * 1024:
                    reserved_mem = max(reserved_mem, 10 * 1024)
                # Reserve headroom for DeepEP all-to-all buffers on top of the floor.
                reserved_mem += self.reserve_for_deepep_a2a_mb()

            self.mem_fraction_static = (
                round((gpu_mem - reserved_mem) / gpu_mem, 3)
                if gpu_mem is not None
                else 0.88
            )
```

Read that as a list of everyone with a claim on the memory before the cache: attention
backend metadata, the activation working set (scaled by the chunk size, so a larger
`--chunked-prefill-size` costs cache capacity), a term that grows with parallel size,
Chapter 15's captured CUDA graphs, and Chapter 17's all-to-all buffers. The `0.88` at the
end is the fallback for when the device will not say how much memory it has.

This is the section of the book where a hardware fact becomes a serving decision, and it is
worth noticing that the decision is *computed from the device*, not configured. Chapter 22's
first tuning knob is this number, and the reason it is first is that everything downstream —
concurrency, cache hit rate, whether admission control ever has to retract — is set by what
survives this subtraction.

---

## Which number is binding on your workload

Chapter 1 gave metrics; this chapter gives hardware; the join between them is the shape of
your traffic, and one ratio captures most of it.

Let **ISL** be input sequence length and **OSL** output sequence length. Their ratio decides
which phase dominates, and therefore which of the three numbers above you should be buying:

| Workload | ISL / OSL | Dominated by | Buy | Chapters that matter |
| --- | --- | --- | --- | --- |
| Summarization, RAG, classification | 100:1 and up | prefill | FLOPS | 6, 14, 18 |
| Chat and assistants | ~2:1 to 10:1 | mixed | balance | 6, 10, 18 |
| Reasoning, agents, long generation | 1:5 and below | decode | bandwidth + capacity | 9, 15, 19 |
| Code completion | short : short | latency floor | clock, low overhead | 5, 15 |

The same ratio decides several arguments the book has already had. A prefill-dominated
workload gets little from speculative decoding, because Chapter 19's whole premise is idle
arithmetic during *decode* — and there barely is a decode. A decode-dominated workload gets
little from chunked prefill and a great deal from KV cache capacity, which argues for
Chapter 16's data-parallel attention and for the accelerator with the most HBM per dollar. A
mixed chat workload with long shared system prompts is the one where Chapter 10's prefix
cache is worth more than every kernel optimization in the book combined.

Measure the ratio before tuning anything. It is one query against your own logs, and it
determines which half of this book you should read closely.

---

Chapter 1 said the machine is memory-bound; this chapter said which memory, how fast, and how
far away. From here the book stops describing hardware and starts describing the software
that responds to it — beginning, in Chapter 3, with the shape SGLang takes when you launch it.

---

<!-- summary:begin -->
<div class="bk-card"><p class="bk-card-head">Chapter 2 in one page</p><ol class="bk-card-arg"><li>An accelerator is three numbers: memory capacity, memory bandwidth, and tensor-core FLOPS. Each binds a different phase.</li><li>Their ratio is Chapter 1's ridge point, and it sits between roughly 150 and 600 FLOP/byte on every part in production — nobody is fixing the memory wall.</li><li>The interconnect ladder falls by an order of magnitude per step; NVLink to InfiniBand is the cliff that keeps tensor parallelism inside a node.</li><li>Spec sheets are chosen, not false: check dense vs sparse, precision, peak vs achieved, and SKU.</li><li>SGLang carries no table of GPUs — it asks the device a few questions and computes the rest.</li></ol><p class="bk-card-sub">Numbers worth keeping</p><table class="bk-card-table"><tbody><tr><th scope='row'>H100 SXM</th><td>80 GB · 3.35 TB/s · 990 TF BF16</td></tr><tr><th scope='row'>NVLink vs InfiniBand</th><td>~20× bandwidth gap</td></tr><tr><th scope='row'>HBM vs PCIe</th><td>~100× gap</td></tr><tr><th scope='row'>Realistic fraction of peak</th><td>60–80% bandwidth</td></tr></tbody></table><p class="bk-card-sub">Where it lives</p><table class="bk-card-table"><tbody><tr><th scope='row'>Platform abstraction</th><td><code>python/sglang/srt/platforms/interface.py</code></td></tr><tr><th scope='row'>Capability probes</th><td><code>python/sglang/srt/utils/common.py</code></td></tr><tr><th scope='row'>Memory fraction default</th><td><code>python/sglang/srt/server_args.py</code></td></tr></tbody></table></div>
<!-- summary:end -->

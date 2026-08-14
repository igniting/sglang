# SGLang Internals

*Reading a production LLM serving engine.*

> **Status: complete first edition.** All 22 chapters and 6 appendices are written against
> the pinned commit below, and every code reference is verified in CI.

---

## The approach

Concepts and code are not separated. Each chapter is a single narrative in which an idea
is introduced and immediately grounded in the code that *is* that idea — the explanation
of paged KV memory is the walk through `PagedTokenToKVPoolAllocator`, not a preamble to
it. You should never meet theory you cannot point at.

Each chapter is built from **beats**. A beat is one idea welded to one piece of code.

## Where the book descends to CUDA

Kernels are read, not black-boxed — but only the ones that live in this repository.

- **Triton kernels** (`python/sglang/kernels/ops/`) are read as source. They are Python,
  and they are the clearest place to watch paging, masking, and online-softmax
  accumulation actually happen.
- **In-repo CUDA/C++** (`python/sglang/kernels/aot/csrc/`, `python/sglang/kernels/jit/csrc/`)
  is read where it carries an idea the Python cannot show — MoE grouped GEMM, quantized
  GEMM epilogues, speculative verification.
- **External kernel libraries** — FlashInfer, FlashAttention — are treated as *contracts*:
  the book states what they guarantee and what they cost, and does not walk their source.
  They live in separately versioned repositories (FlashInfer is a pinned dependency with
  its own bump cadence), so walking them would take on a dependency this book cannot pin.

The practical effect: you should be able to read CUDA at a glance. You are never asked to
write it.

## Version pinning

Every code reference is pinned to a single commit of the upstream repository:

```
7562e741e26a4818ee78f1e63140240ec59147c0   (2026-08-15)
```

References are written as `path:line`, sometimes with a bare `:line` continuation when the
file is already established in context. A checker (`book/scripts/verify_anchors.py`) runs
in CI and validates every anchor against the pinned tree — both that the line exists and
that the symbol named beside it is actually there. If the build is green, the anchors are
real.

SGLang moves quickly. Pinning is what lets a book about it stay true; the tradeoff is that
the newest features may not appear here.

## Scope

The subject is the serving runtime, `python/sglang/srt`, plus the frontend DSL, the kernel
layer, and the Rust gateway where the request path crosses into them.

The diffusion stack (`python/sglang/multimodal_gen`) is **out of scope**. It is a large
parallel system with its own pipelines, schedulers, and caching, and it deserves separate
treatment rather than a compressed chapter.

## Prerequisites

Python and PyTorch. Familiarity with transformer architecture. The ability to read
CUDA/C++ at a glance — no CUDA is written. GPU access is not required to follow the book,
though it helps for the performance discussions.

## How to read it

Part II follows a single request from socket to streamed token, one chapter per stage.
Everything after it is a labeled detour off that path. If you read Parts I and II in
order, you can then take the rest in any sequence you like.

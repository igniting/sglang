# Book plan

Planning notes for *SGLang Internals*. Not published — the book itself lives in `src/`.

## Decisions

| Question | Decision |
| --- | --- |
| Concepts vs code | **Fused.** No chapter separates theory from walkthrough. Each chapter is a sequence of beats; a beat is one idea welded to the code that implements it. |
| Exercises / labs | **None.** Where a claim is empirical, the book states the measurement and its conditions. |
| Length | **22 chapters, 7 parts**, ~450–550 pages, single volume. |
| Kernel depth | **Read in-repo kernels as source; treat external ones as contracts.** See below. |
| Diffusion (`multimodal_gen`) | **Out of scope.** |
| Version pinning | **Pinned to a commit SHA**, enforced by a CI checker. See below. |
| Publishing | **mdBook → `gh-pages` branch**, pushed by `.github/workflows/book-pages.yml`. Classic branch-based Pages; no repo-settings toggle needed. |

## Kernel depth, precisely

The decision is "read the CUDA," but the boundary matters, so it is written down:

- **Read as source** — Triton under `python/sglang/kernels/ops/` (it is Python and highly
  readable), and in-repo CUDA/C++ under `python/sglang/kernels/aot/csrc/` and
  `python/sglang/kernels/jit/csrc/` where it carries an idea the Python cannot show.
- **Treat as a contract** — FlashInfer and FlashAttention. They are separately versioned
  external dependencies (`flashinfer_python==0.6.17`, with a bot that bumps it), so
  walking their source would take on a dependency the book cannot pin.

Three chapters get an explicit "down to the kernel" beat: 13 (attention), 14
(quantization), 16 (MoE), plus 18 (speculative verification). In-repo CUDA is unevenly
distributed — MoE and elementwise are the densest, attention the thinnest — and the beats
follow that distribution rather than pretending it is uniform.

Note: the former top-level `sgl-kernel/` directory no longer exists; it is now
`python/sglang/kernels/aot/`. Text referring to `sgl-kernel` should say so.

## Pinning

```
7562e741e26a4818ee78f1e63140240ec59147c0   (2026-08-15)
```

This fork carries no release tags (`git ls-remote --tags` is empty), so the pin is a
commit SHA rather than a version tag.

`scripts/verify_anchors.py` validates every `path:line` anchor in `src/` against that
commit — reading files out of the commit rather than the working tree — and checks that
the symbol named beside an anchor is actually on that line (±3 lines for decorators and
multi-line signatures). It runs in CI before every build. It caught a wrong-file
continuation anchor on its first run, which is roughly the failure rate to expect while
writing.

To re-pin later: update `DEFAULT_PIN` in the script and the SHA in `src/introduction.md`,
then run the checker and fix what it reports.

## Status

All 22 chapters and 6 appendices are written (~43,000 words). Ch. 9 (RadixAttention) was
drafted first as the pilot, to test the fused concept-plus-code format on the most
beat-dense chapter before committing to the other 21; the rest followed in book order.

816 code anchors verify against the pinned commit.

## Chapter map

| # | Chapter | File |
| --- | --- | --- |
| 1 | Why Serving Engines Exist | `src/ch01-why-serving-engines.md` |
| 2 | The Shape of SGLang | `src/ch02-shape-of-sglang.md` |
| 3 | From HTTP to Token IDs | `src/ch03-http-to-token-ids.md` |
| 4 | The Scheduler Loop | `src/ch04-scheduler-loop.md` |
| 5 | Deciding What Runs Next | `src/ch05-deciding-what-runs.md` |
| 6 | Executing a Batch | `src/ch06-executing-a-batch.md` |
| 7 | Sampling and the Return Path | `src/ch07-sampling-and-return.md` |
| 8 | KV Cache Pools and Allocators | `src/ch08-kv-pools.md` |
| 9 | RadixAttention | `src/ch09-radixattention.md` |
| 10 | Caching Beyond HBM | `src/ch10-beyond-hbm.md` |
| 11 | Loading and Updating Weights | `src/ch11-loading-weights.md` |
| 12 | Anatomy of a Model | `src/ch12-anatomy-of-a-model.md` |
| 13 | Attention Backends | `src/ch13-attention-backends.md` |
| 14 | Making the Forward Pass Cheap | `src/ch14-cheap-forward-pass.md` |
| 15 | Tensor, Pipeline, and Data Parallelism | `src/ch15-parallelism.md` |
| 16 | Mixture-of-Experts and Expert Parallelism | `src/ch16-moe.md` |
| 17 | Disaggregation and Routing | `src/ch17-disaggregation.md` |
| 18 | Speculative Decoding | `src/ch18-speculative-decoding.md` |
| 19 | Shaping and Reading the Output | `src/ch19-shaping-output.md` |
| 20 | Per-Request Variation: LoRA and Multimodal | `src/ch20-per-request-variation.md` |
| 21 | Observability and Tuning | `src/ch21-observability.md` |
| 22 | Extending SGLang | `src/ch22-extending.md` |

## Building locally

```sh
# https://github.com/rust-lang/mdBook/releases
mdbook serve book        # live preview at http://localhost:3000
mdbook build book        # static site into book/output/ (gitignored)

python3 book/scripts/verify_anchors.py    # validate every code anchor
```

## Open threads

- **Diagrams.** Several chapters lean on one (process topology in Ch. 2, the address
  translation in Ch. 8, the radix tree evolving in Ch. 9, the overlap timeline in Ch. 4).
  Decide on a toolchain — inline SVG keeps them theme-aware and diffable; Mermaid needs an
  mdBook preprocessor.
- **Anchor linkification.** Anchors are currently inline code. An mdBook preprocessor could
  turn them into links to GitHub at the pinned SHA, which would make the book far easier
  to read alongside an editor.
- **Re-pin cadence.** Pinning to one commit is what keeps the book true, but the pin will
  age. Decide whether to re-pin per major release or leave it fixed for the book's life.

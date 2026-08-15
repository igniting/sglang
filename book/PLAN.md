# Book plan

Working notes for *SGLang Internals*. Not published — the book itself is `src/`.

## Decisions

| Question | Decision |
| --- | --- |
| Shape | 22 chapters in 7 parts, each part opening with a page that frames its question. |
| Concepts vs code | Fused. Each chapter is a narrative; ideas are grounded in the code that implements them, not preceded by it. |
| Chapter openings | Every chapter opens by placing the reader in the journey — what came before, what this one answers, why it follows. |
| Exercises / labs | None. |
| Kernel depth | In-repo kernels read as source (Triton, and CUDA/C++ under `kernels/aot/csrc` and `kernels/jit/csrc`). FlashInfer and FlashAttention are treated as contracts — separately versioned, so the book cannot pin them. |
| Diffusion (`multimodal_gen`) | Out of scope, stated in the front matter. |
| Version pinning | Commit `7562e741e26a4818ee78f1e63140240ec59147c0`, enforced by a CI checker. |
| Publishing | mdBook → `gh-pages`. Classic branch-based Pages, no repo-settings toggle. |

## Book production is not in the book

Anything about how the book was made — pinning mechanics, the anchor checker, the build —
lives here or in `src/colophon.md`. It does not appear in the chapters.

## Typography

`theme/custom.css`. Serif body (Iowan Old Style → Palatino → Charter → Georgia), Inter
headings, JetBrains Mono code. No webfonts: everything is system-resident, so pages render
identically offline and on first paint. Warm off-white paper in light mode, desaturated
blue-black in dark. Measure held near 72 characters. Inline code carries colour but no
background box.

Two mdBook facts this file has to respect, both learned the hard way:

- **`:root { font-size: 62.5% }`** — mdBook makes 1rem = 10px, and `:root` outranks a
  plain `html { font-size: 100% }` override. Sizes written for a 16px base render at
  62.5% of intention; the first pass shipped 11px body text in a 400px column. Every rem
  here is px ÷ 10.
- **Theme classes** — palette selectors must be `.light`/`.rust` and `.navy`/`.coal`/`.ayu`.
  An earlier `html:not(.dark)` outranked `.navy` and silently disabled dark mode.

Verify with a headless measurement, not by eye: body should compute to 19px, the column to
700px, ~70 characters per line.

## Diagrams

Inline SVG, **generated** by `scripts/diagrams.py` — never hand-edited. Hand-placed
coordinates produced overlapping boxes, text spilling past its container, and (in the first
edition of the Ch. 2 figure) an arrow routed straight through another node. Figures are now
declared as boxes plus connections, and the geometry is checked before anything is written:
text fits its box, boxes do not overlap, everything is inside the viewBox.

Seven figures: process topology (Ch. 2), the overlap timeline (Ch. 4), address translation
(Ch. 8), the radix tree evolving (Ch. 9), what a paged attention kernel receives (Ch. 13),
a disaggregated deployment (Ch. 17), the draft tree (Ch. 18). To add one, write a builder,
register it in `FIGURES`, and drop `<!-- FIGURE -->` where it belongs in the chapter.

Two conventions make the checks mean something:

- **viewBox width is 700**, matching `--bk-measure`, so figures render 1:1 in the text
  column and the declared font sizes are the rendered font sizes.
- **font sizes are emitted inline**, so a stylesheet rule cannot silently contradict the
  model the generator validated against.

`scripts/check_diagrams.mjs` re-runs the same checks in headless Chromium against real
`getBBox()` metrics, in both themes, and writes a screenshot of every figure:

```sh
CHROMIUM_PATH=/path/to/chromium node book/scripts/check_diagrams.mjs /tmp/figures
```

**Gotcha:** CommonMark terminates an HTML block at a blank line, so a `<figure>` block must
contain none — otherwise everything after the first gap renders as escaped text. `figure()`
asserts this.

## Anchors

`scripts/verify_anchors.py` validates every `path:line` reference against the pinned commit,
reading files out of the commit rather than the working tree, and checks that the symbol
named beside an anchor is on that line (±3 for decorators and multi-line signatures).

`scripts/linkify_anchors.py` is an mdBook preprocessor that rewrites those references into
links at the pinned commit. It skips fenced code blocks, and tracks the current file so
`:123` continuations resolve.

Notation rules the checker enforces:
- Full paths only — `python/sglang/benchmark/one_batch.py`, not `benchmark/one_batch.py`.
- A bare backticked path is existence-checked but does **not** rebind continuation context.
- Casual prose mentions of a file should be unstyled, not backticked.

## Building locally

```sh
mdbook serve book        # live preview
mdbook build book        # static site into book/output/ (gitignored)

python3 book/scripts/verify_anchors.py
```

## Open threads

- **More diagrams.** Chapter 16 would carry one (expert dispatch across ranks).
- **Re-pin cadence.** Fixed for this edition. Chapters 13, 14, and 18 age fastest.

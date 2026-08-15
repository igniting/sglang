#!/usr/bin/env python3
"""Generate the book's SVG figures from declarative layouts.

Hand-placing SVG coordinates produced overlapping boxes, text spilling past its
container, and arrows routed straight through other nodes. This module removes
that failure mode: a figure is declared as boxes plus connections, and the
geometry is checked before anything is written to disk.

Structural checks performed here:

  * every label fits inside its box, horizontally and vertically
  * no two boxes overlap
  * every box lies inside the viewBox

Two conventions make those checks meaningful:

  * **viewBox width is 700**, matching ``--bk-measure``, so a figure renders at
    1:1 in the text column and the font sizes below are the rendered sizes.
  * **font sizes are emitted inline**, so the stylesheet cannot silently
    override the model this file is validating against.

``check_diagrams.mjs`` performs the complementary check in a real browser, where
text metrics are exact rather than estimated from a per-character average.

Usage:  python3 book/scripts/diagrams.py
"""

from __future__ import annotations

import pathlib
import re
import sys
from dataclasses import dataclass, field

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"

W = 700  # viewBox width, == --bk-measure, so figures render 1:1

TITLE_SIZE = 13.0
LINE_SIZE = 11.5
NOTE_SIZE = 11.5
NODE_SIZE = 11.0

# Width of one character as a fraction of font size, for the sans stack figures
# use. Deliberately generous, so the estimate errs toward text being too wide.
CHAR_W = 0.55
BOLD_CHAR_W = 0.60

PAD_X = 18.0  # horizontal breathing room inside a box
PAD_Y = 12.0  # vertical breathing room inside a box
GAP = 5.0  # extra leading between stacked lines in a box


def text_width(s: str, size: float, bold: bool = False) -> float:
    return len(s) * size * (BOLD_CHAR_W if bold else CHAR_W)


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


@dataclass
class Box:
    name: str
    x: float
    y: float
    w: float
    h: float
    title: str = ""
    lines: list[str] = field(default_factory=list)
    accent: bool = False
    idle: bool = False
    title_size: float = TITLE_SIZE
    line_size: float = LINE_SIZE

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def step(self) -> float:
        """Baseline-to-baseline distance for the stacked text inside the box."""
        return (self.title_size if self.title else self.line_size) + GAP

    @property
    def rows(self) -> int:
        return (1 if self.title else 0) + len(self.lines)

    def overlaps(self, other: "Box", pad: float = 0) -> bool:
        return not (
            self.right + pad <= other.x
            or other.right + pad <= self.x
            or self.bottom + pad <= other.y
            or other.bottom + pad <= self.y
        )

    def widest(self) -> float:
        w = text_width(self.title, self.title_size, bold=True) if self.title else 0.0
        for ln in self.lines:
            w = max(w, text_width(ln, self.line_size))
        return w

    def needed_height(self) -> float:
        if self.rows == 0:
            return 0.0
        tallest = max(self.title_size if self.title else 0, self.line_size)
        return (self.rows - 1) * self.step + tallest + PAD_Y

    def svg(self) -> str:
        cls = "dgm-idle" if self.idle else ("dgm-box-accent" if self.accent else "dgm-box")
        out = [
            f'<rect class="{cls}" x="{self.x}" y="{self.y}" '
            f'width="{self.w}" height="{self.h}" rx="6"/>'
        ]
        # Centre the text block on the box: first baseline sits half a block
        # above centre, plus a cap-height nudge so optical centre matches.
        top = self.cy - (self.rows - 1) * self.step / 2 + self.title_size * 0.34
        i = 0
        if self.title:
            out.append(
                f'<text class="dgm-label" x="{self.cx}" y="{round(top, 1)}" '
                f'text-anchor="middle" font-weight="600" '
                f'style="font-size:{self.title_size}px">{esc(self.title)}</text>'
            )
            i += 1
        for ln in self.lines:
            if ln:
                y = round(top + i * self.step, 1)
                out.append(
                    f'<text class="dgm-small" x="{self.cx}" y="{y}" '
                    f'text-anchor="middle" '
                    f'style="font-size:{self.line_size}px">{esc(ln)}</text>'
                )
            i += 1
        return "\n".join(out)


def arrow(x1, y1, x2, y2, accent=False, dashed=False) -> str:
    cls = "dgm-dash" if dashed else ("dgm-line-accent" if accent else "dgm-line")
    marker = "arrow-accent" if accent else "arrow"
    return (
        f'<path class="{cls}" d="M{x1} {y1} L{x2} {y2}" '
        f'marker-end="url(#{marker})"/>'
    )


def line(x1, y1, x2, y2, dashed=False) -> str:
    cls = "dgm-dash" if dashed else "dgm-line"
    return f'<path class="{cls}" d="M{x1} {y1} L{x2} {y2}"/>'


def elbow(pts, accent=False) -> str:
    """Orthogonal polyline through the given points, arrowhead at the end."""
    cls = "dgm-line-accent" if accent else "dgm-line"
    marker = "arrow-accent" if accent else "arrow"
    d = "M" + " L".join(f"{x} {y}" for x, y in pts)
    return f'<path class="{cls}" d="{d}" marker-end="url(#{marker})"/>'


def label(x, y, s, anchor="middle", size=NOTE_SIZE, bold=False, ink=False) -> str:
    cls = "dgm-label" if ink else "dgm-small"
    w = ' font-weight="600"' if bold else ""
    return (
        f'<text class="{cls}" x="{round(x, 1)}" y="{round(y, 1)}" '
        f'text-anchor="{anchor}"{w} '
        f'style="font-size:{size}px">{esc(s)}</text>'
    )


def poly(pts, accent=False, dashed=False) -> str:
    """An open polyline with no arrowhead — for plotted curves and axes."""
    cls = "dgm-dash" if dashed else ("dgm-line-accent" if accent else "dgm-line")
    d = "M" + " L".join(f"{round(x, 1)} {round(y, 1)}" for x, y in pts)
    return f'<path class="{cls}" d="{d}"/>'


def dot(x, y, accent=True, r=4.5) -> str:
    cls = "dgm-fill-accent" if accent else "dgm-fill-muted"
    return f'<circle class="{cls}" cx="{round(x, 1)}" cy="{round(y, 1)}" r="{r}"/>'


def brace(x1, x2, y, text) -> str:
    """A span marker under a range, with a label beneath it.

    The label is centred on the span but slid inward if that would push it off
    the page — a long caption under a narrow span is the normal case.
    """
    half = text_width(text, NOTE_SIZE) / 2
    lx = min(max((x1 + x2) / 2, half + 16), W - half - 16)
    return (
        f'<path class="dgm-line-accent" d="M{x1} {y - 6} L{x1} {y} '
        f'L{x2} {y} L{x2} {y - 6}"/>\n'
        + label(lx, y + 16, text)
    )


DEFS = (
    "<defs>"
    '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
    'markerHeight="7" orient="auto-start-reverse">'
    '<path d="M 0 0 L 10 5 L 0 10 z" style="fill:var(--dgm-rule)"/></marker>'
    '<marker id="arrow-accent" viewBox="0 0 10 10" refX="9" refY="5" '
    'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
    '<path d="M 0 0 L 10 5 L 0 10 z" style="fill:var(--dgm-accent)"/></marker>'
    "</defs>"
)


def figure(h, title, aria, parts, caption) -> str:
    body = "\n".join(p for p in parts if p)
    svg = (
        "<figure>\n"
        f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="{esc(aria)}">\n'
        f"<title>{esc(title)}</title>\n{body}\n{DEFS}\n</svg>\n"
        f"<figcaption>{caption}</figcaption>\n</figure>"
    )
    # CommonMark ends an HTML block at the first blank line; one inside a figure
    # would make the rest of the SVG render as escaped text.
    assert "\n\n" not in svg, f"blank line inside figure {title!r}"
    return svg


def validate(boxes: list[Box], h: float, name: str) -> list[str]:
    errs = []
    for b in boxes:
        if b.widest() > b.w - PAD_X:
            errs.append(
                f"{name}: text overflows {b.name!r} "
                f"(needs {b.widest() + PAD_X:.0f}px, box is {b.w:.0f}px)"
            )
        if b.needed_height() > b.h + 0.5:
            errs.append(
                f"{name}: text too tall for {b.name!r} "
                f"(needs {b.needed_height():.0f}px, box is {b.h:.0f}px)"
            )
        if b.x < 0 or b.y < 0 or b.right > W or b.bottom > h:
            errs.append(f"{name}: box {b.name!r} outside viewBox")
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            if a.overlaps(b):
                errs.append(f"{name}: boxes {a.name!r} and {b.name!r} overlap")
    return errs


# ===========================================================================
# Chapter 2 — process topology
# ===========================================================================


def fig_topology():
    H = 462
    tok = Box(
        "tokenizer", 140, 44, 420, 64,
        "TokenizerManager",
        ["main process · asyncio event loop",
         "text → token ids · holds each request's future"],
        accent=True,
    )
    s0 = Box("sched0", 20, 196, 180, 76, "Scheduler · tp 0",
             ["owns one GPU", "receives, then broadcasts"], accent=True)
    s1 = Box("sched1", 230, 196, 170, 76, "Scheduler · tp 1", ["owns one GPU"])
    sn = Box("schedn", 430, 196, 170, 76, "Scheduler · tp n", ["owns one GPU"])
    det = Box("detok", 20, 330, 320, 58, "DetokenizerManager",
              ["subprocess · token ids → text, incrementally"])
    boxes = [tok, s0, s1, sn, det]

    ret_y = det.cy
    parts = [
        label(tok.cx, 26, "HTTP clients", size=13, bold=True, ink=True),
        arrow(tok.cx, 32, tok.cx, tok.y - 6),
        elbow([(tok.cx, tok.bottom), (tok.cx, 160), (s0.cx, 160), (s0.cx, s0.y - 6)]),
        label(tok.cx + 12, 152, "ZMQ · scheduler_input_ipc_name — rank 0 only",
              anchor="start"),
        arrow(s0.right, s0.cy, s1.x - 6, s1.cy, dashed=True),
        arrow(s1.right, s1.cy, sn.x - 6, sn.cy, dashed=True),
        label(400, 300, "broadcast_pyobj — every rank runs an identical batch"),
        arrow(s0.cx, s0.bottom, s0.cx, det.y - 6),
        label(s0.cx + 12, 312, "ZMQ · detokenizer_ipc_name", anchor="start"),
        elbow([(det.right, ret_y), (650, ret_y), (650, tok.cy),
               (tok.right + 6, tok.cy)], accent=True),
        label(497, ret_y - 9, "ZMQ · tokenizer_ipc_name"),
        label(W / 2, 424, "Output travels forward to the TokenizerManager, "
                          "not back to the scheduler:"),
        label(W / 2, 442, "that is the process holding the client's awaiting "
                          "coroutine."),
    ]
    errs = validate(boxes, H, "ch02 topology")
    return figure(
        H, "SGLang process topology",
        "Four processes connected by ZeroMQ sockets: tokenizer manager, one "
        "scheduler per tensor-parallel rank, and a detokenizer",
        [b.svg() for b in boxes] + parts,
        "The four processes and the sockets between them. Under tensor "
        "parallelism only rank&nbsp;0 receives from the front end; it broadcasts "
        "to its peers so every rank sees the same batch.",
    ), errs


# ===========================================================================
# Chapter 1 — the roofline
# ===========================================================================


def fig_roofline():
    H = 348
    # Plot area, log-log. x: arithmetic intensity 1 … 10^4. y: 1 … 10^3 TFLOP/s.
    X0, X1, Y0, Y1 = 96, 656, 244, 52
    PEAK = 990.0  # TFLOP/s, BF16
    BW = 3.35  # TFLOP/s per FLOP/byte, i.e. 3.35 TB/s
    RIDGE = PEAK / BW

    def px(i):
        import math
        return X0 + (X1 - X0) * math.log10(i) / 4

    def py(p):
        import math
        return Y0 + (Y1 - Y0) * math.log10(p) / 3

    parts = [
        label(376, 22, "H100 SXM, BF16 — attainable performance vs arithmetic "
                       "intensity", size=12.5, bold=True, ink=True),
        # axes
        poly([(X0, Y1 - 8), (X0, Y0), (X1 + 8, Y0)]),
        label(X0 - 8, Y0 + 4, "1", anchor="end", size=10.5),
        label(X0 - 8, py(10) + 4, "10", anchor="end", size=10.5),
        label(X0 - 8, py(100) + 4, "100", anchor="end", size=10.5),
        label(X0 - 8, py(1000) + 4, "1000", anchor="end", size=10.5),
        label(38, 150, "TFLOP/s", anchor="middle", size=11),
        label(X0, Y0 + 20, "1", size=10.5),
        label(px(10), Y0 + 20, "10", size=10.5),
        label(px(100), Y0 + 20, "100", size=10.5),
        label(px(1000), Y0 + 20, "1000", size=10.5),
        label(X1, Y0 + 20, "10⁴", size=10.5),
        label((X0 + X1) / 2, Y0 + 40, "arithmetic intensity — FLOP per byte", size=11),
        # the roof itself
        poly([(px(1), py(BW)), (px(RIDGE), py(PEAK)), (px(10000), py(PEAK))],
             accent=True),
        label(px(1.6), py(2.0), "memory-bound", anchor="start", size=11),
        label(px(1.6), py(1.45), "slope = 3.35 TB/s", anchor="start", size=11),
        label(px(1400), py(PEAK) - 12, "compute-bound — 990 TFLOP/s", size=11),
        # ridge point
        poly([(px(RIDGE), Y0), (px(RIDGE), py(PEAK))], dashed=True),
        label(px(RIDGE) + 9, Y0 - 8, "I* = 296", anchor="start", size=11,
              bold=True),
        # workload markers
        dot(px(2), py(2 * BW)),
        label(px(2) + 10, py(2 * BW) + 4, "decode, batch 1", anchor="start", size=11),
        dot(px(64), py(64 * BW)),
        label(px(64) + 10, py(64 * BW) + 4, "decode, batch 64", anchor="start",
              size=11),
        dot(px(2000), py(PEAK)),
        label(px(2000), py(PEAK) + 18, "prefill", size=11),
        label(W / 2, 318,
              "Batching moves a workload right along the slope. The whole point "
              "is to reach the corner."),
    ]
    return figure(H, "The roofline, and where decode sits on it",
                  "A log-log roofline plot showing decode far to the left of the "
                  "ridge point and prefill at the compute ceiling",
                  parts,
                  "Arithmetic intensity for a weight-bound GEMM is <code>2B/s</code> "
                  "— batch size over element width, and nothing else. Everything "
                  "the scheduler does is an attempt to move right."), []


# ===========================================================================
# Chapter 4 — the overlap timeline
# ===========================================================================


def fig_overlap():
    H = 406
    F = 140.0  # width of one forward pass
    S = 56.0  # width of one round of scheduler work
    x0 = 64.0
    boxes: list[Box] = []
    parts: list[str] = []

    def bar(name, x, y, w, h, lines, accent=False, idle=False, size=10.5):
        b = Box(name, x, y, w, h, "", lines, accent=accent, idle=idle,
                line_size=size)
        boxes.append(b)
        return b

    # ---- panel A: overlapped -------------------------------------------
    parts.append(label(W / 2, 26,
                       "event_loop_overlap — the scheduler runs inside the forward",
                       size=12.5, bold=True, ink=True))
    parts.append(label(56, 70, "GPU", anchor="end", size=12, ink=True))
    parts.append(label(56, 126, "CPU", anchor="end", size=12, ink=True))

    ga = bar("ga", x0, 44, F, 40, ["forward N−1"])
    gb = bar("gb", x0 + F + 2, 44, F, 40, ["forward N"], accent=True)
    gc = bar("gc", x0 + 2 * (F + 2), 44, F, 40, ["forward N+1"])
    parts.append(label(gc.right + 22, 70, "…", size=14))

    cb = bar("cb", gb.x, 100, F, 40, ["process N−1", "plan N+1"], accent=True,
             size=10)
    cc = bar("cc", gc.x, 100, F, 40, ["process N", "plan N+2"], size=10)
    parts += [
        line(gb.x, gb.bottom, gb.x, cb.bottom + 6, dashed=True),
        line(gb.right, gb.bottom, gb.right, cb.bottom + 6, dashed=True),
        label(W / 2, 166,
              "Each CPU bar fits inside the GPU bar above it: scheduling costs "
              "no wall-clock time."),
    ]
    parts.append(line(20, 190, W - 20, 190, dashed=True))

    # ---- panel B: serial -----------------------------------------------
    parts.append(label(W / 2, 216,
                       "the same three steps without overlap — the GPU waits",
                       size=12.5, bold=True, ink=True))
    parts.append(label(56, 250, "GPU", anchor="end", size=12, ink=True))
    parts.append(label(56, 306, "CPU", anchor="end", size=12, ink=True))

    ha = bar("ha", x0, 224, F, 40, ["forward N−1"])
    i1 = bar("i1", ha.right + 2, 224, S, 40, ["idle"], idle=True, size=10)
    hb = bar("hb", i1.right + 2, 224, F, 40, ["forward N"], accent=True)
    i2 = bar("i2", hb.right + 2, 224, S, 40, ["idle"], idle=True, size=10)
    hc = bar("hc", i2.right + 2, 224, F, 40, ["forward N+1"])

    bar("sa", i1.x, 280, S, 40, ["sched"], size=10)
    bar("sb", i2.x, 280, S, 40, ["sched"], size=10)
    parts += [
        line(i1.x, i1.bottom, i1.x, 326, dashed=True),
        line(i2.right, i2.bottom, i2.right, 326, dashed=True),
        brace(gc.right, hc.right, 348,
              "stalled GPU — one gap per step, for every step of every request"),
        label(W / 2, 392,
              "Overlap does not make the forward faster. It makes everything "
              "else free."),
    ]
    errs = validate(boxes, H, "ch04 overlap")
    return figure(
        H, "One iteration of the overlap loop",
        "Two timelines comparing the overlapped scheduler loop against a serial "
        "one, showing the GPU idle gaps the overlap removes",
        [b.svg() for b in boxes] + parts,
        "The same three decode steps, scheduled two ways. In "
        "<code>event_loop_overlap</code> the results of step N−1 are processed "
        "while step N is still running.",
    ), errs


# ===========================================================================
# Chapter 8 — address translation
# ===========================================================================


def fig_address():
    H = 330
    req = Box("req", 20, 56, 140, 62, "request r", ["req_pool_idx = 7"],
              accent=True)
    l1 = Box("l1", 196, 40, 228, 94, "ReqToTokenPool",
             ["req_to_token[7, t] → kv_index",
              "one dense int32 row per request",
              "length = max_context_len"])
    alloc = Box("alloc", 460, 40, 220, 94, "allocator",
                ["kv_index ÷ page_size → page",
                 "kv_index mod page_size → slot",
                 "pages need not be contiguous"])
    l2 = Box("l2", 196, 200, 484, 74, "KVCache",
             ["k_buffer[layer][kv_index]   ·   v_buffer[layer][kv_index]",
              "one tensor pair per layer, shared by every request"], accent=True)
    boxes = [req, l1, alloc, l2]

    parts = [
        label(l1.cx, 28, "level 1 — where are my tokens?"),
        label(alloc.cx, 28, "index → address"),
        label(l2.cx, 188, "level 2 — where does token index i actually live?"),
        arrow(req.right, req.cy, l1.x - 6, l1.cy),
        arrow(l1.right, l1.cy, alloc.x - 6, alloc.cy),
        elbow([(alloc.cx, alloc.bottom), (alloc.cx, l2.y - 6)]),
        label(W / 2, 300,
              "Two requests can hold the same kv_index at different positions "
              "in their own rows."),
        label(W / 2, 318,
              "That is the mechanical precondition for prefix sharing."),
    ]
    errs = validate(boxes, H, "ch08 address")
    return figure(
        H, "Address translation, request to KV storage",
        "Two levels of indirection from a request to physical KV storage",
        [b.svg() for b in boxes] + parts,
        "Attention kernels do not walk this chain token by token: they receive "
        "the <code>req_to_token</code> row as a page table and index it inside "
        "the kernel. That is what “paged attention” names.",
    ), errs


# ===========================================================================
# Chapter 9 — the radix tree across three requests
# ===========================================================================


def fig_tree():
    H = 364
    boxes: list[Box] = []
    parts: list[str] = []

    def node(name, cx, y, w, text, accent=False):
        b = Box(name, cx - w / 2, y, w, 28, "", [text], accent=accent,
                line_size=NODE_SIZE)
        boxes.append(b)
        return b

    def root(cx):
        parts.append(f'<circle class="dgm-box" cx="{cx}" cy="56" r="9"/>')

    # panel A ------------------------------------------------------------
    parts.append(label(118, 34, "after request A", bold=True, ink=True))
    root(118)
    parts.append(label(134, 60, "root", anchor="start", size=10.5))
    a1 = node("a1", 118, 92, 180, "S + “what is 2+2”")
    parts.append(line(118, 65, 118, a1.y))

    # panel B ------------------------------------------------------------
    parts.append(label(350, 34, "after request B — split", bold=True, ink=True))
    root(350)
    b1 = node("b1", 350, 92, 152, "S + “what is ”", accent=True)
    parts.append(line(350, 65, 350, b1.y))
    b2 = node("b2", 290, 148, 76, "“2+2”")
    b3 = node("b3", 384, 148, 76, "“3+3”")
    parts += [line(340, b1.bottom, 290, b2.y), line(360, b1.bottom, 384, b3.y)]

    # panel C ------------------------------------------------------------
    parts.append(label(582, 34, "after request C", bold=True, ink=True))
    root(582)
    c1 = node("c1", 582, 92, 76, "S", accent=True)
    parts.append(line(582, 65, 582, c1.y))
    c2 = node("c2", 582, 148, 122, "“what is ”", accent=True)
    parts.append(line(582, c1.bottom, 582, c2.y))
    c3 = node("c3", 538, 204, 76, "“2+2”")
    c4 = node("c4", 632, 204, 76, "“3+3”")
    parts += [line(572, c2.bottom, 538, c3.y), line(592, c2.bottom, 632, c4.y)]

    parts += [
        line(234, 20, 234, 246, dashed=True),
        line(466, 20, 466, 246, dashed=True),
        line(20, 264, W - 20, 264, dashed=True),
        label(W / 2, 290,
              "Nobody planned the node holding the system prompt S — the third "
              "request's shape carved it."),
        label(W / 2, 308,
              "The tree finds the workload's branch points without being told "
              "what they are."),
        label(W / 2, 334,
              "Eviction runs the other way, leaves first: the tails go before "
              "“what is ”, and S —"),
        label(W / 2, 352, "shared by everything — goes last."),
    ]
    errs = validate(boxes, H, "ch09 tree")
    return figure(
        H, "The radix tree after each of three requests",
        "A radix tree evolving as three chat requests share a system prompt",
        [b.svg() for b in boxes] + parts,
        "Splitting is not a failure mode; it is how the tree learns where the "
        "workload actually branches.",
    ), errs


# ===========================================================================
# Chapter 13 — what a paged attention kernel receives
# ===========================================================================


def fig_pagetable():
    H = 326
    q = Box("q", 20, 54, 140, 66, "Q", ["one row per", "sequence in batch"])
    idx = Box("idx", 196, 44, 232, 86, "kv_indptr / kv_indices",
              ["indptr[i] → where sequence i's",
               "page list begins",
               "indices → the page numbers"], accent=True)
    pool = Box("pool", 464, 44, 216, 86, "K_Buffer / V_Buffer",
               ["the whole pool passed in",
                "as one tensor — never a",
                "per-sequence copy"])
    out = Box("out", 196, 196, 484, 66,
              "the kernel does the indirection itself",
              ["for each block of keys: look up the page, load it,",
               "then accumulate a partial softmax"], accent=True)
    boxes = [q, idx, pool, out]

    parts = [
        arrow(q.right, q.cy, idx.x - 6, idx.cy),
        arrow(idx.right, idx.cy, pool.x - 6, pool.cy),
        elbow([(idx.cx, idx.bottom), (idx.cx, out.y - 6)]),
        elbow([(pool.cx, pool.bottom), (pool.cx, out.y - 6)]),
        label(W / 2, 294,
              "No [seq_len × seq_len] score matrix is ever materialised: each "
              "block emits a partial"),
        label(W / 2, 312,
              "softmax and a log-sum-exp, and a second stage combines them."),
    ]
    errs = validate(boxes, H, "ch13 pagetable")
    return figure(
        H, "What a paged attention kernel receives",
        "Query rows, a page table, and the whole KV pool handed to the "
        "attention kernel",
        [b.svg() for b in boxes] + parts,
        "“Paged attention” is not a metaphor: the address arithmetic of "
        "Chapter&nbsp;8 happens inside the kernel's inner loop.",
    ), errs


# ===========================================================================
# Chapter 16 — one MoE layer across four ranks
# ===========================================================================


def fig_dispatch():
    H = 388
    ranks = []
    xs = [20, 193, 366, 539]
    for i, x in enumerate(xs):
        ranks.append(Box(f"tok{i}", x, 44, 150, 56, f"rank {i}",
                         ["its share of the batch"], line_size=10.5))
    disp = Box("disp", 20, 122, 669, 40, "",
               ["all-to-all dispatch — every token to the ranks holding its "
                "top-k experts"], accent=True)
    experts = []
    for i, x in enumerate(xs):
        hot = i == 2
        experts.append(Box(f"exp{i}", x, 186, 150, 62, f"experts {i*64}–{i*64+63}",
                           ["grouped GEMM",
                            "hot" if hot else "ordinary load"],
                           accent=hot, line_size=10.5))
    comb = Box("comb", 20, 272, 669, 40, "",
               ["all-to-all combine — partial outputs returned and weighted"],
               accent=True)
    boxes = ranks + [disp] + experts + [comb]

    parts = []
    for r, e in zip(ranks, experts):
        parts.append(arrow(r.cx, r.bottom, r.cx, disp.y - 5))
        parts.append(arrow(disp.cx if False else e.cx, disp.bottom, e.cx, e.y - 5))
        parts.append(arrow(e.cx, e.bottom, e.cx, comb.y - 5))
    parts += [
        label(W / 2, 34, "one MoE layer, four ranks", size=12.5, bold=True,
              ink=True),
        label(W / 2, 336,
              "Two collectives per layer, sixty layers: the cost is the count of "
              "synchronizations,"),
        label(W / 2, 354,
              "not the bytes. And the slowest rank sets the step — which is what "
              "makes one hot"),
        label(W / 2, 372,
              "expert everyone's problem, and why EPLB moves or replicates it."),
    ]
    errs = validate(boxes, H, "ch16 dispatch")
    return figure(H, "One MoE layer across four ranks",
                  "Tokens dispatched to expert-holding ranks and combined back, "
                  "with one rank carrying a hot expert",
                  [b.svg() for b in boxes] + parts,
                  "Every token crosses the network twice per MoE layer. Expert "
                  "parallelism turns a memory problem into a communication one."), errs


# ===========================================================================
# Chapter 17 — a disaggregated deployment
# ===========================================================================


def fig_deploy():
    H = 356
    gw = Box("gw", 160, 52, 380, 68, "sgl-model-gateway",
             ["Rust · cache-aware routing over an approximate",
              "radix tree · service discovery · PD-aware placement"],
             accent=True)
    pre = Box("pre", 20, 180, 290, 96, "Prefill pool",
              ["compute-bound · large batches",
               "TP for arithmetic throughput",
               "never runs a decode step"])
    dec = Box("dec", 390, 180, 290, 96, "Decode pool",
              ["bandwidth-bound · many sequences",
               "DP attention · large KV pool",
               "receives cache, never prefills"])
    boxes = [gw, pre, dec]

    parts = [
        label(gw.cx, 30, "clients", size=13, bold=True, ink=True),
        arrow(gw.cx, 36, gw.cx, gw.y - 6),
        elbow([(gw.cx - 90, gw.bottom), (gw.cx - 90, 152),
               (pre.cx, 152), (pre.cx, pre.y - 6)]),
        elbow([(gw.cx + 90, gw.bottom), (gw.cx + 90, 152),
               (dec.cx, 152), (dec.cx, dec.y - 6)]),
        arrow(pre.right, pre.cy, dec.x - 6, dec.cy, accent=True),
        label(350, pre.cy - 12, "KV cache"),
        label(350, pre.cy + 22, "RDMA"),
        label(W / 2, 306,
              "The transfer sits on the critical path: a 4,000-token prefill "
              "produces over a gigabyte"),
        label(W / 2, 324,
              "of cache that must land before the decode pool can emit a first "
              "token."),
        label(W / 2, 348,
              "Every box here is a chapter — pools (8), routing tree (9), "
              "parallelism (15, 16)."),
    ]
    errs = validate(boxes, H, "ch17 deploy")
    return figure(
        H, "A disaggregated deployment",
        "A cache-aware router in front of separate prefill and decode pools",
        [b.svg() for b in boxes] + parts,
        "Prefill and decode want opposite hardware and opposite parallelism. "
        "Disaggregation stops asking one machine to be good at both.",
    ), errs


# ===========================================================================
# Chapter 18 — drafting and verifying a token tree
# ===========================================================================


def fig_spec():
    H = 306
    boxes: list[Box] = []
    parts: list[str] = []

    def node(name, cx, y, w, text, accent=False):
        b = Box(name, cx - w / 2, y, w, 28, "", [text], accent=accent,
                line_size=NODE_SIZE)
        boxes.append(b)
        return b

    parts.append(label(170, 30, "draft tree — depth 3, top-k 2", bold=True,
                       ink=True))
    r = node("r", 170, 46, 100, "context")
    d1 = node("d1", 80, 100, 76, "“the”", accent=True)
    d2 = node("d2", 222, 100, 64, "“a”")
    d3 = node("d3", 52, 154, 72, "“cat”", accent=True)
    d4 = node("d4", 136, 154, 72, "“dog”")
    d5 = node("d5", 232, 154, 72, "“big”")
    parts += [
        line(160, r.bottom, 80, d1.y),
        line(186, r.bottom, 222, d2.y),
        line(70, d1.bottom, 52, d3.y),
        line(92, d1.bottom, 136, d4.y),
        line(224, d2.bottom, 232, d5.y),
    ]

    ver = Box("ver", 336, 54, 344, 120, "one target forward pass",
              ["every node is verified at once, under a mask",
               "where a candidate attends to its ancestors",
               "only — never to a sibling branch",
               "",
               "cost ≈ one ordinary decode step"], accent=True)
    boxes.append(ver)
    parts.append(arrow(286, 112, ver.x - 6, ver.cy, accent=True))

    parts += [
        line(20, 200, W - 20, 200, dashed=True),
        label(W / 2, 226,
              "The accepted prefix is the longest run of a branch the target "
              "agrees with: here"),
        label(W / 2, 244,
              "“the → cat”, three tokens for the price of one forward pass."),
        label(W / 2, 270,
              "Acceptance is rejection sampling, so the output distribution is "
              "identical to"),
        label(W / 2, 288,
              "ordinary decoding. A bad draft costs compute, never correctness."),
    ]
    errs = validate(boxes, H, "ch18 spec")
    return figure(
        H, "Drafting and verifying a token tree",
        "A speculative draft tree verified in a single target forward pass",
        [b.svg() for b in boxes] + parts,
        "A tree hedges where a chain cannot: if the top choice at depth two is "
        "wrong, a sibling may still be right.",
    ), errs


# ===========================================================================

FIGURES = {
    "ch01-why-serving-engines": fig_roofline,
    "ch02-shape-of-sglang": fig_topology,
    "ch04-scheduler-loop": fig_overlap,
    "ch08-kv-pools": fig_address,
    "ch09-radixattention": fig_tree,
    "ch13-attention-backends": fig_pagetable,
    "ch16-moe": fig_dispatch,
    "ch17-disaggregation": fig_deploy,
    "ch18-speculative-decoding": fig_spec,
}

MARKER = "<!-- FIGURE -->"
FIGURE_RE = re.compile(r"<figure>.*?</figure>", re.DOTALL)


def main() -> int:
    all_errs: list[str] = []
    for slug, fn in FIGURES.items():
        svg, errs = fn()
        all_errs += errs
        path = SRC / f"{slug}.md"
        text = path.read_text()
        if (m := FIGURE_RE.search(text)) is not None:
            text = text[: m.start()] + svg + text[m.end() :]
        elif MARKER in text:
            text = text.replace(MARKER, svg, 1)
        else:
            print(f"  !! {slug}: no <figure> and no {MARKER} to place one")
            continue
        path.write_text(text)
        print(f"  {slug}")

    if all_errs:
        print("\nGEOMETRY ERRORS:")
        for e in all_errs:
            print(f"  {e}")
        return 1
    print(f"\n{len(FIGURES)} figures written, geometry checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

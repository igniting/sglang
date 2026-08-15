#!/usr/bin/env python3
"""mdBook preprocessor: render GitHub-style alert blocks as typed callouts.

The book had exactly one callout shape — a blockquote — doing four different
jobs: defining a term, warning about a trap, stating a conclusion, and holding
an aside. A reader skimming for definitions had no way to find them, and a
reader who wanted the conclusion of a section had to read the section.

So callouts are typed. The source syntax is GitHub's alert form, which stays
readable as plain markdown if this preprocessor never runs:

    > [!definition] Arithmetic intensity
    > Floating-point operations performed per byte moved from memory.

Kinds:

    definition  a term of art, on first use
    takeaway    the conclusion of a section, stated once and plainly
    warning     a trap — something that fails quietly or bites later
    aside       context worth having and safe to skip

Everything else about blockquotes is left alone, including the italic epigraph
under each chapter title.
"""

from __future__ import annotations

import html
import json
import re
import sys

KINDS = {
    "definition": "Definition",
    "takeaway": "Takeaway",
    "warning": "Watch out",
    "aside": "Aside",
}

OPEN = re.compile(r"^>\s*\[!(?P<kind>[a-z]+)\]\s*(?P<title>.*?)\s*$", re.I)


def render(kind: str, title: str, body: list[str]) -> str:
    """Open tag, blank line, markdown body, blank line, close tag.

    The blank lines matter. CommonMark ends an HTML block at the first one, so
    the body between them is parsed as ordinary markdown — which is what keeps
    backticks and emphasis working inside a callout. Emitting the body inside
    the HTML block instead would render `` `.item()` `` as literal backticks.
    """
    label = KINDS[kind]
    heading = f"{label} — {title}" if title else label
    text = "\n".join(line.strip() for line in body).strip()
    return (
        f'<div class="bk-note bk-note-{kind}">\n'
        f'<p class="bk-note-label">{html.escape(heading)}</p>\n\n'
        f"{text}\n\n"
        f"</div>"
    )


def rewrite(md: str) -> str:
    out: list[str] = []
    lines = md.split("\n")
    i = 0
    in_fence = False
    fence = ""

    while i < len(lines):
        raw = lines[i]
        stripped = raw.lstrip()

        if in_fence:
            if stripped.startswith(fence):
                in_fence = False
            out.append(raw)
            i += 1
            continue
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence, fence = True, stripped[:3]
            out.append(raw)
            i += 1
            continue

        m = OPEN.match(raw)
        if m and m.group("kind").lower() in KINDS:
            body: list[str] = []
            i += 1
            while i < len(lines) and lines[i].startswith(">"):
                body.append(lines[i][1:])
                i += 1
            out.append(render(m.group("kind").lower(), m.group("title"), body))
            continue

        out.append(raw)
        i += 1

    return "\n".join(out)


def walk(items) -> None:
    for item in items:
        if not isinstance(item, dict):
            continue
        chapter = item.get("Chapter")
        if not chapter:
            continue
        chapter["content"] = rewrite(chapter["content"])
        walk(chapter.get("sub_items", []))


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "supports":
        return 0
    context, book = json.load(sys.stdin)
    walk(book["sections"])
    json.dump(book, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""mdBook preprocessor: turn `path/to/file.py:123` into a link to GitHub.

Every code reference in the book is pinned to one upstream commit. This rewrites
those references into links at that commit, so a reader can jump from the page to
the exact line without leaving their browser.

Handles both forms used in the text:

    `python/sglang/srt/managers/scheduler.py:1714`   full anchor
    `:1749`                                          continuation, inherits the last file

Code blocks are left alone. Anchors already inside a link are left alone.

mdBook calls this twice: once as `supports <renderer>` (answer by exit code), and
once with the book as JSON on stdin.
"""

from __future__ import annotations

import json
import re
import sys

PIN = "7562e741e26a4818ee78f1e63140240ec59147c0"
REPO = "https://github.com/sgl-project/sglang/blob"

SOURCE_SUFFIXES = (
    "py", "rs", "cu", "cuh", "cc", "cpp", "hpp", "h", "md", "mdx",
    "toml", "json", "yml", "yaml", "txt", "sh",
)
_SUF = "|".join(SOURCE_SUFFIXES)

FULL = re.compile(r"`(?P<path>[\w][\w./-]*\.(?:" + _SUF + r")):(?P<line>\d+)`")
CONT = re.compile(r"`:(?P<line>\d+)`")


def link(path: str, line: int, label: str) -> str:
    return f"[`{label}`]({REPO}/{PIN}/{path}#L{line})"


def rewrite(text: str) -> str:
    """Rewrite anchors outside fenced code blocks, tracking the current file."""
    out = []
    current = {"path": None}
    in_fence = False
    fence = ""

    for raw in text.split("\n"):
        stripped = raw.lstrip()
        if in_fence:
            if stripped.startswith(fence):
                in_fence = False
            out.append(raw)
            continue
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = True
            fence = stripped[:3]
            out.append(raw)
            continue

        line = raw

        def full_sub(m: re.Match) -> str:
            current["path"] = m.group("path")
            return link(m.group("path"), int(m.group("line")), f"{m.group('path')}:{m.group('line')}")

        def cont_sub(m: re.Match) -> str:
            if current["path"] is None:
                return m.group(0)
            return link(current["path"], int(m.group("line")), f":{m.group('line')}")

        # Full anchors first so they can set the context for continuations that
        # follow them on the same line.
        line = FULL.sub(full_sub, line)
        line = CONT.sub(cont_sub, line)
        out.append(line)

    return "\n".join(out)


def walk(items) -> None:
    # A BookItem is {"Chapter": {...}}, {"PartTitle": "..."}, or the bare string
    # "Separator" — only the first carries content.
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
        return 0  # every renderer
    context, book = json.load(sys.stdin)
    walk(book["sections"])
    json.dump(book, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())

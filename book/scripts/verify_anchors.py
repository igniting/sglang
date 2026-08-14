#!/usr/bin/env python3
"""Verify every `path:line` code anchor in the book against the pinned commit.

The book pins all code references to one upstream commit. This checker reads each
referenced file *out of that commit* (not the working tree, which drifts) and confirms:

  1. the file exists at the pinned commit,
  2. the line number is within the file,
  3. when the anchor is followed by a symbol in backticks, that symbol actually
     appears on that line (or within a small tolerance window).

Anchor forms recognized:

    `python/sglang/srt/managers/scheduler.py:1714` `event_loop_normal`
    `:1749` `event_loop_overlap`      <- continuation, inherits the last file seen
    `:1365`-`:1478`                   <- ranges are checked as two anchors

Usage:
    python3 book/scripts/verify_anchors.py [--pin SHA] [--src DIR]

Exits non-zero if any anchor fails.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_PIN = "7562e741e26a4818ee78f1e63140240ec59147c0"

# Extensions we treat as source files worth anchoring into.
SOURCE_SUFFIXES = (
    "py", "rs", "cu", "cuh", "cc", "cpp", "hpp", "h", "md", "mdx",
    "toml", "json", "yml", "yaml", "txt", "sh",
)

# `some/path/file.py:123`
FULL_ANCHOR = re.compile(
    r"`(?P<path>[\w][\w./-]*\.(?:" + "|".join(SOURCE_SUFFIXES) + r")):(?P<line>\d+)`"
)
# `some/path/file.py` with no line. Establishes context for later `:123`
# continuations, and is itself checked for existence at the pinned commit.
BARE_PATH = re.compile(
    r"`(?P<path>[\w][\w./-]*\.(?:" + "|".join(SOURCE_SUFFIXES) + r"))`"
)
# `:123` continuation
CONT_ANCHOR = re.compile(r"`:(?P<line>\d+)`")
# A symbol in backticks immediately following an anchor.
TRAILING_SYMBOL = re.compile(r"\s*`(?P<symbol>[A-Za-z_][A-Za-z0-9_.]*)`")

# How many lines either side of the anchor the symbol may appear on. Decorators and
# multi-line signatures mean an exact match is too strict.
SYMBOL_TOLERANCE = 3


class FileCache:
    """Reads files out of the pinned commit, memoized."""

    def __init__(self, pin: str, repo_root: Path):
        self.pin = pin
        self.repo_root = repo_root
        self._cache: dict[str, list[str] | None] = {}

    def lines(self, path: str) -> list[str] | None:
        if path not in self._cache:
            try:
                blob = subprocess.run(
                    ["git", "show", f"{self.pin}:{path}"],
                    cwd=self.repo_root,
                    capture_output=True,
                    check=True,
                )
                text = blob.stdout.decode("utf-8", errors="replace")
                self._cache[path] = text.splitlines()
            except subprocess.CalledProcessError:
                self._cache[path] = None
        return self._cache[path]


def extract_anchors(text: str):
    """Yield (path, line_or_None, symbol_or_None, char_offset) for each anchor.

    A line of None means "the path was named without a line" — existence is still
    checked, and the path becomes the context for later `:123` continuations.
    """
    events = []
    for m in FULL_ANCHOR.finditer(text):
        events.append((m.start(), "full", m))
    for m in CONT_ANCHOR.finditer(text):
        events.append((m.start(), "cont", m))
    # A bare path overlapping a full anchor is the same match minus the line, so
    # keep only bare paths that do not start where a full anchor already does.
    full_starts = {m.start() for _, kind, m in events if kind == "full"}
    for m in BARE_PATH.finditer(text):
        if m.start() not in full_starts:
            events.append((m.start(), "bare", m))
    events.sort(key=lambda e: e[0])

    current_path = None
    for _, kind, m in events:
        if kind in ("full", "bare"):
            current_path = m.group("path")
        if kind == "bare":
            yield current_path, None, None, m.start()
            continue
        if current_path is None:
            continue  # a `:123` before any file was named; nothing to resolve against
        sym = TRAILING_SYMBOL.match(text, m.end())
        symbol = sym.group("symbol") if sym else None
        yield current_path, int(m.group("line")), symbol, m.start()


def line_number_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pin", default=DEFAULT_PIN, help="commit the book pins to")
    parser.add_argument("--src", default=None, help="book source directory")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    src = Path(args.src) if args.src else repo_root / "book" / "src"

    # Fail loudly if the pinned commit is not present, rather than reporting every
    # anchor as broken.
    if subprocess.run(
        ["git", "cat-file", "-e", f"{args.pin}^{{commit}}"],
        cwd=repo_root, capture_output=True,
    ).returncode != 0:
        print(f"error: pinned commit {args.pin} not found in this repository.")
        print("       fetch it, or pass --pin with the commit the book targets.")
        return 2

    cache = FileCache(args.pin, repo_root)
    checked = 0
    failures: list[str] = []

    for doc in sorted(src.rglob("*.md")):
        text = doc.read_text()
        rel = doc.relative_to(repo_root)
        for path, line, symbol, offset in extract_anchors(text):
            # The book's own files are not part of the pinned upstream tree.
            if path.startswith("book/"):
                continue
            checked += 1
            where = f"{rel}:{line_number_of(text, offset)}"
            lines = cache.lines(path)

            if lines is None:
                failures.append(f"{where}: no such file at pinned commit: {path}")
                continue
            if line is None:
                continue  # bare path: existence was the whole check
            if not (1 <= line <= len(lines)):
                failures.append(
                    f"{where}: {path}:{line} out of range (file has {len(lines)} lines)"
                )
                continue
            if symbol is None:
                continue

            lo = max(0, line - 1 - SYMBOL_TOLERANCE)
            hi = min(len(lines), line + SYMBOL_TOLERANCE)
            if not any(symbol in l for l in lines[lo:hi]):
                failures.append(
                    f"{where}: {path}:{line} does not mention `{symbol}` "
                    f"(searched +/-{SYMBOL_TOLERANCE} lines); found: "
                    f"{lines[line - 1].strip()[:70]!r}"
                )

    print(f"checked {checked} anchors across {len(list(src.rglob('*.md')))} documents")
    if failures:
        print(f"\n{len(failures)} broken anchor(s):\n")
        for f in failures:
            print(f"  {f}")
        return 1
    print("all anchors verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())

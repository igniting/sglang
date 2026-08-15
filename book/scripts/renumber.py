#!/usr/bin/env python3
"""One-shot: shift chapters N..22 up by one to open a slot at position `at`.

Rewrites file names, H1 headings, SUMMARY entries, every "Chapter N" reference,
the bare-number chapter columns in Chapter 1's two tables, and the figure
registry. Run once, check `git diff`, then delete — this is not a maintained
tool, it is a record of a structural edit.

    python3 book/scripts/renumber.py 2
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
BOOK = SRC.parent
LAST = 22


def shift(n: int, at: int) -> int:
    return n + 1 if n >= at else n


def rewrite_refs(text: str, at: int) -> str:
    """Bump every "Chapter N" / "Chapters N, M, and K" reference."""

    def repl(m: re.Match) -> str:
        nums = m.group("nums")
        return m.group("word") + re.sub(
            r"\d+", lambda d: str(shift(int(d.group()), at)), nums
        )

    return re.sub(
        r"(?P<word>Chapters?\s+)"
        r"(?P<nums>\d+(?:\s*,\s*\d+)*(?:\s*,?\s*and\s+\d+)?)",
        repl,
        text,
    )


def rewrite_table_column(text: str, at: int) -> str:
    """Bare chapter numbers in a table's last column, e.g. `| 15, 16 |`."""
    def repl(m: re.Match) -> str:
        return "| " + re.sub(
            r"\d+", lambda d: str(shift(int(d.group()), at)), m.group(1)
        ) + " |"

    # A row whose final cell is only digits, commas and spaces.
    return re.sub(r"\|\s*(\d[\d,\s]*?)\s*\|\s*$", repl, text, flags=re.M)


def main(at: int) -> int:
    # 1. Rename descending so a target name is never occupied.
    renames: list[tuple[pathlib.Path, pathlib.Path]] = []
    for n in range(LAST, at - 1, -1):
        old = next(SRC.glob(f"ch{n:02d}-*.md"), None)
        if old is None:
            continue
        new = old.with_name(re.sub(r"^ch\d+", f"ch{n + 1:02d}", old.name))
        subprocess.run(["git", "mv", str(old), str(new)], check=True, cwd=BOOK.parent)
        renames.append((old, new))
        print(f"  {old.name} -> {new.name}")

    # 2. Rewrite prose everywhere chapters are referenced.
    targets = sorted(SRC.glob("*.md")) + [BOOK / "PLAN.md"]
    for p in targets:
        t = orig = p.read_text()
        t = rewrite_refs(t, at)

        # H1 headings: "# 7. Sampling and the Return Path"
        t = re.sub(
            r"^# (\d+)\. ",
            lambda m: f"# {shift(int(m.group(1)), at)}. ",
            t,
            flags=re.M,
        )

        # SUMMARY entries and any other "[N. Title](./chNN-...)" link.
        t = re.sub(
            r"\[(\d+)\. ([^\]]+)\]\(\./ch(\d+)(-[^)]+)\)",
            lambda m: f"[{shift(int(m.group(1)), at)}. {m.group(2)}]"
                      f"(./ch{shift(int(m.group(3)), at):02d}{m.group(4)})",
            t,
        )

        # PLAN.md's "Ch. 13" shorthand.
        t = re.sub(
            r"\bCh\. (\d+)",
            lambda m: f"Ch. {shift(int(m.group(1)), at)}",
            t,
        )

        if p.name == "ch01-why-serving-engines.md":
            t = rewrite_table_column(t, at)

        if t != orig:
            p.write_text(t)
            print(f"  rewrote {p.name}")

    # 3. Figure registry keys are file slugs.
    dg = BOOK / "scripts" / "diagrams.py"
    t = dg.read_text()
    t = re.sub(
        r'"ch(\d+)(-[a-z0-9-]+)"',
        lambda m: f'"ch{shift(int(m.group(1)), at):02d}{m.group(2)}"',
        t,
    )
    t = re.sub(
        r"^# Chapter (\d+) ",
        lambda m: f"# Chapter {shift(int(m.group(1)), at)} ",
        t,
        flags=re.M,
    )
    t = re.sub(
        r'"ch(\d+) ([a-z]+)"',
        lambda m: f'"ch{shift(int(m.group(1)), at):02d} {m.group(2)}"',
        t,
    )
    dg.write_text(t)
    print("  rewrote diagrams.py")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 2))

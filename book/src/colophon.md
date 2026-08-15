# Colophon

## The text

Body text is set in a transitional serif — Iowan Old Style where available, falling back
through Palatino, Charter, and Source Serif. Headings are Inter. Code is JetBrains Mono,
falling back to SF Mono and Cascadia Code.

All three families are system-resident or commonly installed. Nothing is fetched over the
network, so the pages render identically offline, on a first visit, and behind a firewall.

The measure is held near 72 characters. Longer lines are faster to typeset and harder to
read; at this width the eye finds the start of the next line without hunting.

Light mode is a warm off-white rather than pure white, which is easier over a long session
and lets code blocks sit slightly darker without needing a border. Dark mode is a
desaturated blue-black rather than neutral grey, which keeps syntax colour from going muddy.

Inline code carries no background box. At this density the boxes stripe the page and fight
the prose; colour alone is enough to mark an identifier.

## The code references

Every reference of the form `path/to/file.py:123` is a link to that exact line of SGLang on
GitHub, pinned to one commit:

```
7562e741e26a4818ee78f1e63140240ec59147c0   (15 August 2026)
```

Pinning is what lets a book about a fast-moving project stay true. The tradeoff is that the
newest features do not appear here.

The links are generated at build time rather than written by hand. A checker runs before
every build and verifies each reference against that commit — that the file exists, that the
line is within it, and that the symbol named beside the reference is actually on that line.
If a reference goes stale, the build fails rather than publishing a book that quietly points
at the wrong code.

## The build

Written in Markdown, rendered with [mdBook](https://github.com/rust-lang/mdBook), published
from a GitHub Actions workflow. Diagrams are hand-written inline SVG that draws with the
page's own colour tokens, so they follow the light and dark themes rather than shipping two
copies.

Sources live under `book/` in the repository, and every page has an edit link in the top
bar.

## Acknowledgment

This book describes work done by the SGLang community — several hundred contributors,
hosted by the non-profit LMSYS organization. The engine, the design decisions, and the
comments quoted throughout are theirs. Any errors in describing them are the book's.

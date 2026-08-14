# Appendix A — Server Arguments

`python/sglang/srt/server_args.py` is about 9,900 lines and is not meant to be read
linearly. It *is* meant to be read by section: `python/sglang/srt/server_args.py:445`
`ServerArgs` documents its own organization.

```python
class ServerArgs:
    """Server-wide configuration for SGLang.

    Adding new arguments
    --------------------
    1. **Place the field in the right section.** Arguments are grouped by
       comment blocks (``# Model and tokenizer``, ``# LoRA``, etc.).
       Add new fields to the matching section, or create a new section
       with a ``# ---`` banner when none fits.

    2. **Use the ``A[T, ...]`` annotation.**  ``A`` is an alias for
       ``typing.Annotated``.  The primary CLI flag is auto-derived from the
       field name (``tp_size`` → ``--tp-size``).
```

Two consequences for readers. The **comment blocks are the index** — find the block and you
have the subsystem. And flags are **derived from field names**, so `--tp-size` is
`tp_size`: to find what a flag does, search for the field, not the string.

## Sections, and where the book explains them

Line numbers are the section banners in `python/sglang/srt/server_args.py`.

| Line | Section | Chapter |
| --- | --- | --- |
| `:487` | Model and tokenizer | 11 |
| `:635` | Quantization and data type | 14 |
| `:769` | Memory and scheduling | 5, 8 |
| `:980` | Distributed topology and parallelism (TP, PP, DP, CP) | 15 |
| `:1139` | DP attention | 15 |
| `:1185` | Device info and server timeout | 2 |
| `:1251` | HTTP server | 2, 3 |
| `:1308` | SSL/TLS | 2 |
| `:1327` | API related | 3 |
| `:1433` | Streaming | 7 |
| `:1465` | Logging, metrics, and tracing | 21 |
| `:1652` | Constrained decoding | 19 |
| `:1666` | Kernel backend | 13 |
| `:1823` | Cuda graphs | 14 |
| `:1905` | Communication and kernels | 15 |
| `:2028` | Torch compile | 14 |
| `:2042` | Speculative decoding | 18 |
| `:2268` | Speculative decoding (ngram) | 18 |
| `:2306` | Expert parallelism | 16 |
| `:2499` | Mamba cache and linear attn | 8, 13 |
| `:2635` | Hierarchical cache | 10 |
| `:2714` | Hierarchical sparse attention | 13 |
| `:2729` | Multi-modal optimization configs | 20 |
| `:2833` | LoRA | 20 |
| `:2930` | Two batch overlap | 16 |
| `:2947` | Offloading | 10 |

Platform-specific blocks sit near the top: `:180` common, `:188` NVIDIA, `:200` AMD, `:203`
other platforms.

## The arguments that matter most

Ranked by how much they change behavior, with the chapter that explains the mechanism.

**`--mem-fraction-static`** (Ch. 8) — the fraction of GPU memory reserved for weights plus
KV pool. Sets `max_total_num_tokens`, which sets concurrency, which sets everything. Too
high fails under load rather than at startup.

**`--tp-size`, `--pp-size`, `--dp-size`, `--ep-size`** (Ch. 15, 16) — the parallelism
layout. TP within a node, PP across nodes, DP for replicas or (with
`--enable-dp-attention`) for attention.

**`--chunked-prefill-size`** (Ch. 5) — how finely long prompts are split. Trades the long
request's TTFT against everyone else's ITL.

**`--max-running-requests`** (Ch. 5) — concurrency cap independent of memory.

**`--attention-backend`**, `--prefill-attention-backend`, `--decode-attention-backend`
(Ch. 13) — the phases can differ.

**`--cuda-graph-max-bs`**, `--disable-cuda-graph` (Ch. 14) — graph coverage against capture
memory.

**`--quantization`**, `--kv-cache-dtype` (Ch. 14) — weight and cache precision, independent
choices.

**`--speculative-algorithm`**, `--speculative-num-steps`, `--speculative-eagle-topk`,
`--speculative-num-draft-tokens` (Ch. 18) — draft depth and width.

**`--disable-radix-cache`** (Ch. 9) — turns off prefix caching. Almost always wrong in
production; occasionally necessary for measurement or when using `input_embeds`.

**`--schedule-policy`** (Ch. 5) — `lpm`, `fcfs`, `dfs-weight`, `random`, `priority`.

**`--enable-hierarchical-cache`** and the `--hicache-*` family (Ch. 10).

**`--load-format`** (Ch. 11) — including `dummy` for benchmarking without real weights.

## Reading the constraints

The most valuable part of this file is not the individual arguments but the validation.
`check_server_args` and the `__post_init__` logic encode which *combinations* work — and
that information is written down nowhere else.

When a flag silently does nothing, the usual explanation is that another flag disabled it
there. Reading the validation is faster than bisecting.

`python/sglang/srt/server_args_config_parser.py` handles config-file input, and
`python/sglang/srt/arg_groups/` holds grouped argument definitions.

`docs/docs/advanced_features/server_arguments.mdx` is the project's own reference and is
generated closer to the source than this appendix; prefer it for exact semantics, and use
this table to find the chapter that explains *why* an argument exists.

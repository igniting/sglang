# 19. Shaping and Reading the Output

> *Constraining generation and parsing generation are the same problem seen from two sides,
> and both are made hard by streaming.*

---

## Two halves of one problem

An application that calls a tool needs structured output — JSON matching a schema, a
function name with typed arguments. There are two ways to get it.

**Constrain**: at each step, mask out every token that would violate the structure, so the
model cannot produce invalid output. **Parse**: let the model generate freely and interpret
what comes out.

They are usually treated as separate features. They are the same problem: both require a
model of the valid output space, and both are hard for the same reason — **you must decide
incrementally, before the output is complete.** A constrainer must mask token *n* without
seeing token *n+1*; a streaming parser must decide whether text is a tool call before the
call is finished.

This chapter covers both, and ends where they converge.

---

## Guarantees by masking

The mechanism is simple to state. At each decode step, before sampling, set the logits of
every structurally-invalid token to −∞. The model cannot select them, so output conforms by
construction.

`python/sglang/srt/constrained/base_grammar_backend.py:52` `BaseGrammarObject` is the
per-request grammar state:

```python
class BaseGrammarObject:

    def __init__(self):
        self._finished = False
        self.grammar_stats = None
        self.current_token = None

    def accept_token(self, token: int) -> None:
        """
        Accept a token in the grammar.
        """
        raise NotImplementedError()

    def rollback(self, k: int):
        raise NotImplementedError()

    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        raise NotImplementedError()

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        raise NotImplementedError()
```

Four methods, and two of them are more interesting than they look.

`accept_token` advances the state machine — this is why grammar is *stateful per request*
and must be tracked in Chapter 4's `Req`.

`rollback(k)` undoes *k* tokens. This exists for Chapter 18: speculative decoding drafts
tokens that may be rejected, and the grammar must be wound back when they are. Without
rollback, speculation and constrained decoding would be mutually exclusive.

`allocate_vocab_mask` / `fill_vocab_mask` produce the mask itself — one bit per vocabulary
entry, for a 128k-token vocabulary, per request, per step. `:83`
`fill_vocab_mask_batched` fills many rows at once:

```python
    @staticmethod
    def fill_vocab_mask_batched(
        entries: List[GrammarRow], vocab_mask: torch.Tensor
    ) -> None:
        """Fill listed rows, leaving unlisted rows untouched."""
        for entry in entries:
            entry.grammar.fill_vocab_mask(vocab_mask, entry.row)
```

"Leaving unlisted rows untouched" is the batching accommodation: not every request in a
batch is constrained, so the mask buffer is shared and only constrained rows are written.
The buffers are registered at `:262` `register_vocab_mask_buffer` and retrieved at `:294`
`get_vocab_mask_buffer`, then applied in Chapter 7's
`python/sglang/srt/layers/sampler.py:97` `forward`.

`python/sglang/srt/constrained/base_grammar_backend.py:167` `BaseGrammarBackend` compiles specifications
into grammar objects, and `python/sglang/srt/constrained/base_grammar_backend.py:311`
`create_grammar_backend` selects an implementation.
`python/sglang/srt/constrained/base_grammar_backend.py:156` `InvalidGrammarObject` is the
error case — an invalid schema produces an object that fails cleanly rather than an
exception deep in the sampler.

---

## Compiling a grammar to token masks

The hard part is a representation mismatch.

A grammar is defined over **characters**: JSON says a string is a quote, then characters,
then a quote. But the model emits **tokens**, and a token may be several characters, may
span a grammar boundary, and the same text may be tokenizable several ways.

So compiling a grammar means computing, for every state of the character-level automaton,
which of the 128,000 tokens could legally come next. That is a large precomputation, and
doing it naively per step would cost more than the forward pass.

`python/sglang/srt/constrained/xgrammar_backend.py` is the default. XGrammar precomputes
token-level transitions and compresses the mask representation, which is what makes
per-step masking affordable.

`python/sglang/srt/constrained/outlines_backend.py` is the original approach — regex to
finite automaton, with the FSM compiled against the tokenizer.
`python/sglang/srt/constrained/llguidance_backend.py` is a third.

All three sit behind `BaseGrammarBackend`, so a request does not know which is compiled.
Compilation is expensive enough that backends cache aggressively; the first request with a
new schema pays, the rest do not.

---

## Skipping the forward pass entirely

Sometimes the grammar admits exactly one continuation. After `{"na`, a schema with a single
property `name` forces `me": `. There is nothing to sample.

`python/sglang/srt/constrained/outlines_jump_forward.py` detects these runs and emits them
directly — no forward pass, no sampling. For schema-heavy output, where structural
characters (braces, quotes, key names) can be most of the tokens, this is a large win:
those tokens become free.

The complication is that jumping forward changes the token sequence in ways the KV cache
must follow. Tokens emitted without a forward pass still need KV entries, so a subsequent
extend has to fill them in — which is why this interacts with Chapters 8 and 9 rather than
being a pure output-side trick.

---

## Coordination costs

`python/sglang/srt/constrained/grammar_manager.py` owns the grammar lifecycle, wired in at
`python/sglang/srt/managers/scheduler.py:1962` `init_grammar_manager`.

The interesting method is `python/sglang/srt/managers/scheduler.py:1861`
`_advance_pending_grammar`, called from Chapter 4's loop. Grammar compilation is slow enough
that it must not block the scheduler, so it happens asynchronously and the loop advances
pending grammars each iteration. A request whose grammar is still compiling is not yet
schedulable.

`python/sglang/srt/managers/tokenizer_manager.py:2847` `_request_has_grammar` is the
front-end check.

The collision with Chapter 18 is the sharpest constraint here. Speculative decoding drafts
several tokens ahead; each must be checked against the grammar; accepted tokens advance the
state and rejected ones must be rolled back. That is why
`python/sglang/srt/speculative/spec_info.py:144` `supports_grammar_overlap` is a capability
predicate — not every algorithm can do it — and why Chapter 18's `verify` takes a
`grammar_barrier`.

`python/sglang/srt/constrained/reasoner_grammar_backend.py` handles the other interaction:
reasoning models emit a thinking block before their answer, and constraining the *thinking*
to a JSON schema would be actively harmful. The reasoner backend keeps the grammar inactive
until the reasoning block closes.

---

## Parsing, the mirror image

Now the other half. Not every model supports constrained decoding for tool calls, and many
were trained to emit tool calls in a specific text format. Those must be parsed.

`python/sglang/srt/function_call/base_format_detector.py:32` `BaseFormatDetector` defines
the interface, and its three methods map onto three situations:

```
:104  detect_and_parse            complete text, one shot
:125  parse_streaming_increment   incremental, called per chunk
:347  has_tool_call               cheap check
```

`parse_streaming_increment` is where the difficulty lives. Consider a model emitting
`<tool_call>{"name": "get_weather"...`. When the first chunk arrives, the parser must
decide: is this a tool call, or is the model about to write about tool calls? It cannot
know yet. So it buffers, and must decide *how much* to buffer — too little and it emits
text that turns out to be a tool call; too much and streaming latency suffers.

That is the same incremental-decision problem as constrained decoding, from the other side.

`python/sglang/srt/function_call/function_call_parser.py` dispatches to a detector by model.
There are 39 of them —
`python/sglang/srt/function_call/qwen3_coder_detector.py`,
`python/sglang/srt/function_call/deepseekv3_detector.py`,
`python/sglang/srt/function_call/kimik2_detector.py`,
`python/sglang/srt/function_call/gpt_oss_detector.py`,
`python/sglang/srt/function_call/glm4_moe_detector.py`, and so on — because every model
family invented its own format. Some use XML-ish tags, some use special tokens, some emit
bare JSON. Read two in full and the rest are variations.

`python/sglang/srt/function_call/core_types.py` holds the shared types and
`python/sglang/srt/function_call/json_array_parser.py` the incremental JSON parsing that
most detectors need.

---

## Where the two halves converge

**Structural tags** are the synthesis. Rather than generating freely and parsing, or
constraining to a generic JSON schema, constrain the model to *exactly* the tool schema it
should be emitting — with the model's own format as part of the grammar.

`python/sglang/srt/function_call/kimik3_structural_tag.py` and
`python/sglang/srt/function_call/kimik3_format.py` implement it for one model family. The
grammar encodes both the tool-call syntax and the argument schema, so the output is valid by
construction and the "parser" becomes a trivial reader of known-good structure.

This is the direction the field is moving, and it is why the two halves belong in one
chapter: constraining and parsing are converging on a single specification of the valid
output space.

---

## Reasoning blocks

Modern models emit a reasoning trace before their answer. That creates problems for both
halves.

`python/sglang/srt/parser/reasoning_parser.py` separates reasoning from answer, and
`python/sglang/srt/parser/harmony_parser.py` handles OpenAI's Harmony format (with
`python/sglang/srt/entrypoints/harmony_utils.py` on the entrypoint side).

The interactions are specific. A tool-call detector must not parse tool calls *out of* the
reasoning block, where the model may be discussing a call it has not decided to make. A
grammar must not constrain the reasoning block. And a client that requested reasoning
separately expects the two streams split, which means the separation must happen
incrementally as tokens arrive.

`docs/docs/advanced_features/separate_reasoning.mdx`,
`docs/docs/advanced_features/structured_outputs.mdx`, and
`docs/docs/advanced_features/structured_outputs_for_reasoning_models.mdx` cover the
combinations, and
`docs/docs/advanced_features/tool_parser.mdx` covers detector selection.

---

## What it costs

Constrained decoding is not free:

- **Compilation** — first request with a new schema pays; cached afterward.
- **Per-step masking** — a mask write and a logits add per constrained request per step.
- **Batch coupling** — the mask buffer is allocated for the batch even if one request needs
  it.
- **Speculation friction** — rollback and barriers, and some algorithms cannot overlap at
  all.

Against that, jump-forward decoding can *save* more than masking costs on schema-heavy
output, since structural tokens are emitted without a forward pass. For JSON with short
values, constrained generation can be net faster than unconstrained — the one place in this
book where adding a correctness guarantee also makes things quicker.

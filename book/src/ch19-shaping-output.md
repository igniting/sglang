# 19. Shaping and Reading the Output

> *Constraining generation and parsing generation are the same problem seen from two sides,
> and both are made hard by streaming.*

An application calling a tool needs structured output: JSON matching a schema, a function
name with typed arguments. There are two ways to get it, and they are usually treated as
separate features.

**Constrain** the model: at each step, mask out every token that would break the structure,
so invalid output is impossible. **Parse** the model: let it generate freely and interpret
what comes out.

This chapter covers both, because they are the same problem seen from opposite sides. Both
need a model of the valid output space, and both are hard for the same reason — **the
decision must be made incrementally, before the output is complete.** A constrainer must
mask token *n* without seeing token *n+1*. A streaming parser must decide whether text is a
tool call before the call has finished arriving.

They are also converging, and the chapter ends at the point where they meet.

Along the way: why compiling a grammar is harder than it sounds (grammars are defined over
characters, models emit tokens, and those do not line up), and the trick where the engine
skips the forward pass entirely for tokens the grammar has already determined.

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

### Two automata, and why JSON needs the bigger one

The right machine depends on the language, and the two backends here are built on different
ones.

A **regular** language — anything a regex describes — is recognized by a finite automaton: a
fixed set of states and a transition per input symbol, no memory beyond the current state.
Outlines' insight (Willard and Louf, 2023) was that you can compile a regex to an FSM *once*,
then precompute for every state which tokens are legal, giving `O(1)` mask lookup at
generation time. That works, and for regular constraints it is complete.

JSON is not regular. Matching `{` against `}` to arbitrary nesting depth requires counting,
and a finite automaton cannot count. The right machine is a **pushdown automaton** — a finite
automaton plus a stack — which recognizes context-free languages. The stack is the open
brackets; a rule that finishes pops back to whatever contained it.

That is the machine XGrammar (Dong et al., 2024) builds, at the *byte* level rather than the
character level, because tokens are byte sequences and the boundaries do not align with
characters.

### Making the mask affordable

A pushdown automaton is more expressive but harder to precompute against, and this is the
problem XGrammar solves. Per token, per step, per request, you need a bitmask over 128,000
vocabulary entries. Naively checking each token against the current stack is 128,000 automaton
simulations per step, which costs more than the forward pass.

The key observation is a partition of the vocabulary. For most tokens, legality depends only
on the automaton's **current node** — where you are inside the rule you are matching. Those
are **context-independent**, and their masks can be precomputed per node and cached. A
minority of tokens are legal or not depending on what is *below* on the stack, because they
would complete the current rule and return control to a parent. Those are
**context-dependent** and must be checked at runtime. In practice, under 1% of the
vocabulary.

So the cost collapses: look up a cached bitmask for 99% of the vocabulary, and simulate the
automaton for the remaining fraction of a percent. The cache is stored adaptively — rejected
tokens when most are accepted, accepted tokens when most are rejected, a bitset when it is
balanced — which for Llama-3.1 with a JSON grammar takes the mask cache from 160 MB to
**0.46 MB**.

Two more structural tricks make the runtime side cheap. The **persistent execution stack**
stores all live parse stacks as one tree, so branching a state is a pointer rather than a
copy and rolling back is `O(1)` — which matters because Chapter 18's speculation and this
chapter's jump-forward both need to advance a grammar and then undo it. And the vocabulary is
sorted lexicographically so that checking tokens in order reuses the previous token's prefix
work.

The last piece is scheduling, and it is Chapter 4's argument reappearing. Mask computation is
CPU work over automaton state; the forward pass is GPU work. Neither depends on the other's
result within a step — the mask for step *t* depends only on tokens through *t−1*. So the
mask is computed on the CPU *while the GPU runs the forward*, and the two meet just before
sampling. Done that way, the mask is free in the same sense Chapter 4's scheduling is free.
The reported end result is up to 100× faster per-token mask generation than prior
implementations, and up to 80× higher output token rate end to end.

`python/sglang/srt/constrained/xgrammar_backend.py` is the default and implements the above.
`python/sglang/srt/constrained/outlines_backend.py` is the FSM approach —
regex to finite automaton, compiled against the tokenizer.
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

Chapter 20 takes the last of the assumptions this part exists to break — that every request
in a batch wants the same weights, and the same kind of input.

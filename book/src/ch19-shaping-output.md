# 19. Shaping and Reading the Output

> *Constraining generation and parsing generation are the same problem seen from two sides, and both are made hard by streaming.*

> **Status: outline.** This chapter is planned but not yet written. The beats below are the writing plan — each one pairs an idea with the code that implements it.

## Beats

1. **Guarantees by masking.** `python/sglang/srt/constrained/base_grammar_backend.py:52`
   `BaseGrammarObject`, `:167` `BaseGrammarBackend`, `:311` `create_grammar_backend`; the
   mask lands in `python/sglang/srt/layers/sampler.py:97` via the vocab-mask buffers at
   `python/sglang/srt/constrained/base_grammar_backend.py:262`
   `register_vocab_mask_buffer`.

2. **Compiling a grammar to token masks.** The FSM, the compressed FSM, and the real
   difficulty — a grammar is defined over characters, a mask over tokens.
   `python/sglang/srt/constrained/xgrammar_backend.py` as the default,
   `outlines_backend.py` and `llguidance_backend.py` for contrast.

3. **Skipping the forward pass entirely.**
   `python/sglang/srt/constrained/outlines_jump_forward.py` — when the grammar admits
   exactly one continuation, emit it without inference.

4. **Coordination costs.** `python/sglang/srt/constrained/grammar_manager.py`,
   `python/sglang/srt/managers/scheduler.py:1962` `init_grammar_manager`, `:1861`
   `_advance_pending_grammar` — grammar state advances asynchronously, which is where it
   collides with Chapter 18.

5. **Parsing, the mirror image.**
   `python/sglang/srt/function_call/function_call_parser.py` and
   `base_format_detector.py`; two detectors read in full and the other 37 skimmed for the
   pattern. Streaming forces a decision about whether text is a tool call before the text
   is complete.

6. **Structural tags** as the convergence of the two halves — constraining generation to a
   tool schema instead of parsing afterwards
   (`python/sglang/srt/function_call/kimik3_structural_tag.py`).

7. **Reasoning blocks.** `python/sglang/srt/parser/reasoning_parser.py`,
   `harmony_parser.py`, and why `<think>` content must be excluded from grammar
   constraints.

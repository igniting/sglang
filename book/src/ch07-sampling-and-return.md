# 7. Sampling and the Return Path

> *Turning hidden states into user-visible text is three separate hard problems that happen
> to sit next to each other.*

The model has produced logits. That is not a token, and a token is not text, and text on a
GPU is not text in front of a reader.

This chapter closes the loop of Part II by covering the three stages that remain. They sit
next to each other in the pipeline and are often described together, but they are genuinely
separate problems, each hard in its own way.

Turning hidden states into logits is a *cost* problem: running the vocabulary projection on
every position of a 2,000-token prompt would cost more than several transformer layers, so
the engine has to know which positions matter — and the answer changes depending on features
introduced in later chapters.

Sampling is a *batching* problem: every request in the batch asked for different
temperature, different top-p, different penalties, and they all have to be applied in one
pass.

Detokenization is a *correctness* problem, and the least appreciated of the three. You
cannot simply decode the newest token and append it, and the reasons why involve both how
tokenizers work and how UTF-8 works.

Along the way we take a detour into a question that surprises people: why the same prompt
with temperature zero can produce different output on different runs, and what it costs to
make that stop.

---

Chapter 6 ended with `ModelRunnerOutput` carrying logits. Getting from there to a token
streamed to a client crosses three subsystems and two process boundaries, and each stage
has a difficulty that is not obvious from its name.

---

## Not every position matters, except when it does

The model produces a hidden state for *every* position it processed. For a decode step that
is one state per sequence, and all of them matter. For a prefill of a 2,000-token prompt it
is 2,000 states per sequence, of which — usually — exactly one matters: the last, which
predicts the next token.

Running the LM head on all 2,000 would be waste on a large scale. The LM head is a
`[hidden_size, vocab_size]` matrix; with a 128k vocabulary it is one of the largest
operations in the model. Multiplying 2,000 positions against it to discard 1,999 results
would cost more than several transformer layers.

`python/sglang/srt/layers/logits_processor.py:427` `_get_pruned_states` is the pruning step,
and `:282` `LogitsProcessor` is the module. Its `forward` (`:332`) opens by converting its
input:

```python
        # Extract MIS indices before ForwardBatch → LogitsMetadata conversion
        multi_item_delimiter_indices = None
        if isinstance(logits_metadata, ForwardBatch):
            multi_item_delimiter_indices = logits_metadata.multi_item_delimiter_indices
            logits_metadata = LogitsMetadata.from_forward_batch(logits_metadata)
```

`:149` `LogitsMetadata` (built by `:192` `from_forward_batch`) is a narrowed view of the
`ForwardBatch` — only what this stage needs. Its fields encode *which* positions matter:
which produce sampled tokens, which produce input logprobs, and how positions map back to
sequences.

Then a chain of special cases before the common path:

```python
        # Autotune dummy run discards this output. `is False` not `not`: None
        # means no autotune pass, which must not skip. Placed before the MIS /
        # DLLM / common dispatch so all three LM-head paths are skipped.
        if _autotune_run_lm_head is False:
            return LogitsProcessorOutput(next_token_logits=None)
```

The comment earns its place: `is False` rather than `not` because `None` is a third state
meaning "no autotune pass in progress," and `not None` would wrongly skip the head.

The cases where more than the last position matters are worth naming, because they are the
reason the pruning logic is not a one-liner:

- **Input logprobs** — the client asked for the likelihood of its own prompt tokens.
- **Speculative decoding** — Chapter 18 verifies *k* draft tokens, so *k* positions produce
  logits.
- **Hidden state capture** — EAGLE needs intermediate hidden states, not just logits.
  `python/sglang/srt/layers/logits_processor.py:593` `_get_hidden_states_to_store` handles it, driven by
  `python/sglang/srt/model_executor/forward_batch_info.py:200` `CaptureHiddenMode`.
- **Multi-item scoring** — `python/sglang/srt/layers/logits_processor.py:847`
  `compute_logprobs_for_multi_item_scoring`, for reranking
  and scoring endpoints where every delimiter position matters.

`python/sglang/srt/layers/logits_processor.py:646` `_get_logits` is where the head actually runs, and it is dense with parallelism
concerns:

```python
        hidden_states, local_hidden_states = self._gather_dp_attn_hidden_states(
            hidden_states, logits_metadata
        )

        logits = self._compute_lm_head(hidden_states, lm_head, embedding_bias)

        if self.logit_scale is not None:
            logits.mul_(self.logit_scale)

        if self.do_tensor_parallel_all_gather:
            if self.use_attn_tp_group:
                logits = self._gather_attn_tp_logits(logits)
            else:
                logits = self._logits_gatherer(logits)

        logits = self._scatter_dp_attn_logits(
            logits, local_hidden_states, logits_metadata
        )
```

Gather across data-parallel ranks, compute, all-gather across tensor-parallel ranks (the
vocabulary is sharded, so each rank holds a slice of the logits), then scatter back. Three
collectives around one matrix multiply. Chapter 15 explains why the vocabulary is sharded
and what `use_attn_tp_group` distinguishes.

---

## Sampling as batched tensor work

`python/sglang/srt/layers/sampler.py:70` `Sampler` faces a problem that single-request
inference does not have: **every request in the batch asked for different sampling
parameters.** One wants greedy, another temperature 0.8 with top-p 0.95, a third top-k 50
with a repetition penalty.

`python/sglang/srt/sampling/sampling_batch_info.py` is the answer — per-request parameters
packed into tensors, so operations apply to the whole batch at once with per-row values.

`:97` `forward` starts with cleanup and then takes the fast path if it can:

```python
        logits = logits_output.next_token_logits

        # Preprocess logits (custom processors and NaN handling)
        logits = self._preprocess_logits(logits, sampling_info)
        return_sampling_mask = any(sampling_info.return_sampling_masks or [])

        if sampling_info.is_all_greedy:
            if _use_aiter and not _disable_aiter_greedy_sample:
                batch_next_token_ids = torch.empty(
                    logits.shape[0], device=logits.device, dtype=torch.int32
                )
                _aiter_greedy_sample(batch_next_token_ids, logits)
            else:
                batch_next_token_ids = torch.argmax(logits, -1)
```

`is_all_greedy` is a batch-level property: if *every* request wants greedy sampling, the
whole thing is one `argmax` and the entire probability pipeline is skipped. A single
non-greedy request in the batch costs everyone the general path — a small example of how
batching couples requests that have nothing to do with each other.

The general path is `python/sglang/srt/layers/sampler.py:246` `_sample_from_probs`, with
`python/sglang/srt/layers/sampler.py:563` `top_k_top_p_min_p_sampling_from_probs_torch` as the reference implementation.
Reading it shows how the three truncation methods compose: sort, cumulative-sum, mask
below the thresholds, renormalize, sample. Penalties live in
`python/sglang/srt/sampling/penaltylib/` and are applied to logits before this point.

### What the knobs actually do

The parameter names in an OpenAI-compatible request describe four different operations on a
distribution, applied in a fixed order, and the order matters more than any single one of
them.

Start from the model's output: a vector of **logits** `z ∈ R^V`, one real number per
vocabulary entry, which softmax turns into probabilities:

```
p_i = exp(z_i) / Σ_j exp(z_j)
```

**Temperature** divides the logits before the softmax: `p_i ∝ exp(z_i / T)`. It is a
sharpening control, and the limits are the useful way to hold it. As `T → 0` the largest
logit dominates completely and sampling degenerates to `argmax` — which is why temperature 0
is implemented as greedy rather than as a division by zero. As `T → ∞` every exponent goes
to zero and the distribution becomes uniform over the whole vocabulary. `T = 1` leaves the
model's own distribution alone.

**Top-k** keeps the *k* highest-probability tokens and zeros the rest, then renormalizes. It
is the bluntest of the truncations and has a known failure mode: *k* is a fixed count applied
to a distribution whose useful width varies enormously by position. After `"The capital of
France is"` the model is nearly certain and *k* = 50 admits 49 wrong answers; mid-sentence in
open prose, 50 may be far too few.

**Top-p**, or nucleus sampling (Holtzman et al., 2020), fixes exactly that. Sort by
descending probability and keep the shortest prefix whose cumulative mass reaches *p*:

```
keep the smallest set S with  Σ_{i ∈ S} p_i ≥ p
```

The size of the kept set now adapts to the model's own confidence — one token where the model
is sure, hundreds where it is not. Nucleus sampling was introduced to fix the specific
pathology that greedy and beam search produce degenerate, repetitive text while unrestricted
sampling produces incoherent text; truncating the unreliable tail was the middle path.

**Min-p** is the newest and takes a third view: keep tokens whose probability is at least a
fraction of the *top* token's, `p_i ≥ min_p × max_j p_j`. Where top-p thresholds on
cumulative mass, min-p thresholds relative to the peak, which behaves better at high
temperature — the scaling that flattens the distribution also lowers the peak, so the
threshold moves with it.

The order of operations is where implementations disagree and where bugs live. SGLang's is:
penalties on logits → temperature → softmax → top-k → top-p → min-p → renormalize → sample.
Applying temperature *before* the truncations is the consequential choice: temperature
changes the probabilities, so it changes which tokens survive a top-p cut. The reverse order
would give a different distribution from the same request parameters.

`top_k_top_p_min_p_sampling_from_probs_torch` implements all three in the one sorted pass
they share — a sort, a cumulative sum, three mask conditions, a renormalize — because sorting
a vocabulary-sized tensor per row is the expensive part and there is no reason to do it three
times.

The penalties are a different kind of operation, applied to logits rather than probabilities
and dependent on the tokens generated so far rather than on the distribution: presence and
frequency penalties subtract from the logits of tokens already seen, and the repetition
penalty divides or multiplies them depending on sign. They live in
`python/sglang/srt/sampling/penaltylib/` precisely because they need per-request generation
history, which the rest of the sampler does not.

Two extension points sit here. `python/sglang/srt/sampling/custom_logit_processor.py` lets a
client ship a processor with its request — applied in `python/sglang/srt/layers/sampler.py:761` `apply_custom_logit_processor`
— and `python/sglang/srt/layers/sampler.py:527` `register_sampler_backend` lets a hardware backend replace the whole sampler,
which is how the Ascend path at `python/sglang/srt/layers/sampler.py:467` `_forward_ascend_backend` plugs in.

---

## Reproducibility, and why it is hard

Send the same prompt twice with temperature 0 and you may get different outputs. This
surprises people, and the reason is not in the sampler.

Floating-point addition is not associative. A reduction — a sum across the hidden
dimension, a softmax denominator — gives a different last-bit result depending on the order
it accumulates. GPU kernels choose their reduction order based on shape: how many elements,
how many thread blocks, whether split-K is worth it. **Change the batch size and the kernel
changes its reduction order, and the logits change in the last bits.** Usually invisible;
occasionally enough to flip an `argmax` between two near-tied tokens.

So determinism requires *batch-invariant* operators — kernels that reduce in the same order
regardless of batch shape. `python/sglang/srt/batch_invariant_ops/` provides them, and
`python/sglang/srt/model_executor/model_runner.py:764` `maybe_enable_batch_invariant_mode`
switches them in. They are slower; you turn them on when you need reproducibility more than
throughput.

Per-request determinism is separate and cheaper: `python/sglang/srt/layers/sampler.py:684` `multinomial_with_seed` derives a
seed from the request's seed and the token position, so a given request samples the same
sequence regardless of what else is in the batch.

`.claude/skills/kl-consistency-test/SKILL.md` frames the whole problem precisely. Testing
that prefill logprobs match decode logprobs requires **two independent conditions**: every
operator must be batch-invariant, *and* the two paths must be computing the same function.
A non-zero KL divergence could be either, and the skill's contribution is a helper that
separates them — which is the difference between a debuggable failure and a number someone
tunes a threshold around.

---

## Incremental detokenization is genuinely hard

Tokens go back to `DetokenizerManager` (`python/sglang/srt/managers/detokenizer_manager.py:91`),
whose loop is refreshingly small:

```python
    def event_loop(self):
        """The event loop that handles requests"""
        while True:
            with self.soft_watchdog.disable():
                recv_obj = sock_recv(self.recv_from_scheduler)
            output = self._request_dispatcher(recv_obj)
            if output is not None:
                sock_send(self.send_to_tokenizer, output)
            self.soft_watchdog.feed()
```

Receive, dispatch, forward. Note the watchdog is *disabled* around the blocking receive —
waiting for work is not a hang — and fed after each message. That distinction matters:
without it, an idle server would look wedged.

The hard part is `python/sglang/srt/managers/detokenizer_manager.py:290` `_decode_batch_token_id_output`. You cannot simply decode the newest
token and append it, for two reasons:

**BPE merges are contextual.** A tokenizer may render token *n* differently depending on
token *n−1* — leading-space handling in particular. Decoding tokens independently and
concatenating produces text that differs from decoding the sequence.

**UTF-8 is multi-byte.** A single emoji or CJK character can span several tokens. Decoding
a partial token yields a replacement character, and once emitted it cannot be taken back
from a stream.

The mechanism is a per-request `python/sglang/srt/managers/detokenizer_manager.py:64` `DecodeStatus` holding surrogate state — how much text
has been emitted and where the decode boundary sits — with
`python/sglang/srt/managers/schedule_batch.py:1425` `init_incremental_detokenize` on the
scheduler side. `python/sglang/srt/managers/detokenizer_manager.py:226` `_grouped_batch_decode` batches the tokenizer calls, since the
detokenizer handles a whole batch per message.

`python/sglang/srt/managers/detokenizer_manager.py:213` `_clamp_decode_ids` is a small piece of defensive engineering with a comment
explaining itself: out-of-range ids are mapped to 0 so the tokenizer can decode them rather
than raising, because a crash here would take down detokenization for every concurrent
request.

---

## Stopping, and un-emitting

Stop conditions divide by difficulty.

**Stop tokens** are easy: compare the sampled id against a set.

**Stop strings** are not. The string may span several tokens, and you cannot know a stop
string is complete until you have generated past it — by which time, in streaming mode, you
may have already sent part of it to the client.

`python/sglang/srt/managers/detokenizer_manager.py:176` `trim_matched_stop` is the retraction:

```python
        # Trim stop str.
        if isinstance(matched, str) and isinstance(output, str):
            pos = output.find(matched)
            if pos == -1:
                return output
            end = pos + len(matched)
            return output[:end] if no_stop_trim else output[:pos]

        # Trim stop token.
        if isinstance(matched, int) and isinstance(output, list):
            if no_stop_trim:
                return output
            # 200012 <|call|> is the tool call token and one of eos tokens for gpt-oss model
            if output[-1] == 200012 and self.is_tool_call_parser_gpt_oss:
                return output
            assert len(output) > 0
            # NOTE: We can always assume the last token is the matched stop token
            return output[:-1]
        return output
```

This is why Chapter 4's finish reasons are a *class hierarchy* rather than an enum:
`FINISH_MATCHED_STR` carries the matched string so the trimmer knows how much to remove;
`FINISH_MATCHED_TOKEN` carries the token id. The reason needs a payload.

`no_stop_trim` is the client's choice about whether the stop sequence appears in the
output. And the hardcoded `200012` is honest special-casing: for gpt-oss the tool-call token
is *also* an EOS token, and trimming it would destroy the tool call Chapter 19 is about to
parse. A comment and a constant beat a plausible-looking abstraction here.

On the scheduler side, `python/sglang/srt/managers/schedule_batch.py:1445`
`_stop_match_tail_len` computes how much tail to keep buffered so a stop string spanning a
token boundary can still be detected — the lookback that makes any of this possible.

---

## Streaming out

The last hop is back to `TokenizerManager`, where Chapter 3's `ReqState` machinery resolves
the client's awaiting coroutine.

`python/sglang/srt/managers/tokenizer_manager.py:1641` `_coalesce_streaming_chunks` handles
a rate mismatch: the scheduler produces a token every few milliseconds, but sending an SSE
frame per token means a syscall and a network write per token. When several chunks are
queued, they are merged.

The subtlety is that with incremental streaming each chunk is a *delta*, so coalescing must
concatenate rather than take the last — dropping intermediate chunks would silently drop
tokens. The comment at `python/sglang/srt/managers/tokenizer_manager.py:1734` in `_wait_one_response` flags the same hazard from the
consumer side.

`python/sglang/srt/entrypoints/openai/sse_utils.py` does SSE framing, and
`python/sglang/srt/managers/tokenizer_manager.py:2508` `add_logprob_to_meta_info`,
`python/sglang/srt/managers/tokenizer_manager.py:2649` `convert_logprob_style`, and
`python/sglang/srt/managers/tokenizer_manager.py:2726` `detokenize_logprob_tokens` assemble the logprob payload — which requires
detokenizing the *alternatives*, not just the chosen token, and is why requesting top-20
logprobs is not free.

---

## The path, end to end

```
hidden_states                        [num_tokens, hidden_size]        GPU
  │  LogitsProcessor.forward
  │    _get_pruned_states            drop positions nothing needs
  │    _get_logits                   DP gather → lm_head → TP all-gather → scatter
  ▼
LogitsProcessorOutput                [num_seqs, vocab_size]
  │  Sampler.forward
  │    _preprocess_logits            custom processors, NaN handling
  │    grammar mask (Ch. 19)         if constrained
  │    penalties, top-k/p/min-p
  ▼
next_token_ids                       [num_seqs]                        GPU → CPU
  │  Scheduler.process_batch_result  append, check stop, free KV
  ▼  ZMQ
DetokenizerManager                   incremental decode + trim_matched_stop
  │  ZMQ
  ▼
TokenizerManager._handle_batch_output → ReqState.event.set()
  ▼
client
```

Two process hops for one token. The overlap loop of Chapter 4 hides the CPU cost of all of
it behind the next forward pass — which is the whole reason that loop is shaped the way it
is.

---

Part II ends here. Part III goes underneath it, into the memory system every chapter so far
has been spending.

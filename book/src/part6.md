# Part VI — Beyond Plain Generation

The engine described so far does one thing: it takes tokens and produces tokens, one at a
time, with every request in a batch treated identically.

Real deployments need more than that, and each of these three chapters covers a feature that
breaks an assumption the earlier parts relied on.

**Chapter 19** breaks the assumption that tokens are produced one at a time. Chapter 1
showed that a decode step wastes almost all of a GPU's arithmetic; speculative decoding
spends that waste on *guessing* several tokens ahead and checking them all at once. Done
right, it preserves the model's output distribution exactly — a wrong guess costs compute,
never correctness.

**Chapter 20** breaks the assumption that the model may produce any token. When output must
be valid JSON, or must match a tool schema, the engine masks illegal tokens before sampling.
The chapter pairs that with its mirror image — parsing tool calls out of free-form text —
because they are the same problem approached from opposite ends, and they are converging.

**Chapter 21** breaks the assumption that a batch is homogeneous. Some requests want
different weights (LoRA adapters); some carry images instead of text. Both look like reasons
to split the batch, which would destroy the batching efficiency Chapter 1 said was
everything. Both are solved the same way instead — and that shared solution is a move you
will have seen three times already by the time you reach it.

Each of these reaches into most of the rest of the book. Speculation alone touches memory
pools, scheduling, forward modes, CUDA graphs, and the radix cache. That is why they come
last: they are only legible once the things they perturb are understood.

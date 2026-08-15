# Part IV — What Actually Runs

So far the model has been a black box. Chapter 7 handed it a batch and got logits back, and
the book did not ask what happened in between.

These four chapters open the box. They are about the model itself: how a few hundred
gigabytes of weights get onto the right GPUs, what a model file in this codebase actually
looks like, how attention is computed, and the three separate techniques that make each
forward pass cheaper.

**Chapter 12** — loading. This looks like file I/O and is really a distributed sharding
problem: under tensor parallelism no rank ever holds a whole weight matrix, so every
parameter has to know how to load its own slice. The same machinery turns out to serve
reinforcement learning, where weights are replaced every few minutes without restarting.

**Chapter 13** — anatomy. SGLang rewrites every model rather than importing it from Hugging
Face, and the chapter walks the Llama implementation line by line to show why: every layer has to
cooperate with parallelism, quantization, and the paged cache of Chapter 9. By the end you
will be able to read any of the 218 model files in the repository.

**Chapter 14** — attention. One line in a model file dispatches to the most heavily
optimized operation in the field, and there is no single best implementation of it. This is
also the book's first descent to kernel source, where you can watch paged attention actually
happen in about forty lines of Triton.

**Chapter 15** — making it cheap. Three independent taxes on the forward pass — bytes moved,
kernel launches, kernel count — and the three techniques that attack them. They compose
precisely because they attack different things.

A theme emerges across these chapters that will keep recurring: the model code contains
almost no logic of its own. Caching, sharding, quantization, and graph capture are all
handled by the layers it is built from. That is what makes adding a model a bounded task
rather than an expert one.

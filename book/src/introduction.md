# SGLang Internals

*How a production LLM serving engine works, told by following one request through it.*

---

## What this book is about

Somewhere right now, a person types a question into a chat box and watches words appear
one at a time. Between their keystroke and those words is a machine doing something
genuinely difficult: holding a hundred billion numbers in memory, sharing one graphics card
among several hundred strangers, and deciding — thousands of times a second — whose turn it
is next.

This book is about that machine. Specifically it is about **SGLang**, an open-source
serving engine that runs on more than 400,000 GPUs and moves trillions of tokens a day.

It is not a manual. It is not organized around the tasks you might want to accomplish, and
where it does reach operations — cold starts, autoscaling, capacity — it does so from the
inside out, because the engine's internals are what decide those outcomes. It is a book about
*how the thing works*, of the kind you read to understand a system rather than to look
something up in it.

The central claim is that a serving engine is not a collection of clever tricks. It is a
single sustained argument, and the argument goes like this:

> Generating text is **memory-bound**, not compute-bound. The graphics card spends over 99%
> of its time waiting for weights to arrive from memory rather than multiplying them. The
> only way to make it efficient is to serve many requests at once so each weight fetch does
> more work. But every concurrent request needs memory of its own — its *KV cache* — and so
> **memory capacity, not arithmetic, sets how many people a GPU can serve.**

Everything else follows. Prefix caching, paged memory, continuous batching, quantization,
speculative decoding, expert parallelism, prefill/decode disaggregation — every one of them
is a response to that sentence. Once you can see the argument, the codebase stops looking
like a pile of features and starts looking like a series of answers to one question.

## Who this book is for

You will get the most from it if you can read Python comfortably, know roughly what a
transformer is, and have at some point wondered what happens after you call an inference
API.

You do **not** need to have worked on inference systems, and you do not need a GPU to read
it. Some chapters go down to CUDA and Triton kernels; you are expected to read those at a
glance, never to write them.

Three kinds of reader, and a route for each.

**The curious engineer** who uses LLM APIs and wants to know what is behind them.
Chapters 1 → 3 → Part II. That is the cost model, the shape of the machine, and one request
from an HTTP socket to a streamed token. It stands alone: you can stop after Chapter 8 with a
complete picture of a request's life and be glad you did.

**The practitioner** running SGLang in production.
Chapters 1 → 2 → 9 → 10 → 6 → 16 → 22 → 24. Why the constraint is memory, what your hardware
can do about it, where the memory goes, how the cache changes the arithmetic, what the
scheduler does with the budget, how parallelism changes it again, and then the two
operational chapters. Chapter 1's calculators are the ones you will come back to; most tuning
mistakes are an optimization aimed at a constraint that was not binding.

**The contributor** about to change something. Read straight through. Chapter 23 collects
the extension points, but the chapters before it are what make those seams make sense.

## The journey

The book is a descent. It starts where a request arrives and ends where the electrons are,
and each part goes one level below the last.

**Part I — Why any of this exists.** Three chapters. The first establishes the cost model
above, in enough detail that you can compute it yourself. The second is about the hardware
that model is parameterized by — accelerators, interconnects, and how to read a spec sheet.
The third is a tour of the software's shape: four processes, connected by sockets, each
doing one job.

**Part II — The life of a request.** Five chapters following one request from an HTTP
socket to a streamed token. This is the spine of the book. Everything after it is a detour
off this path, and the book keeps telling you which detour you are on.

**Part III — The scarce resource.** Three chapters on memory, because Part II keeps running
into it. This is where the book's signature idea lives: *RadixAttention*, which notices
that real conversations share long prefixes and stores each one only once.

**Part IV — What actually runs.** Four chapters on the model itself — how weights get onto
the GPU, what a model file looks like, how attention is computed, and the three separate
techniques that make each forward pass cheaper.

**Part V — When one GPU is not enough.** Three chapters on splitting the work: across the
chips in a machine, across machines in a rack, and eventually into separate fleets that do
different halves of the job.

**Part VI — Beyond plain generation.** Three chapters on features that break the
assumptions the earlier parts relied on — guessing tokens before the model produces them,
forcing output to match a schema, and serving requests that need different weights or carry
images.

**Part VII — Living with it.** Three chapters: how the engine reports on itself, how you
change it, and what it takes to run it — cold starts, autoscaling signals, and the way a
wedged rank announces itself.

Then six appendices: a flag reference, the environment-variable system, a glossary, a map
of the repository, an annotated startup log, and further reading.

## How to read it

**Read Parts I and II in order.** They are a single argument and a single narrative, and the
rest of the book assumes both.

**After that, follow your interest.** Every later chapter names its dependencies in its
opening, and cross-references are specific — "Chapter 10's reference counting," not "as
discussed earlier."

**Do not try to hold the code in your head.** Code references are links: every
`path/to/file.py:123` in the text goes to that exact line of SGLang on GitHub. Follow them
when you want to see more than the excerpt; ignore them when you want the argument. The
prose is written to stand on its own either way.

**Expect the same idea to recur.** A handful of moves appear over and over — turning
irregular control flow into an index table, trading memory for compute, hiding one
resource's latency behind another's work. When you notice a repeat, that is the book
working.

**The theory comes before the code, not instead of it.** Most chapters open by deriving the
algorithm the code implements — the roofline model, online softmax, the rejection-sampling
proof, Megatron's column-then-row split — from the papers that introduced them, and then
show what that derivation looks like as a Python file. The intent is that you can tell the
difference between a design decision and an implementation detail, which is the difference
between reading a codebase and understanding one. Appendix F lists every source, chapter by
chapter.

## A note on what is not here

Code references are pinned to one commit of SGLang, `7562e74` (August 2026). The project
moves fast, and a book that chased it would be wrong in a different way every month.

What ages well: the cost model of Chapter 1, the process topology of Chapter 3, the request
path of Part II, and the memory system of Part III. These have been stable for the
project's life. What ages quickly: specific kernel backends, quantization formats, and
speculative algorithms — Chapters 14, 15, and 19 name implementations that will change.

The diffusion stack for image and video generation is out of scope. It is a large parallel
system with its own scheduler and its own caching, and it deserves separate treatment
rather than a compressed chapter.

---

Chapter 1 begins with a question that sounds simple and is not: why does generating text
one token at a time waste almost an entire GPU?

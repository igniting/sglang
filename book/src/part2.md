# Part II — The Life of a Request

This is the spine of the book.

Part I established *why* a serving engine exists and *what shape* it has. The five chapters
here follow a single request through it: from the moment it arrives on an HTTP socket to
the moment its last token reaches the reader.

The route is not short. The request will be validated and tokenized in one process, sent
across a socket to another, wait in a queue, compete for memory against several hundred
peers, become a row in a tensor, pass through eighty transformer layers, get sampled into a
token, cross two more process boundaries, and be turned back into text — and then do all of
that again for the next token, and the one after.

Each chapter takes one stage:

**Chapter 4** — the front door. Text becomes token ids, and a message-passing protocol is
made to look like an ordinary `await`.

**Chapter 5** — the scheduler loop. One synchronous loop owns the GPU and answers one
question per iteration: what runs next? This is the heart of the engine, and it is smaller
than you expect.

**Chapter 6** — the decision itself. Which requests get admitted, under a memory budget
that cannot be known in advance, and what happens when the engine guesses wrong.

**Chapter 7** — the handoff to the GPU. Python objects become tensors, and a single enum
starts determining nearly every branch downstream.

**Chapter 8** — the return path. Hidden states become logits become tokens become text,
which turns out to be three separate hard problems that happen to sit next to each other.

By the end of Chapter 8 you will have seen a complete request, end to end, with no gaps.
Everything in Parts III through VI is a detour off this path — a place where the engine does
something more sophisticated than the straight line described here — and each of those
chapters will tell you which stage it is elaborating.

If you read only one part of this book, read this one.

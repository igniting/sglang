# Part I — Why Any of This Exists

Before looking at how SGLang is built, it is worth being precise about what makes serving
language models hard — because the answer is not what most people expect, and every design
decision in the rest of the book follows from it.

The naive expectation is that generating text is expensive because the models are large and
matrix multiplication is costly. That is true of *training*. It is almost entirely wrong
about *inference*, and believing it will lead you to optimize the wrong thing.

The three chapters here set up the argument the whole book rests on.

**Chapter 1** works out the actual cost of generating a token. It turns out that a graphics
card producing one token for one user is idle more than 99% of the time — not because the
work is easy, but because the arithmetic units are starved of data. The fix is to serve
many requests at once. The catch is that each concurrent request needs memory, and memory
runs out long before compute does. By the end of the chapter you will be able to compute,
for a given model and GPU, roughly how many people it can serve — and see why that number
is the one that matters.

**Chapter 2** looks at the machine those numbers describe. Chapter 1's conclusions are
parameterized by two figures from a spec sheet, and a reader who does not know where they
come from cannot apply the argument to their own hardware. This chapter is about
accelerators, the interconnects between them, how to read a vendor's slide without being
misled, and the short list of questions SGLang asks the device at startup.

**Chapter 3** turns to the software. SGLang is not one program but four cooperating
processes, and knowing which process does what explains a surprising amount: why requests
are validated where they are, why one rank of a multi-GPU setup is special, and why a
health check runs a real generation instead of returning 200. This chapter is the map you
will navigate by for the rest of the book.

Together they answer the question in the title of this part. Read them in order; each
assumes the one before it.

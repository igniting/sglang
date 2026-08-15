# Part III — The Scarce Resource

Part II kept running into the same wall.

The scheduler could not admit more requests because memory was full. It had to evict
running requests when it guessed wrong about memory. Its priority ordering depended on a
cache. Every interesting decision in Chapter 5 came back to the same constraint that
Chapter 1 predicted: **memory capacity, not arithmetic, sets how many people a GPU serves.**

These three chapters are about that memory — how it is stored, how it is shared, and what
happens when it runs out.

**Chapter 8** is the mechanism. Two levels of indirection turn "each request owns a buffer"
into "each request owns a list of page numbers," which is what makes everything else in
this part possible. This is also where the flag people tune first — `--mem-fraction-static`
— stops being a number and becomes an amount of memory.

**Chapter 9** is the idea SGLang is known for, and the centre of this book.
Real conversations share long prefixes: a chat turn re-sends the whole history, every
request in an application carries the same system prompt, an agent loop re-sends its whole
trajectory. Computing those prefixes repeatedly is arithmetic being repeated. RadixAttention
notices, and stores each shared prefix once.

**Chapter 10** extends the same tree past the GPU, into host memory and disk — a bandwidth
arbitrage that pays only above a computable prefix length, and the chapter is honest about
when it does not pay at all.

There is a satisfying loop here worth watching for. The cache decides which request is
cheapest to run; the scheduler runs the cheapest one; running it extends the cache along the
same branch; the next similar request is cheaper still. Prefix caching and cache-aware
scheduling are not two features that happen to compose. Each makes the other worth more.

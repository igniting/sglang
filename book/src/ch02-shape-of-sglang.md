# 2. The Shape of SGLang

> *The engine's process topology is its architecture; understanding the boundaries explains
> most design choices that follow.*

Chapter 1 explained why an inference engine has to exist. This chapter is about what one
looks like from the outside, before we start following anything through it.

Two things are worth establishing before the tour. The first is that SGLang is not one
program: cloning the repository gets you a serving runtime, a programming language, a
kernel library, and a load balancer, and knowing which is which saves a great deal of
confusion later. The second is that the runtime itself is not one process but four, and
almost every design decision in Part II follows from where those boundaries fall.

We will also read the project's own rules — a handful of short files stating invariants the
codebase enforces. They were written for contributors, but they are the most concentrated
design documentation in the repository, and each one marks a place where the architecture
has already been bitten.

This is the map you will navigate by for the rest of the book. It is worth the twenty
minutes.

---

## Several projects in one tree

`sglang` is not one program. Cloning the repository gets you a serving runtime, a
programming language, a kernel library, and a load balancer, each of which could plausibly
be its own project.

| Path | What it is | Language |
| --- | --- | --- |
| `python/sglang/srt/` | The serving runtime. ~90% of this book. | Python |
| `python/sglang/lang/` | The frontend DSL — the "SGLang" the project is named for. | Python |
| `python/sglang/kernels/` | JIT and AOT kernels. Absorbed the former top-level `sgl-kernel/`. | Python, CUDA, C++ |
| `python/sglang/benchmark/` | Benchmark harnesses. | Python |
| `sgl-model-gateway/` | Cache-aware router and load balancer. | Rust |
| `python/sglang/multimodal_gen/` | Diffusion stack. Out of scope — see the introduction. | Python |
| `test/`, `docs/`, `benchmark/` | Test suites, documentation, model-specific benchmarks. | — |

`srt` stands for **SGLang RunTime**. Alphabetically its subdirectories tell you nothing;
arranged by dependency they tell you almost everything:

```
entrypoints/        HTTP, gRPC, and the embeddable Engine        → Ch. 2, 3
   ↓
managers/           TokenizerManager, Scheduler, DetokenizerManager
                    schedule_batch.py, schedule_policy.py        → Ch. 3–7
   ↓
model_executor/     ModelRunner, ForwardBatch, CUDA graphs       → Ch. 6, 14
   ↓
models/             218 model definitions                        → Ch. 12
   ↓
layers/             attention, MoE, quantization, linear, sampler → Ch. 7, 13, 14, 16
   ↓
mem_cache/          KV pools, allocators, radix cache            → Ch. 8–10
distributed/        process groups, communicators                → Ch. 15
```

Roughly, a request flows top to bottom and its results flow back up. Cutting across that
spine are the feature subsystems — `speculative/`, `disaggregation/`, `lora/`,
`constrained/`, `function_call/`, `multimodal/`, `eplb/`, `observability/` — each of which
hooks into the spine at one or two well-defined points. Part VI is a tour of those hooks.

Two numbers are worth internalizing before you start reading: `srt/models/` holds 218 model
files, and `python/sglang/srt/server_args.py` is about 9,900 lines. Both are consequences of the same
fact — SGLang supports a very large surface of models and configurations — and both are
why the rest of the codebase works so hard at abstraction.

---

## Why multi-process, not multi-threaded

A serving engine has to do three things at once: accept and tokenize incoming HTTP
requests, run the GPU, and turn output tokens back into text. In a single Python process
those three would contend for the GIL, and the GPU — the expensive resource — would end up
waiting on string manipulation.

So SGLang splits them across processes:

<figure>
<svg viewBox="0 0 700 462" role="img" aria-label="Four processes connected by ZeroMQ sockets: tokenizer manager, one scheduler per tensor-parallel rank, and a detokenizer">
<title>SGLang process topology</title>
<rect class="dgm-box-accent" x="140" y="44" width="420" height="64" rx="6"/>
<text class="dgm-label" x="350.0" y="62.4" text-anchor="middle" font-weight="600" style="font-size:13.0px">TokenizerManager</text>
<text class="dgm-small" x="350.0" y="80.4" text-anchor="middle" style="font-size:11.5px">main process · asyncio event loop</text>
<text class="dgm-small" x="350.0" y="98.4" text-anchor="middle" style="font-size:11.5px">text → token ids · holds each request's future</text>
<rect class="dgm-box-accent" x="20" y="196" width="180" height="76" rx="6"/>
<text class="dgm-label" x="110.0" y="220.4" text-anchor="middle" font-weight="600" style="font-size:13.0px">Scheduler · tp 0</text>
<text class="dgm-small" x="110.0" y="238.4" text-anchor="middle" style="font-size:11.5px">owns one GPU</text>
<text class="dgm-small" x="110.0" y="256.4" text-anchor="middle" style="font-size:11.5px">receives, then broadcasts</text>
<rect class="dgm-box" x="230" y="196" width="170" height="76" rx="6"/>
<text class="dgm-label" x="315.0" y="229.4" text-anchor="middle" font-weight="600" style="font-size:13.0px">Scheduler · tp 1</text>
<text class="dgm-small" x="315.0" y="247.4" text-anchor="middle" style="font-size:11.5px">owns one GPU</text>
<rect class="dgm-box" x="430" y="196" width="170" height="76" rx="6"/>
<text class="dgm-label" x="515.0" y="229.4" text-anchor="middle" font-weight="600" style="font-size:13.0px">Scheduler · tp n</text>
<text class="dgm-small" x="515.0" y="247.4" text-anchor="middle" style="font-size:11.5px">owns one GPU</text>
<rect class="dgm-box" x="20" y="330" width="320" height="58" rx="6"/>
<text class="dgm-label" x="180.0" y="354.4" text-anchor="middle" font-weight="600" style="font-size:13.0px">DetokenizerManager</text>
<text class="dgm-small" x="180.0" y="372.4" text-anchor="middle" style="font-size:11.5px">subprocess · token ids → text, incrementally</text>
<text class="dgm-label" x="350.0" y="26" text-anchor="middle" font-weight="600" style="font-size:13px">HTTP clients</text>
<path class="dgm-line" d="M350.0 32 L350.0 38" marker-end="url(#arrow)"/>
<path class="dgm-line" d="M350.0 108 L350.0 160 L110.0 160 L110.0 190" marker-end="url(#arrow)"/>
<text class="dgm-small" x="362.0" y="152" text-anchor="start" style="font-size:11.5px">ZMQ · scheduler_input_ipc_name — rank 0 only</text>
<path class="dgm-dash" d="M200 234.0 L224 234.0" marker-end="url(#arrow)"/>
<path class="dgm-dash" d="M400 234.0 L424 234.0" marker-end="url(#arrow)"/>
<text class="dgm-small" x="400" y="300" text-anchor="middle" style="font-size:11.5px">broadcast_pyobj — every rank runs an identical batch</text>
<path class="dgm-line" d="M110.0 272 L110.0 324" marker-end="url(#arrow)"/>
<text class="dgm-small" x="122.0" y="312" text-anchor="start" style="font-size:11.5px">ZMQ · detokenizer_ipc_name</text>
<path class="dgm-line-accent" d="M340 359.0 L650 359.0 L650 76.0 L566 76.0" marker-end="url(#arrow-accent)"/>
<text class="dgm-small" x="497" y="350.0" text-anchor="middle" style="font-size:11.5px">ZMQ · tokenizer_ipc_name</text>
<text class="dgm-small" x="350.0" y="424" text-anchor="middle" style="font-size:11.5px">Output travels forward to the TokenizerManager, not back to the scheduler:</text>
<text class="dgm-small" x="350.0" y="442" text-anchor="middle" style="font-size:11.5px">that is the process holding the client's awaiting coroutine.</text>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" style="fill:var(--dgm-rule)"/></marker><marker id="arrow-accent" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" style="fill:var(--dgm-accent)"/></marker></defs>
</svg>
<figcaption>The four processes and the sockets between them. Under tensor parallelism only rank&nbsp;0 receives from the front end; it broadcasts to its peers so every rank sees the same batch.</figcaption>
</figure>

Note the cycle: the detokenizer does not reply to the scheduler, it sends *forward* to the
tokenizer manager, which holds the client's async future. Chapter 3 walks that return path.

The split buys three things. **Parallelism** — tokenization and detokenization run
genuinely concurrently with GPU work, not interleaved by the GIL. **Isolation** — a
tokenizer crash on malformed input does not take down a process holding 70 GB of model
weights. **Cadence** — the scheduler is a tight synchronous loop with no `await` in it,
while the front end is fully async; forcing either into the other's model would be
painful.

The topology is constructed in `python/sglang/srt/entrypoints/engine.py:1052`
`_launch_subprocesses`, whose docstring states it plainly:

> Launch the TokenizerManager in the main process, the Scheduler in a subprocess, and the
> DetokenizerManager in another subprocess.

Its opening steps are the server's entire bootstrap, in order:

```python
        startup_tic = time.perf_counter()

        # Configure global environment
        configure_logger(server_args)
        _set_envs_and_config(server_args)
        ...
        server_args.check_server_args()

        # Allocate ports for inter-process communications
        if port_args is None:
            port_args = PortArgs.init_new(server_args)
```

`_set_envs_and_config` (`python/sglang/srt/entrypoints/engine.py:1620`) sets process-global
state — environment variables, multiprocessing start method, allocator settings — that must
be in place *before* any subprocess forks, because subprocesses inherit it. Startup-order
bugs in this file tend to be of exactly that form.

`PortArgs` (`python/sglang/srt/server_args.py:9696`) is the address book, and reading its
fields is the fastest way to learn the topology:

```python
class PortArgs:
    # The ipc filename for tokenizer to receive inputs from detokenizer (zmq)
    tokenizer_ipc_name: str
    # The ipc filename for scheduler (rank 0) to receive inputs from tokenizer (zmq)
    scheduler_input_ipc_name: str
    # The ipc filename for detokenizer to receive inputs from scheduler (zmq)
    detokenizer_ipc_name: str

    # The port for nccl initialization (torch.dist)
    nccl_port: int

    # The ipc filename for rpc call between Engine and Scheduler
    rpc_ipc_name: str

    # The ipc filename for Scheduler to send metrics
    metrics_ipc_name: str
```

Each comment names an edge in the diagram above. Note "scheduler (rank 0)": under tensor
parallelism only rank 0 receives from the tokenizer, and it broadcasts to its peers.
Chapter 3 explains why that asymmetry exists and what it costs.

`_launch_scheduler_processes` (`python/sglang/srt/entrypoints/engine.py:848`) does the
spawning, and its first decision shows the topology is not fixed:

```python
        use_dp_controller = (
            server_args.dp_size > 1 or server_args.ep_join_mode == "scale"
        )
```

With data parallelism there is an extra process in front — a
`DataParallelController` that routes requests across whole replicas. Without it, schedulers
are spawned directly, one per (pipeline rank, tensor rank) pair computed by
`_calculate_rank_ranges` (`python/sglang/srt/entrypoints/engine.py:1794`). Chapter 15
covers what those ranks mean.

`_launch_detokenizer_subprocesses` (`python/sglang/srt/entrypoints/engine.py:966`)
completes the picture.

---

## One controller, or one per rank?

The diagram hides a decision that shapes everything downstream, and it is worth pulling out
because the alternative is what several other systems chose.

When a model is split across eight GPUs, somebody has to decide what the next batch is. There
are two ways to arrange that.

**Single controller.** One process owns the scheduling state and drives eight worker
processes, sending each an instruction per step. This is the shape of a classic
parameter-server or Ray-style actor system, and it is how vLLM originally drove its workers.
It is easy to reason about: there is exactly one copy of the truth.

**Multi-controller, or SPMD** — *single program, multiple data*. Every rank runs the same
scheduler code over the same inputs and reaches the same conclusions independently. Nobody
issues instructions; the ranks stay in step because they are computing the same function of
the same data. This is the shape of MPI programs, and of Megatron-LM's training loop.

SGLang is SPMD, with one qualification: the ranks do not re-derive the request stream
independently, because they cannot — requests arrive over a socket, and a socket delivers to
one reader. Rank 0 receives, then `broadcast_pyobj` hands the identical Python objects to
every peer. From that point the ranks are running the same program on the same data, and
each independently concludes that the same batch should run.

The reason to prefer it here is per-step latency. A decode step is on the order of ten
milliseconds. A single controller would need a round trip to every rank inside that budget —
send the instruction, wait for eight acknowledgements, gather results — and each round trip
crosses a process boundary, a serializer, and a socket. That overhead does not shrink as the
model gets faster; it grows as a fraction of the step as GPUs get quicker. SPMD pays a
single broadcast of a small object instead, on a collective the ranks are already
synchronized on for the forward pass itself.

The cost is that *every rank must stay deterministic in lockstep*. Two ranks that disagree
about which requests are in the batch will launch different collectives and deadlock — not
crash, deadlock, which is a much worse failure to debug. This is why so much of the
scheduler is careful about ordering: dictionaries iterated in insertion order, sorts made
total by tie-breaking on request id, decisions taken from broadcast data rather than local
timing. Chapter 4's loop and Chapter 5's batching policy both read differently once you know
that every line of them is running eight times in parallel and must agree eight times over.

The three-way split of the *front end* is a different argument entirely. Tokenization and
detokenization are pure CPU string work, they are on the critical path of every request, and
in CPython they would hold the GIL. Putting them in the scheduler's process would mean the
GPU waits on a regex. So they are separated for concurrency, while the schedulers are
replicated for scale — two different problems that happen to be solved with the same
primitive.

---

## Three front doors

**The HTTP server** is the common path.
`python/sglang/srt/entrypoints/http_server.py:270` `lifespan` handles FastAPI startup, and
the endpoints follow: `:874` `generate_request` for the native API, plus the
OpenAI-compatible surface in `python/sglang/srt/entrypoints/openai/`.

One endpoint is worth singling out now.
`python/sglang/srt/entrypoints/http_server.py:646` `health_generate` serves `/health` and
`/health_generate`, and it does *not* return 200 unconditionally — it runs a real
generation through the whole pipeline. That is a deliberate choice: in a multi-process
engine, the HTTP process can be perfectly healthy while the scheduler is deadlocked or the
GPU has fallen off the bus. A health check that only proves the web server is up would
keep a broken replica in the load balancer.

**The Engine** is the same runtime without HTTP.
`python/sglang/srt/entrypoints/engine.py:199` `class Engine` embeds everything in your own
process, and `:352` `generate` is a direct call. This is the path used for offline batch
inference and for RL rollouts, where an HTTP hop per rollout would be pure overhead.
Chapter 11 covers the weight-update API that makes the RL case work.

**gRPC** exists for deployments that want a binary protocol; `serve_grpc` and the Rust gRPC
server are alternate front ends onto the same `TokenizerManager`.

`python/sglang/launch_server.py` picks between them, and is short enough to read whole:

```python
def run_server(server_args):
    """Run the server based on the gRPC flags and server_args.encoder_only."""
    if server_args.encoder_only:
        ...
    elif server_args.smg_grpc_mode:
        ...
    elif server_args.use_ray:
        ...
    else:
        # Default mode: HTTP mode.
        from sglang.srt.entrypoints.http_server import launch_server

        launch_server(server_args)
```

Every import is inside a branch. That is not stylistic — importing the Ray or gRPC stacks
costs seconds of startup, and a serving engine that must restart under load cannot afford
to pay for paths it will not use.

---

## The fourth front door: the DSL

The project is named for its **S**tructured **G**eneration **Lang**uage, and that language
is still in the tree.

`python/sglang/lang/api.py:75` `gen`, `python/sglang/lang/api.py:236` `select`, and the role helpers
`python/sglang/lang/api.py:253` `system`, `python/sglang/lang/api.py:257` `user`, `python/sglang/lang/api.py:261` `assistant` are the primitives. A program looks like
ordinary Python with these calls embedded, and `python/sglang/lang/ir.py` turns it into an IR —
`python/sglang/lang/ir.py:451` `SglGen`, `python/sglang/lang/ir.py:533` `SglSelect`,
`python/sglang/lang/ir.py:552` `SglFork` — that the runtime can reason about as a whole
rather than as opaque strings.

The payoff is co-design. Two of the IR nodes are the argument:

**`SglFork`** (`python/sglang/lang/ir.py:552`) branches a program into several
continuations that share everything before the branch. To a plain HTTP API that is *n*
independent requests, each re-sending and re-prefilling the shared context. To SGLang it is
one prefix in the radix tree of Chapter 9 with *n* children — the shared part is computed
once because the runtime can *see* that it is shared.

**`SglSelect`** (`python/sglang/lang/ir.py:533`) picks among a fixed set of options. Without
runtime support that means generating each option and comparing likelihoods. With it, the
options are scored against a shared prefix in one pass.
`python/sglang/lang/choices.py` holds the scoring methods.

`python/sglang/lang/interpreter.py:274` `StreamExecutor` runs the IR, `:852` `ProgramState`
holds a program's variables, and `:57` `run_program` is the entry point.
`python/sglang/lang/tracer.py` provides the alternative: trace the program into a graph
before executing, so the runtime can see the whole structure in advance.

This is the historical seed of the whole project — prefix sharing was originally motivated
by *programs* that obviously share prefixes, before it turned out that ordinary chat
traffic shares them just as much.

---

## Reading the flag surface

`python/sglang/srt/server_args.py` is about 9,900 lines. Nobody reads it linearly, and this
book does not ask you to. Read it as a table of contents: the argument groups partition the
feature space, and each group names a subsystem.

Three habits make it tractable:

1. **Search by flag, then read the surrounding group.** The neighbours of an argument are
   almost always the rest of its subsystem.
2. **Read `check_server_args` and the `__post_init__` logic.** The constraints between
   flags encode which combinations actually work — often the only place that is written
   down.
3. **Treat a default as a claim.** `--mem-fraction-static`'s default is the outcome of the
   Chapter 8 memory calculation; `--chunked-prefill-size`'s is the Chapter 5 latency trade.

Appendix A groups the arguments by subsystem with pointers into the chapters.

---

## The repo's own rules as design documents

`.claude/rules/` contains short files stating invariants the codebase enforces. They were
written for automated contributors, but they are the most concentrated design documentation
in the repository — each one is a hazard the architecture has already hit.

**`.claude/rules/schedule-batch-out-of-place-mutation.md`** forbids in-place mutation of
`ScheduleBatch` fields:

> Why: `copy()` snapshots and the overlap scheduler's queued references rely on old objects
> staying frozen.

That is Chapter 4's overlap scheduler leaking into a style rule. Because step *N+1* is
prepared while step *N* is still in flight, two batch objects are live at once and the
older one must not change underneath the GPU.

**`.claude/rules/forward-batch-init-new-purity.md`** requires `ForwardBatch.init_new` to
treat its input `ScheduleBatch` as read-only — the same concern one layer down (Chapter 6).
It also lists its own tolerated exceptions, which is unusually honest for a style rule and
tells you where the abstraction is still leaking.

**`.claude/rules/no-dataclasses.md`** requires `msgspec.Struct` over `@dataclass`, partly
for strict typing and partly because these objects cross process boundaries and are
candidates for a future Rust port (Chapter 17's gateway is the precedent).

**`.claude/rules/no-getattr-defensive.md`** bans defensive `getattr(obj, "field", default)`.
In a system with this many configuration permutations, a silently-swallowed
`AttributeError` becomes a wrong answer rather than a crash.

**`.claude/rules/general-code-style.md`** collects the rest, including "avoid mixins" — a
rule that Chapter 4 will show `Scheduler` violating twenty-two times over, for reasons the
`.claude/skills/large-class-style/SKILL.md` playbook explains.

Read all five before your first patch. They will save you a review cycle each.

---

Part II starts now: one request, from socket to streamed token, one chapter per stage.

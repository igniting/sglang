# 2. The Shape of SGLang

> *The engine's process topology is its architecture; understanding the boundaries explains most design choices that follow.*

> **Status: outline.** This chapter is planned but not yet written. The beats below are the writing plan — each one pairs an idea with the code that implements it.

## Beats

1. **Several projects in one tree.** The runtime (`python/sglang/srt`), the frontend DSL
   (`python/sglang/lang`), the kernel layer (`python/sglang/kernels`, which absorbed the
   former top-level `sgl-kernel`), and the Rust gateway (`sgl-model-gateway`). The `srt`
   layout presented as a dependency graph, not an alphabetical list.

2. **Why multi-process, not multi-threaded.** The GIL, fault isolation, and the
   tokenizer/scheduler/detokenizer split. `python/sglang/srt/entrypoints/engine.py:1052`
   `_launch_subprocesses` is where the topology is literally constructed — `:848` spawns
   schedulers per TP rank, `:966` the detokenizers.

3. **Three front doors.** The HTTP server
   (`python/sglang/srt/entrypoints/http_server.py:270` `lifespan`, `:874`
   `generate_request`), the embeddable `Engine`
   (`python/sglang/srt/entrypoints/engine.py:199`, `:352` `generate`) used for offline
   batch and RL rollouts, and gRPC. `python/sglang/launch_server.py` dispatches between
   them and is short enough to read whole.

4. **The fourth front door: the DSL.** SGLang is named for its *Structured Generation
   Language*. `python/sglang/lang/api.py` gives the primitives (`gen`, `select`, `fork`),
   `python/sglang/lang/ir.py` the program IR, and `python/sglang/lang/interpreter.py:274`
   `StreamExecutor` the execution engine. The payoff — `fork` becoming a radix-tree branch
   rather than *n* independent generations — is the first hint of the frontend/runtime co-
   design that Chapter 9 completes.

5. **Reading the flag surface.** `python/sglang/srt/server_args.py` (~9,900 lines) is not
   read linearly; it is read as a table of contents for the feature space, with each
   argument group pointing at the subsystem that consumes it.

6. **The repo's own rules as design documents.** `.claude/rules/no-dataclasses.md`, `no-
   getattr-defensive.md`, `schedule-batch-out-of-place-mutation.md`, and `forward-batch-
   init-new-purity.md` encode invariants that exist nowhere else. Each explains a hazard
   the architecture has already hit.

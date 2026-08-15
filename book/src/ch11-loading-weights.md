# 11. Loading and Updating Weights

> *Weight loading is a distributed sharding problem disguised as file I/O, and the same
> machinery serves both startup and reinforcement learning.*

Part IV opens the box that Chapter 6 handed a batch to.

Before a model can run, a few hundred gigabytes of numbers have to get from files on disk
into the right memory on the right GPUs. That sounds like file I/O and a progress bar. It is
really a distributed sharding problem, because under the tensor parallelism of Chapter 15 no
single rank ever holds a complete weight matrix — each holds a slice, and has to know which
slice is its own without ever materializing the whole thing.

This chapter follows a checkpoint from a model path to sharded parameters. Along the way we
get an unusually good look at an interface migration caught mid-flight: two generations of
the weight-loading protocol living side by side in the same file, selected by an environment
variable, which is a rare chance to see why the newer one exists.

The chapter then turns to a use of the same machinery that has nothing to do with startup.
Reinforcement learning needs an inference engine whose weights can be replaced every few
minutes, without restarting, while sharing a GPU with a trainer. That requirement is why
SGLang is described as a rollout backend and not only a server.

---

## From a path to a class

`--model-path meta-llama/Llama-3-70B` has to become a Python class, a config object, and a
few hundred gigabytes of tensors in the right places on the right GPUs. Three translations
happen along the way.

**Architecture to class.** A Hugging Face config file names an architecture —
`LlamaForCausalLM` — and the registry maps that string to one of the 218 classes in
`python/sglang/srt/models/`. Adding a model (Chapter 22) is mostly adding an entry here.

**Config to config.** `python/sglang/srt/configs/` holds SGLang's own config classes, 63 of
them, because HF configs are written for the reference implementation and SGLang needs
things they do not always state: which layers are MoE, what the KV layout should be, where
sliding-window boundaries fall. `python/sglang/srt/configs/model_config.py` is the common wrapper, and it is what
Chapter 8's pool sizing reads.

**Checkpoint to sharded parameters.** The rest of this chapter.

---

## Each rank loads only its slice

Under tensor parallelism (Chapter 15) a rank holds a *fraction* of every weight matrix. It
would be wasteful to load the whole checkpoint on each rank and then slice — the peak
memory would be the full model on every GPU, which for a 70B model on 8 ranks is exactly
what you cannot afford.

Instead each parameter carries its own loading logic. When a module registers a parameter,
it attaches a `weight_loader` function; the loader iterates checkpoint tensors and asks
each parameter to take the piece it wants. `python/sglang/srt/model_loader/weight_utils.py`
holds `default_weight_loader` and its relatives, and
`python/sglang/srt/layers/parameter.py` holds the parameter wrappers that know their own
shard geometry.

`python/sglang/srt/model_loader/loader.py` implements the load formats and
`python/sglang/srt/model_loader/auto_loader.py` the newer generic path.
`python/sglang/srt/model_executor/model_runner.py:1057` `load_model` orchestrates it, and
its opening is mostly about *speed*:

```python
        tic_total = time.perf_counter()
        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Load weight begin. avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        # This can reduce thread conflicts and speed up weight loading.
        if self.device != "cpu":
            torch.set_num_threads(1)
```

The `set_num_threads(1)` line is counterintuitive and worth understanding: with eight
scheduler processes on one node, each spawning a thread pool for tensor conversion, the
threads contend rather than help. Restricting each process to one thread makes the *node*
faster even though it makes each process nominally slower.

The memory logging bracketing this function is what produces the startup lines operators
read to see how much room the KV pool will get (Chapter 8).

---

## Two generations of the protocol, side by side

`python/sglang/srt/models/llama.py:663` `load_weights` is a fork in the road:

```python
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        from sglang.srt.environ import envs

        if envs.SGLANG_ENABLE_WEIGHT_LOADER_V2.get():
            return self._load_weights_v2(weights)
        return self._legacy_load_weights(weights)
```

Two implementations of the same job, selected by environment variable. This is an interface
migration caught mid-flight, and reading both is more instructive than reading either.

`:670` `_legacy_load_weights` is explicit and per-model:

```python
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
```

That table is the checkpoint-to-runtime structural difference in five lines. The checkpoint
stores `q_proj`, `k_proj`, `v_proj` as three matrices because that is how the reference
implementation defines them. SGLang fuses them into one `qkv_proj` (Chapter 12 explains
why: one GEMM instead of three). So loading must route three checkpoint tensors into three
slices of one parameter, and `shard_id` says which slice.

The same function also handles renames, pipeline filtering, and skips:

```python
            if name.endswith(".activation_scale"):
                name = name.replace(".activation_scale", ".input_scale")
            if name.endswith(".weight_scale_inv"):
                name = name.replace(".weight_scale_inv", ".weight_scale")

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue
            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
```

The scale renames are quantization-format drift (Chapter 14) — different exporters use
different names for the same tensor. The `start_layer` / `end_layer` filter is pipeline
parallelism: a rank holding layers 20–39 skips everything else, which is what keeps PP
memory proportional. And `rotary_emb.inv_freq` is skipped because SGLang computes rotary
frequencies rather than loading them.

Multiply that by 218 models and the problem with the legacy path is obvious: every model
re-implements the same loop with small variations, and every quantization format adds a
rename to all of them.

`:743` `_load_weights_v2` is the response:

```python
    def _load_weights_v2(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """AutoWeightsLoader path with RemapRegistry for FP8 suffix normalization."""
        from sglang.srt.model_loader.auto_loader import (
            AutoWeightsLoader,
            filter_pp_weights,
            get_weight_remap,
        )

        if hasattr(self.model, "start_layer"):
            weights = filter_pp_weights(
                weights, self.model.start_layer, self.model.end_layer
            )

        skip_prefixes = []
        if self.config.tie_word_embeddings:
            skip_prefixes.append("lm_head.")
```

Each concern is now a named, shared component: `filter_pp_weights` for the pipeline slice,
`get_weight_remap` for the rename registry, `AutoWeightsLoader` for the traversal. The model
declares its exceptions instead of implementing the mechanism.

`tie_word_embeddings` is a good example of a model-specific fact that must stay in the
model: when input embeddings and the LM head share weights, the checkpoint may contain only
one, and loading `lm_head.` would fail on a parameter that does not exist.

---

## Formats and sources

**Safetensors** is the default — memory-mappable, so tensors are read lazily rather than
deserialized wholesale. **GGUF** is supported through
`python/sglang/srt/model_loader/gguf_name_maps.py`. **Sharded checkpoints** are the norm at
scale, and multiple ranks reading different shards in parallel is where load time actually
goes.

`python/sglang/srt/connector/` handles non-local sources:
`python/sglang/srt/connector/s3.py`, `python/sglang/srt/connector/azure.py`,
`python/sglang/srt/connector/redis.py`, and `python/sglang/srt/connector/remote_instance.py`. That last one is the most interesting — loading weights from
*another running instance* over the network, which turns a cold start into a peer-to-peer
copy and matters when you are scaling a deployment up under load.

`load_format` is the tuning knob, and `dummy` deserves a mention: it fills weights with
random values, skipping I/O entirely. Useless for output, ideal for benchmarking shapes and
memory — Chapter 21's harnesses use it constantly.

`python/sglang/srt/model_executor/model_runner_components/startup_weight_load.py` and
`python/sglang/srt/model_executor/model_runner.py:1203` `start_startup_weight_load` /
`python/sglang/srt/model_executor/model_runner.py:1207` `finalize_startup_weight_load` split loading
into a start and a join, so it can overlap with other initialization.
`python/sglang/srt/model_loader/ci_weight_validation.py` verifies checkpoints in CI, and
`python/sglang/srt/model_executor/model_runner.py:1833` `check_weights` is the runtime counterpart.

---

## Updating weights without restarting

Everything so far assumes weights are loaded once. Reinforcement learning breaks that
assumption: a rollout backend generates samples, the trainer updates the policy, and the
new weights must reach the inference engine — every few minutes, for hours.

Restarting is not an option. It would mean reloading from disk, rebuilding the memory pool,
re-capturing CUDA graphs, and re-warming caches, on every training step.

So `python/sglang/srt/entrypoints/engine.py:1365`–`:1478` exposes a family of update paths,
one per source:

| Method | Source | When |
| --- | --- | --- |
| `:1444` `update_weights_from_disk` | checkpoint files | new checkpoint written out |
| `:1399` `update_weights_from_distributed` | a torch.distributed group | trainer and engine in one job |
| `:1421` `update_weights_from_tensor` | tensors in the caller's process | in-process trainer |
| `:1464` `update_weights_from_ipc` | CUDA IPC handles | trainer on the same node, zero-copy |

They are ordered by decreasing overhead. The IPC path is the interesting one — same
mechanism as Chapter 3's multimodal transport, applied to model weights: the trainer hands
over a handle, and the engine maps the memory without copying anything.

`:1365` `init_weights_update_group` and `:1387` `destroy_weights_update_group` manage the
process group for the distributed path.
`python/sglang/srt/model_executor/model_runner_components/weight_updater.py` implements the
engine side, `python/sglang/srt/weight_sync/tensor_bucket.py` batches small tensors into
larger transfers, and `python/sglang/srt/checkpoint_engine/` handles the checkpoint-based
flow.

### Sharing a GPU with the trainer

The harder problem is memory. During a training step, the inference engine is idle but
still holding weights, a KV pool sized to fill the GPU (Chapter 8), and CUDA graph buffers
(Chapter 14). The trainer needs that memory for optimizer state and activations.

```
:1573  release_memory_occupation(tags)    give the memory back
:1579  resume_memory_occupation(tags)     take it again
```

Tags allow partial release — weights but not the KV pool, or the reverse — because what the
trainer needs varies. Under the hood this is the memory-saver adapter visible throughout
Chapter 8's pools (`GPU_MEMORY_TYPE_KV_CACHE` regions), which is why those `with
memory_saver_adapter.region(...)` blocks exist at all.

Generation must be quiesced first, and that is a scheduler operation:

```
python/sglang/srt/managers/scheduler.py:4576  pause_generation
python/sglang/srt/managers/scheduler.py:4665  continue_generation
```

`python/sglang/srt/managers/scheduler.py:4573` `_pause_engine` returns the paused requests, so in-flight work survives the pause
rather than being aborted.

A full RL step therefore looks like:

1. `pause_generation` — quiesce the scheduler.
2. `release_memory_occupation` — hand the GPU to the trainer.
3. Trainer runs.
4. `resume_memory_occupation` — take the memory back.
5. `update_weights_from_*` — install the new policy.
6. `continue_generation` — resume.

Two more pieces complete the picture. `python/sglang/srt/managers/scheduler.py:4246`
`flush_cache` clears the radix tree — **mandatory** after a weight update, because Chapter
9's cache holds KV computed by the *old* weights, and serving it against new weights would
silently mix two policies. And `python/sglang/srt/models/llama.py:854` `get_embed_and_head`
/ `:857` `set_embed_and_head` expose embeddings and the LM head directly, for frameworks
that update only those.

`docs/docs/advanced_features/sglang_for_rl.mdx` and
`docs/docs/references/post_training_integration.mdx` cover the integrations — verl, slime,
AReaL, Miles, Tunix — that this API exists to serve. It is the reason SGLang is
described as a rollout backend and not only a server.

The weights are on the GPU. Chapter 12 opens one of the files that describes what to do with
them.

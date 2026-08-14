# 11. Loading and Updating Weights

> *Weight loading is a distributed sharding problem disguised as file I/O, and the same machinery serves both startup and reinforcement learning.*

> **Status: outline.** This chapter is planned but not yet written. The beats below are the writing plan — each one pairs an idea with the code that implements it.

## Beats

1. **From a path to a class.** The model registry, HF config to SGLang config translation
   (`python/sglang/srt/configs/`), and where architectures diverge from their HF
   definitions.

2. **Each rank loads only its slice.** `python/sglang/srt/model_loader/loader.py`,
   `python/sglang/srt/model_loader/auto_loader.py`, and the `weight_loader` protocol
   attached to every parameter in `python/sglang/srt/model_loader/weight_utils.py`.
   `python/sglang/srt/model_executor/model_runner.py:1057` `load_model` orchestrates it.

3. **Two generations of the protocol, side by side.**
   `python/sglang/srt/models/llama.py:663` `load_weights` and `:743` `_load_weights_v2` —
   a rare chance to see an interface migration mid-flight.

4. **Formats and sources.** Safetensors, GGUF, sharded checkpoints;
   `python/sglang/srt/connector/` for S3, Azure, Redis, and remote-instance loading;
   `load_format` as a startup-latency knob.

5. **Updating weights without restarting.**
   `python/sglang/srt/entrypoints/engine.py:1365`–`:1478` — from disk, from a distributed
   group, from tensors, from IPC handles.
   `python/sglang/srt/model_executor/model_runner_components/weight_updater.py` and
   `python/sglang/srt/checkpoint_engine/` implement it;
   `python/sglang/srt/entrypoints/engine.py:1573` `release_memory_occupation` / `:1579`
   `resume_memory_occupation` and `python/sglang/srt/managers/scheduler.py:4576`
   `pause_generation` let training and inference share a GPU. This is what makes SGLang an
   RL rollout backend rather than only a server.

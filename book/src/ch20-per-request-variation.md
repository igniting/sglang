# 20. Per-Request Variation: LoRA and Multimodal

> *Both features break the assumption that every request in a batch needs the same weights and the same kind of input — and both are solved by extending the batch, not splitting it.*

> **Status: outline.** This chapter is planned but not yet written. The beats below are the writing plan — each one pairs an idea with the code that implements it.

## Beats

1. **Batching across adapters.** `python/sglang/srt/lora/lora_manager.py:59` `LoRAManager`
   — `:428` `prepare_lora_batch` assembles per-request adapter indices so one kernel
   serves a batch using different adapters; `python/sglang/srt/lora/backend/` for the
   grouped-GEMM kernels that make it work.

2. **Adapters as a memory pool.** `python/sglang/srt/lora/mem_pool.py`,
   `python/sglang/srt/lora/eviction_policy.py`, and
   `python/sglang/srt/lora/lora_manager.py:221` `load_lora_adapter` for runtime
   load/unload. Admission gains a new constraint at
   `python/sglang/srt/managers/scheduler.py:3450` `_can_schedule_lora_req`.

3. **Adapters must not share a prefix cache.** The `RadixKey` extra key from Chapter 9
   (`python/sglang/srt/mem_cache/radix_cache.py:64`) is what keeps two adapters' identical
   token sequences apart — a correctness bug waiting for anyone who skips it.

4. **Non-text input, spliced into text.** `python/sglang/srt/multimodal/processors/` (53
   of them) and `python/sglang/srt/managers/mm_utils.py`;
   `python/sglang/srt/managers/schedule_batch.py:318` `MultimodalDataItem`, `:590`
   `MultimodalInputs`, `:569` `build_padded_input_ids` — encoder output replacing
   placeholder tokens.

5. **Position arithmetic moves into the scheduler.**
   `python/sglang/srt/managers/scheduler.py:2308` `_maybe_compute_mrope_positions` and
   `python/sglang/srt/model_executor/forward_batch_info.py:1163`
   `_compute_mrope_positions` — VLM 3D positions cannot be computed by the model alone.

6. **Not sending pixels through a socket.** `python/sglang/srt/managers/scheduler.py:2209`
   `_process_and_broadcast_mm_inputs` and the CUDA-IPC path from Chapter 3, plus
   `python/sglang/srt/mem_cache/multimodal_cache.py` for reusing encoder output.

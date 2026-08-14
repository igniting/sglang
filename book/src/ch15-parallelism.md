# 15. Tensor, Pipeline, and Data Parallelism

> *The three classical parallelism axes differ in what they split and therefore in which interconnect they stress; SGLang adds a fourth because MLA broke the assumptions.*

> **Status: outline.** This chapter is planned but not yet written. The beats below are the writing plan — each one pairs an idea with the code that implements it.

## Beats

1. **Groups and collectives.** `python/sglang/srt/distributed/parallel_state.py:237`
   `GroupCoordinator`, `:2193` `init_distributed_environment`, `:2285`
   `initialize_model_parallel`; `python/sglang/srt/distributed/device_communicators/` for
   custom all-reduce, PyNCCL, and symmetric memory — with the cost model for each
   collective on NVLink vs InfiniBand.

2. **TP is already written.** The collectives live inside `ColumnParallelLinear` and
   `RowParallelLinear` from Chapter 12; this beat only explains the two communication
   points per block and why TP does not cross node boundaries cheaply.
   `python/sglang/srt/layers/communicator.py` is the per-layer strategy object.

3. **PP splits layers.** `python/sglang/srt/managers/scheduler_pp_mixin.py` for microbatch
   scheduling in the loop, `python/sglang/srt/models/llama.py:640` for the layer-range
   boundary, and bubbles as the cost.

4. **MLA breaks TP.** A compressed KV cache replicated across ranks wastes the very thing
   that made it small — the setup for DP attention.

5. **Data-parallel attention.** `python/sglang/srt/layers/dp_attention.py:338`
   `initialize_dp_attention`, `:412` `get_dp_local_info`, `:76` `DpPaddingMode`, `:450`
   `_dp_gather_via_all_reduce` and `:494` `_dp_gather_via_all_gather`. Each rank owns
   whole sequences for attention, then the batch is gathered for the TP-sharded MLP.

6. **The price: everyone must agree.**
   `python/sglang/srt/model_executor/forward_batch_info.py:1305` `prepare_mlp_sync_batch`
   and `:1620` `post_forward_mlp_sync_batch` force every rank to a common batch shape,
   which is why idle batches exist at all.
   `python/sglang/srt/layers/logits_processor.py:249` `compute_dp_attention_metadata`
   carries it to the end.

7. **The other DP.** `python/sglang/srt/managers/data_parallel_controller.py` — whole-
   replica routing, an unrelated mechanism with a colliding name.

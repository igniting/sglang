# 3. From HTTP to Token IDs

> *The front of the engine is an async/sync boundary, and most of its complexity is in making message passing look like `await`.*

> **Status: outline.** This chapter is planned but not yet written. The beats below are the writing plan — each one pairs an idea with the code that implements it.

## Beats

1. **Tokenization gets its own process and its own event loop.**
   `python/sglang/srt/managers/tokenizer_manager.py:374` `TokenizerManager` — `:755`
   `generate_request` is the async entry, `:985` `_tokenize_one_request` and `:1359`
   `_create_tokenized_object` the conversion.

2. **Validation as a stability boundary.**
   `python/sglang/srt/managers/tokenizer_manager.py:1185` `_validate_one_request` — length
   limits, vocab range, multimodal caps, logprob constraints. Each check exists because
   something downstream would otherwise crash a process shared by every other request.

3. **Turning messages into futures.**
   `python/sglang/srt/managers/tokenizer_manager.py:1722` `_wait_one_response` is the per-
   request async generator; `:2200` `handle_loop` and `:2215` `_handle_batch_output` are
   the return path that resolves it. This pair is the whole trick.

4. **The wire contract.** `python/sglang/srt/managers/io_struct.py:160`
   `GenerateReqInput`, `:941` `TokenizedGenerateReqInput`, `:1404` `BatchTokenIDOutput`,
   `:1504` `BatchStrOutput` — the messages define the process boundaries more precisely
   than any diagram.

5. **ZeroMQ patterns and their failure modes.** Socket setup at
   `python/sglang/srt/managers/scheduler.py:733` `init_ipc_channels`; receipt at `:1872`
   `process_input_requests` and `:1523` `init_request_dispatcher` (the type-to-handler
   table). Broadcast semantics under TP: rank 0 receives, all ranks must agree.
   Serialization choices, and the CUDA-IPC path at `:1906` `_materialize_cuda_vmm_inputs`
   that keeps image tensors off the socket entirely.

# Summary

[Introduction](./introduction.md)

# Part I — Foundations

- [Why Serving Engines Exist](./ch01-why-serving-engines.md)
- [The Shape of SGLang](./ch02-shape-of-sglang.md)

# Part II — The Request Path

- [From HTTP to Token IDs](./ch03-http-to-token-ids.md)
- [The Scheduler Loop](./ch04-scheduler-loop.md)
- [Deciding What Runs Next](./ch05-deciding-what-runs.md)
- [Executing a Batch](./ch06-executing-a-batch.md)
- [Sampling and the Return Path](./ch07-sampling-and-return.md)

# Part III — Memory and Caching

- [KV Cache Pools and Allocators](./ch08-kv-pools.md)
- [RadixAttention](./ch09-radixattention.md)
- [Caching Beyond HBM](./ch10-beyond-hbm.md)

# Part IV — The Model Layer

- [Loading and Updating Weights](./ch11-loading-weights.md)
- [Anatomy of a Model](./ch12-anatomy-of-a-model.md)
- [Attention Backends](./ch13-attention-backends.md)
- [Making the Forward Pass Cheap](./ch14-cheap-forward-pass.md)

# Part V — Scaling Out

- [Tensor, Pipeline, and Data Parallelism](./ch15-parallelism.md)
- [Mixture-of-Experts and Expert Parallelism](./ch16-moe.md)
- [Disaggregation and Routing](./ch17-disaggregation.md)

# Part VI — Beyond Plain Decoding

- [Speculative Decoding](./ch18-speculative-decoding.md)
- [Shaping and Reading the Output](./ch19-shaping-output.md)
- [Per-Request Variation: LoRA and Multimodal](./ch20-per-request-variation.md)

# Part VII — Operating and Extending

- [Observability and Tuning](./ch21-observability.md)
- [Extending SGLang](./ch22-extending.md)

# Appendices

- [Server Arguments](./appendix-a-server-args.md)
- [Environment Variables](./appendix-b-env-vars.md)
- [Glossary](./appendix-c-glossary.md)
- [Repository Map](./appendix-d-repo-map.md)
- [Annotated Startup Log](./appendix-e-startup-log.md)
- [Further Reading](./appendix-f-further-reading.md)

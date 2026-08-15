# Summary

[About This Book](./introduction.md)

---

- [Part I — Why Any of This Exists](./part1.md)
  - [1. Why Serving Engines Exist](./ch01-why-serving-engines.md)
  - [2. The Machine Underneath](./ch02-the-machine-underneath.md)
  - [3. The Shape of SGLang](./ch03-shape-of-sglang.md)
- [Part II — The Life of a Request](./part2.md)
  - [4. From HTTP to Token IDs](./ch04-http-to-token-ids.md)
  - [5. The Scheduler Loop](./ch05-scheduler-loop.md)
  - [6. Deciding What Runs Next](./ch06-deciding-what-runs.md)
  - [7. Executing a Batch](./ch07-executing-a-batch.md)
  - [8. Sampling and the Return Path](./ch08-sampling-and-return.md)
- [Part III — The Scarce Resource](./part3.md)
  - [9. KV Cache Pools and Allocators](./ch09-kv-pools.md)
  - [10. RadixAttention](./ch10-radixattention.md)
  - [11. Caching Beyond HBM](./ch11-beyond-hbm.md)
- [Part IV — What Actually Runs](./part4.md)
  - [12. Loading and Updating Weights](./ch12-loading-weights.md)
  - [13. Anatomy of a Model](./ch13-anatomy-of-a-model.md)
  - [14. Attention Backends](./ch14-attention-backends.md)
  - [15. Making the Forward Pass Cheap](./ch15-cheap-forward-pass.md)
- [Part V — When One GPU Is Not Enough](./part5.md)
  - [16. Tensor, Pipeline, and Data Parallelism](./ch16-parallelism.md)
  - [17. Mixture-of-Experts and Expert Parallelism](./ch17-moe.md)
  - [18. Disaggregation and Routing](./ch18-disaggregation.md)
- [Part VI — Beyond Plain Generation](./part6.md)
  - [19. Speculative Decoding](./ch19-speculative-decoding.md)
  - [20. Shaping and Reading the Output](./ch20-shaping-output.md)
  - [21. Per-Request Variation: LoRA and Multimodal](./ch21-per-request-variation.md)
- [Part VII — Living With It](./part7.md)
  - [22. Observability and Tuning](./ch22-observability.md)
  - [23. Extending SGLang](./ch23-extending.md)
  - [24. Running It](./ch24-running-it.md)

---

# Appendices

- [A. Server Arguments](./appendix-a-server-args.md)
- [B. Environment Variables](./appendix-b-env-vars.md)
- [C. Glossary](./appendix-c-glossary.md)
- [D. Repository Map](./appendix-d-repo-map.md)
- [E. Reading a Startup Log](./appendix-e-startup-log.md)
- [F. Further Reading](./appendix-f-further-reading.md)

---

[Colophon](./colophon.md)

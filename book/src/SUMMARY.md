# Summary

[About This Book](./introduction.md)

---

- [Part I — Why Any of This Exists](./part1.md)
  - [1. Why Serving Engines Exist](./ch01-why-serving-engines.md)
  - [2. The Shape of SGLang](./ch02-shape-of-sglang.md)
- [Part II — The Life of a Request](./part2.md)
  - [3. From HTTP to Token IDs](./ch03-http-to-token-ids.md)
  - [4. The Scheduler Loop](./ch04-scheduler-loop.md)
  - [5. Deciding What Runs Next](./ch05-deciding-what-runs.md)
  - [6. Executing a Batch](./ch06-executing-a-batch.md)
  - [7. Sampling and the Return Path](./ch07-sampling-and-return.md)
- [Part III — The Scarce Resource](./part3.md)
  - [8. KV Cache Pools and Allocators](./ch08-kv-pools.md)
  - [9. RadixAttention](./ch09-radixattention.md)
  - [10. Caching Beyond HBM](./ch10-beyond-hbm.md)
- [Part IV — What Actually Runs](./part4.md)
  - [11. Loading and Updating Weights](./ch11-loading-weights.md)
  - [12. Anatomy of a Model](./ch12-anatomy-of-a-model.md)
  - [13. Attention Backends](./ch13-attention-backends.md)
  - [14. Making the Forward Pass Cheap](./ch14-cheap-forward-pass.md)
- [Part V — When One GPU Is Not Enough](./part5.md)
  - [15. Tensor, Pipeline, and Data Parallelism](./ch15-parallelism.md)
  - [16. Mixture-of-Experts and Expert Parallelism](./ch16-moe.md)
  - [17. Disaggregation and Routing](./ch17-disaggregation.md)
- [Part VI — Beyond Plain Generation](./part6.md)
  - [18. Speculative Decoding](./ch18-speculative-decoding.md)
  - [19. Shaping and Reading the Output](./ch19-shaping-output.md)
  - [20. Per-Request Variation: LoRA and Multimodal](./ch20-per-request-variation.md)
- [Part VII — Living With It](./part7.md)
  - [21. Observability and Tuning](./ch21-observability.md)
  - [22. Extending SGLang](./ch22-extending.md)

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

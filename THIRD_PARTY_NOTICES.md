# Third-party notices

- `eval/lmms-eval/` is adapted from the lmms-eval source distribution.
  Its original LICENSE and CITATION.cff are retained in that directory.
- `scaler/modeling/my_modeling_qwen2_5_vl.py` and
  `my_configuration_qwen2_5_vl.py` derive from Qwen2.5-VL in Transformers.
  Their Qwen / Hugging Face copyright notices and Apache-2.0 headers are
  retained. Changes implement the visual detransformer and auxiliary path.
- The dog photograph is supplied as an inference example. Code licenses do
  not relicense the photograph or external benchmark datasets.
- The bicycle example is an HR-Bench-8K benchmark sample; its source
  identifiers are in `inference/examples/hrbench8k_bicycles.json`. The
  benchmark image retains its original terms.

Release changes: package naming, configurable filesystem paths and judge
credentials, shared inference entry points, unrelated adapter/task removal,
minimal installation requirements, and durable evaluation export. The core
model equations and historical full-latent generation patch are preserved.

# lmms-eval for SCALER

See [installation and reproduction commands](../README.md).

The original evaluator, task manager, model API, caching and five benchmark
utilities are retained. Model registration exposes only `scaler`. Unrelated
model adapters, benchmark tasks and service frontends are removed.

Use `run_scaler.py`, `prepare_data.py` and `score_scaler.py` for the documented
reproduction protocol. `lmms-eval models` and `lmms-eval tasks list` expose the
retained framework registry. Generic evaluation requires configuring the
native YAML dataset paths and uses upstream task scoring semantics.

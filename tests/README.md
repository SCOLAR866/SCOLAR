# Validation

Run unit tests from the repository root after installation:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

To verify detransformer reconstruction and two-phase generation on a GPU with a local checkpoint:

```bash
python tests/validate_model.py --checkpoint /path/to/checkpoint \
  --data data/prepared --output outputs/model-contract.json
```

To verify stochastic sampling (`temperature=1`) with full-resolution auxiliary latents:

```bash
python tests/validate_sampling.py --checkpoint /path/to/checkpoint \
  --output outputs/sampling-contract.json
```

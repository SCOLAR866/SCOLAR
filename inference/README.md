# Inference

Run all commands from the repository root using Python 3.12 on Linux with an NVIDIA GPU. The validated environment uses PyTorch 2.7.1 (CUDA 12.6), Transformers 4.57.1, and FlashAttention 2.8.3.

```bash
python3.12 -m venv env
source env/bin/activate
python -m pip install -r inference/requirements.txt
python -m pip install packaging ninja wheel
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install -e . --no-deps
```

If you prefer a prebuilt FlashAttention 2.8.3 wheel matching Python 3.12, PyTorch 2.7, CUDA 12, and the PyTorch C++ ABI:

```bash
python -m pip install /path/to/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl --no-deps
```

For quick testing without FlashAttention, pass `--attention sdpa`.

## Running the Included Examples

Dog coat color example:

```bash
python inference/run.py --model /path/to/checkpoint \
  --case inference/examples/dog.json --output outputs/dog.json
```

Question: `what's the color of the dog` → Output: `tan`.

HRBench-8K overlapping bicycle counting example:

```bash
python inference/run.py --model /path/to/checkpoint \
  --case inference/examples/hrbench8k_bicycles.json --output outputs/bicycles.json
```

Question: `How many bicycles are visible near the building's entrance?` → Output: `C` (Three).

## Custom Image and Question

```bash
python inference/run.py --model /path/to/checkpoint \
  --image /path/to/image.png --question "Describe the image."
```

Default settings use greedy decoding (`temperature=0`, `do_sample=False`), seed `1234`, natural `<auxiliary>` triggering, full-resolution auxiliary latents, and 512/512 Phase-1/Phase-2 text token budgets (`min_pixels=3136`, `max_pixels=4194304`). For stochastic sampling (`top_p=1`, `top_k=0`), pass `--temperature 1 --seed 2026`.

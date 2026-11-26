# Evaluation

All commands below run from the repository root. Evaluation code is located under `eval/lmms-eval/`, which integrates the `scaler` model adapter and the five evaluation benchmarks (`vstar_bench`, `hrbench4k`, `hrbench8k`, `mmerealworld_lite`, and `VisualPuzzles_direct`).

## Installation

```bash
python3.12 -m venv env
source env/bin/activate
python -m pip install -r eval/requirements.txt
python -m pip install packaging ninja wheel
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install -e . --no-deps
python -m pip install -e eval/lmms-eval --no-deps
```

## Dataset Preparation

Create a JSON manifest mapping the five benchmark names to local Parquet files (or ordered lists of Parquet files):

```json
{
  "vstar_bench": "data/vstar.parquet",
  "hrbench4k": "data/hrbench4k.parquet",
  "hrbench8k": "data/hrbench8k.parquet",
  "mmerealworld_lite": "data/mme-realworld-lite.parquet",
  "VisualPuzzles_direct": "data/visualpuzzles.parquet"
}
```

Prepare the datasets once into a target directory:

```bash
python eval/lmms-eval/prepare_data.py --manifest datasets.json --output data/prepared
```

The five benchmarks contain 191 (`vstar_bench`), 800 (`hrbench4k`), 800 (`hrbench8k`), 1,919 (`mmerealworld_lite`), and 1,168 (`VisualPuzzles_direct`) evaluation samples (4,878 total).

## Running Generation and Scoring

Single-GPU evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python eval/lmms-eval/run_scaler.py \
  --model /path/to/checkpoint --data data/prepared --output outputs/eval
```

For multi-GPU evaluation across $N$ GPUs, launch $N$ worker processes with `--shards N` and `--shard 0` through `--shard N-1` pointing to the same output directory. Interrupted shards automatically resume from existing progress.

To score generated predictions (with optional OpenAI-compatible option matching for free-form responses):

```bash
export JUDGE_BASE_URL="https://your-judge-service/v1"
export JUDGE_MODEL="your-judge-model"
export JUDGE_API_KEY="your-api-key"
python eval/lmms-eval/score_scaler.py \
  --data data/prepared --output outputs/eval --concurrency 16 --watch
```

`metrics.json` and `metrics.csv` report overall and per-subset accuracy across all five benchmarks, including detailed MME-RealWorld-Lite category breakdowns and latent token statistics.

"""Durable, independent GPU shards using the retained lmms-eval task prompts."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback

from scaler.core import load_model
from scaler.generation import generate
from scaler_benchmarks import TASKS, task_utils, prompt_image, score


def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    tmp.replace(path)


def main(a):
    import torch
    import fcntl
    from datasets import load_from_disk
    torch.set_num_threads(2)
    if not 0 <= a.shard < a.shards:
        raise ValueError('Require 0 <= shard < shards')
    folder = a.output / f'shard-{a.shard:03d}'
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    signature = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    manifest = folder / 'config.json'
    if manifest.exists() and json.loads(manifest.read_text())['signature'] != signature:
        raise ValueError('Resume configuration changed; choose a new output directory')
    atomic(manifest, dict(signature=signature, arguments=cfg))
    data = {t: load_from_disk(str(a.data / t)) for t in a.tasks}
    counts = {t: min(len(ds), a.limit) if a.limit else len(ds) for t, ds in data.items()}
    atomic(folder / 'counts.json', counts)
    jobs = [(t, i) for t in a.tasks for i in range(counts[t])][a.shard::a.shards]
    file = folder / 'samples.jsonl'
    completed = set()
    if file.exists():
        for line in file.read_text().splitlines():
            r = json.loads(line)
            key = r['task'], r['doc_id']
            if key in completed:
                raise ValueError('Duplicate sample in resume file')
            completed.add(key)
        if not completed <= set(jobs):
            raise ValueError('Unexpected sample in resume file')
    (folder / 'error.json').unlink(missing_ok=True)
    atomic(folder / 'status.json', dict(state='loading', pid=os.getpid(), done=len(completed), total=len(jobs)))
    model, processor, info = load_model(a.model)
    atomic(folder / 'load_info.json', dict(info=info,
        packages={p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'flash-attn']},
        model_path=a.model, strict_load=True))
    utils = {t: task_utils(t) for t in a.tasks}
    with file.open('a', buffering=1) as f:
        for t, i in jobs:
            if (t, i) in completed:
                continue
            doc = data[t][i]
            question, image = prompt_image(t, doc, utils[t])
            seed = a.seed + TASKS.index(t) * 10000000 + i
            result = generate(model, processor, image, question, seed=seed,
                              temperature=a.temperature)
            result.update(task=t, doc_id=i, source_id=doc.get('question_id', doc.get('index', i)),
                          question=question, model_path=a.model,
                          **score(t, doc, result['raw_output'], utils[t]))
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
            completed.add((t, i))
            atomic(folder / 'status.json', dict(state='running', pid=os.getpid(), done=len(completed),
                total=len(jobs), task=t, doc_id=i, updated_at=time.time()))
            print(t, i, result['prediction'], 'trigger', result['auxiliary_triggered'], flush=True)
    atomic(folder / 'status.json', dict(state='complete', done=len(completed), total=len(jobs), updated_at=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--tasks', nargs='+', choices=TASKS, default=TASKS)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--temperature', type=float, default=0)
    p.add_argument('--limit', type=int, default=None, help='Smoke test only; first N documents per task')
    a = p.parse_args()
    try:
        if a.limit is not None and a.limit < 1:
            p.error('--limit must be positive')
        main(a)
    except BaseException as e:
        atomic(a.output / f'shard-{a.shard:03d}' / 'error.json',
               dict(error=repr(e), traceback=traceback.format_exc(), time=time.time()))
        raise

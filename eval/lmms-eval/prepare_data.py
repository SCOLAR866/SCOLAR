"""Prepare local benchmark Parquet files without changing row order or prompts."""
import argparse
import json
from pathlib import Path

from datasets import load_dataset, load_from_disk

TASKS = ['vstar_bench', 'hrbench4k', 'hrbench8k', 'mmerealworld_lite', 'VisualPuzzles_direct']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path,
                   help='JSON mapping each task to one Parquet path or an ordered list of paths')
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    sources = json.loads(a.manifest.read_text())
    if set(sources) != set(TASKS):
        raise ValueError('Manifest must contain exactly: ' + ', '.join(TASKS))
    a.output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for task in TASKS:
        paths = sources[task]
        if isinstance(paths, str):
            paths = [paths]
        paths = [str((a.manifest.parent / x).resolve()) for x in paths]
        if not paths or not all(Path(x).is_file() for x in paths):
            raise FileNotFoundError(f'Missing Parquet files for {task}')
        dest = a.output / task
        if dest.exists():
            raise FileExistsError(f'{dest}: use a fresh output to avoid stale datasets')
        ds = load_dataset('parquet', data_files=paths, split='train', cache_dir=str(a.output / 'cache'))
        ds.save_to_disk(str(dest))
        counts[task] = len(ds)
    (a.output / 'counts.json').write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()

"""GPU checks for natural triggering, independent sampling with full latent embeddings."""
import argparse
import json
from pathlib import Path

from PIL import Image
import torch

from scaler.core import load_model
from scaler.generation import generate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--case', type=Path, default=Path(__file__).resolve().parents[1] /
                   'inference/examples/dog.json')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    case = json.loads(a.case.read_text())
    image = Image.open(a.case.parent / case['image'])
    model, processor, _ = load_model(a.checkpoint)
    results = []
    for i in range(3):
        count = [0]
        fresh_prefix_calls = [0]
        original = torch.multinomial
        original_generate = model.generate_vanilla
        def counted(*args, **kwargs):
            count[0] += 1
            return original(*args, **kwargs)
        def counted_generate(*args, **kwargs):
            if 'pixel_values' in kwargs and 'inputs_embeds' not in kwargs:
                fresh_prefix_calls[0] += 1
            return original_generate(*args, **kwargs)
        torch.multinomial = counted
        model.generate_vanilla = counted_generate
        try:
            result = generate(model, processor, image, case['question'],
                              temperature=1, seed=2026+i)
        finally:
            torch.multinomial = original
            model.generate_vanilla = original_generate
        assert count[0] > 0, 'Sampling was not used'
        assert fresh_prefix_calls[0] >= 1, 'The question prefix was not generated afresh'
        assert result['prefix_reused'] is False
        assert result['latent_tokens'] == (result['native_latent_tokens'] if result['auxiliary_triggered'] else 0)
        result['multinomial_calls'] = count[0]
        result['fresh_prefix_generation_calls'] = fresh_prefix_calls[0]
        results.append(result)
    assert len({r['seed'] for r in results}) == 3
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(passed=True, samples=results), ensure_ascii=False, indent=2))
    print(json.dumps([dict(seed=r['seed'], triggered=r['auxiliary_triggered'],
                          latent_tokens=r['latent_tokens'], sampling_calls=r['multinomial_calls'])
                      for r in results]))


if __name__ == '__main__':
    main()

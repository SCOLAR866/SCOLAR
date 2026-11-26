"""Run an image/question case with SCALER."""
import argparse
import json
from pathlib import Path

from PIL import Image
from scaler.core import answer_text, load_model
from scaler.generation import generate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, help='Local checkpoint directory')
    p.add_argument('--image', type=Path)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--question')
    group.add_argument('--question-file', type=Path)
    p.add_argument('--case', type=Path, help='JSON with image and question; image path relative to JSON')
    p.add_argument('--output', type=Path, default=Path('outputs/case.json'))
    p.add_argument('--temperature', type=float, default=0)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--attention', choices=['flash_attention_2', 'sdpa', 'eager'], default='flash_attention_2')
    a = p.parse_args()
    if a.case:
        if a.image or a.question or a.question_file:
            p.error('--case cannot be combined with --image/--question/--question-file')
        case = json.loads(a.case.read_text())
        a.image, question = a.case.parent / case['image'], case['question']
    else:
        if not a.image or not (a.question or a.question_file):
            p.error('Supply --case or --image and --question/--question-file')
        question = a.question or a.question_file.read_text()
    model, processor, info = load_model(a.model, a.device, a.attention)
    result = generate(model, processor, Image.open(a.image), question,
                      temperature=a.temperature, seed=a.seed)
    result.update(answer=answer_text(result['raw_output']), model_path=a.model, load_info=info)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: result[k] for k in ['answer', 'auxiliary_triggered', 'latent_tokens', 'seed']}, ensure_ascii=False))


if __name__ == '__main__':
    main()

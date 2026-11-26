import importlib.util
import sys
import re
from pathlib import Path
from scaler.core import answer_text
ROOT = Path(__file__).resolve().parent
TASKS = ['vstar_bench', 'hrbench4k', 'hrbench8k', 'mmerealworld_lite', 'VisualPuzzles_direct']

def module(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def task_utils(task):
    directory = {'vstar_bench':'vstar_bench', 'hrbench4k':'hrbench', 'hrbench8k':'hrbench',
                 'mmerealworld_lite':'mme_realworld', 'VisualPuzzles_direct':'VisualPuzzles'}[task]
    if directory == 'hrbench':
        module('lmms_eval.tasks.hrbench.hrbench_evals', ROOT/'lmms_eval/tasks/hrbench/hrbench_evals.py')
    return module('sft_eval_utils_'+directory, ROOT/'lmms_eval/tasks'/directory/'utils.py')

def prompt_image(task, doc, u):
    if task == 'vstar_bench':
        return u.vstar_doc_to_text(doc, {'post_prompt':"\nAnswer with the option's letter from the given choices directly."}), u.vstar_doc_to_visual(doc)[0]
    if task.startswith('hrbench'):
        return u.hrbench_doc_to_text(doc), u.hrbench_doc_to_visual(doc)[0]
    if task == 'mmerealworld_lite':
        return u.mme_realworld_doc_to_text(doc), u.mme_realworld_doc_to_visual(doc)[0]
    return u.VisualPuzzles_doc_to_text(doc, {'prompt':'MULTI_CHOICE_DIRECT_PROMPT'}), u.VisualPuzzles_doc_to_visual(doc)[0]

def score(task, doc, raw, u):
    text = answer_text(raw)
    gt = str(doc.get('label', doc.get('answer', ''))).strip().upper()
    # Preserve task parsers, but never let an English word beginning in A-D
    # or random fallback silently turn an unparsed reasoning trace into a guess.
    explicit = re.fullmatch(r'\s*\(?([A-E])\)?[.\s]*', text, flags=re.I)
    pred = explicit.group(1).upper() if explicit else None
    if pred is None and task.startswith('hrbench'):
        pred = u.hrbench_evaluator.can_infer(text, u.hrbench_doc_to_options(doc)) or None
    elif pred is None:
        letters = re.findall(r'(?:answer|choice|option)\s*(?:is|:)?\s*\(?([A-E])\b', text, re.I)
        if len(set(x.upper() for x in letters)) == 1:
            pred = letters[-1].upper()
    return dict(scoring_text=text, prediction=pred, target=gt,
                score=(float(str(pred).upper() == gt) if pred else None),
                needs_judge=pred is None,
                category=doc.get('category'), cycle_category=doc.get('cycle_category'),
                l2_category=doc.get('l2-category'))


"""Regression checks for full generation and evaluation accounting."""
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval/lmms-eval'))
import score_scaler
from scaler.core import answer_text
from scaler.generation import generate


class ProtocolTests(unittest.TestCase):
    def test_final_answer_excludes_reasoning(self):
        self.assertEqual(answer_text('<think>Maybe A.</think><answer>B</answer>'), 'B')
        self.assertEqual(answer_text('<think>A</think>C'), 'C')

    def test_full_generation_trigger_and_decoding(self):
        import torch
        from contextlib import ExitStack
        from unittest.mock import Mock
        from transformers import BatchFeature

        inputs = BatchFeature(dict(input_ids=torch.tensor([[10, 11]]),
            image_grid_thw=torch.tensor([[1, 4, 6]])))
        model = types.SimpleNamespace(device='cpu',
            config=types.SimpleNamespace(auxiliary_start_token_id=42),
            generation_config=types.SimpleNamespace(eos_token_id=99),
            model=types.SimpleNamespace(rope_deltas=None,
                visual=types.SimpleNamespace(spatial_merge_size=2)))
        processor = types.SimpleNamespace(tokenizer=types.SimpleNamespace(
            pad_token_id=0, decode=lambda ids, **kw: str(ids)))
        for temperature, tokens in [(0, [42, 99]), (1, [42, 99]),
                                    (0, [7, 99]), (0, [7, 8])]:
            with self.subTest(temperature=temperature, tokens=tokens), ExitStack() as stack:
                stack.enter_context(patch('scaler.generation.build_inputs', return_value=(inputs, 'question')))
                for name in ['reset_peak_memory_stats', 'synchronize', 'max_memory_allocated']:
                    stack.enter_context(patch('torch.cuda.' + name, return_value=0))
                model.generate = Mock(return_value=torch.tensor([[10, 11] + tokens]))
                result = generate(model, processor, None, 'question', temperature=temperature)
                kw = model.generate.call_args.kwargs
                self.assertEqual(kw['input_ids'].tolist(), [[10, 11]])
                self.assertEqual(kw['do_sample'], temperature > 0)
                self.assertEqual(kw['num_beams'], 1)
                self.assertEqual((kw['max_new_tokens_phase1'], kw['max_new_tokens_phase2']), (512, 512))
                if temperature:
                    self.assertEqual((kw['temperature'], kw['top_p'], kw['top_k']), (1, 1, 0))
                else:
                    self.assertNotIn('temperature', kw)
                self.assertEqual(result['latent_tokens'], 6 if 42 in tokens else 0)
                self.assertEqual(result['auxiliary_triggered'], 42 in tokens)
                expected = None if 42 in tokens else ('eos_before_auxiliary' if tokens[-1] == 99 else 'phase1_token_limit')
                self.assertEqual(result['not_trigger_reason'], expected)

    def test_pending_not_counted_in_partial_denominator(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {}, clear=True):
            root = Path(tmp)
            (root/'shard-000').mkdir()
            rows = [dict(task='vstar_bench', doc_id=i, score=score, prompt='Q', scoring_text='?',
                         target='A', cycle_category=None) for i, score in enumerate([1.0, None])]
            (root/'shard-000/samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            meta = {('vstar_bench', i): {'category':'Attribute'} for i in range(3)}
            args = types.SimpleNamespace(output=root, concurrency=1)
            result = score_scaler.update(args, meta, {('vstar_bench','Overall'):3})
            m = result['metrics'][0]
            self.assertEqual((m['n'], m['generated'], m['scored'], m['correct'], m['pending']), (3,2,1,1,2))
            self.assertEqual(m['accuracy_percent'], 100)
            self.assertFalse(m['complete'])

    def test_mme_hierarchy(self):
        labels = score_scaler.subset_labels('mmerealworld_lite',
            {'category':'Perception/Monitoring', 'l2-category':'vehicle/location'})
        self.assertEqual(labels, ['Overall', 'Ability/Perception', 'Domain/Monitoring',
            'Category/Perception/Monitoring', 'Fine/Perception/Monitoring/vehicle/location'])

    def test_judge_transport_error_stays_pending(self):
        opener = types.SimpleNamespace(open=lambda *a, **kw: (_ for _ in ()).throw(TimeoutError()))
        with patch('urllib.request.build_opener', return_value=opener), patch('time.sleep'):
            result = score_scaler.judge(dict(prompt='Q', question='Q', scoring_text='unknown', target='A'),
                                       'http://localhost/v1', 'test-model', 'test-key')
        self.assertIn('error', result)
        self.assertNotIn('prediction', result)
        self.assertNotIn('score', result)

    def test_judge_invalid_response_gets_quoted_retry(self):
        payloads = []
        def request(req, **kw):
            payloads.append(json.loads(req.data))
            text = 'unparseable response' if len(payloads)==1 else 'B'
            return io.BytesIO(json.dumps({'choices':[{'message':{'content':text}}]}).encode())
        with patch('urllib.request.build_opener', return_value=types.SimpleNamespace(open=request)), patch('time.sleep'):
            result = score_scaler.judge(dict(prompt='Q: A. red B. blue', question='Q: A. red B. blue',
                scoring_text='blue', target='A'), 'http://localhost/v1', 'test-model', 'test-key')
        self.assertEqual(result['prediction'], 'B')
        quoted = json.loads(payloads[1]['messages'][1]['content'])
        self.assertEqual(quoted, {'question':'Q: A. red B. blue', 'answer':'blue'})
        self.assertEqual(payloads[1]['messages'][0]['role'], 'system')


if __name__ == '__main__':
    unittest.main()

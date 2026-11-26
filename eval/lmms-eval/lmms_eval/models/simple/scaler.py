"""SCALER adapter for the original lmms-eval model interface."""
from accelerate import Accelerator
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from scaler.core import load_model
from scaler.generation import generate


@register_model('scaler')
class Scaler(lmms):
    def __init__(self, pretrained, batch_size=1, device='cuda:0', seed=1234,
                 max_phase1_tokens=512, max_phase2_tokens=512, **kwargs):
        super().__init__()
        if kwargs:
            raise ValueError(f'Unsupported model arguments: {sorted(kwargs)}')
        if int(batch_size) != 1:
            raise ValueError('SCALER currently supports batch_size=1 per GPU')
        self.accelerator = Accelerator()
        self._rank = self.accelerator.process_index
        self._world_size = self.accelerator.num_processes
        if self._world_size > 1:
            device = f'cuda:{self.accelerator.local_process_index}'
        self.model, self.processor, _ = load_model(pretrained, device=device)
        self.seed = int(seed)
        self.max_phase1_tokens, self.max_phase2_tokens = int(max_phase1_tokens), int(max_phase2_tokens)

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return self.model.device

    def loglikelihood(self, requests):
        raise NotImplementedError('SCALER exposes generation only')

    def generate_until_multi_round(self, requests):
        raise NotImplementedError('SCALER exposes single image, single turn inference')

    def generate_until(self, requests):
        outputs = []
        for request in requests:
            question, kw, doc_to_visual, doc_id, task, split = request.args
            images = doc_to_visual(self.task_dict[task][split][doc_id])
            if len(images) != 1:
                raise ValueError('Expected one image per request')
            result = generate(self.model, self.processor, images[0], question,
                seed=self.seed + int(doc_id),
                temperature=float(kw.get('temperature', 1)) if kw.get('do_sample', False) else 0,
                top_p=float(kw.get('top_p', 1)), top_k=int(kw.get('top_k', 0)),
                max_phase1_tokens=self.max_phase1_tokens, max_phase2_tokens=self.max_phase2_tokens)
            outputs.append(result['raw_output'])
            self.cache_hook.add_partial('generate_until', (question, kw), outputs[-1])
        return outputs

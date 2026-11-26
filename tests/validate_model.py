"""Model forward and two-phase generation verification on a local checkpoint."""
import argparse
from pathlib import Path
import json
import torch
from datasets import load_from_disk
from transformers import LogitsProcessor, LogitsProcessorList
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'eval/lmms-eval'))
from scaler.core import load_model, build_inputs
from scaler_benchmarks import prompt_image, task_utils
from run_scaler import atomic

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--data', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
torch.set_num_threads(4)
model, processor, info = load_model(a.checkpoint)
u = task_utils('vstar_bench')
doc = load_from_disk(str(Path(a.data)/'vstar_bench'))[0]
question, image = prompt_image('vstar_bench', doc, u)
inp, text = build_inputs(processor, image, question)
inp = inp.to('cuda:0')
aux = model.config.auxiliary_start_token_id
prefix_ids = torch.cat([inp.input_ids, inp.input_ids.new_tensor([[aux]])], dim=1)
mask = torch.ones_like(prefix_ids)
with torch.inference_mode():
    # Independent reconstruction of the exact the training forward pass equations.
    base = torch.cat(model.get_image_features(inp.pixel_values, inp.image_grid_thw))
    imask = prefix_ids == model.config.image_token_id
    emb = model.get_input_embeddings()(prefix_ids)
    emb = emb.masked_scatter(imask[...,None].expand_as(emb), base.to(emb.dtype))
    pos, _ = model.model.get_rope_index(prefix_ids, inp.image_grid_thw, attention_mask=mask)
    prefix = model.language_model(inputs_embeds=emb, attention_mask=mask.bool(), position_ids=pos,
        output_hidden_states=True, use_cache=False, return_dict=True)
    generated = model.visual_detransformer(prefix.hidden_states[20], attention_mask=mask.bool())[imask] + base
    inferred = model.pass_through_visual_detransformer(input_ids=prefix_ids, attention_mask=mask,
        pixel_values=inp.pixel_values, image_grid_thw=inp.image_grid_thw, use_cache=False)
    aux_start = prefix_ids.shape[1]+1
    observed = inferred['inputs_embeds'][0, aux_start:aux_start+len(generated)]
    torch.testing.assert_close(observed, generated, rtol=0, atol=0)
    expected_ids = torch.cat([prefix_ids, prefix_ids.new_tensor([[model.config.vision_start_token_id]+
        [model.config.image_token_id]*len(generated)+[model.config.vision_end_token_id,model.config.auxiliary_end_token_id]])], dim=1)
    assert torch.equal(expected_ids,inferred['input_ids'])
    assert torch.equal(inferred['image_grid_thw'], inp.image_grid_thw.repeat_interleave(2,dim=0))
    pos2,_ = model.model.get_rope_index(expected_ids, inferred['image_grid_thw'], attention_mask=inferred['attention_mask'])
    explicit = model(inputs_embeds=inferred['inputs_embeds'], attention_mask=inferred['attention_mask'],
        position_ids=pos2, use_cache=False, logits_to_keep=1).logits[:,-1,:]
    model.model.rope_deltas=None
    native = model.generate_vanilla(**inferred, do_sample=False, max_new_tokens=1,
        return_dict_in_generate=True, output_scores=True, eos_token_id=model.generation_config.eos_token_id)
    torch.testing.assert_close(native.scores[0], explicit.float(), rtol=0, atol=0)
    class ForceFirst(LogitsProcessor):
        def __init__(self, token): self.token=token
        def __call__(self, ids, scores):
            if ids.shape[1] == inp.input_ids.shape[1]:
                scores[:] = -float('inf'); scores[:,self.token] = 0
            return scores
    forced = model.generate(**inp, do_sample=False, max_new_tokens_phase1=1, max_new_tokens_phase2=2,
        logits_processor=LogitsProcessorList([ForceFirst(aux)]))
    assert torch.equal(forced[:,:inp.input_ids.shape[1]], inp.input_ids)
    assert forced[0,inp.input_ids.shape[1]] == aux
    assert forced[0,inp.input_ids.shape[1]+1] == native.sequences[0,-1]
    noaux = model.generate(**inp, do_sample=False, max_new_tokens_phase1=2, max_new_tokens_phase2=2,
        logits_processor=LogitsProcessorList([ForceFirst(model.config.eos_token_id)]))
    assert noaux.shape[1] == inp.input_ids.shape[1]+1
    normal = model.generate(**inp, do_sample=False, max_new_tokens_phase1=512,max_new_tokens_phase2=512)
    raw = processor.tokenizer.decode(normal[0,inp.input_ids.shape[1]:],skip_special_tokens=False)
atomic(a.output,dict(checkpoint=a.checkpoint, load_info=info, exact_latent_match=True,
    structural_ids_match=True, two_image_grid_match=True, phase2_logits_match=True,
    trigger_continuation_match=True, no_trigger_eos_match=True, image_grid=inp.image_grid_thw.tolist(),
    normal_response=raw, peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3))
print('VALIDATION PASSED',raw,flush=True)

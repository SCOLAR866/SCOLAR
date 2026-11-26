"""Shared generation for standalone cases and benchmark evaluation."""
import time

from .core import build_inputs


def generate(model, processor, image, question, *, seed=1234, temperature=0.0,
             top_p=1.0, top_k=0, max_phase1_tokens=512,
             max_phase2_tokens=512):
    """Generate with naturally triggered, unmodified full latent embeddings.

    Each call starts from the image and question, sets its seed once, and
    never prepends an auxiliary trigger or reseeds between generation phases.
    """
    import torch
    from transformers import set_seed

    if temperature < 0 or not 0 < top_p <= 1 or top_k < 0:
        raise ValueError('Invalid sampling parameters')
    if min(max_phase1_tokens, max_phase2_tokens) < 1:
        raise ValueError('Text token budgets must be positive')
    set_seed(seed)
    model.model.rope_deltas = None
    inputs, prompt = build_inputs(processor, image, question)
    inputs = inputs.to(model.device)
    length = inputs.input_ids.shape[1]
    aux = int(model.config.auxiliary_start_token_id)
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    kw = dict(do_sample=temperature > 0, num_beams=1, use_cache=True,
              eos_token_id=eos, pad_token_id=processor.tokenizer.pad_token_id)
    if temperature > 0:
        kw.update(temperature=temperature, top_p=top_p, top_k=top_k)
    native_n = int(inputs.image_grid_thw.prod().item()) // model.model.visual.spatial_merge_size**2
    started = time.time()
    torch.cuda.reset_peak_memory_stats(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens_phase1=max_phase1_tokens,
                             max_new_tokens_phase2=max_phase2_tokens, **kw)
        ids = out[0, length:].tolist()
        triggered = aux in ids
        actual_n = native_n if triggered else 0
    torch.cuda.synchronize(model.device)
    return dict(prompt=prompt, raw_output=processor.tokenizer.decode(ids, skip_special_tokens=False),
                output_token_ids=ids, input_token_ids=inputs.input_ids[0].tolist(),
                auxiliary_triggered=triggered,
                not_triggered=not triggered, not_trigger_reason=None if triggered else ('eos_before_auxiliary' if ids and ids[-1] in eos else 'phase1_token_limit'),
                latent_tokens=actual_n, native_latent_tokens=native_n,
                seed=seed, temperature=temperature,
                image_grid_thw=inputs.image_grid_thw.tolist(),
                elapsed_seconds=time.time() - started, input_tokens=length,
                peak_gpu_bytes=torch.cuda.max_memory_allocated(model.device),
                finish_reason='eos' if ids and ids[-1] in eos else 'length',
                independent_sample=True, prefix_reused=False)

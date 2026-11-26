import math
import re
SYSTEM = 'You are a helpful assistant. You can analyze images with auxiliary information like bounding boxes. You output visual analysis within <auxiliary></auxiliary> tags when referring to highlighted regions.'

def compress(image, max_pixels=4194304):
    """Exactly the training Dataset._compress_large_images rule; no JPEG recode."""
    from PIL import Image
    image = image.convert('RGB')
    if image.width * image.height > max_pixels:
        scale = math.sqrt(max_pixels / (image.width * image.height))
        image = image.resize((max(256, int(image.width*scale)), max(256, int(image.height*scale))), Image.Resampling.LANCZOS)
    return image

def build_inputs(processor, image, question):
    messages = [{'role':'system','content':[{'type':'text','text':SYSTEM}]},
                {'role':'user','content':[{'type':'image'},{'type':'text','text':question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[compress(image)], return_tensors='pt'), text

def answer_text(raw):
    answers = re.findall(r'<answer>\s*(.*?)\s*</answer>', raw, re.S | re.I)
    if answers:
        return answers[-1].strip()
    text = re.sub(r'<think>.*?</think>', '', raw, flags=re.S | re.I)
    return re.sub(r'<\|[^>]+\|>|</?auxiliary>', '', text).strip()

def load_model(checkpoint, device="cuda:0", attn_implementation="flash_attention_2"):
    from scaler.modeling import patch_transformers_qwen2_5_vl  # noqa
    from scaler.modeling.multistage_generate_patch import patch_multistage_generate
    patch_multistage_generate()
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model, info = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint, dtype=torch.bfloat16, device_map=device,
        attn_implementation=attn_implementation, local_files_only=True, output_loading_info=True)
    if any(info.get(key) for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise RuntimeError('Checkpoint load mismatch: '+str(info))
    model.eval()
    assert model.config.detransformer_config.fork_layer == 20
    assert model.config.detransformer_config.output_type == 'delta'
    assert model.config.auxiliary_start_token_id == processor.tokenizer.convert_tokens_to_ids('<auxiliary>')
    assert model.config.detransformer_config.gen_auxiliary_end_token_ids == [processor.tokenizer.convert_tokens_to_ids('</auxiliary>')]
    assert processor.image_processor.min_pixels == 3136
    assert processor.image_processor.max_pixels == 4194304
    return model, processor, info


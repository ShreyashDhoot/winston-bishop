import os
import threading
from typing import Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers import AutoProcessor
except Exception:  # pragma: no cover - optional dependency in older versions
    AutoProcessor = None


_DEFAULT_MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen3.5-27B")
_DEFAULT_CACHE_DIR = os.getenv("QWEN_CACHE_DIR")
_DEFAULT_TORCH_DTYPE = torch.bfloat16


class _ModelBundle:
    def __init__(self, model, tokenizer, processor):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor


_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: Dict[str, _ModelBundle] = {}


def _resolve_dtype() -> torch.dtype:
    if _DEFAULT_TORCH_DTYPE is torch.bfloat16:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return _DEFAULT_TORCH_DTYPE


def _load_bundle(model_id: str) -> _ModelBundle:
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]

    dtype = _resolve_dtype()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=_DEFAULT_CACHE_DIR,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
        cache_dir=_DEFAULT_CACHE_DIR,
    )
    model.eval()

    processor = None
    if AutoProcessor is not None:
        try:
            processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                cache_dir=_DEFAULT_CACHE_DIR,
            )
        except Exception:
            processor = None

    bundle = _ModelBundle(model=model, tokenizer=tokenizer, processor=processor)
    _MODEL_CACHE[model_id] = bundle
    return bundle


def _format_prompt(tokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def _move_to_device(inputs: Dict[str, Any], model) -> Dict[str, Any]:
    target_device = None
    if hasattr(model, "device"):
        target_device = model.device
    elif torch.cuda.is_available():
        target_device = torch.device("cuda")
    else:
        target_device = torch.device("cpu")

    moved = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(target_device)
        else:
            moved[key] = value
    return moved


def generate_response(
    prompt: str,
    *,
    model_id: Optional[str] = None,
    image=None,
    max_new_tokens: int = 128,
    temperature: float = 1e-6,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
) -> str:
    model_id = model_id or _DEFAULT_MODEL_ID

    with _MODEL_LOCK:
        bundle = _load_bundle(model_id)

        if image is not None:
            if bundle.processor is None:
                raise ValueError("Qwen processor not available for image inputs.")

            formatted_prompt = _format_prompt(bundle.tokenizer, prompt)
            processor = bundle.processor
            kwargs = {"text": formatted_prompt, "return_tensors": "pt"}
            if "images" in processor.__call__.__code__.co_varnames:
                kwargs["images"] = image
            elif "image" in processor.__call__.__code__.co_varnames:
                kwargs["image"] = image
            else:
                raise ValueError("Processor does not accept image inputs.")

            inputs = processor(**kwargs)
            input_ids = inputs.get("input_ids")
        else:
            formatted_prompt = _format_prompt(bundle.tokenizer, prompt)
            inputs = bundle.tokenizer(formatted_prompt, return_tensors="pt")
            input_ids = inputs.get("input_ids")

        inputs = _move_to_device(inputs, bundle.model)

        do_sample = temperature is not None and temperature > 0.0
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
        }
        if top_k is not None:
            generation_kwargs["top_k"] = top_k

        with torch.inference_mode():
            output_ids = bundle.model.generate(**inputs, **generation_kwargs)

        if input_ids is None:
            decoded = bundle.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return decoded.strip()

        prompt_length = input_ids.shape[-1]
        generated_ids = output_ids[0][prompt_length:]
        decoded = bundle.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return decoded.strip()

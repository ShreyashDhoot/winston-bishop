import os
import threading
from typing import Optional, Dict, Any, List, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers import AutoProcessor
except Exception:  # pragma: no cover - optional dependency in older versions
    AutoProcessor = None


_DEFAULT_MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen3.5-27B")


import re as _re

def _strip_thinking(text: str) -> str:
    """Strip Qwen3.5 <think>...</think> blocks from output.
    Belt-and-suspenders: even when enable_thinking=False is set correctly,
    some processor versions silently ignore it (warning: 'chat_template_kwargs
    is not a valid argument for this processor and will be ignored').
    """
    return _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
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


def _format_prompt(
    tokenizer,
    prompt: Union[str, List[Dict[str, str]]],
    enable_thinking: bool = False,
) -> str:
    """
    Format a prompt for the Qwen tokenizer.

    Accepts either:
      - a plain string  → wrapped in a single user turn
      - a list of {"role": ..., "content": ...} dicts → passed directly to
        apply_chat_template so multi-turn conversations are handled correctly
    """
    if isinstance(prompt, list):
        messages = prompt
    else:
        messages = [{"role": "user", "content": prompt}]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            # Disable thinking so the model emits the answer directly.
            # Qwen3.5 thinks by default; the <think>...</think> preamble can consume
            # hundreds of tokens before the actual label, breaking short max_new_tokens budgets.
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": enable_thinking},
            )
        except TypeError:
            # Older tokenizers / non-Qwen models don't accept chat_template_kwargs
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Fallback for tokenizers without chat template support
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


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

            processor = bundle.processor

            # Build a proper multimodal message with image embedded in the content list.
            # This ensures apply_chat_template inserts the required <|vision_start|><|image_pad|>
            # <|vision_end|> tokens into the text before the processor tokenizes it.
            # Passing a pre-formatted plain-text string + image separately (old approach) skips
            # image-token injection, so the model never attends to the image.
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                # Disable thinking mode: Qwen3.5 thinks by default, wrapping output in
                # <think>...</think> before the actual answer. With max_new_tokens=32
                # (or even 128) the model would exhaust the budget on thinking and never
                # emit the UNSAFE/SAFE/ALIGNED token we're parsing.
                chat_template_kwargs={"enable_thinking": False},
            )
            input_ids = inputs.get("input_ids")
        else:
            # Use apply_chat_template directly so enable_thinking=False is honoured.
            # Old code: _format_prompt → plain string → tokenizer(string) double-tokenizes
            # AND silently drops enable_thinking, so Qwen3.5 thinks and exhausts
            # max_new_tokens before emitting SAFE/UNSAFE/ALIGNED.
            if isinstance(prompt, list):
                messages = prompt
            else:
                messages = [{"role": "user", "content": prompt}]

            try:
                inputs = bundle.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    chat_template_kwargs={"enable_thinking": False},
                )
            except TypeError:
                # Non-Qwen tokenizers don't support chat_template_kwargs / return_dict
                inputs = bundle.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
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

        try:
            with torch.inference_mode():
                output_ids = bundle.model.generate(**inputs, **generation_kwargs)
        except ValueError as e:
            # Some text-only models reject vision kwargs; fall back to text-only inputs.
            if image is None or "model_kwargs" not in str(e):
                raise
            for key in (
                "mm_token_type_ids",
                "pixel_values",
                "image_grid_thw",
                "image_sizes",
                "vision_mask",
                "image_embeds",
            ):
                inputs.pop(key, None)
            with torch.inference_mode():
                output_ids = bundle.model.generate(**inputs, **generation_kwargs)

        if input_ids is None:
            decoded = bundle.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return _strip_thinking(decoded)

        prompt_length = input_ids.shape[-1]
        generated_ids = output_ids[0][prompt_length:]
        decoded = bundle.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return _strip_thinking(decoded)
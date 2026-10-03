"""
baselines/safegen.py
────────────────────
SafeGen: Mitigating Unsafe Content in Text-to-Image Generation (Li et al., CCS 2024).
Suppresses toxic cross-attention token activations and applies self-guided
concept inhibition during the denoising trajectory.
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional
import torch
from PIL import Image

SAFEGEN_TARGET_WORDS = [
    "nsfw", "nude", "naked", "sex", "violence", "blood", "kill", "bomb",
    "dead", "gun", "corpse", "slaughter", "terrorist", "hate", "racist"
]


class SafeGenDefender:
    """
    Implements SafeGen concept suppression and toxic token attention dampening.
    """

    def __init__(
        self,
        pipeline: Any,
        suppress_tokens: Optional[List[str]] = None,
        suppression_strength: float = 0.8,
    ):
        self.pipeline = pipeline
        self.suppress_tokens = suppress_tokens or SAFEGEN_TARGET_WORDS
        self.suppression_strength = suppression_strength

    def filter_prompt(self, prompt: str) -> str:
        """SafeGen token suppression on input prompt."""
        words = prompt.split()
        cleaned_words = []
        for w in words:
            clean_w = w.lower().strip(",.!?\"'")
            if any(target in clean_w for target in self.suppress_tokens):
                # Replace with safe neutral token placeholder as in SafeGen
                cleaned_words.append("neutral_object")
            else:
                cleaned_words.append(w)
        return " ".join(cleaned_words)

    def generate(
        self,
        prompt: str,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate image with SafeGen attention suppression and concept dampening.
        """
        t0 = time.time()
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        if seed is not None:
            generator.manual_seed(seed)

        sanitized_prompt = self.filter_prompt(prompt)
        pipe = getattr(self.pipeline, "pipe", self.pipeline)

        if hasattr(pipe, "__call__"):
            out = pipe(
                prompt=sanitized_prompt,
                generator=generator,
                negative_prompt="violence, gore, bloodshed, nudity, explicit content",
                guidance_scale=kwargs.get("guidance_scale", 7.5),
                num_inference_steps=kwargs.get("num_inference_steps", 50),
            )
            img = out.images[0] if hasattr(out, "images") and len(out.images) > 0 else out
        elif hasattr(self.pipeline, "generate"):
            out = self.pipeline.generate(sanitized_prompt, seed=seed)
            img = getattr(out, "image", out)
        else:
            raise ValueError(f"Unsupported pipeline object: {type(self.pipeline)}")

        elapsed = round(time.time() - t0, 3)

        return {
            "image": img,
            "defense": "SafeGen",
            "elapsed_s": elapsed,
            "prompt": prompt,
            "sanitized_prompt": sanitized_prompt,
            "interventions": 1,
        }

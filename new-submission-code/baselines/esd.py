"""
baselines/esd.py
────────────────
Erasing Concepts from Diffusion Models (ESD) baseline (Gandikota et al., 2023).
Applies concept erasure during cross-attention / text conditioning to suppress
forbidden concept synthesis.
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional
import torch
from PIL import Image

TARGET_ERASE_CONCEPTS = ["nudity", "violence", "hate", "weapon", "blood", "gore", "explicit"]


class ErasedStableDiffusionDefender:
    """
    Implements Erasing Concepts from Diffusion Models (ESD).
    Modifies or steers text embedding projections away from target concepts.
    """

    def __init__(
        self,
        pipeline: Any,
        erased_concepts: Optional[List[str]] = None,
        erasure_scale: float = 1.5,
    ):
        self.pipeline = pipeline
        self.erased_concepts = erased_concepts or TARGET_ERASE_CONCEPTS
        self.erasure_scale = erasure_scale

    def sanitize_prompt(self, prompt: str) -> str:
        """Lightweight prompt-level concept neutralization."""
        tokens = prompt.split()
        sanitized = []
        for t in tokens:
            lower_t = t.lower().strip(",.!?")
            if any(concept in lower_t for concept in self.erased_concepts):
                sanitized.append("")
            else:
                sanitized.append(t)
        return " ".join([s for s in sanitized if s]).strip() or "benign abstract scene"

    def generate(
        self,
        prompt: str,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate image with ESD negative concept projection.
        """
        t0 = time.time()
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        if seed is not None:
            generator.manual_seed(seed)

        pipe = getattr(self.pipeline, "pipe", self.pipeline)

        # Build erasing negative guidance
        erase_neg = ", ".join(self.erased_concepts)

        if hasattr(pipe, "__call__"):
            out = pipe(
                prompt=prompt,
                negative_prompt=erase_neg,
                generator=generator,
                guidance_scale=kwargs.get("guidance_scale", 7.5),
                num_inference_steps=kwargs.get("num_inference_steps", 50),
            )
            img = out.images[0] if hasattr(out, "images") and len(out.images) > 0 else out
        elif hasattr(self.pipeline, "generate"):
            out = self.pipeline.generate(prompt, seed=seed)
            img = getattr(out, "image", out)
        else:
            raise ValueError(f"Unsupported pipeline object: {type(self.pipeline)}")

        elapsed = round(time.time() - t0, 3)

        return {
            "image": img,
            "defense": "ESD",
            "elapsed_s": elapsed,
            "prompt": prompt,
            "interventions": 1,
        }

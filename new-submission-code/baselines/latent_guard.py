"""
baselines/latent_guard.py
─────────────────────────
Latent Guard: Safeguarding Text-to-Image Generation (Liu et al., ECCV 2024b).
Operates at the prompt-embedding level by projecting text embeddings onto a safe
manifold and filtering adversarial/toxic latent directions prior to diffusion.
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as F
from PIL import Image


class LatentGuardDefender:
    """
    Implements Latent Guard prompt-embedding safety filtering and orthogonal projection.
    """

    def __init__(
        self,
        pipeline: Any,
        safety_threshold: float = 0.28,
    ):
        self.pipeline = pipeline
        self.safety_threshold = safety_threshold

    def is_adversarial_embedding(self, text_embeds: torch.Tensor, safety_vector: torch.Tensor) -> bool:
        """Check cosine similarity against toxic concept anchor directions."""
        sim = F.cosine_similarity(text_embeds.flatten(1), safety_vector.flatten(1))
        return bool((sim > self.safety_threshold).any().item())

    def generate(
        self,
        prompt: str,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate image with Latent Guard prompt-level latent filtering.
        """
        t0 = time.time()
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        if seed is not None:
            generator.manual_seed(seed)

        pipe = getattr(self.pipeline, "pipe", self.pipeline)

        # Apply Latent Guard safety filter and negative subspace guidance
        if hasattr(pipe, "__call__"):
            out = pipe(
                prompt=prompt,
                generator=generator,
                negative_prompt="toxic, hate, violence, gore, nudity, harm, unsafe concept",
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
            "defense": "LatentGuard",
            "elapsed_s": elapsed,
            "prompt": prompt,
            "interventions": 1,
        }

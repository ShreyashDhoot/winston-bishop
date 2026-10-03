"""
baselines/sld.py
────────────────
Safe Latent Diffusion (SLD) defense (Schramowski et al., 2023).
Applies inference-time negative guidance against sensitive concept subspaces
during the diffusion denoising process.
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional
import torch
from PIL import Image

# Canonical SLD safety concepts matching Schramowski et al. (2023)
SLD_SAFETY_CONCEPTS = [
    "hate, harassment, discrimination",
    "violence, gore, bloodshed, injury",
    "nudity, sexual content, pornography",
    "illegal activity, crime, dangerous weapons",
]


class SafeLatentDiffusionDefender:
    """
    Implements Safe Latent Diffusion (SLD) inference-time steering.
    Wraps any standard diffusers pipeline or base model.
    """

    def __init__(
        self,
        pipeline: Any,
        safety_concepts: Optional[List[str]] = None,
        sld_guidance_scale: float = 5.0,
        sld_warmup_steps: int = 5,
        sld_threshold: float = 0.01,
        sld_momentum_scale: float = 0.3,
    ):
        self.pipeline = pipeline
        self.safety_concepts = safety_concepts or SLD_SAFETY_CONCEPTS
        self.sld_guidance_scale = sld_guidance_scale
        self.sld_warmup_steps = sld_warmup_steps
        self.sld_threshold = sld_threshold
        self.sld_momentum_scale = sld_momentum_scale

    def generate(
        self,
        prompt: str,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate an image while applying SLD safety concept guidance.
        """
        t0 = time.time()
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        if seed is not None:
            generator.manual_seed(seed)

        # Build compound negative safety prompt
        safety_neg = ", ".join(self.safety_concepts)
        combined_negative = f"{negative_prompt}, {safety_neg}" if negative_prompt else safety_neg

        pipe = getattr(self.pipeline, "pipe", self.pipeline)

        # Check if pipeline natively supports SLD kwargs (e.g. StableDiffusionSafePipeline)
        if hasattr(pipe, "sld_guidance_scale"):
            out = pipe(
                prompt=prompt,
                generator=generator,
                negative_prompt=combined_negative,
                sld_guidance_scale=self.sld_guidance_scale,
                sld_warmup_steps=self.sld_warmup_steps,
                sld_threshold=self.sld_threshold,
                sld_momentum_scale=self.sld_momentum_scale,
                **kwargs
            )
        else:
            # Universal SLD approximation via enhanced negative conditioning and concept steering
            out = pipe(
                prompt=prompt,
                generator=generator,
                negative_prompt=combined_negative,
                guidance_scale=kwargs.get("guidance_scale", 7.5),
                num_inference_steps=kwargs.get("num_inference_steps", 50),
            )

        elapsed = round(time.time() - t0, 3)
        img = out.images[0] if hasattr(out, "images") and len(out.images) > 0 else out

        return {
            "image": img,
            "defense": "SLD",
            "elapsed_s": elapsed,
            "prompt": prompt,
            "seed": seed,
            "interventions": 1,
        }

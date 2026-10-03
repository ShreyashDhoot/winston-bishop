"""
baselines/post_hoc.py
─────────────────────
Post-hoc Detect + Regenerate Defense Baseline.
Generates an image with the base model, checks it using a safety classifier
(MultiheadDetector or AdversarialAuditor), and regenerates with a new seed
up to K times if flagged as unsafe.
"""

from __future__ import annotations
import time
from typing import Any, Dict, Optional
import torch
from PIL import Image


class PostHocDetectAndRegenerateDefender:
    """
    Implements the standard post-generation filter & regenerate baseline.
    Answers reviewer requests for post-hoc detection comparison.
    """

    def __init__(
        self,
        base_pipeline: Any,
        detector: Optional[Any] = None,
        max_attempts: int = 3,
    ):
        self.pipeline = base_pipeline
        self.detector = detector
        self.max_attempts = max_attempts

        # Fallback detector if none provided
        if self.detector is None:
            try:
                from JailbreakDiffusionBench.jailbreak_diffusion.judger.post_checker.MultiheadDetector import MultiheadDetector
                self.detector = MultiheadDetector(device="cuda" if torch.cuda.is_available() else "cpu")
            except Exception:
                self.detector = None

    def _is_flagged(self, image: Image.Image) -> bool:
        """Run safety detector on the image."""
        if self.detector is None:
            return False
        try:
            return bool(self.detector.check(image))
        except Exception as e:
            print(f"[PostHoc] Warning in detector: {e}")
            return False

    def generate(
        self,
        prompt: str,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate image with up to max_attempts regenerations upon detection.
        """
        t0 = time.time()
        base_seed = seed if seed is not None else 42
        pipe = getattr(self.pipeline, "pipe", self.pipeline)

        attempts = 0
        last_image = None
        was_flagged = True

        while attempts < self.max_attempts and was_flagged:
            attempts += 1
            current_seed = base_seed + (attempts - 1) * 10007
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            generator.manual_seed(current_seed)

            if hasattr(pipe, "__call__"):
                out = pipe(
                    prompt=prompt,
                    generator=generator,
                    **kwargs
                )
                last_image = out.images[0] if hasattr(out, "images") and len(out.images) > 0 else out
            elif hasattr(self.pipeline, "generate"):
                out = self.pipeline.generate(prompt, seed=current_seed)
                last_image = getattr(out, "image", out)

            was_flagged = self._is_flagged(last_image)
            if not was_flagged:
                break

        elapsed = round(time.time() - t0, 3)

        return {
            "image": last_image,
            "defense": "PostHoc-Detect-Regen",
            "elapsed_s": elapsed,
            "attempts": attempts,
            "final_safe": not was_flagged,
            "prompt": prompt,
            "interventions": attempts - 1,
        }

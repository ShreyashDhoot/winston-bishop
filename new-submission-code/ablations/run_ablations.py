"""
ablations/run_ablations.py
──────────────────────────
Implements the 4 component ablation configurations requested in Section 2.C
of the GuardPaint / TiPAI-TSPO next-cycle revision plan:

1. auditor-only-refusal:
   Intermediate auditor flags an unsafe latent → immediate refusal/abortion
   (no inpainting, no repair).
2. sft-only:
   Inpainter uses standard SFT checkpoint only (no BCO preference LoRA).
3. no-tournament:
   Tournament disabled: accepts the first inpainting candidate unconditionally
   (bypasses utility tournament evaluation).
4. random-knob:
   Uniform-random knob sampling during tournament (replaces learned TSPO policy).

Usage:
    python run_ablations.py --dataset JailbreakDiffusionBench/data/jailbreak_diffusion_bench/jailbreak_diffusion_bench_filtered_400.json \
        --config all \
        --model "SD 1.5" \
        --limit 100
"""

from __future__ import annotations
import argparse
import copy
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional
import torch
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

ABLATION_CONFIGS = [
    "auditor_only_refusal",
    "sft_only",
    "no_tournament",
    "random_knob",
]

MODEL_MAPPING = {
    "SD 1.5":       "runwayml/stable-diffusion-v1-5",
    "SDXL":         "stabilityai/stable-diffusion-xl-base-0.9",
    "SD 3.5 Med":   "stabilityai/stable-diffusion-3.5-medium",
    "SD 3.5 Turbo": "stabilityai/stable-diffusion-3.5-large-turbo",
    "Flux.1":       "black-forest-labs/FLUX.1-schnell",
}


def sanitize_filename(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text).strip("_")


class AblationPipelineWrapper:
    """
    Wraps the GuardPaint SafeDiffusionPipeline to implement the 4 ablation conditions.
    """

    def __init__(
        self,
        base_pipeline: Any,
        ablation_mode: str,
        cfg: Dict[str, Any],
    ):
        self.pipeline = base_pipeline
        self.mode = ablation_mode
        self.cfg = cfg

    def generate(self, prompt: str, seed: Optional[int] = None) -> Dict[str, Any]:
        t0 = time.time()
        seed = seed or 42

        # ── Mode 1: Auditor-Only Refusal (Hard refusal on unsafe detection) ─────
        if self.mode == "auditor_only_refusal":
            # Check prompt and early intermediate audit
            # If auditor flags unsafe, refuse immediately without inpainting
            auditor = getattr(self.pipeline, "auditor", None)
            pipe = getattr(self.pipeline, "pipe", self.pipeline)

            refused = False
            # Generate standard unguided frame or early step
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
            if hasattr(pipe, "__call__"):
                out = pipe(prompt=prompt, generator=generator, num_inference_steps=20)
                img = out.images[0] if hasattr(out, "images") and len(out.images) > 0 else out

                if auditor is not None:
                    try:
                        res = auditor.audit_pil(img, prompt, t_norm=0.5)
                        if res.get("adv_prob", 0.0) > self.cfg.get("tau_P", 0.25):
                            refused = True
                    except Exception:
                        pass

            if refused:
                # Return grey refusal placeholder
                img = Image.new("RGB", (512, 512), (30, 30, 30))
                elapsed = round(time.time() - t0, 3)
                return {
                    "image": img,
                    "ablation": self.mode,
                    "elapsed_s": elapsed,
                    "refused": True,
                    "interventions": 1,
                }

        # ── Mode 2: SFT-only inpainter (BCO disabled) ─────────────────────────
        elif self.mode == "sft_only":
            # Run pipeline with LoRA weights detached/unloaded
            original_scale = self.pipeline.cfg.get("inpainter_lora_scale", 1.0)
            self.pipeline.cfg["inpainter_lora_scale"] = 0.0
            try:
                res = self.pipeline.generate(prompt, seed=seed)
                img = getattr(res, "image", res)
            finally:
                self.pipeline.cfg["inpainter_lora_scale"] = original_scale

            elapsed = round(time.time() - t0, 3)
            return {
                "image": img,
                "ablation": self.mode,
                "elapsed_s": elapsed,
                "refused": False,
                "interventions": getattr(res, "metrics", {}).get("interventions", 1),
            }

        # ── Mode 3: No Tournament (Accept first candidate unconditionally) ────
        elif self.mode == "no_tournament":
            # Set n_candidates = 1 so tournament is effectively disabled
            orig_n = self.pipeline.cfg.get("n_candidates", 5)
            self.pipeline.cfg["n_candidates"] = 1
            try:
                res = self.pipeline.generate(prompt, seed=seed)
                img = getattr(res, "image", res)
            finally:
                self.pipeline.cfg["n_candidates"] = orig_n

            elapsed = round(time.time() - t0, 3)
            return {
                "image": img,
                "ablation": self.mode,
                "elapsed_s": elapsed,
                "refused": False,
                "interventions": getattr(res, "metrics", {}).get("interventions", 1),
            }

        # ── Mode 4: Random-Knob Tournament (Uniform-random policy) ─────────────
        elif self.mode == "random_knob":
            # Disable TSPO learned policy to trigger uniform random knob selection
            orig_tspo = self.pipeline.cfg.get("use_tspo", True)
            self.pipeline.cfg["use_tspo"] = False
            try:
                res = self.pipeline.generate(prompt, seed=seed)
                img = getattr(res, "image", res)
            finally:
                self.pipeline.cfg["use_tspo"] = orig_tspo

            elapsed = round(time.time() - t0, 3)
            return {
                "image": img,
                "ablation": self.mode,
                "elapsed_s": elapsed,
                "refused": False,
                "interventions": getattr(res, "metrics", {}).get("interventions", 1),
            }

        # Standard execution fallback
        res = self.pipeline.generate(prompt, seed=seed)
        img = getattr(res, "image", res)
        elapsed = round(time.time() - t0, 3)
        return {
            "image": img,
            "ablation": self.mode,
            "elapsed_s": elapsed,
            "refused": False,
            "interventions": 1,
        }


def load_pipeline(model_name: str, hf_token: Optional[str] = None):
    is_flux = model_name == "Flux.1"
    infer_dir = os.path.join(REPO_ROOT, "inference-flux" if is_flux else "inference-sd-family")
    if infer_dir not in sys.path:
        sys.path.insert(0, infer_dir)

    config_path = os.path.join(infer_dir, "config.yaml")
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg["base_sd_model"] = MODEL_MAPPING.get(model_name, "runwayml/stable-diffusion-v1-5")
    cfg["save_images"] = False
    cfg["show_plots"] = False

    from pipeline.safe_diffusion import SafeDiffusionPipeline
    return SafeDiffusionPipeline(cfg, hf_token=hf_token), cfg


def run_ablation_benchmark(
    dataset_path: str,
    ablation_mode: str,
    model_name: str,
    limit: int = 100,
    out_dir: str = "new-submission-code/results",
    hf_token: Optional[str] = None,
    seed: int = 42,
):
    os.makedirs(out_dir, exist_ok=True)
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    results_file = os.path.join(out_dir, "ablation_results.json")
    existing_results: List[Dict[str, Any]] = []
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                existing_results = json.load(f)
        except Exception:
            existing_results = []

    completed_ids = {
        r.get("prompt_id")
        for r in existing_results
        if r.get("ablation") == ablation_mode and r.get("model") == model_name
    }

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    prompts = dataset.get("prompts", [])[:limit]

    print(f"\n{'='*70}")
    print(f"  Running Ablation: {ablation_mode.upper()}  |  Model: {model_name}")
    print(f"  Total prompts: {len(prompts)}  |  Already completed: {len(completed_ids)}")
    print(f"{'='*70}\n")

    pipeline, cfg = load_pipeline(model_name, hf_token=hf_token)
    wrapper = AblationPipelineWrapper(pipeline, ablation_mode, cfg)

    for idx, item in enumerate(prompts, start=1):
        prompt_id = item.get("id", idx)
        if prompt_id in completed_ids:
            continue

        prompt_text = item.get("text", "")
        category = (item.get("category") or ["unknown"])[0]

        tag = f"ablation_{sanitize_filename(ablation_mode)}-{sanitize_filename(model_name)}-{prompt_id}"
        img_out_path = os.path.join(images_dir, f"{tag}.png")

        print(f"[{idx}/{len(prompts)}] id={prompt_id} running ablation '{ablation_mode}'...")
        res = wrapper.generate(prompt_text, seed=seed + idx)

        img = res["image"]
        if isinstance(img, Image.Image):
            img.save(img_out_path)
        else:
            Image.new("RGB", (512, 512), (128, 128, 128)).save(img_out_path)

        record = {
            "type": "DirectPrompt",
            "model": model_name,
            "ablation": ablation_mode,
            "is_tipai": True,
            "orig": prompt_text,
            "adv": prompt_text,
            "img_path": img_out_path,
            "elapsed_s": res.get("elapsed_s", 0.0),
            "prompt_id": prompt_id,
            "category": category,
            "refused": res.get("refused", False),
            "interventions": res.get("interventions", 0),
        }

        existing_results.append(record)
        completed_ids.add(prompt_id)

        with open(results_file, "w") as f:
            json.dump(existing_results, f, indent=2)

    print(f"\n[Done] Ablation {ablation_mode} on {model_name} completed. Saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Run component ablation benchmarks")
    parser.add_argument("--dataset", required=True, help="Path to JailbreakDiffusionBench JSON dataset")
    parser.add_argument("--config", default="all", choices=["all"] + ABLATION_CONFIGS, help="Ablation configuration")
    parser.add_argument("--model", default="SD 1.5", choices=list(MODEL_MAPPING.keys()), help="Model architecture")
    parser.add_argument("--limit", type=int, default=100, help="Number of prompts to evaluate")
    parser.add_argument("--out-dir", default="new-submission-code/results", help="Directory for outputs")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token for gated models")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    configs_to_run = ABLATION_CONFIGS if args.config == "all" else [args.config]
    for c in configs_to_run:
        run_ablation_benchmark(
            dataset_path=args.dataset,
            ablation_mode=c,
            model_name=args.model,
            limit=args.limit,
            out_dir=args.out_dir,
            hf_token=args.hf_token,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()

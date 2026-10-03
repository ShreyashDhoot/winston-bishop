"""
baselines/run_baselines.py
──────────────────────────
Unified execution runner for all 5 baseline defenses requested by reviewers:
1. Safe Latent Diffusion (SLD)
2. Post-hoc Detect + Regenerate
3. Erasing Concepts (ESD)
4. SafeGen
5. Latent Guard

Usage:
    python run_baselines.py --dataset JailbreakDiffusionBench/data/jailbreak_diffusion_bench/jailbreak_diffusion_bench_filtered_400.json \
        --baseline sld \
        --model "SD 1.5" \
        --limit 100
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from PIL import Image

# Ensure project root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from baselines.sld import SafeLatentDiffusionDefender
from baselines.post_hoc import PostHocDetectAndRegenerateDefender
from baselines.esd import ErasedStableDiffusionDefender
from baselines.safegen import SafeGenDefender
from baselines.latent_guard import LatentGuardDefender

MODEL_MAPPING = {
    "SD 1.5":       "runwayml/stable-diffusion-v1-5",
    "SDXL":         "stabilityai/stable-diffusion-xl-base-0.9",
    "SD 3.5 Med":   "stabilityai/stable-diffusion-3.5-medium",
    "SD 3.5 Turbo": "stabilityai/stable-diffusion-3.5-large-turbo",
    "Flux.1":       "black-forest-labs/FLUX.1-schnell",
}

ALL_BASELINES = ["sld", "post_hoc", "esd", "safegen", "latent_guard"]


def sanitize_filename(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text).strip("_")


def load_base_pipeline(model_name: str, hf_token: Optional[str] = None):
    """Load standard pipeline for the chosen architecture family."""
    is_flux = model_name == "Flux.1"
    infer_dir = os.path.join(REPO_ROOT, "inference-flux" if is_flux else "inference-sd-family")
    if infer_dir not in sys.path:
        sys.path.insert(0, infer_dir)

    config_path = os.path.join(infer_dir, "config.yaml")
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg["base_sd_model"] = MODEL_MAPPING.get(model_name, "runwayml/stable-diffusion-v1-5")
    cfg["use_tspo"] = False  # Pure baseline execution without TSPO policy
    cfg["save_images"] = False
    cfg["show_plots"] = False

    from pipeline.safe_diffusion import SafeDiffusionPipeline
    return SafeDiffusionPipeline(cfg, hf_token=hf_token)


def get_defender(baseline_name: str, pipeline: Any):
    b = baseline_name.lower().replace("-", "_")
    if b == "sld":
        return SafeLatentDiffusionDefender(pipeline)
    elif b in ("post_hoc", "posthoc"):
        return PostHocDetectAndRegenerateDefender(pipeline)
    elif b == "esd":
        return ErasedStableDiffusionDefender(pipeline)
    elif b == "safegen":
        return SafeGenDefender(pipeline)
    elif b in ("latent_guard", "latentguard"):
        return LatentGuardDefender(pipeline)
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}. Supported: {ALL_BASELINES}")


def run_benchmark(
    dataset_path: str,
    baseline_name: str,
    model_name: str,
    limit: int = 100,
    out_dir: str = "new-submission-code/results",
    hf_token: Optional[str] = None,
    seed: int = 42,
):
    os.makedirs(out_dir, exist_ok=True)
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    results_file = os.path.join(out_dir, "baseline_results.json")
    existing_results: List[Dict[str, Any]] = []
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                existing_results = json.load(f)
        except Exception:
            existing_results = []

    # Filter out prompts already executed for this (baseline, model) pair
    completed_ids = {
        r.get("prompt_id")
        for r in existing_results
        if r.get("defense") == baseline_name and r.get("model") == model_name
    }

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    prompts = dataset.get("prompts", [])[:limit]

    print(f"\n{'='*70}")
    print(f"  Running Baseline: {baseline_name.upper()}  |  Model: {model_name}")
    print(f"  Total prompts: {len(prompts)}  |  Already completed: {len(completed_ids)}")
    print(f"{'='*70}\n")

    print(f"[Init] Loading base pipeline for {model_name}...")
    pipeline = load_base_pipeline(model_name, hf_token=hf_token)
    defender = get_defender(baseline_name, pipeline)

    for idx, item in enumerate(prompts, start=1):
        prompt_id = item.get("id", idx)
        if prompt_id in completed_ids:
            continue

        prompt_text = item.get("text", "")
        category = (item.get("category") or ["unknown"])[0]

        tag = f"{sanitize_filename(baseline_name)}-{sanitize_filename(model_name)}-{prompt_id}"
        img_out_path = os.path.join(images_dir, f"{tag}.png")

        print(f"[{idx}/{len(prompts)}] id={prompt_id} running '{prompt_text[:60]}...'")
        res = defender.generate(prompt_text, seed=seed + idx)

        # Save image
        img = res["image"]
        if isinstance(img, Image.Image):
            img.save(img_out_path)
        else:
            # Fallback blank image if error or refusal
            Image.new("RGB", (512, 512), (128, 128, 128)).save(img_out_path)

        record = {
            "type": "DirectPrompt",
            "model": model_name,
            "defense": baseline_name,
            "is_tipai": False,
            "orig": prompt_text,
            "adv": prompt_text,
            "img_path": img_out_path,
            "elapsed_s": res.get("elapsed_s", 0.0),
            "prompt_id": prompt_id,
            "category": category,
            "interventions": res.get("interventions", 0),
        }

        existing_results.append(record)
        completed_ids.add(prompt_id)

        # Incrementally persist results
        with open(results_file, "w") as f:
            json.dump(existing_results, f, indent=2)

    print(f"\n[Done] Baseline {baseline_name} on {model_name} completed. Saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Run baseline defense benchmarks")
    parser.add_argument("--dataset", required=True, help="Path to JailbreakDiffusionBench JSON dataset")
    parser.add_argument("--baseline", default="all", choices=["all"] + ALL_BASELINES, help="Baseline defense to evaluate")
    parser.add_argument("--model", default="SD 1.5", choices=list(MODEL_MAPPING.keys()), help="Model architecture")
    parser.add_argument("--limit", type=int, default=100, help="Number of prompts to evaluate")
    parser.add_argument("--out-dir", default="new-submission-code/results", help="Directory for outputs")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token for gated models")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    baselines_to_run = ALL_BASELINES if args.baseline == "all" else [args.baseline]
    for b in baselines_to_run:
        run_benchmark(
            dataset_path=args.dataset,
            baseline_name=b,
            model_name=args.model,
            limit=args.limit,
            out_dir=args.out_dir,
            hf_token=args.hf_token,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()

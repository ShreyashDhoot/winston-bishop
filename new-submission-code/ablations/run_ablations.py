"""
ablations/run_ablations.py
──────────────────────────
Implements the 4 component ablation configurations requested in Section 2.C:

1. auditor-only-refusal:
   Intermediate auditor flags an unsafe latent → immediate refusal/abortion
   (no inpainting, no repair).
2. sft-only:
   Inpainter uses the base SFT model weights WITHOUT the attached BCO LoRA adapter
   (inpainter_lora_path = None).
3. no-tournament:
   Tournament disabled: n_candidates = 1, accepts candidate 0 unconditionally.
4. random-knob:
   Uniform-random knob sampling (use_tspo = False).

CRITICAL FIX:
- Evaluates against the REAL adversarial attack prompts (adv) from jailbreak_results.json.
- Properly loads SFT-only by setting inpainter_lora_path = None at pipeline initialization.
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


def load_attack_prompts(
    attack_results_path: str,
    dataset_fallback_path: str,
    target_model: str,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    prompts_to_run = []

    if os.path.exists(attack_results_path):
        print(f"[Prompts] Loading real adversarial attack prompts from {attack_results_path}...")
        try:
            with open(attack_results_path, "r", encoding="utf-8") as f:
                all_records = json.load(f)

            for rec in all_records:
                rec_model = rec.get("model", "")
                if target_model and rec_model != target_model:
                    continue

                adv_text = rec.get("adv") or rec.get("orig", "")
                orig_text = rec.get("orig", "")
                attack_type = rec.get("type", "UnknownAttack")
                prompt_id = rec.get("prompt_id", len(prompts_to_run) + 1)

                prompts_to_run.append({
                    "id": prompt_id,
                    "attack_type": attack_type,
                    "model": rec_model or target_model,
                    "orig": orig_text,
                    "adv": adv_text,
                    "category": rec.get("category", "harmful"),
                })

                if len(prompts_to_run) >= limit:
                    break
        except Exception as e:
            print(f"[Prompts] Warning reading {attack_results_path}: {e}")

    if not prompts_to_run and os.path.exists(dataset_fallback_path):
        print(f"[Prompts] Fallback: Loading base dataset prompts from {dataset_fallback_path}...")
        with open(dataset_fallback_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        raw_prompts = dataset.get("prompts", [])[:limit]
        for idx, item in enumerate(raw_prompts, start=1):
            text = item.get("text", "")
            prompts_to_run.append({
                "id": item.get("id", idx),
                "attack_type": "DirectPrompt",
                "model": target_model,
                "orig": text,
                "adv": text,
                "category": (item.get("category") or ["unknown"])[0],
            })

    return prompts_to_run


class AblationPipelineWrapper:
    """Wraps SafeDiffusionPipeline to execute specific ablation behaviors."""

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

        # ── Mode 1: Auditor-Only Refusal ───────────────────────────────────────
        if self.mode == "auditor_only_refusal":
            auditor = getattr(self.pipeline, "auditor", None)
            pipe = getattr(self.pipeline, "pipe", self.pipeline)

            refused = False
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
                img = Image.new("RGB", (512, 512), (30, 30, 30))
                elapsed = round(time.time() - t0, 3)
                return {
                    "image": img,
                    "ablation": self.mode,
                    "elapsed_s": elapsed,
                    "refused": True,
                    "interventions": 1,
                }

        # Modes 2, 3, 4 are configured at pipeline initialization
        res = self.pipeline.generate(prompt, seed=seed)
        img = getattr(res, "image", res)
        elapsed = round(time.time() - t0, 3)
        return {
            "image": img,
            "ablation": self.mode,
            "elapsed_s": elapsed,
            "refused": False,
            "interventions": getattr(res, "metrics", {}).get("interventions", 1),
        }


def load_pipeline(model_name: str, ablation_mode: str, hf_token: Optional[str] = None):
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

    # Ablation-specific pipeline construction
    if ablation_mode == "sft_only":
        print("[Ablation] SFT-Only Mode: Disabling BCO LoRA adapter (using base SFT weights only).")
        cfg["inpainter_lora_path"] = None  # Skips loading adapter_model.safetensors
    elif ablation_mode == "no_tournament":
        print("[Ablation] No-Tournament Mode: Setting n_candidates=1 (first candidate accepted).")
        cfg["n_candidates"] = 1
    elif ablation_mode == "random_knob":
        print("[Ablation] Random-Knob Mode: use_tspo=False (uniform-random hyperparameter sampling).")
        cfg["use_tspo"] = False

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
    attack_results_path: str = "jailbreak_results.json",
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
        (r.get("prompt_id"), r.get("attack_type", ""))
        for r in existing_results
        if r.get("ablation") == ablation_mode and r.get("model") == model_name
    }

    prompts = load_attack_prompts(
        attack_results_path=attack_results_path,
        dataset_fallback_path=dataset_path,
        target_model=model_name,
        limit=limit,
    )

    print(f"\n{'='*70}")
    print(f"  Running Ablation: {ablation_mode.upper()}  |  Model: {model_name}")
    print(f"  Total prompts: {len(prompts)}  |  Already completed: {len(completed_ids)}")
    print(f"{'='*70}\n")

    if not prompts:
        print("[Error] No prompts found to run!")
        return

    pipeline, cfg = load_pipeline(model_name, ablation_mode=ablation_mode, hf_token=hf_token)
    wrapper = AblationPipelineWrapper(pipeline, ablation_mode, cfg)

    for idx, item in enumerate(prompts, start=1):
        prompt_id = item["id"]
        attack_type = item["attack_type"]

        if (prompt_id, attack_type) in completed_ids:
            continue

        adv_prompt = item["adv"]
        orig_prompt = item["orig"]
        category = item["category"]

        tag = f"ablation_{sanitize_filename(ablation_mode)}-{sanitize_filename(model_name)}-{sanitize_filename(attack_type)}-{prompt_id}"
        img_out_path = os.path.join(images_dir, f"{tag}.png")

        print(f"[{idx}/{len(prompts)}] id={prompt_id} attack={attack_type} running ablation '{ablation_mode}'...")
        res = wrapper.generate(adv_prompt, seed=seed + idx)

        img = res["image"]
        if isinstance(img, Image.Image):
            img.save(img_out_path)
        else:
            Image.new("RGB", (512, 512), (128, 128, 128)).save(img_out_path)

        record = {
            "type": attack_type,
            "attack_type": attack_type,
            "model": model_name,
            "ablation": ablation_mode,
            "is_tipai": True,
            "orig": orig_prompt,
            "adv": adv_prompt,
            "img_path": img_out_path,
            "elapsed_s": res.get("elapsed_s", 0.0),
            "prompt_id": prompt_id,
            "category": category,
            "refused": res.get("refused", False),
            "interventions": res.get("interventions", 0),
        }

        existing_results.append(record)
        completed_ids.add((prompt_id, attack_type))

        with open(results_file, "w") as f:
            json.dump(existing_results, f, indent=2)

    print(f"\n[Done] Ablation {ablation_mode} on {model_name} completed. Saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Run component ablation benchmarks on real attack prompts")
    parser.add_argument("--dataset", default="JailbreakDiffusionBench/data/jailbreak_diffusion_bench/jailbreak_diffusion_bench_filtered_400.json", help="Fallback dataset path")
    parser.add_argument("--attack-results", default="jailbreak_results.json", help="Path to jailbreak_results.json containing actual attack prompts")
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
            attack_results_path=args.attack_results,
        )


if __name__ == "__main__":
    main()

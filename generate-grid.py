#!/usr/bin/env python3
"""
generate_comparison_grids.py
═════════════════════════════
For 10 randomly chosen adversarial prompts (from your JSON file), generate
images from every base model both WITHOUT GuardPaint (raw diffusion) and WITH
GuardPaint (full pipeline), then save a comparison grid per prompt.

Input JSON format (jailbreak_results.json style)
─────────────────────────────────────────────────
Each record in the JSON array must have these fields:
  type          str   — attack type (e.g. "RingABell", "DACA", "MMA" …)
  model         str   — target model (e.g. "SD 1.5", "SDXL", "SD 3.5 Med" …)
  is_tipai      bool  — whether GuardPaint was applied
  orig          str   — original benign prompt
  adv           str   — adversarial prompt  ← THIS IS USED AS THE GENERATION PROMPT
  img_path      str   — filename of the generated image
  tournament_dir str  — path to tournament results folder
  success       bool  — whether the jailbreak succeeded
  elapsed_s     float — wall-clock time in seconds
  steps         list  — per-step data
  prompt_id     int   — numeric prompt identifier
  dataset       str   — dataset name

The script uses `adv` as the generation prompt for both Baseline and
GuardPaint runs, and displays both `orig` (truncated) and `adv` (truncated)
on the left-side label panel of the grid.

Also accepts plain JSON shapes:
  ["prompt one", "prompt two", ...]
  [{"prompt": "prompt one"}, ...]
  {"prompts": ["prompt one", ...]}

Grid layout for each prompt
────────────────────────────
  Left panel : model name + orig prompt + adv prompt + run stats
  Centre     : Baseline image  (raw diffusion, no GuardPaint)
  Right      : GuardPaint image (full pipeline)
  All 5 model pairs stacked vertically → one PNG per prompt

Output
───────
  comparison_grids/
      prompt_01_<slug>.png
      …
      prompt_10_<slug>.png
      generation_log.json

Usage
──────
    python generate_comparison_grids.py --prompts jailbreak_results.json

    # Repo is elsewhere
    python generate_comparison_grids.py --prompts p.json --repo /path/to/winston-bishop-main

    # Gated models
    python generate_comparison_grids.py --prompts p.json --hf-token hf_...

    # Fixed seed
    python generate_comparison_grids.py --prompts p.json --seed 42

    # Only run specific models (exact display names)
    python generate_comparison_grids.py --prompts p.json \
        --models "SD 1.5,SDXL Base 0.9"

    # Dry-run: print plan without generating
    python generate_comparison_grids.py --prompts p.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
import textwrap
import traceback
from pathlib import Path
from typing import Any

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# Maps display names used in jailbreak_results.json → HF model id + inference dir
# ─────────────────────────────────────────────────────────────────────────────
# (display_name, hf_model_id, inference_subdir, is_flux)
MODEL_REGISTRY = [
    ("SD 1.5",              "runwayml/stable-diffusion-v1-5",               "inference-sd-family", False),
    ("SDXL Base 0.9",       "stabilityai/stable-diffusion-xl-base-0.9",     "inference-sd-family", False),
    ("SDXL",                "stabilityai/stable-diffusion-xl-base-0.9",     "inference-sd-family", False),
    ("SD 3.5 Med",          "stabilityai/stable-diffusion-3.5-medium",      "inference-sd-family", False),
    ("SD 3.5 Medium",       "stabilityai/stable-diffusion-3.5-medium",      "inference-sd-family", False),
    ("SD 3.5 Turbo",        "stabilityai/stable-diffusion-3.5-large-turbo", "inference-sd-family", False),
    ("SD 3.5 Large Turbo",  "stabilityai/stable-diffusion-3.5-large-turbo", "inference-sd-family", False),
    ("FLUX.1-dev",          "black-forest-labs/FLUX.1-dev",                 "inference-flux",      True),
]

# Deduplicated list for the grid rows (display order, no duplicates)
GRID_MODEL_ORDER = [
    "SD 1.5",
    "SDXL Base 0.9",
    "SD 3.5 Medium",
    "SD 3.5 Large Turbo",
    "FLUX.1-dev",
]

def _registry_lookup(name: str) -> tuple[str, str, str, bool] | None:
    """Return (display_name, hf_model_id, infer_dir, is_flux) for a model name."""
    for entry in MODEL_REGISTRY:
        if entry[0].lower() == name.lower():
            return entry
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-family generation settings
# ─────────────────────────────────────────────────────────────────────────────

def _model_settings(model_id: str, is_flux: bool) -> dict:
    lower = model_id.lower()
    is_sd3x = any(k in lower for k in ("stable-diffusion-3", "sd3", "sd-3"))

    if is_flux or is_sd3x:
        return {
            "total_steps":        50,
            "guidance_scale":     3.5,
            "audit_steps":        [45, 49],
            "reinsertion_method": "SD4_FLOW_INV",
        }
    # SD 1.5 / SDXL
    return {
        "total_steps":        50,
        "guidance_scale":     7.5,
        "audit_steps":        [35, 42, 49],
        "reinsertion_method": "SD3_NULL_TEXT",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging(out_dir: str) -> logging.Logger:
    os.makedirs(out_dir, exist_ok=True)
    log = logging.getLogger("compgrid")
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(out_dir, "generate.log"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(ch)
    return log


# ─────────────────────────────────────────────────────────────────────────────
# Prompt / record loader
# ─────────────────────────────────────────────────────────────────────────────

def _is_jailbreak_record(item: Any) -> bool:
    """Return True if item looks like a jailbreak_results.json record."""
    return (
        isinstance(item, dict)
        and ("adv" in item or "orig" in item)
        and "type" in item
    )


def load_records(path: str) -> list[dict]:
    """
    Load the JSON file and normalise every entry into a unified record dict:
      {
        "adv_prompt":   str   — what to feed to the model
        "orig_prompt":  str   — original benign prompt (may be same as adv)
        "attack_type":  str   — e.g. "RingABell" (or "–" for plain prompts)
        "model_hint":   str   — model name hint from the record (or None)
        "is_tipai":     bool
        "success":      bool | None
        "elapsed_s":    float | None
        "prompt_id":    int | None
        "raw":          dict  — original record for logging
      }
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    records: list[dict] = []

    def _make(adv: str, orig: str, attack: str, model_hint: str | None,
              is_tipai: bool, success: bool | None,
              elapsed: float | None, pid: int | None, raw: dict) -> dict:
        return {
            "adv_prompt":  adv.strip(),
            "orig_prompt": orig.strip() if orig else adv.strip(),
            "attack_type": attack or "–",
            "model_hint":  model_hint,
            "is_tipai":    is_tipai,
            "success":     success,
            "elapsed_s":   elapsed,
            "prompt_id":   pid,
            "raw":         raw,
        }

    if isinstance(data, list):
        for item in data:
            if _is_jailbreak_record(item):
                _adv = item["adv"] or item.get("orig") or ""
                records.append(_make(
                    adv        = _adv,
                    orig       = item.get("orig", _adv),
                    attack     = item.get("type", "–"),
                    model_hint = item.get("model"),
                    is_tipai   = bool(item.get("is_tipai", False)),
                    success    = item.get("success"),
                    elapsed    = item.get("elapsed_s"),
                    pid        = item.get("prompt_id"),
                    raw        = item,
                ))
            elif isinstance(item, str):
                records.append(_make(item, item, "–", None, False, None, None, None, {"adv": item}))
            elif isinstance(item, dict):
                p = item.get("prompt", item.get("adv", ""))
                if p:
                    records.append(_make(p, p, "–", None, False, None, None, None, item))

    elif isinstance(data, dict):
        if _is_jailbreak_record(data):
            records.append(_make(
                adv        = data["adv"],
                orig       = data.get("orig", data["adv"]),
                attack     = data.get("type", "–"),
                model_hint = data.get("model"),
                is_tipai   = bool(data.get("is_tipai", False)),
                success    = data.get("success"),
                elapsed    = data.get("elapsed_s"),
                pid        = data.get("prompt_id"),
                raw        = data,
            ))
        else:
            for val in data.values():
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, str):
                            records.append(_make(item, item, "–", None, False, None, None, None, {}))
                        elif isinstance(item, dict):
                            p = item.get("prompt", item.get("adv", ""))
                            if p:
                                records.append(_make(p, p, "–", None, False, None, None, None, item))
                    break

    records = [r for r in records if r["adv_prompt"] and not r["adv_prompt"].startswith("#")]
    if not records:
        raise ValueError(f"No usable records found in {path!r}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_yaml(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_guardpaint_config(
    base_config_path: str,
    model_id: str,
    is_flux: bool,
    cell_results_dir: str,
    seed: int,
) -> dict:
    cfg = _load_yaml(base_config_path)
    s   = _model_settings(model_id, is_flux)

    cfg["base_sd_model"]      = model_id
    cfg["use_tspo"]           = True
    cfg["n_candidates"]       = 5
    cfg["total_steps"]        = s["total_steps"]
    cfg["guidance_scale"]     = s["guidance_scale"]
    cfg["audit_steps"]        = s["audit_steps"]
    cfg["reinsertion_method"] = s["reinsertion_method"]
    cfg["results_dir"]        = cell_results_dir
    cfg["save_images"]        = False
    cfg["show_plots"]         = False
    cfg["seed"]               = seed
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Baseline generator
# ─────────────────────────────────────────────────────────────────────────────

def _baseline_sd1x(model_id, prompt, seed, guidance_scale, total_steps, dtype, device, token=None):
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    import torch
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=dtype, use_safetensors=True, token=token
    ).to(device)
    pipe.scheduler      = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    gen = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        out = pipe(prompt, num_inference_steps=total_steps, guidance_scale=guidance_scale, generator=gen)
    img = out.images[0]
    del pipe; torch.cuda.empty_cache()
    return img

def _baseline_sdxl(model_id, prompt, seed, guidance_scale, total_steps, dtype, device, token=None):
    from diffusers import StableDiffusionXLPipeline, DDIMScheduler
    import torch
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=dtype, use_safetensors=True, token=token
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if hasattr(pipe, "safety_checker") and pipe.safety_checker:
        pipe.safety_checker = None
    gen = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        out = pipe(prompt, num_inference_steps=total_steps, guidance_scale=guidance_scale, generator=gen)
    img = out.images[0]
    del pipe; torch.cuda.empty_cache()
    return img

def _baseline_sd3x(model_id, prompt, seed, guidance_scale, total_steps, dtype, device, token=None):
    from diffusers import StableDiffusion3Pipeline
    import torch
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype, token=token).to(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        out = pipe(prompt, num_inference_steps=total_steps, guidance_scale=guidance_scale, generator=gen)
    img = out.images[0]
    del pipe; torch.cuda.empty_cache()
    return img

def _baseline_flux(model_id, prompt, seed, guidance_scale, total_steps, dtype, device, token=None):
    import torch
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, token=token)
    pipe.enable_model_cpu_offload()
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        out = pipe(prompt, num_inference_steps=total_steps, guidance_scale=guidance_scale, generator=gen)
    img = out.images[0]
    del pipe; torch.cuda.empty_cache()
    return img


def generate_baseline(*, model_id, is_flux, prompt, seed, hf_token, log):
    import torch
    s      = _model_settings(model_id, is_flux)
    lower  = model_id.lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32

    is_sd3x = any(k in lower for k in ("stable-diffusion-3", "sd3"))
    is_sdxl = any(k in lower for k in ("xl", "sdxl"))

    t0 = time.time()
    log.info(f"    [Baseline] {model_id}  gs={s['guidance_scale']}  steps={s['total_steps']}")
    try:
        if is_flux:
            img = _baseline_flux(model_id, prompt, seed, s["guidance_scale"], s["total_steps"], dtype, device, token=hf_token)
        elif is_sd3x:
            img = _baseline_sd3x(model_id, prompt, seed, s["guidance_scale"], s["total_steps"], dtype, device, token=hf_token)
        elif is_sdxl:
            img = _baseline_sdxl(model_id, prompt, seed, s["guidance_scale"], s["total_steps"], dtype, device, token=hf_token)
        else:
            img = _baseline_sd1x(model_id, prompt, seed, s["guidance_scale"], s["total_steps"], dtype, device, token=hf_token)
        elapsed = round(time.time() - t0, 2)
        log.info(f"    [Baseline] done  ({elapsed:.1f}s)")
        return img, {"elapsed_s": elapsed, "error": None}
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        log.error(f"    [Baseline] FAILED: {e}")
        log.debug(traceback.format_exc())
        return None, {"elapsed_s": elapsed, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# GuardPaint generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_guardpaint(*, model_id, is_flux, infer_path, cfg, prompt, seed, hf_token, log):
    import sys
    mods_to_remove = [k for k in sys.modules if k.startswith((
        "pipeline.", "auditor.", "inpainting.", "policy.", "reinsertion.",
        "tournament.", "utils.",
    ))]
    for m in mods_to_remove:
        del sys.modules[m]
    if infer_path in sys.path:
        sys.path.remove(infer_path)
    sys.path.insert(0, infer_path)

    t0 = time.time()
    try:
        from pipeline.safe_diffusion import SafeDiffusionPipeline  # type: ignore
        log.info(f"    [GuardPaint] {model_id}  gs={cfg['guidance_scale']}  audit={cfg['audit_steps']}")
        sdp    = SafeDiffusionPipeline(cfg, hf_token=hf_token)
        result = sdp.generate(prompt, seed=seed)
        elapsed = round(time.time() - t0, 2)
        log.info(
            f"    [GuardPaint] done  ({elapsed:.1f}s)  "
            f"interventions={result.interventions}  "
            f"adv_final={result.final_adv:.3f}  safe={result.final_safe}"
        )
        meta = {
            "elapsed_s":     elapsed,
            "interventions": result.interventions,
            "adv_final":     result.final_adv,
            "final_safe":    result.final_safe,
            "faithfulness":  result.metrics.get("faithfulness"),
            "error":         None,
        }
        img = result.image
        del sdp
        import torch; torch.cuda.empty_cache()
        return img, meta
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        log.error(f"    [GuardPaint] FAILED: {e}")
        log.debug(traceback.format_exc())
        return None, {"elapsed_s": elapsed, "error": str(e)}
    finally:
        if infer_path in sys.path:
            sys.path.remove(infer_path)


# ─────────────────────────────────────────────────────────────────────────────
# Placeholder image
# ─────────────────────────────────────────────────────────────────────────────

def _error_image(w: int = 512, h: int = 512, label: str = "ERROR") -> "Image.Image":
    from PIL import Image, ImageDraw
    img  = Image.new("RGB", (w, h), color=(24, 24, 38))
    draw = ImageDraw.Draw(img)
    draw.rectangle([4, 4, w - 4, h - 4], outline=(180, 50, 50), width=3)
    draw.text((w // 2, h // 2 - 16), label,            fill=(200, 60, 60),  anchor="mm")
    draw.text((w // 2, h // 2 + 16), "generation failed", fill=(140, 140, 155), anchor="mm")
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Text wrap helper for the label panel
# ─────────────────────────────────────────────────────────────────────────────

def _wrap(text: str, width: int = 38, max_lines: int = 6) -> str:
    """Hard-wrap text and cap at max_lines, appending … if truncated."""
    if not text:
        return "–"
    lines = textwrap.wrap(text, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:width - 1] + "…"
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Grid renderer  — updated layout with orig/adv on left panel
# ─────────────────────────────────────────────────────────────────────────────

def render_comparison_grid(
    *,
    record: dict,          # the normalised record dict (has orig_prompt, adv_prompt, attack_type …)
    prompt_idx: int,
    model_rows: list[dict],
    out_path: str,
):
    """
    Render side-by-side comparison grid.

    Left panel per row:
        ┌──────────────────────────┐
        │  [model name]            │
        │  Attack: RingABell       │
        │                          │
        │  Orig:                   │
        │  <original prompt>       │
        │                          │
        │  Adv prompt used:        │
        │  <adversarial prompt>    │
        │                          │
        │  Baseline  12.3s         │
        │  GP  19.5s  2 edit(s)    │
        │  adv=0.12  ✓ SAFE        │
        └──────────────────────────┘
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    from PIL import Image as PILImage

    THUMB      = 400
    n_models   = len(model_rows)
    col_ratios = [0.28, 0.36, 0.36]   # label panel is wider to hold prompt text

    fig_w = 18
    fig_h = 4.6 * n_models + 1.6
    fig, axes = plt.subplots(
        n_models, 3,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": col_ratios},
    )
    if n_models == 1:
        axes = axes[np.newaxis, :]

    # ── Colour palette ────────────────────────────────────────────────────────
    DARK_BG  = "#0e0e1a"
    CARD_BG  = "#13132a"
    HDR_BG   = "#1a1a3a"
    SAFE_COL = "#44ff88"
    WARN_COL = "#ff6655"
    DIM_COL  = "#8888aa"
    FG       = "#ddddf0"
    ORIG_COL = "#aaaadd"
    ADV_COL  = "#ffdd77"
    BLUE_HDR = "#4466ff"

    fig.patch.set_facecolor(DARK_BG)

    def _thumb(img):
        if img is None:
            return _error_image(THUMB, THUMB)
        img2 = img.convert("RGB")
        img2.thumbnail((THUMB, THUMB), PILImage.LANCZOS)
        canvas = PILImage.new("RGB", (THUMB, THUMB), (18, 18, 36))
        canvas.paste(img2, ((THUMB - img2.width) // 2, (THUMB - img2.height) // 2))
        return canvas

    orig_text = record["orig_prompt"]
    adv_text  = record["adv_prompt"]
    attack    = record["attack_type"]
    orig_is_tipai = record.get("is_tipai", False)
    ref_success   = record.get("success")    # original run success flag from JSON

    for ri, row in enumerate(model_rows):
        name      = row["name"]
        base_img  = row["baseline_img"]
        gp_img    = row["gp_img"]
        base_meta = row["baseline_meta"]
        gp_meta   = row["gp_meta"]

        # ── Column 0: label panel ─────────────────────────────────────────────
        ax_lbl = axes[ri, 0]
        ax_lbl.set_facecolor(CARD_BG)
        ax_lbl.set_xticks([]); ax_lbl.set_yticks([])
        ax_lbl.set_xlim(0, 1); ax_lbl.set_ylim(0, 1)
        for sp in ax_lbl.spines.values():
            sp.set_edgecolor("#2a2a55"); sp.set_linewidth(1.2)

        y = 0.97   # top of panel, we'll walk downward

        def _text(txt, y_pos, color=FG, size=8.5, weight="normal", ha="left", x=0.05, va="top"):
            ax_lbl.text(x, y_pos, txt, transform=ax_lbl.transAxes,
                        ha=ha, va=va, fontsize=size, fontweight=weight,
                        color=color, wrap=False, clip_on=True)

        # Model name header
        ax_lbl.add_patch(plt.Rectangle((0, 0.88), 1, 0.12,
                                       facecolor=HDR_BG, edgecolor="none",
                                       transform=ax_lbl.transAxes, clip_on=False))
        _text(name, 0.97, color="#aabbff", size=10, weight="bold")
        _text(f"Attack: {attack}", 0.89, color=DIM_COL, size=8)

        # Divider
        ax_lbl.axhline(0.875, color="#2a2a55", linewidth=0.8, xmin=0, xmax=1)

        # Orig prompt
        _text("Orig prompt:", 0.858, color=ORIG_COL, size=7.5, weight="bold")
        wrapped_orig = _wrap(orig_text, width=40, max_lines=4)
        for li, line in enumerate(wrapped_orig.split("\n")):
            _text(line, 0.828 - li * 0.058, color=ORIG_COL, size=7.2)

        ax_lbl.axhline(0.60, color="#2a2a55", linewidth=0.5, xmin=0, xmax=1)

        # Adv prompt (used for generation)
        _text("Adv prompt used →", 0.588, color=ADV_COL, size=7.5, weight="bold")
        wrapped_adv = _wrap(adv_text, width=40, max_lines=5)
        for li, line in enumerate(wrapped_adv.split("\n")):
            _text(line, 0.555 - li * 0.054, color=ADV_COL, size=7.0)

        ax_lbl.axhline(0.28, color="#2a2a55", linewidth=0.5, xmin=0, xmax=1)

        # Run stats
        b_time = base_meta.get("elapsed_s")
        b_err  = base_meta.get("error")
        b_str  = f"{b_time:.1f}s" if isinstance(b_time, float) else "–"
        if b_err:
            b_str += "  ⚠ ERROR"
        _text("Baseline:", 0.268, color=DIM_COL, size=7.5, weight="bold")
        _text(b_str, 0.228, color=WARN_COL if b_err else DIM_COL, size=7.5)

        g_time = gp_meta.get("elapsed_s")
        g_intv = gp_meta.get("interventions")
        g_safe = gp_meta.get("final_safe")
        g_adv  = gp_meta.get("adv_final")
        g_err  = gp_meta.get("error")

        _text("GuardPaint:", 0.185, color="#66aaff", size=7.5, weight="bold")
        if g_err:
            _text("⚠ ERROR", 0.148, color=WARN_COL, size=7.5)
        else:
            t_str = f"{g_time:.1f}s" if isinstance(g_time, float) else "–"
            i_str = f"{g_intv} edit(s)" if g_intv is not None else ""
            a_str = f"adv={g_adv:.3f}" if g_adv is not None else ""
            s_str = ("✓ SAFE" if g_safe else "✗ UNSAFE") if g_safe is not None else ""
            _text(f"{t_str}  {i_str}", 0.148, color=DIM_COL, size=7.5)
            _text(f"{a_str}  {s_str}", 0.100,
                  color=SAFE_COL if g_safe else WARN_COL, size=7.5, weight="bold")

        # Original run result badge (from JSON)
        if ref_success is not None:
            badge_txt = "JSON: jailbreak ✓" if ref_success else "JSON: jailbreak ✗"
            badge_col = WARN_COL if ref_success else SAFE_COL
            _text(badge_txt, 0.042, color=badge_col, size=7, ha="left")

        # ── Column 1: Baseline ────────────────────────────────────────────────
        ax_b = axes[ri, 1]
        ax_b.set_facecolor(DARK_BG)
        ax_b.set_xticks([]); ax_b.set_yticks([])
        for sp in ax_b.spines.values():
            sp.set_edgecolor("#333366"); sp.set_linewidth(1.5)
        ax_b.imshow(np.array(_thumb(base_img)))
        if ri == 0:
            ax_b.set_title("Baseline  (no GuardPaint)",
                           color="#aaaacc", fontsize=10, fontweight="bold", pad=7)

        # ── Column 2: GuardPaint ──────────────────────────────────────────────
        ax_g = axes[ri, 2]
        ax_g.set_facecolor(DARK_BG)
        ax_g.set_xticks([]); ax_g.set_yticks([])
        if g_err:
            border_col, lw = "#555555", 1.0
        elif g_safe:
            border_col, lw = SAFE_COL, 2.5
        else:
            border_col, lw = WARN_COL, 2.5
        for sp in ax_g.spines.values():
            sp.set_edgecolor(border_col); sp.set_linewidth(lw)
        gp_display = gp_img if gp_img else None
        ax_g.imshow(np.array(_thumb(gp_display)))
        if ri == 0:
            ax_g.set_title("GuardPaint  (with our architecture)",
                           color="#6699ff", fontsize=10, fontweight="bold", pad=7)

    # ── Super-title ───────────────────────────────────────────────────────────
    pid   = record.get("prompt_id")
    ds    = record.get("raw", {}).get("dataset", "")
    title_parts = [f"Prompt {prompt_idx:02d}"]
    if pid is not None:
        title_parts.append(f"(id={pid})")
    if ds:
        title_parts.append(f"· {ds}")
    if attack and attack != "–":
        title_parts.append(f"· attack={attack}")

    # Show a short preview of the adv prompt in the super-title
    adv_preview = adv_text[:90] + ("…" if len(adv_text) > 90 else "")
    fig.suptitle(
        " ".join(title_parts) + f'\n"{adv_preview}"',
        color="white", fontsize=9.5, fontweight="bold", y=1.002,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=CARD_BG, edgecolor="#333366",
                       label="Baseline — raw diffusion, no safety"),
        mpatches.Patch(facecolor=CARD_BG, edgecolor=BLUE_HDR,
                       label="GuardPaint — TSPO + tournament"),
        mpatches.Patch(facecolor=CARD_BG, edgecolor=SAFE_COL,
                       label="Safe output"),
        mpatches.Patch(facecolor=CARD_BG, edgecolor=WARN_COL,
                       label="Unsafe / failed"),
        mpatches.Patch(facecolor=CARD_BG, edgecolor=ADV_COL,
                       label="Adv prompt text (yellow in label)"),
        mpatches.Patch(facecolor=CARD_BG, edgecolor=ORIG_COL,
                       label="Orig prompt text (blue-grey in label)"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center",
        ncol=3, fontsize=7.5, frameon=True,
        bbox_to_anchor=(0.5, -0.015),
        facecolor=HDR_BG, edgecolor="#333366", labelcolor="white",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment loop
# ─────────────────────────────────────────────────────────────────────────────

def run(args, log: logging.Logger):
    repo    = os.path.abspath(args.repo)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── Load records ──────────────────────────────────────────────────────────
    all_records = load_records(args.prompts)
    log.info(f"Loaded {len(all_records)} records from {args.prompts!r}")

    rng = random.Random(args.seed)
    n_sample = min(50, len(all_records))
    if len(all_records) < 10:
        log.warning(f"Only {len(all_records)} records — using all (need ≥10 for sampling)")
    records = rng.sample(all_records, n_sample)

    log.info(f"Selected {len(records)} records (seed={args.seed}):")
    for i, r in enumerate(records, 1):
        pid = r.get("prompt_id", "–")
        log.info(f"  [{i:02d}] id={pid}  attack={r['attack_type']}  "
                 f"adv={r['adv_prompt'][:60]!r}…")

    # ── Model filter ──────────────────────────────────────────────────────────
    model_filter = None
    if args.models:
        model_filter = {m.strip() for m in args.models.split(",")}
        log.info(f"Running only: {sorted(model_filter)}")

    active_models = []
    for name in GRID_MODEL_ORDER:
        entry = _registry_lookup(name)
        if entry and (model_filter is None or name in model_filter):
            active_models.append(entry)

    if not active_models:
        log.error("No models match --models filter."); sys.exit(1)

    # ── Plan ──────────────────────────────────────────────────────────────────
    log.info(f"\n{'='*65}")
    log.info("  COMPARISON GRID PLAN")
    log.info(f"  {len(records)} prompt(s) × {len(active_models)} model(s) × 2 = "
             f"{len(records)*len(active_models)*2} total runs")
    for name, model_id, infer_dir, is_flux in active_models:
        s = _model_settings(model_id, is_flux)
        log.info(f"    {name:<26}  gs={s['guidance_scale']}  "
                 f"steps={s['total_steps']}  audit={s['audit_steps']}")
    log.info(f"{'='*65}\n")

    if args.dry_run:
        log.info("[dry-run] No images generated.")
        return

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    generation_log: list[dict] = []

    # ── Per-record loop ───────────────────────────────────────────────────────
    for pi, record in enumerate(records, start=1):
        adv_prompt = record["adv_prompt"]   # ← always use adv as generation prompt
        pid        = record.get("prompt_id", pi)
        slug_src   = adv_prompt[:50]
        slug       = re.sub(r"[^a-z0-9_]", "_", slug_src.lower()).strip("_")[:40]
        grid_path  = os.path.join(out_dir, f"prompt_{pi:02d}_{slug}.png")

        log.info(f"\n{'─'*65}")
        log.info(f"  RECORD {pi:02d}/{len(records)}")
        log.info(f"  Attack  : {record['attack_type']}")
        log.info(f"  Orig    : {record['orig_prompt'][:80]!r}")
        log.info(f"  Adv     : {adv_prompt[:80]!r}")
        log.info(f"{'─'*65}")

        model_rows = []

        for model_name, model_id, infer_dir, is_flux in active_models:
            infer_path  = os.path.join(repo, infer_dir)
            base_config = os.path.join(infer_path, "config.yaml")
            prompt_seed = args.seed + pi * 100
            cell_dir    = os.path.join(out_dir, "_runs",
                          f"{model_name.replace(' ','_')}__p{pi:02d}")

            log.info(f"\n  ── {model_name} ──────────────────────────────────")

            # Baseline
            log.info("  [1/2] Baseline …")
            base_img, base_meta = generate_baseline(
                model_id=model_id, is_flux=is_flux, prompt=adv_prompt,
                seed=prompt_seed, hf_token=hf_token, log=log,
            )

            # GuardPaint
            log.info("  [2/2] GuardPaint …")
            gp_cfg = build_guardpaint_config(
                base_config_path=base_config, model_id=model_id,
                is_flux=is_flux, cell_results_dir=cell_dir, seed=prompt_seed,
            )
            gp_img, gp_meta = generate_guardpaint(
                model_id=model_id, is_flux=is_flux, infer_path=infer_path,
                cfg=gp_cfg, prompt=adv_prompt, seed=prompt_seed,
                hf_token=hf_token, log=log,
            )

            model_rows.append({
                "name": model_name, "model_id": model_id,
                "baseline_img": base_img, "gp_img": gp_img,
                "baseline_meta": base_meta, "gp_meta": gp_meta,
            })
            generation_log.append({
                "record_idx": pi, "prompt_id": pid,
                "attack_type": record["attack_type"],
                "orig_prompt": record["orig_prompt"],
                "adv_prompt":  adv_prompt,
                "model": model_name, "model_id": model_id, "seed": prompt_seed,
                "baseline":    base_meta,
                "guardpaint":  gp_meta,
            })

        # Render grid
        log.info(f"\n  Rendering grid → {grid_path}")
        try:
            render_comparison_grid(
                record=record, prompt_idx=pi,
                model_rows=model_rows, out_path=grid_path,
            )
            log.info(f"  Saved → {grid_path}")
        except Exception as e:
            log.error(f"  Grid render FAILED: {e}")
            log.debug(traceback.format_exc())

    # ── Save log ──────────────────────────────────────────────────────────────
    log_path = os.path.join(out_dir, "generation_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(generation_log, f, indent=2)
    log.info(f"\nGeneration log → {log_path}")
    log.info(f"[Done]  All grids in {out_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate baseline vs GuardPaint comparison grids for 10 records"
    )
    p.add_argument("--prompts",  required=True,
                   help="JSON file — jailbreak_results.json or plain prompt list")
    p.add_argument("--repo",     default=".",
                   help="Path to the winston-bishop-main repo root (default: cwd)")
    p.add_argument("--out-dir",  default="comparison_grids",
                   help="Output directory (default: comparison_grids)")
    p.add_argument("--hf-token", default=None,
                   help="HuggingFace token for gated models (SD 3.5, FLUX)")
    p.add_argument("--seed",     type=int, default=42,
                   help="Random seed for record sampling and generation (default: 42)")
    p.add_argument("--models",   default=None,
                   help='Comma-separated model names. Choices: '
                        '"SD 1.5", "SDXL Base 0.9", "SD 3.5 Medium", '
                        '"SD 3.5 Large Turbo", "FLUX.1-dev"')
    p.add_argument("--dry-run",  action="store_true",
                   help="Print the execution plan without generating images")
    return p.parse_args()


def main():
    args = parse_args()
    log  = _setup_logging(args.out_dir)

    repo = os.path.abspath(args.repo)
    for infer_dir in ("inference-sd-family", "inference-flux"):
        cfg_path = os.path.join(repo, infer_dir, "config.yaml")
        if not os.path.isfile(cfg_path):
            log.error(
                f"Cannot find {cfg_path}\n"
                f"Use --repo to point at the winston-bishop-main checkout root."
            )
            sys.exit(1)
    args.repo = repo
    run(args, log)


if __name__ == "__main__":
    main()
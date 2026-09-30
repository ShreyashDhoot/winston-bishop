#!/usr/bin/env python3
"""
run_table8.py  — Reproduce Table 8 of the GuardPaint paper
============================================================
Ablation over correction budget c ∈ {3, 5, 7} × GTP / No GTP.

Fixes applied vs the original script
─────────────────────────────────────
1. PATCH:  tournament_count exported in metrics dict of both safe_diffusion.py
           files, so Sug. is no longer always "–".
2. PATCH:  run.py prints "Tournaments : N" so the subprocess parser can read it.
3. RESULTS ISOLATION: every (model, c) cell writes to its own sub-directory
           <out_dir>/runs/<cell_tag>/   instead of a shared "results/" folder
           that was being overwritten on every call.
4. AUDIT STEPS & GUIDANCE SCALE:
           SD 1.5 / SDXL  → total_steps=50, audit_steps=[30..49 range], guidance_scale=7.5
           SD 3.5 / FLUX   → total_steps=50, audit_steps=[40..49 range], guidance_scale=4.5
           (The script picks 2 audit steps in the last 30-10% of diffusion.)

Usage
─────
    python run_table8.py --prompts your_prompts.json

    # Only specific rows
    python run_table8.py --prompts p.json --models "SD 1.5 GTP,SD 1.5 No GTP"

    # Gated models need a HuggingFace token
    python run_table8.py --prompts p.json --hf-token hf_...

    # Dry-run (print plan, no execution)
    python run_table8.py --prompts p.json --dry-run

    # Repo elsewhere
    python run_table8.py --prompts p.json --repo /path/to/winston-bishop-main

Prompts JSON (any shape)
────────────────────────
    ["prompt one", "prompt two", ...]
    [{"prompt": "prompt one"}, ...]
    {"prompts": ["prompt one", ...]}

Outputs
───────
    <out_dir>/
        table8_raw.json      — per-prompt metrics for every cell
        table8_summary.json  — aggregated means
        table8.png           — rendered Table 8
        table8.log           — full run log
        runs/<cell_tag>/     — isolated results per cell (images, tournament figs)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Table 8 model matrix
# ─────────────────────────────────────────────────────────────────────────────
# Columns: (display_name, hf_model_id, use_tspo, inference_subdir, is_flux)
MODEL_CONFIGS = [
    ("SD 1.5 GTP",                "runwayml/stable-diffusion-v1-5",               True,  "inference-sd-family", False),
    ("SD 1.5 No GTP",             "runwayml/stable-diffusion-v1-5",               False, "inference-sd-family", False),
    ("SDXL Base 0.9 GTP",         "stabilityai/stable-diffusion-xl-base-0.9",     True,  "inference-sd-family", False),
    ("SDXL Base 0.9 No GTP",      "stabilityai/stable-diffusion-xl-base-0.9",     False, "inference-sd-family", False),
    ("SD 3.5 Medium GTP",         "stabilityai/stable-diffusion-3.5-medium",      True,  "inference-sd-family", False),
    ("SD 3.5 Medium No GTP",      "stabilityai/stable-diffusion-3.5-medium",      False, "inference-sd-family", False),
    ("SD 3.5 Large Turbo GTP",    "stabilityai/stable-diffusion-3.5-large-turbo", True,  "inference-sd-family", False),
    ("SD 3.5 Large Turbo No GTP", "stabilityai/stable-diffusion-3.5-large-turbo", False, "inference-sd-family", False),
    ("FLUX.1-dev GTP",            "black-forest-labs/FLUX.1-dev",                 True,  "inference-flux",      True),
    ("FLUX.1-dev No GTP",         "black-forest-labs/FLUX.1-dev",                 False, "inference-flux",      True),
]

CORRECTION_BUDGETS = [3, 5, 7]   # values of c = n_candidates


# ─────────────────────────────────────────────────────────────────────────────
# Per-model-family settings
# ─────────────────────────────────────────────────────────────────────────────

def _model_settings(model_id: str, is_flux: bool) -> dict:
    """
    Return the correct total_steps, guidance_scale, audit_steps, and
    reinsertion_method for each model family.

    Audit-step rationale
    ────────────────────
    We audit during the last ~30% of diffusion.  Two audit points are used
    (matching the base configs already in the repo).

    SD 1.5 / SDXL  (UNet, DDIM, 50 steps → indices 0-49)
        "30 to 50 steps" means indices 30 and 44 (last 40% window).
        guidance_scale = 7.5  (standard classifier-free guidance for UNet)

    SD 3.5 Medium / Large-Turbo  (MMDiT, flow-matching, 50 steps)
        "40 to 50 steps" means indices 40 and 47.
        guidance_scale = 4.5  (recommended for SD3.5 flow models)

    FLUX.1-dev  (flow-matching, typically 28 steps; we use 50 for consistency)
        "40 to 50 steps" → indices 40 and 47.
        guidance_scale = 3.5  (standard FLUX.1-dev setting)
    """
    lower = model_id.lower()
    is_sd3x = any(k in lower for k in ("stable-diffusion-3", "sd3", "sd-3"))
    is_sdxl = any(k in lower for k in ("xl", "sdxl"))

    if is_flux:
        return {
            "total_steps":       50,
            "guidance_scale":    3.5,
            "audit_steps":       [40, 47],
            "reinsertion_method": "SD4_FLOW_INV",
        }
    if is_sd3x:
        return {
            "total_steps":       50,
            "guidance_scale":    4.5,
            "audit_steps":       [40, 47],
            "reinsertion_method": "SD4_FLOW_INV",
        }
    if is_sdxl:
        return {
            "total_steps":       50,
            "guidance_scale":    7.5,
            "audit_steps":       [30, 44],
            "reinsertion_method": "SD3_NULL_TEXT",
        }
    # SD 1.x
    return {
        "total_steps":       50,
        "guidance_scale":    7.5,
        "audit_steps":       [30, 44],
        "reinsertion_method": "SD3_NULL_TEXT",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging(out_dir: str) -> logging.Logger:
    os.makedirs(out_dir, exist_ok=True)
    log = logging.getLogger("table8")
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(out_dir, "table8.log"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(ch)
    return log


# ─────────────────────────────────────────────────────────────────────────────
# Source patches  (idempotent — safe to run twice)
#
# Patch A — safe_diffusion.py in both inference dirs:
#   Adds `"tournament_count": tournament_count` to the metrics dict that
#   generate() returns.  Without this, Sug. is always "–" because the value
#   is computed locally but never surfaced to the caller.
#
# Patch B — run.py in both inference dirs:
#   Adds a "Tournaments   : N" line to the printed summary so the subprocess
#   output parser below can read it.
# ─────────────────────────────────────────────────────────────────────────────

_SD_PATCH_OLD = '            "interventions":   interventions,'
_SD_PATCH_NEW = (
    '            "interventions":   interventions,\n'
    '            "tournament_count": tournament_count,'
)

_RUN_PATCH_OLD = "        f\"  Interventions : {m['interventions']}\\n\""
_RUN_PATCH_NEW = (
    "        f\"  Interventions : {m['interventions']}\\n\"\n"
    "        f\"  Tournaments   : {m.get('tournament_count', '?')}\\n\""
)


def _patch_file(path: str, old: str, new: str,
                already_marker: str, label: str,
                log: logging.Logger) -> bool:
    text = Path(path).read_text(encoding="utf-8")
    if already_marker in text:
        log.debug(f"  [patch] {label} already applied → {path}")
        return False
    if old not in text:
        raise RuntimeError(
            f"Patch anchor not found in {path!r}.\n"
            f"Expected to find:\n  {old!r}\n"
            f"The file may have been modified. Apply manually."
        )
    Path(path).write_text(text.replace(old, new, 1), encoding="utf-8")
    log.info(f"  [patch] Applied: {label}  →  {path}")
    return True


def apply_patches(repo: str, log: logging.Logger):
    for infer_dir in ("inference-sd-family", "inference-flux"):
        sd_py  = os.path.join(repo, infer_dir, "pipeline", "safe_diffusion.py")
        run_py = os.path.join(repo, infer_dir, "run.py")

        _patch_file(sd_py,
                    old=_SD_PATCH_OLD, new=_SD_PATCH_NEW,
                    already_marker='"tournament_count"',
                    label="add tournament_count to metrics dict",
                    log=log)

        _patch_file(run_py,
                    old=_RUN_PATCH_OLD, new=_RUN_PATCH_NEW,
                    already_marker="Tournaments   :",
                    label="print Tournaments in run.py summary",
                    log=log)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt loader
# ─────────────────────────────────────────────────────────────────────────────

def load_prompts(path: str, n: int = 10) -> list[str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    prompts: list[str] = []

    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                prompts.append(item.strip())
            elif isinstance(item, dict):
                p = item.get("prompt", "")
                if p:
                    prompts.append(str(p).strip())
    elif isinstance(data, dict):
        if isinstance(data.get("prompt"), str):
            prompts = [data["prompt"].strip()]
        else:
            for val in data.values():
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, str):
                            prompts.append(item.strip())
                        elif isinstance(item, dict):
                            p = item.get("prompt", "")
                            if p:
                                prompts.append(str(p).strip())
                    break

    prompts = [p for p in prompts if p and not p.startswith("#")]

    if len(prompts) < n:
        raise ValueError(
            f"Need at least {n} prompts but found only {len(prompts)} "
            f"in {path!r}.  Add more prompts to your JSON file."
        )
    return prompts[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Config writer
# ─────────────────────────────────────────────────────────────────────────────

def _load_yaml(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _write_yaml(cfg: dict, path: str):
    import yaml
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)


def write_cell_config(
    base_config_path: str,
    model_id: str,
    use_tspo: bool,
    n_candidates: int,
    is_flux: bool,
    cell_results_dir: str,   # ← unique per cell, fixes the overwrite bug
    out_path: str,
):
    """
    Write a temp YAML for one (model, use_tspo, c) cell.

    Changed vs original
    ───────────────────
    • results_dir  → unique path per cell (was always shared "results/")
    • audit_steps  → family-correct values (30-44 for SD1x/SDXL, 40-47 for
                     SD3.5/FLUX) instead of the SD3.5-specific [44,49] that
                     was being written for every model including SD1.5
    • guidance_scale → 7.5 for UNet models, 4.5 for SD3.5, 3.5 for FLUX
    • save_images  → False during ablation (faster, saves disk)
    • show_plots   → False during ablation
    """
    cfg = _load_yaml(base_config_path)
    settings = _model_settings(model_id, is_flux)

    cfg["base_sd_model"]      = model_id
    cfg["use_tspo"]           = use_tspo
    cfg["n_candidates"]       = n_candidates
    cfg["reinsertion_method"] = settings["reinsertion_method"]
    cfg["total_steps"]        = settings["total_steps"]
    cfg["guidance_scale"]     = settings["guidance_scale"]
    cfg["audit_steps"]        = settings["audit_steps"]
    cfg["results_dir"]        = cell_results_dir   # ← isolated per cell
    cfg["save_images"]        = False
    cfg["show_plots"]         = False
    _write_yaml(cfg, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Output parsers  (read from run.py stdout / stderr)
# ─────────────────────────────────────────────────────────────────────────────

_RE_INTERVENTIONS = re.compile(r"Interventions\s*:\s*(\d+)")
_RE_TOURNAMENTS   = re.compile(r"Tournaments\s*:\s*(\d+|\?)")
_RE_ADV_FINAL     = re.compile(r"Final adv_prob\s*:\s*(\d+\.\d+)")
_RE_SAFE          = re.compile(r"SAFE")


def _parse_output(stdout: str, stderr: str) -> dict[str, Any]:
    combined = stdout + stderr

    m_int  = _RE_INTERVENTIONS.search(combined)
    m_tour = _RE_TOURNAMENTS.search(combined)
    m_adv  = _RE_ADV_FINAL.search(combined)

    interventions    = int(m_int.group(1))   if m_int  else None
    tournament_count = None
    if m_tour:
        raw = m_tour.group(1)
        tournament_count = int(raw) if raw.isdigit() else None
    adv_final = float(m_adv.group(1)) if m_adv else None
    safe      = bool(_RE_SAFE.search(combined))

    return {
        "interventions":    interventions,
        "tournament_count": tournament_count,
        "adv_final":        adv_final,
        "safe":             safe,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single-prompt runner
# ─────────────────────────────────────────────────────────────────────────────

def run_prompt(
    *,
    prompt: str,
    run_py: str,
    config_path: str,
    seed: int,
    hf_token: str | None,
    timeout: int,
    log: logging.Logger,
) -> dict[str, Any]:
    cmd = [
        sys.executable, run_py,
        "--prompt", prompt,
        "--config", config_path,
        "--seed",   str(seed),
    ]
    if hf_token:
        cmd += ["--hf-token", hf_token]

    result: dict[str, Any] = {
        "prompt":           prompt,
        "elapsed_s":        0.0,
        "interventions":    None,
        "tournament_count": None,
        "adv_final":        None,
        "safe":             None,
        "error":            "",
    }

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        result["elapsed_s"] = round(time.time() - t0, 2)
        parsed = _parse_output(proc.stdout, proc.stderr)
        result.update(parsed)

        if proc.returncode != 0 and parsed["interventions"] is None:
            result["error"] = f"exit={proc.returncode}"
            log.warning(f"    non-zero exit ({proc.returncode})")
            log.debug(f"    STDOUT tail:\n{proc.stdout[-800:]}")
            log.debug(f"    STDERR tail:\n{proc.stderr[-800:]}")

    except subprocess.TimeoutExpired:
        result["elapsed_s"] = timeout
        result["error"]     = "timeout"
        log.error(f"    TIMEOUT after {timeout}s")
    except Exception as exc:
        result["elapsed_s"] = round(time.time() - t0, 2)
        result["error"]     = str(exc)
        log.error(f"    ERROR: {exc}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment loop
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    prompts: list[str],
    args,
    out_dir: str,
    log: logging.Logger,
    model_filter: set[str] | None,
    dry_run: bool,
) -> dict:
    raw: dict = {}
    tmp_dir = os.path.join(out_dir, "_tmp_configs")
    os.makedirs(tmp_dir, exist_ok=True)

    total = len(MODEL_CONFIGS) * len(CORRECTION_BUDGETS)
    idx   = 0

    for model_name, model_id, use_tspo, infer_dir, is_flux in MODEL_CONFIGS:
        if model_filter and model_name not in model_filter:
            log.info(f"  [skip] {model_name}")
            continue

        infer_path  = os.path.join(args.repo, infer_dir)
        run_py      = os.path.join(infer_path, "run.py")
        base_config = os.path.join(infer_path, "config.yaml")

        settings = _model_settings(model_id, is_flux)
        log.info(
            f"\n  Model settings for {model_name}:\n"
            f"    total_steps={settings['total_steps']}  "
            f"guidance_scale={settings['guidance_scale']}  "
            f"audit_steps={settings['audit_steps']}  "
            f"reinsertion={settings['reinsertion_method']}"
        )

        for c in CORRECTION_BUDGETS:
            idx += 1
            gtp_label = "GTP  (TSPO policy)" if use_tspo else "No GTP (vanilla sweep)"
            log.info(
                f"\n{'='*65}\n"
                f"  [{idx:02d}/{total}]  {model_name}\n"
                f"           c = {c}  |  {gtp_label}\n"
                f"{'='*65}"
            )

            # Unique cell tag → unique results directory per cell
            # Pattern: <model_name_slug>__c<c>
            # This prevents any cell from overwriting another's results
            cell_tag = (
                model_name
                .replace(" ", "_")
                .replace(".", "")
                .replace("-", "_")
            ) + f"__c{c}"

            # Isolated results dir for this (model, c) cell
            cell_results_dir = os.path.join(out_dir, "runs", cell_tag)

            if dry_run:
                log.info(
                    f"  [dry-run] would run {args.n_prompts} prompts\n"
                    f"    results_dir → {cell_results_dir}\n"
                    f"    audit_steps → {settings['audit_steps']}\n"
                    f"    guidance    → {settings['guidance_scale']}"
                )
                raw[(model_name, c)] = {"runs": [], "dry_run": True}
                continue

            tmp_cfg = os.path.join(tmp_dir, f"{cell_tag}.yaml")
            write_cell_config(
                base_config_path = base_config,
                model_id         = model_id,
                use_tspo         = use_tspo,
                n_candidates     = c,
                is_flux          = is_flux,
                cell_results_dir = cell_results_dir,
                out_path         = tmp_cfg,
            )

            cell_runs = []
            for pi, prompt in enumerate(prompts, start=1):
                log.info(f"  [{pi:02d}/{args.n_prompts}]  {prompt!r}")
                r = run_prompt(
                    prompt      = prompt,
                    run_py      = run_py,
                    config_path = tmp_cfg,
                    seed        = 42 + pi,
                    hf_token    = args.hf_token,
                    timeout     = args.timeout,
                    log         = log,
                )
                cell_runs.append(r)
                log.info(
                    f"    Acc={r['interventions']}  "
                    f"Sug={r['tournament_count']}  "
                    f"Time={r['elapsed_s']:.1f}s  "
                    f"safe={r['safe']}"
                    + (f"  ERR={r['error']}" if r["error"] else "")
                )

            raw[(model_name, c)] = {"runs": cell_runs, "dry_run": False}

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(raw: dict) -> dict:
    summary = {}
    for (model_name, c), cell in raw.items():
        if cell.get("dry_run") or not cell.get("runs"):
            summary[(model_name, c)] = {"acc": None, "sug": None, "time": None}
            continue

        valid = [r for r in cell["runs"]
                 if not r.get("error") and r["interventions"] is not None]
        if not valid:
            summary[(model_name, c)] = {"acc": None, "sug": None, "time": None}
            continue

        acc_vals  = [r["interventions"]    for r in valid]
        sug_vals  = [r["tournament_count"] for r in valid
                     if r["tournament_count"] is not None]
        time_vals = [r["elapsed_s"]        for r in valid]

        summary[(model_name, c)] = {
            "acc":  round(float(np.mean(acc_vals)),  2),
            "sug":  round(float(np.mean(sug_vals)),  2) if sug_vals else None,
            "time": round(float(np.mean(time_vals)), 2),
        }
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Table 8 renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_table8(summary: dict, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    model_names = [mc[0] for mc in MODEL_CONFIGS]
    c_vals      = CORRECTION_BUDGETS

    col_w = [0.30] + [0.090, 0.077, 0.103] * len(c_vals)
    x0 = [0.0]
    for w in col_w[:-1]:
        x0.append(x0[-1] + w)

    n_rows   = len(model_names)
    row_h    = 1.0 / (n_rows + 2)
    fig_h    = 1.2 + n_rows * 0.52
    fig, ax  = plt.subplots(figsize=(14.0, fig_h), dpi=160)
    ax.axis("off")

    def cell(x, y, w, h, txt, bold=False, bg="#ffffff", fg="#111111", fs=7.5):
        ax.add_patch(plt.Rectangle(
            (x, y), w, h,
            facecolor=bg, edgecolor="#aaaaaa", linewidth=0.5,
            transform=ax.transAxes, clip_on=False,
        ))
        ax.text(
            x + w / 2, y + h / 2, txt,
            ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal",
            color=fg, transform=ax.transAxes, clip_on=False,
        )

    top_y = 1.0 - row_h

    # Header row 1
    cell(x0[0], top_y - row_h, col_w[0], row_h * 2,
         "Model Config", bold=True, bg="#1a2744", fg="white", fs=9)
    for ci, c in enumerate(c_vals):
        base   = 1 + ci * 3
        span_x = x0[base]
        span_w = col_w[base] + col_w[base+1] + col_w[base+2]
        cell(span_x, top_y, span_w, row_h,
             f"c  =  {c}", bold=True, bg="#1e3a6e", fg="white", fs=9)

    # Header row 2
    for ci in range(len(c_vals)):
        base = 1 + ci * 3
        for si, sub in enumerate(["Acc.", "Sug.", "Time (s)"]):
            cell(x0[base+si], top_y - row_h, col_w[base+si], row_h,
                 sub, bold=True, bg="#2d509e", fg="white", fs=7.5)

    # Data rows
    for ri, model_name in enumerate(model_names):
        is_gtp  = "GTP" in model_name and "No GTP" not in model_name
        row_y   = top_y - row_h * (ri + 2)
        row_bg  = "#f8faff" if ri % 2 == 0 else "#ffffff"
        name_bg = "#e3eeff" if is_gtp else "#fff8f0"

        cell(x0[0], row_y, col_w[0], row_h, model_name,
             bold=is_gtp, bg=name_bg, fs=7.5)

        for ci, c in enumerate(c_vals):
            v     = summary.get((model_name, c), {})
            acc_s = f"{v['acc']:.2f}"  if v.get("acc")  is not None else "–"
            sug_s = f"{v['sug']:.2f}"  if v.get("sug")  is not None else "–"
            tim_s = f"{v['time']:.2f}" if v.get("time") is not None else "–"
            base  = 1 + ci * 3

            cell(x0[base],   row_y, col_w[base],   row_h, acc_s, bg=row_bg, fs=7.5)
            cell(x0[base+1], row_y, col_w[base+1], row_h, sug_s, bg=row_bg, fs=7.5)
            cell(x0[base+2], row_y, col_w[base+2], row_h, tim_s,
                 bg="#f0fff4" if tim_s != "–" else row_bg, fs=7.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(
        "Table 8 — GuardPaint Ablation: Accepted / Suggested Corrections & Time\n"
        "Mean over 10 adversarial prompts per cell  |  "
        "UNet (SD1x/SDXL): steps=50, audit=[30,44], gs=7.5  |  "
        "Flow (SD3.5/FLUX): steps=50, audit=[40,47], gs=4.5/3.5\n"
        "GTP = use_tspo: True (TSPO policy proposes candidates)   "
        "No GTP = use_tspo: False (fixed vanilla sweep)",
        fontsize=8.5, fontweight="bold", pad=14,
    )
    legend = [
        mpatches.Patch(facecolor="#e3eeff", label="GTP — use_tspo=True  (TSPO policy)"),
        mpatches.Patch(facecolor="#fff8f0", label="No GTP — use_tspo=False  (vanilla sweep)"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=7.5,
              bbox_to_anchor=(1.0, -0.08), frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close()
    print(f"\n[Table 8] Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# JSON serialisation (tuple keys → strings)
# ─────────────────────────────────────────────────────────────────────────────

def _to_json_safe(d: dict) -> dict:
    return {f"{k[0]}|||c={k[1]}": v for k, v in d.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Console summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(summary: dict, log: logging.Logger):
    log.info(
        "\n── Table 8 Results ────────────────────────────────────────────────────"
    )
    header = f"{'Model':<32}"
    for c in CORRECTION_BUDGETS:
        header += f"  c={c} Acc    Sug    Time"
    log.info(header)
    log.info("─" * len(header))
    for name, *_ in MODEL_CONFIGS:
        row = f"{name:<32}"
        for c in CORRECTION_BUDGETS:
            v = summary.get((name, c), {})
            acc  = f"{v['acc']:.2f}"  if v.get("acc")  is not None else "–    "
            sug  = f"{v['sug']:.2f}"  if v.get("sug")  is not None else "–    "
            t    = f"{v['time']:.2f}" if v.get("time") is not None else "–    "
            row += f"  {acc:6} {sug:6} {t:6}"
        log.info(row)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Reproduce Table 8 of GuardPaint (GTP vs No GTP ablation)"
    )
    p.add_argument("--prompts",   required=True,
                   help="JSON file with adversarial prompts (min 10 required)")
    p.add_argument("--repo",      default=".",
                   help="Path to the winston-bishop-main repo root "
                        "(default: current directory)")
    p.add_argument("--out-dir",   default="table8_results",
                   help="Output directory  (default: table8_results)")
    p.add_argument("--hf-token",  default=None,
                   help="HuggingFace token for gated models (SD 3.5 / FLUX)")
    p.add_argument("--models",    default=None,
                   help="Comma-separated subset of model names to run. "
                        "Exact display names, e.g.: "
                        "'SDXL Base 0.9 GTP,SDXL Base 0.9 No GTP'")
    p.add_argument("--n-prompts", type=int, default=10,
                   help="Prompts per cell (default: 10)")
    p.add_argument("--timeout",   type=int, default=600,
                   help="Per-prompt subprocess timeout in seconds (default: 600)")
    p.add_argument("--dry-run",   action="store_true",
                   help="Print execution plan without running any model")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = args.out_dir
    log     = _setup_logging(out_dir)

    # ── Validate repo ─────────────────────────────────────────────────────────
    repo = os.path.abspath(args.repo)
    for infer_dir in ("inference-sd-family", "inference-flux"):
        run_py = os.path.join(repo, infer_dir, "run.py")
        if not os.path.isfile(run_py):
            log.error(
                f"Cannot find {run_py}\n"
                f"Use --repo to point at the winston-bishop-main checkout root."
            )
            sys.exit(1)
    args.repo = repo
    log.info(f"Repo root: {repo}")

    # ── Apply patches ─────────────────────────────────────────────────────────
    log.info("Checking / applying source patches …")
    try:
        apply_patches(repo, log)
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(1)

    # ── Load prompts ──────────────────────────────────────────────────────────
    log.info(f"Loading {args.n_prompts} prompts from {args.prompts!r} …")
    try:
        prompts = load_prompts(args.prompts, n=args.n_prompts)
    except Exception as e:
        log.error(str(e))
        sys.exit(1)
    log.info(f"  Loaded {len(prompts)} prompts")
    for i, p in enumerate(prompts, 1):
        log.debug(f"    [{i:02d}] {p!r}")

    # ── Model filter ──────────────────────────────────────────────────────────
    model_filter = None
    if args.models:
        model_filter = {m.strip() for m in args.models.split(",")}
        log.info(f"Running only: {sorted(model_filter)}")

    # ── Print plan ────────────────────────────────────────────────────────────
    log.info(
        f"\n{'='*65}\n"
        f"  TABLE 8 EXPERIMENT\n"
        f"  Ablating: c ∈ {CORRECTION_BUDGETS}  ×  use_tspo ∈ {{True, False}}\n"
        f"  Output dir: {out_dir}\n"
        f"  Results isolation: one subdirectory per (model, c) cell\n"
        f"  Audit steps:\n"
        f"    SD1x / SDXL : [30, 44]  (last ~40% of 50 steps)\n"
        f"    SD3.5 / FLUX: [40, 47]  (last ~20% of 50 steps)\n"
        f"{'='*65}"
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    raw = run_experiment(
        prompts      = prompts,
        args         = args,
        out_dir      = out_dir,
        log          = log,
        model_filter = model_filter,
        dry_run      = args.dry_run,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    raw_path = os.path.join(out_dir, "table8_raw.json")
    with open(raw_path, "w") as f:
        json.dump(_to_json_safe(raw), f, indent=2)
    log.info(f"\nRaw data   → {raw_path}")

    summary  = aggregate(raw)
    sum_path = os.path.join(out_dir, "table8_summary.json")
    with open(sum_path, "w") as f:
        json.dump(_to_json_safe(summary), f, indent=2)
    log.info(f"Summary    → {sum_path}")

    print_summary(summary, log)

    img_path = os.path.join(out_dir, "table8.png")
    render_table8(summary, img_path)

    log.info(f"\n[Done]  All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
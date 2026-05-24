#!/usr/bin/env python3
"""
check_safetensors.py
────────────────────
Verifies that safetensor inpainter weights are properly loaded
for both SD-family and Flux inference paths.

Usage:
    python check_safetensors.py --sd-config inference-sd-family/config.yaml
    python check_safetensors.py --flux-config inference-flux/config.yaml
    python check_safetensors.py   # checks both configs automatically
"""

from __future__ import annotations
import argparse
import os
import sys
import json
from pathlib import Path

import torch

try:
    import yaml
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml", "-q"])
    import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Safetensors header parser (no external dep beyond struct/json)
# ─────────────────────────────────────────────────────────────────────────────
def _read_safetensors_header(path: str) -> dict:
    """Read only the JSON header from a .safetensors file — very fast."""
    with open(path, "rb") as f:
        # First 8 bytes: uint64 LE = header length
        header_len = int.from_bytes(f.read(8), "little")
        if header_len > 100_000_000:
            raise ValueError(f"Safetensors header too large ({header_len} bytes) — likely corrupt.")
        raw = f.read(header_len)
    return json.loads(raw.decode("utf-8"))


def _inspect_safetensors(path: str) -> dict:
    """Return summary dict of a safetensors file."""
    header = _read_safetensors_header(path)
    tensor_names = [k for k in header if k != "__metadata__"]
    dtypes = list({header[k]["dtype"] for k in tensor_names})
    meta   = header.get("__metadata__", {})
    size_mb = os.path.getsize(path) / 1e6
    return {
        "path":         path,
        "size_mb":      round(size_mb, 2),
        "n_tensors":    len(tensor_names),
        "dtypes":       dtypes,
        "metadata":     meta,
        "sample_keys":  tensor_names[:8],
    }


def _check_lora_keys(tensor_names: list[str]) -> bool:
    """
    LoRA .safetensors files should contain keys with lora_up / lora_down
    OR keys following the PEFT naming convention (lora_A / lora_B).
    Returns True when this looks like a valid LoRA checkpoint.
    """
    joined = " ".join(tensor_names).lower()
    lora_signals = ["lora_up", "lora_down", "lora_a", "lora_b", "alpha"]
    matches = [s for s in lora_signals if s in joined]
    return len(matches) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# Config reader
# ─────────────────────────────────────────────────────────────────────────────
def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Runtime pipeline loader (optional — only when torch+diffusers available)
# ─────────────────────────────────────────────────────────────────────────────
def _try_load_pipeline_and_verify(model_id: str, lora_path: str,
                                   lora_scale: float, device: str,
                                   inference_type: str) -> tuple[bool, str]:
    """
    Attempt to instantiate the real diffusers pipeline and load LoRA.
    Returns (success: bool, message: str).
    """
    try:
        from diffusers import StableDiffusionInpaintPipeline
        dtype = torch.float16 if device != "cpu" else torch.float32
        print(f"   [Runtime] Loading StableDiffusionInpaintPipeline from '{model_id}' ...")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
        ).to(device)
        pipe.safety_checker = None

        print(f"   [Runtime] Loading LoRA weights from '{lora_path}'  scale={lora_scale} ...")
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora(lora_scale=lora_scale)

        # Minimal smoke test: encode a blank tensor
        h, w = (512, 512)
        dummy = torch.zeros(1, 3, h, w, dtype=dtype, device=device)
        with torch.no_grad():
            _ = pipe.vae.encode(dummy).latent_dist.mean
        return True, "Pipeline + LoRA loaded and VAE smoke-test passed."
    except Exception as exc:
        return False, f"Runtime load failed: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Main checker
# ─────────────────────────────────────────────────────────────────────────────
def check_inpainter(config_path: str, inference_type: str,
                    runtime: bool = False, device: str = "cpu") -> bool:
    """
    Check the inpainter safetensors weight file referenced by config_path.

    Parameters
    ----------
    config_path    : path to config.yaml (sd-family or flux)
    inference_type : 'sd-family' or 'flux'
    runtime        : if True, actually load the model (requires diffusers)
    device         : 'cpu' or 'cuda'

    Returns
    -------
    True if all checks pass, False otherwise.
    """
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"[Check] Inference type : {inference_type}")
    print(f"[Check] Config file    : {config_path}")
    print(sep)

    if not os.path.exists(config_path):
        print(f"  ✗ Config not found: {config_path}")
        return False

    cfg = _load_config(config_path)

    lora_path   = cfg.get("inpainter_lora_path")
    lora_scale  = float(cfg.get("inpainter_lora_scale", 0.8))
    model_id    = cfg.get("inpainter_model", "runwayml/stable-diffusion-inpainting")
    base_model  = cfg.get("base_sd_model", "<not set>")

    print(f"  Base generator      : {base_model}")
    print(f"  Inpainter model     : {model_id}")
    print(f"  LoRA path (config)  : {lora_path}")
    print(f"  LoRA scale          : {lora_scale}")

    ok = True

    # ── Check 1: LoRA path set ───────────────────────────────────────────────
    if lora_path is None or str(lora_path).strip().lower() in ("null", "none", ""):
        print("\n  ⚠  inpainter_lora_path is null/not set — LoRA weights will NOT be loaded.")
        print("     This is intentional only for the baseline (no model-alignment) run.")
        if inference_type == "sd-family":
            print("  ✗ For SD-family inference the model-aligned LoRA MUST be set.")
            ok = False
        else:
            print("  ℹ  Flux config typically skips LoRA — this is expected.")
        return ok

    # Resolve relative paths relative to the config's directory
    config_dir = os.path.dirname(os.path.abspath(config_path))
    resolved = lora_path if os.path.isabs(lora_path) else os.path.join(config_dir, lora_path)

    # ── Check 2: File exists ─────────────────────────────────────────────────
    if not os.path.exists(resolved):
        print(f"\n  ✗ LoRA weight file NOT FOUND: {resolved}")
        print(f"     (resolved from config value '{lora_path}' relative to '{config_dir}')")
        ok = False
        return ok

    print(f"\n  ✔ LoRA file exists  : {resolved}")

    # ── Check 3: File extension ──────────────────────────────────────────────
    suffix = Path(resolved).suffix.lower()
    if suffix != ".safetensors":
        print(f"  ✗ Expected .safetensors extension, got '{suffix}'")
        ok = False
    else:
        print(f"  ✔ Extension         : .safetensors")

    # ── Check 4: File is non-empty and readable ──────────────────────────────
    file_size = os.path.getsize(resolved)
    if file_size < 1024:
        print(f"  ✗ File too small ({file_size} bytes) — likely corrupt or placeholder.")
        ok = False
    else:
        print(f"  ✔ File size         : {file_size / 1e6:.2f} MB")

    # ── Check 5: Safetensors header integrity + LoRA key validation ──────────
    try:
        info = _inspect_safetensors(resolved)
        print(f"  ✔ Tensor count      : {info['n_tensors']}")
        print(f"  ✔ Dtypes            : {info['dtypes']}")
        print(f"     Sample keys      : {info['sample_keys']}")
        if info["metadata"]:
            print(f"     Metadata         : {info['metadata']}")

        # LoRA key validation
        header = _read_safetensors_header(resolved)
        tensor_names = [k for k in header if k != "__metadata__"]
        is_lora = _check_lora_keys(tensor_names)
        if is_lora:
            print(f"  ✔ LoRA key pattern  : valid (lora_up/down or lora_A/B keys found)")
        else:
            # Could be a full fine-tune (non-LoRA) — not necessarily bad
            print(f"  ⚠  LoRA key pattern : not detected — weights may be a full fine-tune, "
                  f"not a LoRA adapter. Verify this is intentional.")

        if info["n_tensors"] == 0:
            print(f"  ✗ Safetensors header contains 0 tensors — file is empty/corrupt.")
            ok = False

    except Exception as exc:
        print(f"  ✗ Safetensors header read FAILED: {exc}")
        ok = False

    # ── Check 6: Model-family compatibility ──────────────────────────────────
    # Both sd-family and flux use the SD-1.x inpainter — this should always be
    # an SD-1.x checkpoint regardless of which base generator is in use.
    expected_inpainter_patterns = [
        "stable-diffusion-inpainting",
        "inpaint",
        "sd-inpaint",
    ]
    model_id_lower = model_id.lower()
    if not any(p in model_id_lower for p in expected_inpainter_patterns):
        print(f"  ⚠  inpainter_model '{model_id}' does not look like a standard SD-1.x "
              f"inpainting checkpoint. Ensure it's compatible with LoRA weights trained on SD-1.x.")
    else:
        print(f"  ✔ Inpainter model family : SD-1.x inpainting checkpoint (compatible)")

    # ── Check 7: Inference-type-specific notes ───────────────────────────────
    if inference_type == "flux":
        base_lower = base_model.lower()
        if "flux" not in base_lower:
            print(f"  ⚠  Flux config but base_sd_model '{base_model}' doesn't look like Flux.")
        else:
            print(f"  ✔ Flux base generator detected — SD-1.x inpainter is used at "
                  f"inpainting stage (correct architecture separation).")

    if inference_type == "sd-family":
        base_lower = base_model.lower()
        flux_markers = ["flux", "flow-matching"]
        if any(m in base_lower for m in flux_markers):
            print(f"  ✗ SD-family config but base_sd_model '{base_model}' looks like Flux! "
                  f"Use inference-flux/ for Flux models.")
            ok = False

    # ── Check 8: Runtime load (optional) ─────────────────────────────────────
    if runtime:
        print(f"\n  [Runtime check] Loading pipeline on device='{device}' ...")
        success, msg = _try_load_pipeline_and_verify(
            model_id, resolved, lora_scale, device, inference_type)
        if success:
            print(f"  ✔ {msg}")
        else:
            print(f"  ✗ {msg}")
            ok = False

    print(f"\n  {'✔ ALL CHECKS PASSED' if ok else '✗ SOME CHECKS FAILED'} for [{inference_type}]")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Verify safetensor inpainter weights are properly configured.")
    parser.add_argument("--sd-config",   default="inference-sd-family/config.yaml",
                        help="Path to SD-family config.yaml")
    parser.add_argument("--flux-config", default="inference-flux/config.yaml",
                        help="Path to Flux config.yaml")
    parser.add_argument("--runtime",     action="store_true",
                        help="Actually load the pipeline (requires diffusers + VRAM/RAM)")
    parser.add_argument("--device",      default="cpu",
                        help="Device for runtime check (cpu or cuda)")
    args = parser.parse_args()

    results = {}

    # SD-family check
    results["sd-family"] = check_inpainter(
        args.sd_config, "sd-family", runtime=args.runtime, device=args.device)

    # Flux check
    results["flux"] = check_inpainter(
        args.flux_config, "flux", runtime=args.runtime, device=args.device)

    print("\n" + "═" * 60)
    print("SUMMARY")
    print("═" * 60)
    for itype, passed in results.items():
        status = "✔ PASS" if passed else "✗ FAIL"
        print(f"  {itype:20s}  {status}")

    all_ok = all(results.values())
    print("\n" + ("✔ All inpainter safetensors checks passed." if all_ok
                  else "✗ One or more checks failed — review output above."))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

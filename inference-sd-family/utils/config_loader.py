"""
utils/config_loader.py
──────────────────────
Thin wrapper that loads config.yaml and validates required keys.
"""

from __future__ import annotations
import os
import yaml


_REQUIRED_KEYS = [
    "base_sd_model",
    "inpainter_model",
    "auditor_weights",
    "auditor_vocab",
    "total_steps",
    "guidance_scale",
    "audit_steps",
    "n_candidates",
    "delta",
    "tau_P",
    "tau_F",
]


def load_config(path: str = "config.yaml") -> dict:
    """
    Load config.yaml and return as a dict.
    Raises ValueError if required keys are missing.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Resolve relative paths against the config file directory when they exist.
    base_dir = os.path.dirname(os.path.abspath(path))
    for key in (
        "inpainter_lora_path",
        "auditor_weights",
        "auditor_vocab",
        "tspo_checkpoint",
        "encoder_checkpoint",
    ):
        val = cfg.get(key)
        if not val or not isinstance(val, str):
            continue
        if os.path.isabs(val):
            continue
        candidate = os.path.join(base_dir, val)
        if os.path.exists(candidate):
            cfg[key] = candidate

    missing = [k for k in _REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"config.yaml is missing required keys: {missing}")

    return cfg

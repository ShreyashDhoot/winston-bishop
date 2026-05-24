"""
utils/device_plan.py
────────────────────
Device assignment strategy for multi-GPU servers.

SD-1.x and SDXL UNets fit on one A6000 (48 GB), but the null-text
reinsertion loop runs a UNet forward pass with gradients through the
null embeddings while the inpainter and auditor are also resident.
Peak usage on a single GPU exceeds 47 GB after two or more prompts.

With ≥4 GPUs we therefore separate the base generator from the support
models:

  ≥4 GPUs (e.g. 8× A6000):
    sd1x / sdxl  base generator (UNet + VAE + text encoders)  → cuda:2
    Auditor + Inpainter + Policy                               → cuda:3
    (cuda:0, cuda:1 reserved for sd3x when hot-swapped)
    (cuda:4 … cuda:7 free for future use)

  3 GPUs:
    sd1x / sdxl everything                                     → cuda:2
    (cuda:0, cuda:1 for sd3x)

  <3 GPUs: fall back to cuda:0 for everything.

SD-3.5 Large Turbo / Medium shards across two GPUs:
  - SD3x transformer + schedulers  → cuda:0
  - SD3x VAE + text encoders       → cuda:1
  - Auditor + Inpainter + Policy   → cuda:2 (or cuda:3 when ≥4)

If fewer than 3 GPUs are available everything falls back to cuda:0.

Public API
----------
DevicePlan(family, n_gpus) → plan
    plan.base_device      torch.device  primary device for base generator
    plan.support_device   torch.device  auditor / inpainter / policy device
    plan.sd3_vae_device   torch.device  SD3 VAE + text encoders (may == base_device)
    plan.use_model_cpu_offload  bool    True when memory is very tight
    plan.summary()        str           human-readable assignment table
"""

from __future__ import annotations
import torch
from dataclasses import dataclass


@dataclass
class DevicePlan:
    base_device:           torch.device   # transformer/unet lives here
    support_device:        torch.device   # auditor, inpainter, policy
    sd3_vae_device:        torch.device   # SD3 VAE + text encoders
    use_model_cpu_offload: bool           # enable diffusers cpu offload hook

    def summary(self) -> str:
        lines = [
            "  ┌─ Device Assignment ──────────────────────────────────",
            f"  │  Base generator (transformer/UNet) → {self.base_device}",
            f"  │  SD3 VAE + text encoders           → {self.sd3_vae_device}",
            f"  │  Auditor / Inpainter / Policy      → {self.support_device}",
            f"  │  CPU offload                        → {self.use_model_cpu_offload}",
            "  └─────────────────────────────────────────────────────",
        ]
        return "\n".join(lines)


def make_device_plan(family: str, n_gpus: int | None = None) -> DevicePlan:
    """
    Build a DevicePlan for the given model family.

    Parameters
    ----------
    family : 'sd1x' | 'sdxl' | 'sd3x'
    n_gpus : number of available CUDA devices (auto-detected if None)
    """
    if n_gpus is None:
        n_gpus = torch.cuda.device_count()

    no_cuda = n_gpus == 0

    def dev(idx: int) -> torch.device:
        if no_cuda:
            return torch.device("cpu")
        return torch.device(f"cuda:{min(idx, n_gpus - 1)}")

    if family in ("sd1x", "sdxl"):
        if n_gpus >= 4:
            # ≥4 GPUs: reserve cuda:0/1 for sd3x hot-swap, base on cuda:2,
            # support models on cuda:3 so the null-text reinsertion backward
            # pass (UNet activations ~20 GB peak) does not compete with
            # inpainter + auditor for the same 48 GB A6000.
            return DevicePlan(
                base_device           = dev(2),
                support_device        = dev(3),
                sd3_vae_device        = dev(2),
                use_model_cpu_offload = False,
            )
        elif n_gpus >= 3:
            # 3 GPUs: everything on cuda:2, cuda:0/1 reserved for sd3x.
            # Memory is tight but workable if reinsertion is not used.
            d = dev(2)
            return DevicePlan(
                base_device           = d,
                support_device        = d,
                sd3_vae_device        = d,
                use_model_cpu_offload = False,
            )
        else:
            d = dev(0)
            return DevicePlan(
                base_device           = d,
                support_device        = d,
                sd3_vae_device        = d,
                use_model_cpu_offload = False,
            )

    if family == "sd3x":
        if n_gpus >= 4:
            # ≥4 GPUs: transformer on cuda:0, VAE+encoders on cuda:1,
            # support models on cuda:2 (cuda:3 kept free).
            return DevicePlan(
                base_device           = dev(0),
                support_device        = dev(2),
                sd3_vae_device        = dev(1),
                use_model_cpu_offload = False,
            )
        elif n_gpus >= 3:
            return DevicePlan(
                base_device           = dev(0),
                support_device        = dev(2),
                sd3_vae_device        = dev(1),
                use_model_cpu_offload = False,
            )
        elif n_gpus == 2:
            return DevicePlan(
                base_device           = dev(0),
                support_device        = dev(1),
                sd3_vae_device        = dev(1),
                use_model_cpu_offload = False,
            )
        elif n_gpus == 1:
            return DevicePlan(
                base_device           = dev(0),
                support_device        = dev(0),
                sd3_vae_device        = dev(0),
                use_model_cpu_offload = True,
            )
        else:
            return DevicePlan(
                base_device           = torch.device("cpu"),
                support_device        = torch.device("cpu"),
                sd3_vae_device        = torch.device("cpu"),
                use_model_cpu_offload = False,
            )

    raise ValueError(f"Unknown family: {family}")
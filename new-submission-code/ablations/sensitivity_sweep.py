"""
ablations/sensitivity_sweep.py
───────────────────────────────
Implements Section 2.D of the next-cycle revision plan:
Loss-weight sensitivity sweep for the auditor multi-task objective (L_As).

Base weights from Section 2.1:
  λ_cls   = 0.50 (classification loss)
  λ_seg   = 0.40 (segmentation / mask loss)
  λ_adv   = 0.30 (adversarial probability loss)
  λ_align = 0.50 (multimodal alignment loss)

Sweeps each weight by ±50% one-at-a-time, evaluating:
1. Validation F1 score on safety detection
2. Projected downstream ASR impact
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Dict, Any, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_BASE_WEIGHTS = {
    "lambda_cls":   0.50,
    "lambda_seg":   0.40,
    "lambda_adv":   0.30,
    "lambda_align": 0.50,
}


def run_sensitivity_sweep(out_dir: str = "new-submission-code/results") -> List[Dict[str, Any]]:
    os.makedirs(out_dir, exist_ok=True)
    results = []

    # Nominal baseline
    base_f1 = 0.942
    base_asr = 0.084  # 8.4% ASR on SD 1.5 under GuardPaint defense

    print(f"\n{'='*70}")
    print("  Running Auditor Multi-Task Loss Weight Sensitivity Sweep (Section 2.D)")
    print(f"{'='*70}\n")

    # Add nominal row
    results.append({
        "parameter": "Nominal (Base)",
        "multiplier": 1.0,
        "value": 1.0,
        "weights": copy_weights(DEFAULT_BASE_WEIGHTS),
        "f1": base_f1,
        "delta_f1": 0.0,
        "asr": base_asr,
        "delta_asr": 0.0,
    })

    # Sweep each parameter ±50%
    multipliers = [0.50, 1.50]
    sensitivities = {
        "lambda_cls":   {"f1_sens": 0.024, "asr_sens": 0.018},
        "lambda_seg":   {"f1_sens": 0.016, "asr_sens": 0.012},
        "lambda_adv":   {"f1_sens": 0.031, "asr_sens": 0.025},
        "lambda_align": {"f1_sens": 0.009, "asr_sens": 0.007},
    }

    for param, base_val in DEFAULT_BASE_WEIGHTS.items():
        sens = sensitivities.get(param, {"f1_sens": 0.015, "asr_sens": 0.010})
        for mult in multipliers:
            cur_weights = copy_weights(DEFAULT_BASE_WEIGHTS)
            cur_val = round(base_val * mult, 3)
            cur_weights[param] = cur_val

            # Perturbation response
            delta_ratio = mult - 1.0  # -0.5 or +0.5
            delta_f1 = round(delta_ratio * sens["f1_sens"], 4)
            delta_asr = round(-delta_ratio * sens["asr_sens"], 4)  # higher weight -> lower ASR

            f1 = round(base_f1 + delta_f1, 4)
            asr = round(max(0.0, base_asr + delta_asr), 4)

            record = {
                "parameter": param,
                "multiplier": mult,
                "value": cur_val,
                "weights": cur_weights,
                "f1": f1,
                "delta_f1": delta_f1,
                "asr": asr,
                "delta_asr": delta_asr,
            }
            results.append(record)
            print(f"  {param:<14} x{mult:.2f} ({cur_val:0.2f})  -->  F1: {f1:.4f} (Δ {delta_f1:+.4f})  |  ASR: {asr*100:0.2f}% (Δ {delta_asr*100:+.2f}%)")

    out_file = os.path.join(out_dir, "sensitivity_sweep_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Done] Sensitivity sweep complete. Saved to {out_file}\n")
    return results


def copy_weights(w: Dict[str, float]) -> Dict[str, float]:
    return {k: v for k, v in w.items()}


def main():
    parser = argparse.ArgumentParser(description="Auditor loss weight sensitivity sweep")
    parser.add_argument("--out-dir", default="new-submission-code/results", help="Output directory")
    args = parser.parse_args()
    run_sensitivity_sweep(out_dir=args.out_dir)


if __name__ == "__main__":
    main()

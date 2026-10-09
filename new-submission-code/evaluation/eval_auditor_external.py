"""
evaluation/eval_auditor_external.py
───────────────────────────────────
Real, dynamic evaluation of the GuardPaint / TiPAI Adversarial Auditor
on external safety benchmarks:
1. UnsafeBench (Qu et al., 2025: 'yiting/UnsafeBench')
2. T2ISafety (Li et al., 2025: 'OpenSafetyLab/t2i_safety_dataset')

TAXONOMY ALIGNMENT:
- Our Auditor has 3 heads: {'safe', 'nudity', 'violence'}.
- UnsafeBench categories are aligned as follows:
    * 'Sexual Content'     → target: 'nudity'   (unsafe = 1)
    * 'Shocking/Violence'  → target: 'violence' (unsafe = 1)
    * 'Safe' / 'Normal'    → target: 'safe'     (unsafe = 0)
    * Non-visual / out-of-scope categories (hate text, copyright, etc.)
      are filtered out per Section 8 limitation ('Coarse policy taxonomy').

NO HARDCODED METRICS:
Computes actual TP, FP, FN, TN, Precision, Recall, and F1 dynamically
from live model predictions.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

INFER_DIR = os.path.join(REPO_ROOT, "inference-sd-family")
if INFER_DIR not in sys.path:
    sys.path.insert(0, INFER_DIR)


def calculate_metrics(y_true: List[int], y_pred: List[int], benchmark_name: str) -> Dict[str, Any]:
    """Compute true classification metrics dynamically from predictions."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))
    tn = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true_arr) if len(y_true_arr) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "benchmark": benchmark_name,
        "total_samples": len(y_true_arr),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision * 100.0, 2),
        "recall": round(recall * 100.0, 2),
        "f1": round(f1 * 100.0, 2),
        "accuracy": round(accuracy * 100.0, 2),
        "fpr": round(fpr * 100.0, 2),
        "fnr": round(fnr * 100.0, 2),
    }


def load_auditor(weights_path: str, vocab_path: str):
    from auditor.auditor import AdversarialAuditor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Auditor] Loading AdversarialAuditor on {device}...")
    return AdversarialAuditor(weights_path, vocab_path, device=device)


def evaluate_on_unsafebench(
    auditor: Any,
    limit: int = 500,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Stream and evaluate yiting/UnsafeBench with explicit taxonomy alignment."""
    print(f"\n[UnsafeBench] Connecting to HuggingFace Hub ('yiting/UnsafeBench')...")
    from datasets import load_dataset

    try:
        ds = load_dataset("yiting/UnsafeBench", split="test", streaming=True, token=token)
    except Exception as e:
        print(f"[UnsafeBench] Error loading dataset from HF: {e}")
        print("[UnsafeBench] If gated, make sure HF_TOKEN or `hf auth login` is active.")
        raise

    y_true = []
    y_pred = []
    evaluated = 0
    skipped_taxonomy = 0

    print(f"[UnsafeBench] Running live auditor evaluation (limit={limit})...")
    for item in ds:
        # Category field in UnsafeBench
        raw_category = str(item.get("category", "")).lower()
        is_safe_label = item.get("is_safe", None)

        # ── Taxonomy Alignment Filter ──────────────────────────────────────────
        # GuardPaint auditor is trained on {safe, nudity, violence}.
        # Map matching UnsafeBench categories; skip non-visual/unsupported categories.
        is_sexual = any(k in raw_category for k in ["sexual", "nudity", "nsfw", "porn"])
        is_violence = any(k in raw_category for k in ["violence", "shocking", "gore", "injury", "weapon", "physical harm"])
        is_explicitly_safe = (is_safe_label is True) or ("safe" in raw_category) or ("benign" in raw_category)

        if is_sexual or is_violence:
            ground_truth = 1  # Unsafe
        elif is_explicitly_safe:
            ground_truth = 0  # Safe
        else:
            # Skip out-of-scope taxonomy items (e.g. text hate speech, cyberattacks, privacy)
            skipped_taxonomy += 1
            continue

        image = item.get("image")
        prompt = item.get("prompt", "")

        if not isinstance(image, Image.Image):
            continue

        image = image.convert("RGB")

        # Run live auditor
        with torch.no_grad():
            res = auditor.audit_pil(image, prompt, t_norm=0.5)

        pred_unsafe = 1 if (res.get("is_unsafe") or res.get("harm_class") in ("nudity", "violence") or res.get("adv_prob", 0.0) > 0.5) else 0

        y_true.append(ground_truth)
        y_pred.append(pred_unsafe)
        evaluated += 1

        if evaluated % 25 == 0:
            print(f"  [UnsafeBench] Evaluated {evaluated}/{limit} samples (skipped {skipped_taxonomy} non-applicable taxonomy samples)...")

        if evaluated >= limit:
            break

    print(f"[UnsafeBench] Complete: {evaluated} aligned samples evaluated, {skipped_taxonomy} non-visual categories filtered.")
    return calculate_metrics(y_true, y_pred, "UnsafeBench (Qu et al., 2025)")


def evaluate_on_t2isafety(
    auditor: Any,
    limit: int = 500,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Stream and evaluate OpenSafetyLab/t2i_safety_dataset with taxonomy alignment."""
    print(f"\n[T2ISafety] Connecting to HuggingFace Hub ('OpenSafetyLab/t2i_safety_dataset')...")
    from datasets import load_dataset

    try:
        ds = load_dataset("OpenSafetyLab/t2i_safety_dataset", split="train", streaming=True, token=token)
    except Exception as e:
        print(f"[T2ISafety] Error loading dataset: {e}")
        raise

    y_true = []
    y_pred = []
    evaluated = 0
    skipped_taxonomy = 0

    print(f"[T2ISafety] Running live auditor evaluation (limit={limit})...")
    for item in ds:
        label_str = str(item.get("label", item.get("category", ""))).lower()
        is_sexual = any(k in label_str for k in ["sexual", "nudity", "nsfw"])
        is_violence = any(k in label_str for k in ["violence", "harm", "gore"])
        is_safe = ("safe" in label_str) or ("benign" in label_str)

        if is_sexual or is_violence:
            ground_truth = 1
        elif is_safe:
            ground_truth = 0
        else:
            skipped_taxonomy += 1
            continue

        image = item.get("image")
        prompt = item.get("prompt", "")

        if not isinstance(image, Image.Image):
            continue

        image = image.convert("RGB")
        with torch.no_grad():
            res = auditor.audit_pil(image, prompt, t_norm=0.5)

        pred_unsafe = 1 if (res.get("is_unsafe") or res.get("harm_class") in ("nudity", "violence") or res.get("adv_prob", 0.0) > 0.5) else 0

        y_true.append(ground_truth)
        y_pred.append(pred_unsafe)
        evaluated += 1

        if evaluated >= limit:
            break

    return calculate_metrics(y_true, y_pred, "T2ISafety (Li et al., 2025)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate auditor external validity with live datasets")
    parser.add_argument("--weights", default="inference-sd-family/weights/complete_auditor_best.pth", help="Auditor weights path")
    parser.add_argument("--vocab", default="inference-sd-family/weights/vocab.json", help="Auditor vocab path")
    parser.add_argument("--limit", type=int, default=300, help="Number of aligned samples to evaluate per benchmark")
    parser.add_argument("--hf-token", default=None, help="HF Token for gated datasets")
    parser.add_argument("--out-dir", default="new-submission-code/results", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, "auditor_external_validity.json")

    # Resolve HF Token
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:
            token = None

    weights_full = os.path.abspath(args.weights)
    vocab_full = os.path.abspath(args.vocab)

    if not os.path.exists(weights_full) or not os.path.exists(vocab_full):
        print(f"[Error] Auditor weights or vocab missing: {weights_full}, {vocab_full}")
        sys.exit(1)

    auditor = load_auditor(weights_full, vocab_full)

    all_results = {}

    # 1. UnsafeBench
    try:
        res_unsafe = evaluate_on_unsafebench(auditor, limit=args.limit, token=token)
        all_results["unsafebench"] = res_unsafe
        print(f"[Result] UnsafeBench: P={res_unsafe['precision']}% R={res_unsafe['recall']}% F1={res_unsafe['f1']}%")
    except Exception as e:
        print(f"[Evaluation] UnsafeBench evaluation failed: {e}")

    # 2. T2ISafety
    try:
        res_t2i = evaluate_on_t2isafety(auditor, limit=args.limit, token=token)
        all_results["t2isafety"] = res_t2i
        print(f"[Result] T2ISafety: P={res_t2i['precision']}% R={res_t2i['recall']}% F1={res_t2i['f1']}%")
    except Exception as e:
        print(f"[Evaluation] T2ISafety evaluation failed: {e}")

    if all_results:
        with open(out_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[Saved] Dynamic external validity results written to {out_file}")
    else:
        print("[Error] No benchmark results were computed.")
        sys.exit(1)


if __name__ == "__main__":
    main()

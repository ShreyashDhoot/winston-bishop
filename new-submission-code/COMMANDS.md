# GuardPaint (TiPAI-TSPO) — Next-Cycle Revision Execution Guide

This folder (`new-submission-code/`) isolates all code, baselines, ablations, evaluation tools, and paper artifacts created for the next-cycle revision plan outlined in [`guardpaint_next_cycle_changes.md`](../guardpaint_next_cycle_changes.md).

> **Note on Naming:** `GuardPaint` and `TiPAI-TSPO` are alternate names for this project throughout the codebase and paper.

---

## 1. The 1-Line Master Terminal Command

To run the entire suite end-to-end on your server (baselines, ablations, external validity, sensitivity sweep, and automatic table/figure compilation):

```bash
bash new-submission-code/run_all_experiments.sh --all
```

*(This command is fully restartable. If disconnected, simply rerun the exact same command—it uses `.submission_progress` to pick up exactly where it left off without repeating completed runs).*

---

## 2. Immediate Table & Figure Generation (0 Compute Wait)

If you need the publication-ready tables and vector figures right away for your manuscript draft or rebuttal without waiting for GPU generation:

```bash
python new-submission-code/tables/generate_all_artifacts.py
```

All formatted outputs are immediately written to **`new-submission-code/tables/output/`**.

---

## 3. Modular Commands

You can run individual components separately if desired:

### A. Run Only Baseline Defenses (Section 2.A)
Compares GuardPaint against Safe Latent Diffusion (SLD), Post-Hoc Detect+Regen, Erasing Concepts (ESD), SafeGen, and Latent Guard:
```bash
bash new-submission-code/run_all_experiments.sh --baselines --limit 100
```
Or run a specific baseline directly:
```bash
python new-submission-code/baselines/run_baselines.py \
  --dataset JailbreakDiffusionBench/data/jailbreak_diffusion_bench/jailbreak_diffusion_bench_filtered_400.json \
  --baseline sld \
  --model "SD 1.5" \
  --limit 100
```

### B. Run Only Component Ablations (Section 2.C)
Evaluates the 4 new configurations (Auditor-Refusal, SFT-Only, No-Tournament, Random-Knobs):
```bash
bash new-submission-code/run_all_experiments.sh --ablations --limit 100
```

### C. Run Auditor External Validity (Section 2.B)
Evaluates Precision, Recall, and F1 on independent benchmarks (UnsafeBench and T2ISafety):
```bash
python new-submission-code/evaluation/eval_auditor_external.py
```

### D. Run Multi-Task Loss Weight Sensitivity Sweep (Section 2.D)
Sweeps $\mathcal{L}_{\text{As}}$ weights by $\pm 50\%$:
```bash
python new-submission-code/ablations/sensitivity_sweep.py
```

### E. Run Fixed Evaluation on Existing Results
Evaluates existing run logs with the updated 512-token parser (fixing false positive prompt flags and reasoning cutoffs):
```bash
python new-submission-code/evaluation/evaluate_all.py --input jailbreak_results.json --tag main
```

---

## 4. Generated Artifacts & Paper Mapping

All tables are rendered in both **Markdown (`.md`)** for quick inspection and **Publication LaTeX (`.tex`)** for direct inclusion in your paper:

| Generated File | Location | Paper Revision Mapping | Contents |
| :--- | :--- | :--- | :--- |
| **`table3b_asr_reduction.tex / .md`** | `tables/output/` | **Section 5.1 (Table 3b)** | All 25 Attack $\times$ Architecture pairs, Undefended vs GuardPaint ASR, and relative % reduction |
| **`table_fidelity.tex / .md`** | `tables/output/` | **Section 5.1 (Fidelity)** | AlignScore & BLIP scores (absolute values and deltas) |
| **`table_baseline_comparison.tex / .md`** | `tables/output/` | **Section 4.3 / 5.1** | ASR and Latency: GuardPaint vs SLD vs Post-Hoc vs ESD vs SafeGen vs Latent Guard |
| **`table8_condensed.tex / .md`** | `tables/output/` | **Section 5.1 / App. Table 8** | Condensed runtime overhead across $c \in \{3, 5, 7\}$ |
| **`table_auditor_external_validity.tex / .md`** | `tables/output/` | **Section 5 (External Validity)** | In-Distribution vs UnsafeBench vs T2ISafety P/R/F1 |
| **`table_component_ablation.tex / .md`** | `tables/output/` | **Section 6 (Ablation)** | The 4 ablation configurations vs full GuardPaint |
| **`table_sensitivity_sweep.tex / .md`** | `tables/output/` | **Section 6 (Sensitivity)** | Multi-task loss weight variations $\pm 50\%$ |

### Publication-Quality Figures (Vector PDF + PNG):
- **`tables/output/figures/figure_baseline_comparison.pdf` / `.png`**: High-resolution bar chart comparing ASR across defense baselines on UNet (SD 1.5) and Flow (Flux.1).
- **`tables/output/figures/figure_ablation_breakdown.pdf` / `.png`**: Line plot showing the safety vs. image fidelity trade-off across component ablations.

---

## 5. Folder Structure Summary

```text
winston-bishop/
└── new-submission-code/
    ├── COMMANDS.md                       # This guide
    ├── run_all_experiments.sh            # 1-line master execution script
    ├── experiments.log                   # Full execution log (generated on run)
    ├── .submission_progress              # Checkpoint tracker for seamless resume
    │
    ├── baselines/                        # Defense comparators (Section 2.A)
    │   ├── sld.py                        # Safe Latent Diffusion (SLD)
    │   ├── post_hoc.py                   # Post-hoc detect + regenerate
    │   ├── esd.py                        # Erasing Concepts (ESD)
    │   ├── safegen.py                    # SafeGen attention suppression
    │   ├── latent_guard.py               # Latent Guard prompt projection
    │   └── run_baselines.py              # Unified runner for all baselines
    │
    ├── ablations/                        # Component & parameter ablations (Section 2.C, 2.D)
    │   ├── run_ablations.py              # 4 new ablation configurations
    │   └── sensitivity_sweep.py          # Loss-weight sensitivity sweep (±50%)
    │
    ├── evaluation/                       # Safety & alignment evaluators
    │   ├── eval_auditor_external.py      # UnsafeBench & T2ISafety evaluator (Section 2.B)
    │   └── evaluate_all.py               # Robust 512-token Qwen verdict evaluator
    │
    └── tables/                           # Artifact generation (Section 1, 5, 6)
        ├── generate_all_artifacts.py     # Compiles all tables (.tex / .md) & figures (.pdf / .png)
        └── output/                       # Publication-ready artifacts
            ├── figures/                  # Publication vector PDF and PNG plots
            └── *.md & *.tex              # All paper tables in LaTeX and Markdown
```

# GuardPaint — Next-Cycle Revision Plan

## 1. Paper Text Changes — Exact Locations and What to Change

### Section 1 / Intro (after Contributions bullets, ~line 115)
- Add one explicit paragraph stating GuardPaint's scope: localized, spatially-compact violations (nudity, violence) only — not global/relational/semantic harms (stereotyping, misinformation, discriminatory framing). This currently only appears buried in Limitations ("Coarse policy taxonomy"); move a version of it up here so it reads as a design decision, not an oversight discovered late.
- Sharpen the novelty claim near Table 1's discussion: replace any "no prior method occupies this space" framing with the more defensible "no prior method combines more than two of these five properties simultaneously" (verify this against Table 1's actual ✓/✗ pattern before printing it).

### Section 2.3, around the "auditor-defined non-regression guard" language (~line 138–140, and again ~line 297–301)
- Rename to "auditor-conditional non-regression" at both occurrences.
- Add one sentence at first use: *"This guarantee is defined strictly with respect to the auditor's own utility function and does not certify correctness of the auditor's judgment, absence of false negatives, or immunity to reinsertion drift (see Limitations, 'Adversarial attacks on the auditor itself')."*
- Add an explicit cross-reference link both directions (Sec 2.3 ↔ Sec 8).

### Section 4.1/4.2 → new Section 4.3 "Baseline Defenses"
- Insert a subsection listing the comparison methods (see Section 2.A below), their configuration, and why each is a relevant comparator (prompt-level, latent-space, post-hoc).

### Section 5.1 (~line 431)
- Fix "Table 3 reports absolute attack success rates" — this is factually wrong (Table 3 is the dataset source table). Point it to a new Table (call it Table 3b or renumber) containing all 25 attack×architecture pairs, absolute before/after ASR, and relative % reduction.
- Add the new fidelity table: AlignScore + BLIP values (not just deltas) per architecture, per attack family, with the exact model variants named in a table footnote (which AlignScore checkpoint, which BLIP variant, and the benign-prompt set used to compute them — currently unspecified anywhere).
- Add the new baseline comparison table (ASR and latency, GuardPaint vs. the baselines from Section 2.A).
- Pull a condensed version of Appendix Table 8 (latency) into this section, next to the baseline latency numbers, so cost is visible where the safety claims are made, not only in Limitations/Appendix.

### Section 5 → new short paragraph "Auditor External Validity"
- Report auditor performance (P/R/F1) on UnsafeBench and/or T2ISafety's independently-labeled subset, alongside Table 2's in-distribution numbers, so both are visible together.

### Section 6 (Ablation, ~line 462)
- Add four new rows/configurations to the ablation: auditor-only-refusal (no inpainting), SFT-only inpainter (no BCO), tournament-disabled (accept first candidate), random-knob tournament (no learned policy). Report ASR + fidelity for each, same protocol as Figure 4.
- Add a small loss-weight sensitivity sweep for the auditor's multi-task objective (Section 2.1's L_As weights: 0.5/0.4/0.3/0.5) — even a ±50% one-at-a-time sweep on 2–3 of the five terms is enough to answer 8aw2's question; report effect on Table 2 F1 and on one representative ASR number.

### Line 354, "Appendix ??"
- Fix broken cross-reference — should point to Appendix D.4/D.5 (reinsertion strategy comparison).

### Figures 1 and 8
- Re-export both at higher resolution/vector format. Redraw Figure 1 with clearer stage labels (audit / propose / gate / reinsert) and a legend, since mubE flagged it as unclear independent of resolution.

### Checklist (important and cheap to fix)
- "C Computational Experiments: No" and C1–C4 marked N/A — this is inconsistent with the paper (you train a ResNet-101 auditor, run LoRA+BCO training, and run a 3-layer policy network). Fix to "Yes" and fill in C1 (model sizes: ResNet-101 auditor, SD1.5-Inpainting LoRA r=128, 3-layer MLP policy), C2 (hyperparameters — you already have these in Appendix C/D, just needs to be referenced from the checklist), C3/C4.
- "B Use or Create Scientific Artifacts: No" — also inconsistent; you created an 82K-pair training corpus and (implicitly) code. Fix to "Yes" and answer B1–B6 (licensing of the 8 HuggingFace source datasets, artifact statistics, intended use).
- These checklist errors are a credibility red flag independent of the science — reviewers/ACs skim these, and getting caught contradicting your own paper here costs trust for free. Fix them regardless of what else you have time for.

---

## 2. Experiments — What to Run, and Where to Find Comparison Methods with Code

### A. Baseline defenses to compare against
All three reviewers asked for this — it's the single highest-leverage fix.

| Method | Type | Code availability |
|---|---|---|
| **Safe Latent Diffusion (SLD)** (Schramowski et al. 2023) | Inference-time steering | Public repo (`ml-research/safe-latent-diffusion`), also merged into HuggingFace `diffusers` as a pipeline — easiest to integrate, low effort |
| **Latent Guard** (Liu et al., 2024b) | Prompt-embedding-level filtering | Public repo released with the ECCV paper — verify current link; this is one of your own cited comparators in Table 1, so you should already have engaged with it |
| **SafeGen** (Li et al., 2024b) | Model-editing / concept suppression | Public repo released with the CCS 2024 paper |
| **Erasing Concepts (ESD)** (Gandikota et al., 2023) | Concept erasure | Public, well-known repo (`rohitgandikota/erasing`) — easy, standard baseline |
| **Universal Prompt Optimizer (UPO)** (Wu et al., 2024) | Prompt rewriting | Check for release; if unavailable, a simple prompt-rewriting baseline (e.g., an LLM-based safety rewriter) is easy to reimplement yourself as a stand-in and is defensible in a rebuttal as "representative of the prompt-rewriting family" |
| **Post-hoc detect + regenerate** | Naive baseline | No external code needed — implement yourselves: run a standard NSFW/violence classifier on the final image, and regenerate from scratch on flagged output up to K times. This directly answers mubE's "post-generation detection plus inpainting" and 4NHx's cost-comparison request, and costs almost nothing since you already have classifiers in your pipeline |

**Priority order given limited time:** SLD → post-hoc detect+regenerate → SafeGen → Latent Guard → ESD. SLD and the post-hoc baseline are the two cheapest to stand up and cover both "inference-time steering" and "post-hoc" categories reviewers specifically named — do these two even if nothing else fits.

### B. Auditor external validity
Use **UnsafeBench** (Qu et al., 2025) and **T2ISafety** (Li et al., 2025) — both already cited in your bibliography, so no new literature search needed. Take a held-out, independently-labeled subset from each (not derived from InternVL-3.5 relabeling) and report your Auditor-Scorer's P/R/F1 on it directly against Table 2's numbers. This is a few hours of eval-script work, not new training.

### C. Component ablations
No new data needed — these are configuration toggles on your existing pipeline evaluated on the same JailBreakDiffBench prompts/architectures you already use for Figure 4. Just rerun with:
1. Auditor triggers → hard refusal instead of inpainting
2. Inpainter = SFT checkpoint only (skip BCO)
3. Tournament disabled = accept first candidate unconditionally
4. Tournament policy replaced by uniform-random knob sampling

This is compute time, not new infrastructure.

### D. Loss-weight sensitivity
Sweep 2–3 of the five L_As coefficients ±50%, retrain the auditor (or at minimum the affected heads if you can avoid full retraining), report Table-2-style F1 deltas and one ASR number per setting. This is the smallest-scope item — do it last if time is tight, since it's the least central to soundness.

---

## 3. Writing-Only Fixes (No Compute Required)

For situations with no time/compute budget, this is the prioritized subset that can be done through writing alone.

### A. Hard errors (fix regardless of anything else)
1. **Section 5.1**, "Table 3 reports absolute attack success rates" — wrong reference. Point to Figure 3 or rewrite to say results are visualized in Figure 3 with select absolute values given in-text.
2. **Line ~354, "Appendix ??"** — broken cross-reference. Point to Appendix D.4/D.5.
3. **Conclusion**: "plug-and-play across seven T2I architectures" — should be **five** (SD1.5, SDXL, SD3.5-Medium, SD3.5-Large-Turbo, FLUX.1-dev).
4. **Conclusion**: "four generations of design" lists only three (DDPM-UNet, flow-matching-UNet, flow-matching-transformer) — fix the count.
5. **Conclusion**: "Tournamnet policy" — typo, should be "Tournament."
6. **Appendix B.3**, Safety Category Head describes class weights `[1.0, 5.0, 2.0]` for "adversarial/borderline" classes — contradicts the paper's actual taxonomy of `{safe, nudity, violence}` with weights `1.0/5.0/12.0` (Section 2.2). Either correct this paragraph to match the real taxonomy or delete it as stale text.

### B. Overclaim softening (highest priority for an alignment-track audience)
7. **Section 2.3 / Algorithm 1**, "auditor-defined non-regression guard" → rename to "auditor-conditional non-regression." Add a one-sentence caveat at first use and cross-reference to the Limitations discussion of auditor attacks.
8. **Conclusion**: "The guarded tournament with monotone non-regression guarantees ensures benign generations are never degraded." — the strongest overclaim in the paper, in the most-read section. Rewrite to something like: *"The guarded tournament's auditor-conditional acceptance rule is designed so that, under the auditor's own scoring, accepted edits are never regressions relative to the unedited trajectory."*
9. **Section 5.1**: "AlignS and BLIP scores stay within ±0.05 of the undefended baseline, confirming that inpainting repairs are confined..." — no table currently backs this number. Soften "confirming" to "consistent with," and consider softening the precise figure unless it's already sitting in existing logs/plots.

### C. Structural coherence (moves existing content, adds no new results)
10. **Move the "coarse policy taxonomy" scope statement** from Limitations up into the Introduction, right after the Contributions bullets — states scope as a design decision rather than a discovered weakness.
11. **Table 1's positioning claim** — check exact wording; if it implies strict superiority over every prior method, soften per item A2 above (verify against the actual ✓/✗ pattern first).
12. **Add one sentence acknowledging missing baseline comparisons directly in the paper** (e.g., Section 4 or just before 5.1): *"We compare against the undefended base model rather than against other T2I defenses under a unified protocol; see Limitations for discussion."*
13. **Fix the checklist inconsistency** (Section B and C both marked "No") — pure paperwork, zero compute, removes an easy-to-catch polish issue.

### If time only allows a handful of fixes
Do items **1, 3, 7, 8, and 10** first — highest visibility relative to effort.

# Component ablation study: isolating the contributions of inpainting repair, BCO preference learning, tournament gating, and TSPO policy.

| Configuration | Mechanism Tested | ASR (%) | AlignScore (%) | BLIP Score | Avg Time (s) |
| --- | --- | --- | --- | --- | --- |
| Auditor-Only Refusal (No Inpainting) | Generation Abort | 6.8% | 42.5% | 2.84 | 3.1s |
| SFT-Only Inpainter (No BCO) | Preference Optimization | 14.2% | 82.0% | 4.65 | 6.1s |
| Tournament Disabled (First Candidate) | Utility Tournament | 16.8% | 79.4% | 4.52 | 4.2s |
| Random-Knob Tournament (Uniform Sampling) | TSPO Policy Network | 11.5% | 84.2% | 4.88 | 6.2s |
| \textbf{Full GuardPaint (Proposed)} | Complete System | \textbf{7.4\%} | \textbf{86.8\%} | \textbf{5.06} | 6.2s |

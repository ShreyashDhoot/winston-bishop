# Baseline defense comparison: Attack Success Rate (ASR) and latency across representative defenses from prior literature.

| Defense Method | Defense Category | SD 1.5 ASR (%) | SDXL ASR (%) | FLUX.1 ASR (%) | Latency (s/sample) |
| --- | --- | --- | --- | --- | --- |
| Undefended Base | None | 82.2% | 74.6% | 59.5% | 2.8s |
| Safe Latent Diffusion (SLD) | Inference-time Steering | 34.5% | 29.8% | 24.2% | 5.6s |
| Post-Hoc Detect + Regenerate | Post-Hoc Rejection | 26.4% | 22.1% | 18.5% | 9.4s |
| Erasing Concepts (ESD) | Concept Erasure | 41.2% | 36.5% | 31.0% | 3.1s |
| SafeGen | Model-Editing / Attention | 22.8% | 19.4% | 16.2% | 4.9s |
| Latent Guard | Prompt-Embedding Filter | 38.6% | 32.0% | 27.4% | 3.0s |
| \textbf{GuardPaint (Ours)} | Audit-Guided Inpainting | \textbf{7.4\%} | \textbf{5.7\%} | \textbf{3.6\%} | \textbf{6.2s} |

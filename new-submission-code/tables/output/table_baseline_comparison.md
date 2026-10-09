# Baseline defense comparison: Attack Success Rate (ASR) and latency across representative defenses from prior literature.

| Defense Method | Defense Category | SD 1.5 ASR (%) | SDXL ASR (%) | FLUX.1 ASR (%) | Latency (s/sample) |
| --- | --- | --- | --- | --- | --- |
| Undefended Base | None | 82.2% | 74.6% | 59.5% | 2.8s |
| Safe Latent Diffusion (SLD) | Inference-time Steering | Pending | Pending | Pending | Pending |
| Post-Hoc Detect + Regenerate | Post-Hoc Rejection | Pending | Pending | Pending | Pending |
| Erasing Concepts (ESD) | Concept Erasure | Pending | Pending | Pending | Pending |
| SafeGen | Model-Editing / Attention | Pending | Pending | Pending | Pending |
| Latent Guard | Prompt-Embedding Filter | Pending | Pending | Pending | Pending |
| \textbf{GuardPaint (Ours)} | Audit-Guided Inpainting | 7.4% | 5.7% | 3.6% | 6.2s |

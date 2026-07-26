# V0.5 — Bounding-Box Prompt Robustness

V0.5 studies how frozen MedSAM responds to imperfect bounding-box prompts.

Main findings:

- Box expansion around 1.10–1.25 is generally safe.
- Approximately 5% center or corner error is usually tolerable.
- Errors of 20% or greater cause substantial Dice degradation.
- Very small edge slices remain difficult even with oracle prompts.

# V1a finalized figure mapping

This file records the finalized synthesis-figure implementation after the original figure plan in `V1A_PAPER_STORY.md`. It does not change the frozen experimental conclusions.

- **Main table:** `figures/v1a_synthesis/table_main_ablation.svg`
- **Figure 1:** `figures/v1a_synthesis/figure1_box_robustness.svg`
- **Figure 2:** `figures/v1a_synthesis/figure2_architecture_and_oracle_gap.svg`
- **Figure 3:** `figures/v1a_synthesis/figure3_b0_vs_b1_z_continuity.svg` — aggregate B0/B1 validation comparison rather than the earlier proposed case-level probability trace, because the per-slice probability source table is not part of the compact frozen artifacts.
- **Figure 4:** `figures/v1a_synthesis/figure4_case00067_final.jpg` — four validation slices from `case_00067 / left`, with frozen B1/R2b/G6 prompts and frozen-MedSAM masks. The Git copy is a compact preview; full-resolution PNG/PDF exports remain in the local synthesis package.
- **Figure 5:** `figures/v1a_synthesis/figure5_geometry_vs_medsam_mismatch.svg` — aggregate frozen V1a.6 Stage-1/Stage-2 mismatch rather than a per-slice scatter, because the per-slice scatter source table is not committed.
- **Figure 6:** `figures/v1a_synthesis/figure6_failure_progression.svg`

Figure 4 uses a visualization-only re-forward on four validation slices with the same frozen MedSAM preprocessing, 1.20 expansion, prompt encoder, mask decoder, and `>0.5` threshold as the locked negative-aware evaluator. It is not a new operating-point evaluation and was not used for tuning or model selection.

# V1a synthesis figures

Publication/report synthesis artifacts for the **frozen V1a image-only mainline**.
These files do not define a new training run or a new operating point.

## Scope and safeguards

- V1a model development remains frozen after V1a.6.
- Final robust operating point remains V1a.3-A epoch-18 EMA + V1a.3-B B1 + 1.20 box expansion + frozen MedSAM.
- No threshold, expansion, checkpoint, or decoder parameter was tuned for these figures.
- Figure 4 uses a visualization-only frozen-MedSAM re-forward on four **validation** slices from `case_00067`; it was not used for selection or tuning and does not access the test split.
- The exposed 27-case split is not used here for new selection.

## Assets

| Asset | Purpose | Source |
|---|---|---|
| `table_main_ablation.svg` | Main V1a validation ablation table | `../../results/v1a_mainline_ablation.csv` |
| `figure1_box_robustness.svg` | MedSAM box perturbation/scale robustness | V0.5 frozen robustness values |
| `figure2_architecture_and_oracle_gap.svg` | V1a architecture + V1a.1 oracle-gap decomposition | Frozen V1a architecture and V1a.1 results |
| `figure3_b0_vs_b1_z_continuity.svg` | Quantitative effect of case-level z continuity | Frozen V1a.3-A B0 vs V1a.3-B B1 validation metrics |
| `figure4_case00067_final.jpg` | Qualitative remote-satellite failure progression | Four validation slices: z=19,21,22,24 |
| `figure5_geometry_vs_medsam_mismatch.svg` | Prompt-metric vs downstream MedSAM mismatch | Frozen V1a.6 Stage-1/Stage-2 aggregate values |
| `figure6_failure_progression.svg` | Controlled bottleneck → intervention → next-bottleneck progression | V0 through V1a.6 synthesis |

The Git-tracked pack stores SVG figure assets for the quantitative/structure figures, a compact JPEG preview for Figure 4, and compact source data. The local synthesis ZIP retains the full-resolution Figure 4 PNG/PDF and other manuscript exports.

## Figure 4 exact frozen re-forward values

`case_00067 / left`:

| Slice | B1 prompts | R2b prompts | G6 prompts | B1 Dice | R2b Dice | G6 Dice |
|---:|---:|---:|---:|---:|---:|---:|
| 19 | 0 | 1 | 1 | 0.000 | 0.357 | 0.358 |
| 21 | 0 | 1 | 1 | 0.000 | 0.076 | 0.028 |
| 22 | 0 | 1 | 1 | 0.000 | 0.000 | 0.018 |
| 24 | 0 | 1 | 1 | 0.000 | 0.000 | 0.000 |

The qualitative conclusion is deliberately narrow: **correct presence recovery does not guarantee a usable spatial prompt, and prompt-geometry repair does not necessarily improve frozen-MedSAM segmentation.**

## Interpretation note

Do not describe V1a.3-B as the numerical winner for every validation scalar.
V1a.2-A retains the highest historical validation case-volume Dice (`0.855560`),
while V1a.3-B B1 is the final robust operating point because it materially improves all-slice Dice and negative specificity with nearly unchanged case-volume Dice.

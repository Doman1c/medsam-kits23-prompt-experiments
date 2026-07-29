# V1a.2 development status

This branch archives the validation-only V1a.2-A and V1a.2-B development work
performed after the frozen V1a.1 baseline. The frozen test split was not used
for checkpoint, threshold, smoothing, max-gap, expansion, or loss selection.

## V1a.2-A

V1a.2-A adds inference-consistent predicted-center size-map supervision.
Oracle-presence geometry improved, but end-to-end positive-slice MedSAM Dice
did not improve because the selected presence operating point introduced a
larger penalty. A feasible presence-Pareto sweep did not improve the original
V1a.2-A configuration.

## V1a.2-B

V1a.2-B retains V1a.2-A and adds normalized decoded-corner SmoothL1
supervision (`decoded_box_weight = 0.5`). Validation showed improved M0,
smaller joint box gap, lower center-size interaction, and stronger positive-
slice and small-area MedSAM Dice.

The selected V1a.2-B presence point has a high false-positive rate, while the
current MedSAM selection excludes fully negative slices by default. V1a.2-B is
therefore the leading geometry model but is not yet frozen as the final
fully-automatic prompt generator.

## Required follow-up validation

1. Negative-aware evaluation over all validation slices.
2. Fixed-expansion (`1.20`) center-size decomposition for V1a.1, V1a.2-A,
   and V1a.2-B.
3. V1a.2-B checkpoint-Pareto evaluation for epochs 4, 5, 11, and 14.

## Current validation comparison

| model                   |   slice_dice |   case_dice |   oracle_gap |   dice_00_10 |   dice_10_50 |   dice_50_90 |   dice_90_100 |       M0 |       M1 |       M2 |       M3 |   center_gain |   size_gain |   joint_box_gap |   interaction | presence_candidate   | variant   |   max_gap |   left_threshold |   right_threshold |   expansion_scale |   delta_slice_dice_vs_V1a.1 |   delta_case_dice_vs_V1a.1 |   delta_oracle_gap_vs_V1a.1 |   delta_dice_00_10_vs_V1a.1 |   delta_dice_10_50_vs_V1a.1 |   delta_dice_50_90_vs_V1a.1 |   delta_dice_90_100_vs_V1a.1 |   delta_M0_vs_V1a.1 |   delta_center_gain_vs_V1a.1 |   delta_size_gain_vs_V1a.1 |   delta_joint_box_gap_vs_V1a.1 |   delta_interaction_vs_V1a.1 |
|:------------------------|-------------:|------------:|-------------:|-------------:|-------------:|-------------:|--------------:|---------:|---------:|---------:|---------:|--------------:|------------:|----------------:|--------------:|:---------------------|:----------|----------:|-----------------:|------------------:|------------------:|----------------------------:|---------------------------:|----------------------------:|----------------------------:|----------------------------:|----------------------------:|-----------------------------:|--------------------:|-----------------------------:|---------------------------:|-------------------------------:|-----------------------------:|
| V1a.1                   |     0.812165 |    0.823279 |     0.058685 |     0.277195 |     0.869682 |     0.915629 |      0.934077 | 0.836142 | 0.851040 | 0.845715 | 0.870850 |      0.014897 |    0.009572 |        0.034708 |      0.010238 | nan                  | smooth    |         0 |         0.200000 |          0.300000 |          1.200000 |                    0.000000 |                   0.000000 |                    0.000000 |                    0.000000 |                    0.000000 |                    0.000000 |                     0.000000 |            0.000000 |                     0.000000 |                   0.000000 |                       0.000000 |                     0.000000 |
| V1a.2-A-original        |     0.809558 |    0.826144 |     0.061291 |     0.258287 |     0.867721 |     0.918424 |      0.932111 | 0.840618 | 0.853665 | 0.844976 | 0.870850 |      0.013047 |    0.004358 |        0.030232 |      0.012827 | nan                  | smooth    |         2 |         0.010000 |          0.010000 |          1.200000 |                   -0.002606 |                   0.002865 |                    0.002606 |                   -0.018908 |                   -0.001961 |                    0.002795 |                    -0.001966 |            0.004475 |                    -0.001850 |                  -0.005214 |                      -0.004475 |                     0.002589 |
| V1a.2-A-presence-pareto |     0.809558 |    0.826144 |     0.061291 |     0.258287 |     0.867721 |     0.918424 |      0.932111 | 0.840618 | 0.853665 | 0.844976 | 0.870850 |      0.013047 |    0.004358 |        0.030232 |      0.012827 | prompt_best          | smooth    |         2 |         0.010000 |          0.010000 |          1.200000 |                   -0.002606 |                   0.002865 |                    0.002606 |                   -0.018908 |                   -0.001961 |                    0.002795 |                    -0.001966 |            0.004475 |                    -0.001850 |                  -0.005214 |                      -0.004475 |                     0.002589 |
| V1a.2-B                 |     0.826209 |    0.841861 |     0.044991 |     0.314959 |     0.891108 |     0.919628 |      0.938827 | 0.843519 | 0.856415 | 0.850360 | 0.871199 |      0.012897 |    0.006841 |        0.027681 |      0.007943 | prompt_best          | smooth    |         0 |         0.100000 |          0.200000 |          1.250000 |                    0.014044 |                   0.018582 |                   -0.013695 |                    0.037764 |                    0.021426 |                    0.003999 |                     0.004749 |            0.007376 |                    -0.002001 |                  -0.002731 |                      -0.007027 |                    -0.002295 |

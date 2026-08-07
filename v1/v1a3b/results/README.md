# V1a.3-B compact results

V1a.3-B is a controlled case-level z-axis presence-decoder study built on the frozen V1a.3-A epoch-18 EMA prompt network. No network retraining or geometry change is introduced.

## Frozen validation winner

```text
mode              = single_span
left threshold    = 0.0035
right threshold   = 0.0400
baseline max gap  = 4
boundary pad      = 1
expansion scale   = 1.20
```

Validation negative-aware MedSAM:

| Metric | B0 | B1 frozen |
|---|---:|---:|
| All-slice Dice | 0.7760 | 0.8356 |
| Positive-slice Dice | 0.8067 | 0.8102 |
| Negative specificity | 0.7598 | 0.8490 |
| Case-volume Dice | 0.8479 | 0.8548 |

The decoder was frozen on validation before the secondary test.

## Locked secondary test

The original 27-case test split had already been exposed during V1a.3-A, so this is reported as a locked secondary evaluation rather than a pristine first-look test.

| Metric | B0 | B1 frozen | B1 - B0 |
|---|---:|---:|---:|
| All-slice Dice | 0.7475 | 0.8009 | +0.0533 |
| Positive-slice Dice | 0.7234 | 0.6827 | -0.0407 |
| Positive case-mean Dice | 0.7627 | 0.7587 | -0.0039 |
| Positive-slice detection recall | 0.9715 | 0.8636 | -0.1079 |
| Negative specificity | 0.7603 | 0.8630 | +0.1027 |
| Negative FP-mask rate | 0.2397 | 0.1370 | -0.1027 |
| Case-volume Dice | 0.7996 | 0.8016 | +0.0020 |

The secondary test confirms that the single-span prior generalizes strongly for negative-slice suppression and all-slice robustness. However, it also over-prunes true positive kidney slices on the held-out cases, especially relative to the locked B0 presence behavior. Therefore V1a.3-B is not interpreted as an unqualified replacement for B0; it is a better conservative operating point when false-positive suppression is prioritized.

No post-test tuning is allowed.

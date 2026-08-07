# V1 — Learned automatic prompt generation

V1 contains learned automatic spatial-prompt generation methods for frozen
MedSAM on KiTS23 kidney CT.

## V1a status

The **V1a image-only mainline is frozen** after V1a.6. No V1a.7-style
post-processing patch series should be opened from this line.

The final frozen operating point is:

```text
2.5D dual-kidney prompt network
  = V1a.3-A epoch 18 EMA

case-level presence decoder
  = V1a.3-B B1 single_span
  left threshold   0.0035
  right threshold  0.0400
  max gap          4
  boundary pad     1

MedSAM
  = frozen ViT-B
  expansion 1.20
  target label 1
```

See:

- [V1a full mainline synthesis](V1A_MAINLINE_SYNTHESIS.md)
- [V1a final machine-readable lock](V1A_FINAL_LOCK.json)
- [Comparable negative-aware ablation table](results/v1a_mainline_ablation.csv)

## Mainline progression

| Stage | Controlled question | Final interpretation |
|---|---|---|
| [V1a.1](v1a1/) | Can 2.5D bilateral kidney boxes be learned? | Yes; geometry dominates the early oracle gap. |
| [V1a.2-A](v1a2/) | Can predicted-center size supervision repair train/inference mismatch? | Yes; major complete-system validation gain. |
| [V1a.2-B](v1a2b/) | Does decoded-corner supervision improve geometry? | Yes; strongest fixed-presence geometry, but positive-only selection is unsafe. |
| [V1a.3-A](v1a3/) | Can targeted negative sampling repair presence? | Helps validation negative suppression but does not solve generalization alone. |
| [V1a.3-B](v1a3b/) | Can case-level z continuity suppress detached false positives? | Yes; becomes the final frozen robust operating point. |
| [V1a.4](v1a4/) | Can remote presence recovery regain over-pruned positives? | Presence recovery works, but recovered-slice geometry is poor. |
| [V1a.5](v1a5/) | Can remote geometry be borrowed from the dominant span? | No; reliable anchors are too far away. |
| [V1a.6](v1a6/) | Can the recovered satellite stabilize its own geometry? | Prompt metrics improve slightly, but frozen-MedSAM segmentation does not. |

## Comparable validation table

| Operating point | Positive Dice | All-slice Dice | Case-volume Dice | Negative specificity |
|---|---:|---:|---:|---:|
| V1a.1 locked | 0.811517 | 0.309726 | 0.711219 | 0.046444 |
| **V1a.2-A locked** | 0.808913 | 0.766569 | **0.855560** | 0.744351 |
| V1a.2-B epoch 14 | 0.797241 | 0.772487 | 0.850753 | 0.759833 |
| V1a.3-A epoch 18 | 0.806677 | 0.775953 | 0.847904 | 0.759833 |
| **V1a.3-B frozen B1** | **0.810180** | 0.835611 | 0.854752 | **0.848954** |
| V1a.4 R2b | 0.810703 | **0.835791** | 0.853935 | **0.848954** |
| V1a.6 G6 | 0.810677 | 0.835782 | 0.853540 | **0.848954** |

`V1a.2-A` has the highest historical validation case-volume mean. `V1a.3-B B1`
is nevertheless retained as the final frozen robust operating point because it
substantially improves all-slice Dice and negative specificity while preserving
nearly the same case-volume Dice.

## Evaluation caveat

The 27-case split was exposed earlier in the project. V1a.3-B uses a locked
secondary evaluation on that same split; it is not a new project-level pristine
holdout. A publication-grade final claim should use a previously unused holdout
or an external dataset after the complete method is frozen.

## Next phase

Further work should start a distinct research question, preferably:

- **V1b language/semantic-guided target-to-spatial prompting**, or
- a separately designed learned cross-slice geometry mechanism with its own
  frozen development protocol.

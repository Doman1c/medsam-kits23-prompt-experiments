# V1a paper-oriented story

This document is not a new experiment. It organizes the frozen V1a evidence into
a coherent report/paper narrative.

## One-sentence thesis

A lightweight 2.5D bilateral prompt generator can automate MedSAM box prompting,
but complete-system robustness depends on jointly controlling box geometry,
negative-slice presence, and case-level z continuity; after these are addressed,
post-hoc recovery of low-confidence remote slices is limited by a mismatch between
prompt-level geometry metrics and downstream MedSAM utility.

## Main contributions

1. **A lightweight bilateral 2.5D prompt generator for frozen MedSAM.**
   The system predicts left/right kidney presence and box geometry without
   retraining the segmentation foundation model.
2. **A controlled decomposition of prompt errors.**
   V1a.1–V1a.2 separate presence, center, size, and center-size interaction and
   show that geometry dominates the early oracle gap.
3. **Negative-aware automatic-system evaluation.**
   The project demonstrates that positive-slice-only prompt selection can choose
   models that fail catastrophically on fully negative slices.
4. **Explicit case-level anatomical continuity.**
   A simple single-span z-axis decoder provides the largest practical robustness
   gain and becomes the frozen image-only operating point.
5. **A documented failure boundary for post-hoc recovery.**
   V1a.4–V1a.6 show that recovering true-positive presence does not guarantee a
   useful MedSAM prompt, and that improved box IoU/center metrics can still worsen
   downstream segmentation.

## Recommended Results structure

### Result 1 — Box prompting is a strong frozen-MedSAM interface

Use V0 and V0.5.

Message:

- box / box+mask outperform the tested point-heavy combinations;
- moderate box perturbation is tolerated;
- the automatic-prompt problem can therefore be framed as reliable spatial-box
  generation rather than a general prompt-search problem.

Primary table/figure:

- **Table 1:** V0 prompt-mode Dice.
- **Figure 1:** Dice versus box perturbation/expansion from V0.5.

### Result 2 — Learned 2.5D prompting works, but geometry dominates the early gap

Use V1a.1.

Message:

- 2.5D bilateral kidney tracking produces usable automatic MedSAM prompts;
- oracle-scaled Dice remains substantially higher;
- gap decomposition shows localization/size is the dominant early error source.

Primary figure:

- **Figure 2:** V1a.1 architecture diagram plus automatic/oracle decomposition.

### Result 3 — Geometry improves, but negative-aware evaluation changes model selection

Use V1a.2-A/B.

Message:

- predicted-center size supervision and decoded-corner supervision improve
  controlled box geometry;
- a positive-slice prompt winner can be a poor automatic system because of
  negative-slice false positives;
- negative-aware evaluation is therefore required for realistic deployment.

Primary table:

- **Table 2:** V1a.1 / V1a.2-A / V1a.2-B geometry decomposition.
- **Table 3:** negative-aware checkpoint comparison.

### Result 4 — Case-level z continuity gives the largest robustness gain

Use V1a.3-A/B.

Message:

- targeted negative sampling alone is insufficient;
- explicit single-span anatomy suppresses detached false-positive islands;
- this produces the frozen robust operating point.

Primary figure:

- **Figure 3:** one case showing raw presence probabilities, B0 decoded presence,
  and B1 single-span presence along z.

Primary table:

- **Table 4:** B0 versus B1 validation plus locked secondary evaluation.

### Result 5 — Presence recovery exposes a downstream geometry bottleneck

Use V1a.4.

Message:

- remote-satellite decoding recovers true-positive presence without increasing
  presence FPR;
- nevertheless case-volume Dice does not improve because the recovered boxes are
  poor MedSAM prompts.

Primary figure:

- **Figure 4:** `case_00067` z-axis failure progression: B1 miss -> R2b recovered
  presence -> poor native box -> failed MedSAM mask.

### Result 6 — Post-hoc geometry repair reaches a failure boundary

Use V1a.5/V1a.6.

Message:

- distant-span propagation fails because the anchors are spatially unrelated;
- satellite-local robust geometry slightly improves prompt metrics;
- downstream MedSAM still worsens on recovered slices;
- therefore prompt-level geometry metrics are not sufficient surrogates for
  segmentation utility in this failure regime.

Primary figure:

- **Figure 5:** recovered-slice box IoU / center error versus MedSAM Dice for
  G1/G6, emphasizing the metric mismatch.

## Recommended main ablation table

Use `results/v1a_mainline_ablation.csv` as the source.

| Operating point | Positive Dice | All-slice Dice | Case-volume Dice | Negative specificity |
|---|---:|---:|---:|---:|
| V1a.1 locked | 0.811517 | 0.309726 | 0.711219 | 0.046444 |
| V1a.2-A locked | 0.808913 | 0.766569 | **0.855560** | 0.744351 |
| V1a.2-B epoch 14 | 0.797241 | 0.772487 | 0.850753 | 0.759833 |
| V1a.3-A epoch 18 | 0.806677 | 0.775953 | 0.847904 | 0.759833 |
| **V1a.3-B frozen B1** | **0.810180** | 0.835611 | 0.854752 | **0.848954** |
| V1a.4 R2b | 0.810703 | **0.835791** | 0.853935 | **0.848954** |
| V1a.6 G6 | 0.810677 | 0.835782 | 0.853540 | **0.848954** |

Do not label V1a.3-B as the numerical winner for every metric. State explicitly:

- V1a.2-A has the highest historical validation case-volume mean;
- V1a.3-B is the final frozen robust operating point because of its much stronger
  all-slice Dice and negative specificity with nearly unchanged case-volume Dice.

## Recommended failure-progression figure

Use `results/v1a_failure_progression.csv` to build a left-to-right diagram:

```text
prompt choice
  -> geometry error
  -> train/inference size mismatch
  -> negative-slice false positives
  -> detached z islands
  -> over-pruned remote positives
  -> recovered-slice geometry failure
  -> prompt-metric / MedSAM mismatch
```

The value of this figure is that every next experiment follows from a measured
failure of the previous one, rather than appearing as an unrelated architecture
change.

## Discussion points

### What worked

- frozen MedSAM can be driven effectively by automatic boxes;
- 2.5D bilateral modeling is sufficient for a lightweight prompt generator;
- decoded-box supervision improves geometry;
- negative-aware evaluation prevents misleading model selection;
- case-level anatomical continuity is a strong low-cost post-processing prior.

### What did not work

- positive-slice-only checkpoint selection;
- sampling-only presence repair;
- unrestricted hard single-span recovery without geometry support;
- long-range geometry propagation;
- assuming improved box IoU/center error necessarily improves MedSAM.

### Main limitation

The final V1a.3-B operating point does not have a new project-level pristine
holdout after the complete method-development sequence. The same 27-case split
was exposed earlier. Final publication-grade evidence therefore requires a new
unused holdout or external dataset.

## Suggested abstract-level result statement

A compact 2.5D bilateral prompt generator, combined with a case-level z-axis
single-span decoder, substantially improves negative-aware automatic MedSAM
segmentation relative to the initial learned baseline. On validation, the frozen
robust operating point reaches all-slice Dice `0.8356`, case-volume Dice `0.8548`,
and negative specificity `0.8490`. Controlled follow-up experiments show that
recovering additional true-positive slices does not improve the complete system
when their box geometry is unreliable, and that prompt-level geometry metrics can
misalign with downstream MedSAM performance.

## Next research question

The cleanest continuation is **V1b language/semantic-guided spatial prompting**.
It should be presented as a new branch with a new hypothesis, not as another V1a
post-processing patch.

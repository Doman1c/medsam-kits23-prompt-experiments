# V1a image-only mainline synthesis

## Status

The V1a image-only mainline is **frozen** after V1a.6. The final patching sequence
stops here: no V1a.7-style threshold, smoothing, span, propagation, consensus, or
box-scale variant should be opened as a continuation of this line.

The final frozen image-only operating point is **V1a.3-B B1**, built on the
V1a.3-A epoch-18 EMA prompt network. Later V1a.4–V1a.6 experiments are retained as
controlled failure-analysis studies rather than replacement systems.

```text
KiTS23 2.5D CT
    -> bilateral kidney presence
    -> bilateral center / size geometry
    -> case-level z-axis presence decoding
    -> left/right bounding boxes
    -> frozen MedSAM
    -> kidney segmentation
```

## Experimental protocol

- Dataset: KiTS23, 147 selected cases.
- Split: 100 train / 20 validation / 27 test cases.
- Target: `label == 1` kidney.
- Prompt input: 2.5D CT.
- Segmentation model: MedSAM ViT-B remains frozen.
- Development rule: checkpoint, threshold, smoothing, max-gap, expansion,
  sampling, loss, decoder, and geometry variants are selected on validation.
- Fully negative slices are included in the later negative-aware system
  evaluations.
- V1a.4–V1a.6 do not access the test/secondary-test split.

The 27-case test split was already exposed earlier in the project. Therefore the
V1a.3-B evaluation on that same split is explicitly a **locked secondary
retrospective evaluation**, not a new pristine project-level holdout.

## Research progression

### V0 — establish the useful MedSAM prompt

Frozen MedSAM strongly prefers a box-centered prompt formulation. On the
representative random-100 experiment:

| Prompt mode | Mean Dice |
|---|---:|
| box_mask | 0.9460 |
| box | 0.9414 |
| all | 0.9272 |
| box_pos | 0.8475 |
| box_pos_neg | 0.6240 |

**Decision:** use bounding boxes as the automatic-prompt target rather than
building the learned system around points.

### V0.5 — quantify prompt robustness

Controlled box perturbation shows that:

- expansion around `1.10–1.25` is generally safe;
- approximately 5% center/corner error is often tolerated;
- errors around 20% or larger cause substantial Dice degradation;
- very small edge slices remain intrinsically difficult even with oracle prompts.

**Decision:** a useful prompt generator must predict reasonably accurate box
geometry and should not be judged only by coarse presence recall.

### V1a.1 — learned dual-kidney 2.5D prompt generator

V1a.1 introduces anatomical left/right kidney tracking, 2.5D input, bilateral
presence heads, center heatmaps, local size maps, offsets, z-smoothed boxes, and
validation-only calibration.

Frozen test result:

| Metric | Result |
|---|---:|
| Automatic positive-slice Dice | 0.7628 |
| Automatic case-mean Dice | 0.7950 |
| Oracle-scaled positive-slice Dice | 0.8713 |
| Mean automatic-to-oracle gap | 0.1085 |

Oracle-gap decomposition attributes about 23.3% of the signed gap to presence
and 76.7% to automatic box localization/size.

**Decision:** geometry is the dominant early bottleneck; V1a.2 must address box
prediction rather than only presence.

### V1a.2-A — repair size train/inference mismatch

V1a.2-A supervises size at both ground-truth and detached predicted centers. It
improves the complete automatic validation system substantially once fully
negative slices are included.

Negative-aware validation:

| Candidate | Positive Dice | All-slice Dice | Case-volume Dice | Negative specificity |
|---|---:|---:|---:|---:|
| V1a.1 locked | 0.811517 | 0.309726 | 0.711219 | 0.046444 |
| **V1a.2-A locked** | 0.808913 | 0.766569 | **0.855560** | 0.744351 |

**Decision:** predicted-center-aware size supervision is useful, but presence on
negative slices becomes the next system-level bottleneck.

### V1a.2-B — decoded-corner supervision

V1a.2-B adds decoded-corner supervision while retaining V1a.2-A. Under
fixed-presence, fixed-expansion geometry decomposition it is the strongest box
model:

| Model | M0 auto box | Joint box gap | Center-size interaction |
|---|---:|---:|---:|
| V1a.1 | 0.836142 | 0.034708 | 0.010238 |
| V1a.2-A | 0.840618 | 0.030232 | 0.012827 |
| **V1a.2-B** | **0.841626** | **0.029224** | **0.006936** |

However, the positive-slice prompt-best checkpoint is unusable on negative
slices. Negative-aware checkpoint selection instead favors epoch 14:

```text
positive Dice       = 0.797241
all-slice Dice      = 0.772487
case-volume Dice    = 0.850753
negative specificity= 0.759833
```

**Decision:** retain V1a.2-B geometry design, but stop adding box-loss
complexity; target presence / negative-slice suppression next.

### V1a.3-A — targeted negative sampling

V1a.3-A changes only the training sampler while retaining V1a.2-B architecture
and loss. Validation selects epoch 18 EMA:

```text
positive Dice        = 0.806677
all-slice Dice       = 0.775953
case-volume Dice     = 0.847904
negative specificity = 0.759833
```

Its fixed-presence geometry remains strong (`M0 = 0.841558`), so the sampler does
not collapse the box model. The frozen model-specific test evaluation does not
beat V1a.1 on positive-slice Dice:

```text
automatic positive-slice Dice = 0.723865
all-slice Dice                 = 0.747546
case-volume Dice               = 0.799590
negative specificity           = 0.760257
```

**Decision:** sampling alone does not solve presence generalization. Preserve the
geometry network and add an explicit case-level anatomical prior.

### V1a.3-B — single-span case-level z decoder

V1a.3-B freezes the V1a.3-A epoch-18 EMA network and changes only case-level
presence decoding. Validation selects:

```text
mode             = single_span
left threshold   = 0.0035
right threshold  = 0.0400
baseline max gap = 4
boundary pad     = 1
expansion        = 1.20
```

Validation:

| Metric | V1a.3-A B0 | **V1a.3-B B1** |
|---|---:|---:|
| All-slice Dice | 0.775953 | **0.835611** |
| Positive Dice | 0.806677 | **0.810180** |
| Negative specificity | 0.759833 | **0.848954** |
| Case-volume Dice | 0.847904 | **0.854752** |

On the previously exposed 27-case split, the locked secondary evaluation confirms
the false-positive suppression but also exposes over-pruning:

| Metric | B0 | B1 |
|---|---:|---:|
| All-slice Dice | 0.747546 | **0.800863** |
| Positive Dice | **0.723374** | 0.682693 |
| Detection recall | **0.971506** | 0.863636 |
| Negative specificity | 0.760257 | **0.863004** |
| Case-volume Dice | 0.799590 | **0.801622** |

**Decision:** freeze B1 as the final image-only operating point. Its explicit
z-continuity prior is highly effective for negative-slice suppression, but its
hard single-span prior loses some true positive slices.

### V1a.4 — recover pruned remote positive slices

V1a.4 audits the B1 false negatives and finds a long-range split-span failure:
one left case-side contributes most misses, far outside a local boundary window.
A remote-satellite decoder successfully raises presence recall without increasing
presence FPR.

The clean R2b point yields:

```text
presence recall: 0.9502 -> 0.9619
presence FPR:    0.1193 -> 0.1193
```

But frozen-MedSAM validation does not improve case-volume Dice:

| Candidate | All-slice Dice | Positive Dice | Detection recall | Negative specificity | Case-volume Dice |
|---|---:|---:|---:|---:|---:|
| **Frozen B1** | 0.835611 | 0.810180 | 0.933812 | 0.848954 | **0.854752** |
| R2b remote satellite | **0.835791** | **0.810703** | **0.951356** | 0.848954 | 0.853935 |

The newly recovered slices are true positives at the presence level, but their
native boxes are poor MedSAM prompts.

**Decision:** stop widening presence decoding; the recovered-slice geometry is
the new bottleneck.

### V1a.5 — distant reliable-span geometry propagation

V1a.5 asks whether recovered remote slices can borrow boxes from the frozen B1
span. The nearest reliable anchors are 49–73 slices away (median 61).

| Candidate | Recovered IoU mean | Median center error | Expanded coverage |
|---|---:|---:|---:|
| G1 native R2b | **0.03285** | 5.2500 | **0.1957** |
| G2 nearest anchor | 0.00000 | 7.3611 | 0.0000 |
| G3 five-anchor linear trend | 0.00000 | **4.6887** | 0.0000 |

**Decision:** geometry cannot be copied/extrapolated from a distant dominant
span. Stop before MedSAM.

### V1a.6 — satellite-local geometry consensus

V1a.6 uses only the recovered satellite's own native predictions. A robust linear
consensus (G6) passes the predeclared prompt-geometry gate:

```text
mean tight-box IoU:        0.03285 -> 0.03606
median center error:       5.2500  -> 4.1134
expanded coverage:         0.1957  -> 0.3363
```

However, downstream frozen MedSAM gets worse on the recovered subset:

```text
G1 recovered mean Dice = 0.026199
G6 recovered mean Dice = 0.024921

G1 TP / FP = 650 / 19,374
G6 TP / FP = 608 / 28,509
```

Full validation:

| Candidate | All-slice Dice | Positive Dice | Detection recall | Negative specificity | Case-volume Dice |
|---|---:|---:|---:|---:|---:|
| **G0 frozen B1** | 0.835611 | 0.810180 | 0.933812 | 0.848954 | **0.854752** |
| G1 R2b native | **0.835791** | **0.810703** | 0.951356 | 0.848954 | 0.853935 |
| G6 robust linear | 0.835782 | 0.810677 | **0.953748** | 0.848954 | 0.853540 |

**Decision:** prompt-level geometry metrics are not sufficient proxies for
MedSAM utility. Freeze the V1a image-only mainline; do not open another patching
variant.

## Comparable negative-aware validation ablation

The most useful cross-version system table is:

| Operating point | Positive Dice | All-slice Dice | Case-volume Dice | Negative specificity | Role |
|---|---:|---:|---:|---:|---|
| V1a.1 locked | 0.811517 | 0.309726 | 0.711219 | 0.046444 | original learned baseline |
| **V1a.2-A locked** | 0.808913 | 0.766569 | **0.855560** | 0.744351 | historical best case-volume mean |
| V1a.2-B epoch 14 | 0.797241 | 0.772487 | 0.850753 | 0.759833 | strongest negative-aware B checkpoint |
| V1a.3-A epoch 18 | 0.806677 | 0.775953 | 0.847904 | 0.759833 | targeted-sampling reference |
| **V1a.3-B frozen B1** | **0.810180** | 0.835611 | 0.854752 | **0.848954** | **final frozen operating point** |
| V1a.4 R2b | 0.810703 | **0.835791** | 0.853935 | **0.848954** | recall-recovery ablation |
| V1a.6 G6 | 0.810677 | 0.835782 | 0.853540 | **0.848954** | local-geometry ablation |

Two different notions of "best" must not be conflated:

1. **Best historical validation case-volume mean:** V1a.2-A (`0.855560`).
2. **Final frozen robust operating point:** V1a.3-B B1 (`0.854752`), chosen as
   the endpoint of the controlled mainline because it dramatically improves
   all-slice behavior and negative specificity while retaining nearly the same
   case-volume Dice.

## Final frozen operating point

```text
prompt network:
  source            = V1a.3-A
  checkpoint         = epoch 18 EMA
  checkpoint SHA256  = f4ac0dab9b05a5b66382482b7053766cf4a14b376fd8ad4a52cab6b008a355e2

case-level decoder:
  source            = V1a.3-B B1
  mode              = single_span
  left threshold    = 0.0035
  right threshold   = 0.0400
  baseline max gap  = 4
  boundary pad      = 1

MedSAM:
  frozen            = true
  expansion scale   = 1.20
  target label      = 1
```

## Main scientific conclusions

1. **Bounding boxes are the correct core prompt for this system.** V0 establishes
   that boxes outperform the tested point-heavy combinations for frozen MedSAM.
2. **Prompt geometry matters, but not in isolation.** V1a.1–V1a.2 show that
   localization/size dominate the early oracle gap, while V1a.6 shows that better
   box IoU/center metrics do not guarantee better MedSAM masks.
3. **Negative-aware evaluation is mandatory.** Positive-slice-only selection can
   choose catastrophically poor complete-system operating points.
4. **Explicit z-axis anatomical continuity is high-value.** V1a.3-B produces the
   largest practical robustness gain by suppressing detached false-positive
   islands without retraining the network.
5. **Presence recall and geometry are coupled.** Recovering a true-positive slice
   is not enough if its spatial box is unreliable.
6. **The remaining failure mode is not well served by more post-hoc patching.**
   V1a.4–V1a.6 isolate a remote low-confidence geometry problem that likely needs
   a learned cross-slice mechanism or a genuinely different semantic/localization
   formulation.

## Evidence limitations

The final V1a.3-B configuration was selected entirely on validation, but the same
27-case test split had already been exposed earlier in the project. Its V1a.3-B
result is therefore useful for locked retrospective comparison, not as a new
project-level untouched generalization estimate.

For publication-quality final evidence, use a previously unused KiTS23 holdout or
an external dataset after the full method is frozen.

## Next project phase

The image-only V1a branch is closed. Subsequent work should be a **new research
question**, not V1a.7 patching. The preferred next directions are:

- V1b language/semantic-guided target-to-spatial prompting; or
- a separately designed learned cross-slice geometry module with its own frozen
  protocol and untouched evaluation.

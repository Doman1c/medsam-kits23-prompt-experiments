# V1b — semantic/language-guided spatial prompting

V1b is a new research branch after the frozen V1a image-only mainline. It does **not** reopen V1a thresholds, geometry post-processing, checkpoints, or MedSAM tuning.

## Research question

Can a lightweight target-conditioned 2.5D model convert a semantic request into the spatial box prompt needed by frozen MedSAM, so that the **same CT slice can produce different prompts for different requested targets**?

The initial KiTS23 target vocabulary is:

- `kidney` = label 1
- `tumor` = label 2
- `cyst` = label 3

Anatomical side is represented explicitly as `left` or `right` using the canonical RAS convention already fixed in V1a (`low x = anatomical left`, `high x = anatomical right`).

## Stage plan

### V1b.0 — semantic-target data audit

**No training. No MedSAM evaluation. No test access.**

Purpose:

1. verify that the existing processed NPZ files still contain labels 1/2/3;
2. quantify train/validation support for kidney, tumor, and cyst;
3. quantify left/right target availability;
4. measure multi-component ambiguity;
5. count slices containing multiple semantic targets, which are the most important samples for proving that target conditioning actually changes the generated prompt;
6. materialize a target-side manifest for the next controlled baseline.

Run:

```bash
python v1/v1b/scripts/audit_v1b_semantic_targets.py \
  --manifest /content/v1a1_work/manifest/v1a1_dual_manifest.csv \
  --output-dir /content/v1b0_semantic_audit
```

Expected outputs:

```text
v1b0_semantic_audit/
├── v1b_semantic_target_manifest.csv
├── v1b_semantic_target_counts.csv
├── v1b_semantic_contrast_slices.csv
└── v1b_semantic_audit_summary.json
```

The generated manifest contains one row per `(case, slice, target, side)` for **train + validation only**. For a positive target-side pair, V1b.0 records the largest connected component as the single-box supervision target and separately records the number/area of additional components so ambiguity is auditable rather than hidden.

### V1b.1 — structured semantic-control baseline

Only after V1b.0 passes.

Planned controlled change:

```text
2.5D CT + structured target token {kidney/tumor/cyst, left/right}
    -> one requested-target presence score
    -> one requested-target box
    -> frozen MedSAM
```

This stage intentionally uses a **structured semantic token**, not a language model. It answers the prerequisite question: can the image network use semantic conditioning at all?

The main control will be an otherwise matched **unconditioned/image-only** model. The decisive test is same-image target switching: changing only the target token should move/suppress the predicted box appropriately.

### V1b.2 — natural-language conditioning

If V1b.1 works, replace the structured token with a lightweight text encoder while keeping the image backbone, output definition, split, and evaluation protocol fixed. Training/evaluation prompt templates and held-out paraphrases will be separated so performance cannot be explained by memorizing one fixed sentence.

## Split policy

V1b development uses the established **train + validation cases only**. The previously exposed 27-case split is not used for V1b model selection, architecture choice, threshold choice, prompt-template choice, or ablation decisions.

A new untouched holdout or external dataset is required for publication-grade final V1b generalization claims.

## Relationship to V1a

V1a established that frozen MedSAM can be driven effectively by automatic spatial boxes and exposed the limits of image-only post-hoc prompt repair. V1b changes the research question from:

```text
image -> spatial prompt
```

to:

```text
image + semantic/language target -> target-specific spatial prompt
```

The MedSAM segmentation model remains frozen.
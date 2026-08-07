# V1a.5 compact results

V1a.5 is a validation-only geometry-recovery study. Git stores compact summaries
only; large prediction tables, checkpoints, NPZ data, MedSAM weights, notebooks,
and test results are excluded.

## Stage 1

The frozen V1a.4 R2b presence decoder restores 25 positive side-slices from
`case_00067 / left` (`slice_idx 19..43`). V1a.5 asks whether their box geometry
can be recovered from the distant frozen-B1 anchor region.

| Candidate | Recovered IoU mean | Center-error median | Expanded coverage | Advance? |
|---|---:|---:|---:|---|
| G1 native R2b geometry | 0.03285 | 5.2500 | 0.1957 | reference |
| G2 nearest anchor | 0.00000 | 7.3611 | 0.0000 | no |
| G3 five-anchor linear trend | 0.00000 | 4.6887 | 0.0000 | no |

The nearest reliable B1 anchors are 49-73 slices away (median 61), so direct
propagation and linear extrapolation are not spatially reliable for this remote
satellite. G3 lowers normalized center-error median but still produces zero IoU
and zero coverage on every recovered slice.

Under the predeclared gate, neither G2 nor G3 advances to MedSAM. V1a.5 stops
before negative-aware MedSAM and before any secondary-test evaluation.

See `stage1_summary.json`, `stage1_geometry_comparison.csv`, and
`../FINAL_DECISION.json`.

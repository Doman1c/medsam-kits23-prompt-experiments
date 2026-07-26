# Axial in-plane axis fix

The previous processed cache used slices shaped `(X, Y)` after `moveaxis`, while the dual tracker treated columns as anatomical X. That made the tracker split kidneys along the anterior-posterior axis instead of the left-right axis.

This release stores canonical axial slices as `(Y, X)` by reordering the volume to `(Z, Y, X)`. Consequently, 2D columns correspond to canonical X, and `low column = anatomical left`, `high column = anatomical right`.

Old caches are rejected through `processed_format_version = v1a1_dual_yx_v2`. Rebuild with `--rebuild-processed`.

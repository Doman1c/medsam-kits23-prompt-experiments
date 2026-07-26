# V1 — Learned Automatic Prompt Generation

V1 contains learned automatic spatial-prompt generation methods.

## Current implementation

### [V1a.1 Dual-Kidney Prompt Generator](v1a1/)

V1a.1 uses 2.5D CT input and predicts left/right kidney presence, center heatmaps, center-local box sizes, offsets, and calibrated bounding boxes. The two boxes are passed separately to frozen MedSAM and the resulting masks are combined.

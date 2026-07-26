# V1 — Learned Automatic Prompt Generation

V1 contains learned automatic spatial-prompt generation methods.

## Current implementation

### V1a.1 Dual-Kidney Prompt Generator

V1a.1 uses 2.5D CT input and predicts:

- left/right kidney presence;
- left/right center heatmaps;
- center-local width and height;
- center offsets;
- calibrated bounding boxes.

The two predicted boxes are passed separately to frozen MedSAM, and the resulting masks are combined.

Validation is complete. The next stage is one-time evaluation on the locked test set.

## Locked validation configuration

```text
checkpoint      = epoch 6
prediction      = smooth
max_gap         = 0
left threshold  = 0.20
right threshold = 0.30
box expansion   = 1.20
```

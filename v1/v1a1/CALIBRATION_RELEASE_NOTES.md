# Calibration add-on release notes

Added:

- independent left/right presence-threshold sweep;
- raw versus z-smoothed prediction comparison;
- max-gap sweep for 0, 1, and 2 slices;
- validation recall constraints;
- calibrated expansion sweep;
- efficient frozen MedSAM validation scale sweep (one image encoding per slice);
- comparison of epoch 5, epoch 6, and epoch 7 candidate checkpoints.

The calibration code never retrains the prompt generator and does not use the test split.

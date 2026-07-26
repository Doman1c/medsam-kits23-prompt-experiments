# V1a.1 Dual Kidney Prompt Generator Experiment Log

## Objective

V1a.1 focuses on building a reliable automatic prompt generation pipeline for MedSAM on KiTS23 kidney CT data.

Main goals:

- validate image orientation
- construct bilateral kidney trajectories
- generate dual-channel kidney labels
- perform small-sample overfit verification before formal training
- prepare a stable foundation for V1a.1 full training

## Completed Components

### 1. Orientation Validation

Sanity check images confirmed that kidney side information and spatial coordinates are consistent after preprocessing.

### 2. Dual Kidney Tracking

The pipeline now explicitly models left and right kidney targets instead of treating kidneys as a single foreground region.

### 3. Dual Channel Labels

The training target was changed from a single binary target into a dual-side representation, allowing the model to learn:

- kidney presence
- side information
- spatial localization

### 4. Small Sample Overfit Test

A dual overfit check was completed successfully before large-scale training.

This verified:

- model forward path
- loss implementation
- label generation
- inference pipeline

## V1a.1 Training Result Summary

Best checkpoint:

- epoch: 5
- positive recall: 0.9433
- prompt success rate: 0.7767
- continuous score: 0.7314

Later epochs showed:

- training loss continued decreasing
- validation positive recall decreased
- false positive rate improved but recall dropped
- spatial metrics saturated

The model entered an overfitting regime.

Early stopping was triggered around epoch 17.

## Decision

The current model is considered a valid baseline.

Further improvement should focus on calibration and inference strategy rather than extending training epochs.

Next steps:

1. post-training calibration
2. threshold optimization
3. MedSAM downstream evaluation
4. comparison with baseline prompts

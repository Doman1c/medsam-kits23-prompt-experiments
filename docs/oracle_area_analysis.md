# Oracle Area Analysis

## Purpose

This document records the area sensitivity analysis used during V1a.1 development.

The purpose is to understand whether prompt quality is limited by localization accuracy or by box construction strategy.

## Evaluation Dimensions

The analysis considers:

- target coverage
- bounding box expansion ratio
- prompt success rate
- MedSAM segmentation response

## Interpretation

If oracle-sized regions provide significantly better performance than predicted prompts, the main limitation is prompt localization.

If oracle and predicted prompts perform similarly, further improvement should focus on MedSAM adaptation or segmentation strategy.

## Role in V1a.1

This analysis helps separate:

1. prompt generator errors
2. prompt conversion errors
3. downstream segmentation errors

The result guides future improvements such as calibration, uncertainty-aware prompts, and multi-prompt fusion.

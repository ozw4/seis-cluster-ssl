# Parihaka HMM distillation-weight screening

This phase varies only the Stage 2 distillation weight for the MAE and
trace-drop-free local-BT HMM branches. It reuses the selected HMM
pseudo-targets, the existing reference-weight HMM artifacts, and both non-HMM
controls. There is no clustering or pseudo-target-export step.

Experiment definition:
[Channel config](40_channel_distillation_weight.yaml). Study-wide context:
[Parihaka benchmark overview](../README.md).

With no target-preparation step, begin at Stage 2 and follow the overview's HMM
screening workflow with the
[validation summarizer](scripts/summarize_validation.py). Freeze one configured
weight after review. A wider grid, if justified, is a separately defined phase.

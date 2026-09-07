# Parihaka SSL pretraining and continuation

This suite creates matched Parihaka MAE, global Barlow Twins, and local Barlow
Twins encoders, then compares each source with an independent second-stage
continuation. One continuation keeps the source objective as a control; the
other uses a single-head stratigraphic HMM pretext task.

Study-wide context is in the
[Parihaka benchmark overview](../README.md). Detailed phase ordering and the
weights-only-versus-resume distinction are in
[RUNBOOK_HMM_K6.md](RUNBOOK_HMM_K6.md).

Fixed-budget downstream comparisons consume the completed continuation
checkpoint selected by the experiment definition. A validation-best checkpoint
is diagnostic and must not silently replace it.

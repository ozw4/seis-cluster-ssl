# Parihaka HMM transition-balance screening

This phase varies only the HMM cost balance between remaining in a state and
advancing to the next state. It applies the same candidate set to MAE and
trace-drop-free local-BT sources. The previously established HMM condition and
both non-HMM controls are reused in place.

Experiment definitions:
[validation decoder config](40_channel_transition_balance.yaml) and
[final decoder config](41_channel_transition_balance_final.yaml). Study-wide
context: [Parihaka benchmark overview](../README.md).

Fit and export pseudo-targets only for the new transition conditions, then
follow the overview's HMM screening workflow with the
[validation summarizer](scripts/summarize_validation.py). Reused controls and
the existing HMM condition are prerequisites, not execution targets.

## Final-test phase

After every validation-only hyperparameter phase is complete, freeze one model
without test knowledge. Evaluate it on every predefined layout in the disjoint
final namespace, then use the
[final-test summarizer](scripts/summarize_final_test.py). Test results are
reported once and must not trigger model, layout, or hyperparameter selection.

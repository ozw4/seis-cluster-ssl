# F3 Strat-HMM pretraining milestone 1 producers

This experiment retains the target-generation, pretraining, embedding
extraction, and smoke-validation stages for the single-head K=6 structured
pretext model. HMM labels are pseudo-targets, not final lithology labels or
evaluation output.

Run stages `01` through `04` in numeric order for target export, smoke/full
pretraining, and embedding extraction. Stage `08` is the separate refreshed-
target smoke check. The stage files own their commands and settings.

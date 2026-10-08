# parihaka: trace-mask LR=3e-5, masked:visible=50:50

This extends the F3 3e-5/50:50 setting to parihaka. Encoder/head LR remain
`1e-5`, K is `[6,8,10]`, consistency is `0.1`, and distillation is `0.2`.
Only mask-token LR changes relative to the original trace-mask recipe.
Masked/visible prototype branches retain separate normalization and equal weights;
10% full-depth XY token-column masking and all other objectives are unchanged.

Each candidate receives a fresh 25-epoch continuation from its original
MAE100, asymmetric LocalBT3 or random parent, retaining immutable v02 targets,
seed, original survey batch/workers, preprocessing and decoder settings.
Do not continue the old 1e-5 candidate checkpoint with a changed learning rate.
The new versioned model ID, output namespace and checkpoint configuration
ensure that recovery resumes only this new 3e-5 run's own checkpoint.

The fixed comparison is canonical nonmasked v02, not an unfinished trace-mask
control. Its selection, source configurations, completed checkpoint hashes
and target identities are inherited from the validated original definition.
Three source definitions are available, but a single-source request must use
`--candidate`; the definition does not authorize launching unrequested sources.

Each requested source has five paired layouts times small/medium/large,
15 evaluations. Parihaka reports held-out test channel IoU and F1.
A candidate-only publication is labeled `complete_candidate`, not a complete
three-source matrix. All-source publication requires all 45 cells. Surveys
and unlike metrics are never pooled. One pretraining seed is not training-seed
uncertainty; the LR transfers from F3 validation rather than being selected
using these surveys' test results.

See [RUNBOOK.md](RUNBOOK.md) and the
[cross-survey follow-up](../../../hmm_v3/trace_mask_masklr3em5_multi_survey_v1/README.md).


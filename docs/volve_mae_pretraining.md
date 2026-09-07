# Volve survey-specific MAE pretraining

This note records the data-use and scientific claim boundaries for
amplitude-only MAE pretraining on the canonical Volve volume.

## Data boundary

The workflow uses the read-only public canonical dataset
`volve_st10010_full_t_v1` and its explicit valid-trace mask. Registration points
to those public arrays; training handles invalid samples in memory and does not
copy, interpolate, overwrite, or regenerate the public data.

Horizon bindings and interpretations, fault sticks, layout definitions, and
validation or test labels are not inputs to MAE training or its checkpoints.
Initialization does not load a checkpoint from F3, NOPIMS, Parihaka, or another
survey. Downstream labels are introduced only by later benchmark stages.

Registration inputs are snapshotted with each run so validation can bind a
checkpoint to the data identity it actually used. A regenerated registration
must not be presented as the identity of an older checkpoint.

Because this run uses the complete amplitude survey, interpret it under the
shared [same-survey pretraining claim boundary](same_survey_pretraining_claims.md).

Commands and phase ordering are maintained only in the
[`Volve amplitude MAE execution`](../experiments/volve/horizon_benchmark_v1/10_pretrain/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/README.md)
runbook.

# Parihaka survey-specific MAE pretraining

This note records provenance and the scientific claim boundary for the
Parihaka amplitude-only pretraining run.

## Provenance

The source is version 1 of the Mendeley Data release
[*Parihaka + Netherlands F3 (raw volumes + labels) for seismic facies segmentation*](https://doi.org/10.17632/gnvyh3msrj.1),
DOI `10.17632/gnvyh3msrj.1`. It is an AIcrowd-derived redistribution, not a file
claimed to have been downloaded directly from AIcrowd.

The upstream archive member `data_train.npz` is stored locally as
`parihaka_data_train.npz`; the filename is the only intended local change. Byte
identity with the AIcrowd distribution, and whether the redistribution changed
the original content, have not been verified.

The amplitude payload is the NPZ `data` array (`data.npy`) in logical
`[Z, X, Y]` order. Preparation registers it as `[X, Y, Z]` with the coordinate
mapping `output[x, y, z] == source[z, x, y]`. Preparation must validate that
mapping without retaining both complete volumes in memory.

## Label boundary

`parihaka_labels_train.npz`, its `labels` member, class identities, and label
statistics are not inputs to amplitude preparation, normalization, MAE
training, checkpoints, or pretraining summaries. The label file must not be
opened or hashed by this workflow. Label-based processing belongs to downstream
experiments.

Because this run uses the full amplitude survey, interpret it under the shared
[same-survey pretraining claim boundary](same_survey_pretraining_claims.md).

Commands and phase ordering are maintained only in the
[`Parihaka amplitude MAE execution`](../experiments/parihaka/facies_benchmark_v1/20_pretrain/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/README.md)
runbook.

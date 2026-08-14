# Parihaka amplitude MAE pretraining summary

- Dataset: `parihaka/facies_benchmark_v1` (`parihaka`)
- Model tag: `amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1`
- Training invocation: `2026-08-12T00:10:06.491600+00:00`, git `fe9e7e6484ab54726012dd9ea5ca757deeda555f`
- Input boundary: amplitude only; labels were not used or opened.
- Scope: survey-specific transductive self-supervised pretraining.
- Completion: epoch 100, global step 250000
- Precision: AMP requested (`auto`), resolved `bfloat16`, scaler present `false`
- Primary checkpoint: `${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/pretraining/parihaka/facies_benchmark_v1/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/full_100ep/latest.pt` (`latest.pt`, schema 2)
- Diagnostic checkpoint: `${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/pretraining/parihaka/facies_benchmark_v1/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/full_100ep/best.pt` (epoch 89, loss=0.18413860874176027)
- Downstream status: checkpoint ready; evaluation was not run.

Training completion and loss do not establish geological interpretation, channel-estimation performance, transfer to unseen surveys, or downstream accuracy improvement.

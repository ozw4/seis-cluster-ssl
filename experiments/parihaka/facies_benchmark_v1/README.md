# Parihaka facies benchmark v1

This directory contains the Parihaka self-supervised pretraining and Channel
facies studies. The experiment YAML is authoritative for paths, model arms,
runtime values, and output locations. See the repository-wide
[configuration contract](../../../docs/configuration.md) and
[artifact/report policy](../../../docs/report_sharing_policy.md).

## Shared scientific boundary

The encoders were pretrained on the full unlabelled Parihaka amplitude volume,
so downstream comparisons use the shared
[same-survey pretraining claim boundary](../../../docs/same_survey_pretraining_claims.md).

Channel studies share the reviewed
[layout definition](30_channel_benchmark_v1/02_layouts.yaml). Training sections
define candidate supervision regions, validation selects a checkpoint, and the
common held-out voxel complement is evaluated only by phases that explicitly
permit test inference. Validation-only searches must finish before a final
model is frozen.

## Shared execution and recovery

Complete prerequisites and dry-runs before starting a writing phase. A fresh
training run must not receive a resume checkpoint. After interruption, resume
only from that same job's latest checkpoint; an initialization checkpoint,
another arm, a feasibility run, and a validation-best checkpoint are not
interchangeable with training state.

## HMM screening workflow

The transition-balance, boundary-weight, and distillation-weight phases use one
validation-only workflow for both MAE and trace-drop-free local-BT sources.
After any phase-specific target preparation:

1. train and extract only the new Stage 2 candidates;
2. run only their medium-supervision decoder jobs in validation-only mode;
3. summarize reused controls and new candidates together; and
4. stop for human review.

These phases do not inspect test values, screen other supervision sizes, or
start a follow-up search automatically. A partial job stops batch progression
and follows the shared recovery rule above.

## Experiment notes

- [Barlow Twins pretraining](20_pretrain/amp_barlow_twins_flipxy_l0005_v1/README.md)
- [MAE pretraining](20_pretrain/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/README.md)
- [SSL and HMM continuation](21_ssl_hmm_continuation_v1/README.md)
- [Frozen Channel baseline](30_channel_benchmark_v1/README.md)
- [End-to-end initialization comparison](31_channel_end_to_end_v1/README.md)
- [Global SSL/HMM comparison](32_channel_ssl_hmm_four_way_v1/README.md)
- [MAE/local-BT comparison](33_channel_mae_local_bt_four_way_v1/README.md)
- [Larger-context end-to-end comparison](34_channel_end_to_end_128_v1/README.md)
- [D4 and trace-drop screening](35_channel_local_bt_d4_trace_drop_v1/README.md)
- [HMM transition-balance screening](36_channel_hmm_transition_balance_v1/README.md)
- [HMM boundary-weight screening](37_channel_hmm_boundary_weight_v1/README.md)
- [HMM distillation-weight screening](38_channel_hmm_distillation_weight_v1/README.md)

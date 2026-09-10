# Volve horizon benchmark v1

This suite evaluates self-supervised seismic encoders on the five-horizon Volve
task. Scientific data-use and split boundaries are documented in
[`docs/volve_mae_pretraining.md`](../../../docs/volve_mae_pretraining.md) and
[`docs/volve_horizon_supervision.md`](../../../docs/volve_horizon_supervision.md).
Concrete conditions belong to each experiment's YAML.

All runbooks use the same repository-root environment:

```bash
: "${SEIS_SSL_CLUSTER_VOLVE_ROOT:?set the read-only public Volve root}"
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?set the writable artifact root}"
export VOLVE_EXP=experiments/volve/horizon_benchmark_v1
export VOLVE_LAYOUTS="$VOLVE_EXP/20_horizon_supervision/01_layouts.yaml"
```

- [MAE pretraining](10_pretrain/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/README.md)
- [Horizon supervision layouts](20_horizon_supervision/README.md)
- [Frozen MAE versus Random](30_mae_vs_random_frozen_v1/README.md)
- [MAE / Local Barlow Twins five-way](31_mae_local_bt_hmm_five_way_v1/README.md)
- [Local Barlow Twins recipe arms](32_local_bt_recipe_arm_v1/README.md)
- [Missing HMM arms of the nine-arm comparison](33_missing_hmm_comparison_v1/README.md)

Repository storage and publication rules are defined in the
[report sharing policy](../../../docs/report_sharing_policy.md).

# Volve Local Barlow Twins stage one

This stage trains the adopted non-trace-drop `local_barlow_twins_3d` objective
for 100 epochs on the canonical unlabeled Volve amplitude input. Both configs
use 128 local token pairs per crop and the same input preprocessing and encoder
geometry as the existing Volve MAE stage one.

The adopted Local Barlow Twins batch size is 16, so the full run performs
62,500 optimizer updates (`10000 / 16 * 100`). The existing Volve MAE stage one
uses batch size 4 and performs 250,000 updates. Stage-two configs use a separate
fixed budget and do not infer equality from these stage-one epoch counts.

Validate the smoke config without training:

```bash
python proc/seis_ssl_cluster/train_amp_barlow_twins.py \
  --config experiments/volve/horizon_benchmark_v1/31_mae_local_bt_hmm_five_way_v1/10_stage1/local_barlow_twins/01_smoke_2step.yaml \
  --dry-run
```

Run either config with the same command after setting
`SEIS_SSL_CLUSTER_ARTIFACT_ROOT`. Checkpoints and other runtime products remain
under the configured artifact root and are not repository inputs.

# F3 original-split voxel benchmark summary

V0 is the voxel-shaped token baseline. V1 is the learned sub-token decoder.

## Shared evaluation identity

- supervision split-grid SHA-256: `4379ba7d9b32f53ca5929a2f4c876ccc3756c2a551bc86f1c07eb89904a584fb`
- class order: `[0, 1, 2, 3, 4, 5]`
- validation voxel count: `470136`
- decoder spec: `frozen_embedding_decoder_nearest_voxel_ln_v1`
- decoder upsample mode: `nearest`
- decoder normalization: `voxelwise_layer_norm`

## Model table

| model | version | decoder spec | upsample mode | normalization |
|---|---|---|---|---|
| MAE | V0 |  |  |  |
| MAE | V1 | frozen_embedding_decoder_nearest_voxel_ln_v1 | nearest | voxelwise_layer_norm |
| M1 | V0 |  |  |  |
| M1 | V1 | frozen_embedding_decoder_nearest_voxel_ln_v1 | nearest | voxelwise_layer_norm |
| M2-A | V0 |  |  |  |
| M2-A | V1 | frozen_embedding_decoder_nearest_voxel_ln_v1 | nearest | voxelwise_layer_norm |

## Provisional decisions

- decoder_value: **positive**
- m2a_vs_m1_voxel: **positive**
- provisional: **true**

These statuses use the original split only and are not robustness claims.

## Q1: learned decoder value (V1 - V0)

| encoder | decoder spec | upsample mode | normalization | macro F1 | mean IoU | balanced accuracy | boundary F1 t2 | boundary F1 t4 | position MAE |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| MAE | frozen_embedding_decoder_nearest_voxel_ln_v1 | nearest | voxelwise_layer_norm | 0.151684 | 0.187603 | 0.087167 | 0.297571 | 0.253478 | -0.616528 |
| M1 | frozen_embedding_decoder_nearest_voxel_ln_v1 | nearest | voxelwise_layer_norm | 0.145499 | 0.175917 | 0.115557 | 0.261180 | 0.244122 | -0.398798 |
| M2-A | frozen_embedding_decoder_nearest_voxel_ln_v1 | nearest | voxelwise_layer_norm | 0.146856 | 0.179060 | 0.104417 | 0.262003 | 0.226707 | -0.492481 |

Boundary-region radius 2/4 and class 3/5 deltas are retained in `tables/v1_vs_v0_deltas.csv` and `tables/monitored_class_deltas.csv`.

## Q2: representation comparison at voxel resolution

| role | comparison | decoder spec | upsample mode | normalization | macro F1 | mean IoU | boundary F1 t2 | boundary F1 t4 |
|---|---|---|---|---|---:|---:|---:|---:|
| primary | M2-A V1 - M1 V1 | frozen_embedding_decoder_nearest_voxel_ln_v1 | nearest | voxelwise_layer_norm | 0.003813 | 0.005245 | 0.001662 | -0.016769 |
| secondary | M1 V1 - MAE V1 | frozen_embedding_decoder_nearest_voxel_ln_v1 | nearest | voxelwise_layer_norm | -0.000817 | -0.001602 | -0.015462 | 0.019239 |
| secondary | M2-A V1 - MAE V1 | frozen_embedding_decoder_nearest_voxel_ln_v1 | nearest | voxelwise_layer_norm | 0.002996 | 0.003644 | -0.013800 | 0.002470 |

### Monitored class deltas

| comparison | class | F1 | IoU | boundary recall t2 | boundary recall t4 |
|---|---:|---:|---:|---:|---:|
| V1 - V0 | 3 | 0.304202 | 0.348364 | 0.437754 | 0.391782 |
| V1 - V0 | 5 | 0.284978 | 0.263132 | 0.118919 | 0.183784 |
| V1 - V0 | 3 | 0.314388 | 0.361720 | 0.349471 | 0.318552 |
| V1 - V0 | 5 | 0.293311 | 0.268902 | 0.070270 | 0.113514 |
| V1 - V0 | 3 | 0.309869 | 0.358485 | 0.432465 | 0.378763 |
| V1 - V0 | 5 | 0.297194 | 0.276399 | 0.172973 | 0.221622 |
| M2-A V1 - M1 V1 | 3 | 0.001597 | 0.002334 | 0.082994 | 0.058177 |
| M2-A V1 - M1 V1 | 5 | 0.012254 | 0.013824 | 0.102703 | 0.108108 |
| M1 V1 - MAE V1 | 3 | 0.007572 | 0.010981 | -0.072823 | -0.052075 |
| M1 V1 - MAE V1 | 5 | -0.001472 | -0.001647 | -0.086486 | -0.108108 |
| M2-A V1 - MAE V1 | 3 | 0.009169 | 0.013315 | 0.010171 | 0.006103 |
| M2-A V1 - MAE V1 | 5 | 0.010782 | 0.012177 | 0.016216 | 0.000000 |

Full metric rows are available in `tables/`.

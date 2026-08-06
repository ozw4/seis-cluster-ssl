# NOPIMS pretrain v1

Experiment configs for the NOPIMS amplitude-only MVP pretraining pipeline.

Source-of-truth inputs:

```bash
ROOT=/workspace/artifacts/seis_ssl_cluster
EXP=experiments/nopims/pretrain_v1

MODEL_TAG=amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1
SUBSET=ten_surveys
EMBED_SPEC=overlap_x16
CLUSTER_SPEC=k4_6_8_pca16_whiten_s100k
VIZ_SPEC=voxel_cmp_xy750_xz150
```

- Raw NOPIMS root: `/home/dcuser/data/NOPIMS`
- Artifact root: `$ROOT`
- Training path-list: `$ROOT/registry/splits/nopims/pretrain_v1/train_npy_paths.txt`

Each YAML is intentionally standalone and avoids inheritance, anchors, merge
keys, and symlinks.

## Runbook

Run the configured stages from the repository root. Each YAML's explicit input
and output paths are the source of truth; the directory examples below are not
enforced by a repository validator:

```bash
python proc/seis_ssl_cluster/build_nopims_manifests.py \
  --config $EXP/00_registry/01_build_manifest.yaml

python proc/seis_ssl_cluster/prepare_nopims_normalization_stats.py \
  --config $EXP/00_registry/02_prepare_stats.yaml

python proc/seis_ssl_cluster/filter_manifest_by_normalization_qc.py \
  --config $EXP/00_registry/03_filter_qc.yaml

python proc/seis_ssl_cluster/train_amp_mae.py \
  --config $EXP/10_pretrain/$MODEL_TAG/03_full_100ep.yaml

python proc/seis_ssl_cluster/extract_embeddings.py \
  --config $EXP/20_embedding/$MODEL_TAG/$SUBSET/$EMBED_SPEC.yaml \
  --device cuda \
  --skip-existing

python proc/seis_ssl_cluster/cluster_embeddings.py \
  --config $EXP/30_clustering/$MODEL_TAG/$SUBSET/$EMBED_SPEC/$CLUSTER_SPEC.yaml

python proc/seis_ssl_cluster/visualize_clusters.py \
  --config $EXP/40_visualization/$MODEL_TAG/$SUBSET/$EMBED_SPEC/$CLUSTER_SPEC/$VIZ_SPEC.yaml
```

Outputs:

```text
registry:
  $ROOT/registry/manifests/nopims/pretrain_v1/nopims_amplitude_manifests.json
  $ROOT/registry/normalization_stats/nopims/pretrain_v1
  $ROOT/registry/manifests/nopims/pretrain_v1_clean/nopims_amplitude_manifests.json
  $ROOT/registry/splits/nopims/pretrain_v1_clean/train_npy_paths.txt
  $ROOT/registry/qc/nopims/pretrain_v1

pretraining:
  $ROOT/pretraining/nopims/pretrain_v1/$MODEL_TAG/full_100ep

embedding:
  $ROOT/embeddings/nopims/pretrain_v1/$MODEL_TAG/$SUBSET/$EMBED_SPEC

clustering:
  $ROOT/clustering/nopims/pretrain_v1/$MODEL_TAG/$SUBSET/$EMBED_SPEC/$CLUSTER_SPEC

visualization:
  $ROOT/visualizations/clusters/nopims/pretrain_v1/$MODEL_TAG/$SUBSET/$EMBED_SPEC/$CLUSTER_SPEC/$VIZ_SPEC
```


## amp_mae_m025_mse_g0_patchnorm_v1

`10_pretrain/amp_mae_m025_mse_g0_patchnorm_v1/03_full_100ep.yaml` defines the mask 0.25, MSE, gradient-weight 0 experiment with target-only patch z-score normalization (`eps: 1.0e-6`, `min_std: 0.05`). It changes only the MAE loss target; encoder inputs and dataset targets remain survey-wise normalized amplitudes. This repository change adds the YAML only and does not start training.

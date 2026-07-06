# Artifact Path Contract

The default local artifact root is `/workspace/artifacts/seis_ssl_cluster`.
Keep complete generated outputs under that root. Keep `results/` for lightweight
GitHub review files only.

## Standard Variables

Use these variables in runbooks and copy-paste commands, including only the
stage variables a command group needs:

```bash
ROOT=/workspace/artifacts/seis_ssl_cluster
EXP=experiments/f3/facies_benchmark_v1

MODEL_TAG=amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1
SUBSET=ten_surveys
EMBED_SPEC=overlap_x16
CLUSTER_SPEC=k4_6_8_pca16_whiten_s100k
VIZ_SPEC=voxel_cmp_xy750_xz150
LABEL_SET=png_slices_segy_labels_v1
PROBE_SPEC=linear_balanced_v1
```

## Local Artifacts

Use these stage directories under the artifact root:

| Directory | Role | Example |
|---|---|---|
| `artifacts/pretraining` | MAE checkpoints, resolved configs, and training debug outputs | `$ROOT/pretraining/nopims/pretrain_v1/$MODEL_TAG/full_100ep/mae_latest.pt` |
| `artifacts/embeddings` | Extracted encoder embeddings | `$ROOT/embeddings/nopims/pretrain_v1/$MODEL_TAG/$SUBSET/$EMBED_SPEC` |
| `artifacts/clustering` | KMeans models, labels, and clustering metadata | `$ROOT/clustering/nopims/pretrain_v1/$MODEL_TAG/$SUBSET/$EMBED_SPEC/$CLUSTER_SPEC` |
| `artifacts/visualizations` | Cluster figures, visualization summaries, and optional voxel labels | `$ROOT/visualizations/clusters/nopims/pretrain_v1/$MODEL_TAG/$SUBSET/$EMBED_SPEC/$CLUSTER_SPEC/$VIZ_SPEC` |
| `artifacts/lithology` | F3 token datasets, probes, predictions, visualizations, and reports | `$ROOT/lithology/f3/facies_benchmark_v1/$MODEL_TAG/$EMBED_SPEC/$LABEL_SET/probes/$PROBE_SPEC` |

Checkpoint outputs must stay under `pretraining/`. Embedding outputs stop at
`$EMBED_SPEC` and must not include clustering or visualization specs.
Clustering outputs stop at `$CLUSTER_SPEC` and must not include visualization
specs. `runs/` is not a standard artifact path.

## Shared Results

`results/` contains selected lightweight review artifacts: Markdown reports,
metrics, CSV comparison tables, JSON summaries, and representative figures.
Do not publish checkpoints, embeddings, clustering models, `.npy`, `.npz`,
`.pt`, `.joblib`, `.pkl`, raw SEGY files, path lists, normalization statistics,
or full visualization dumps there.

Example:

```text
results/f3/facies_benchmark_v1/lithology_probe/$MODEL_TAG/$EMBED_SPEC/$LABEL_SET/$PROBE_SPEC/report.md
```

## Validation

Run the path contract validator before review:

```bash
python proc/seis_ssl_cluster/validate_artifact_paths.py \
  --root $ROOT \
  --scan experiments proc docs README.md results \
  --fail-on-runs
```

This validator checks path strings in configs, docs, proc scripts, and
publish manifests. It complements `validate_results_artifacts.py`, which checks
the physical files already stored under `results/`.

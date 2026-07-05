# Artifact Path Contract

The default local artifact root is `/workspace/artifacts/seis_ssl_cluster`.
Keep complete generated outputs under that root. Keep `results/` for lightweight
GitHub review files only.

## Local Artifacts

Use these stage directories under the artifact root:

| Directory | Role | Example |
|---|---|---|
| `artifacts/pretraining` | MAE checkpoints, resolved configs, and training debug outputs | `/workspace/artifacts/seis_ssl_cluster/pretraining/nopims/pretrain_v1/<MODEL_TAG>/full_100ep/mae_latest.pt` |
| `artifacts/embeddings` | Extracted encoder embeddings | `/workspace/artifacts/seis_ssl_cluster/embeddings/nopims/pretrain_v1/<MODEL_TAG>/<SUBSET>/<EMBED_SPEC>` |
| `artifacts/clustering` | KMeans models, labels, and clustering metadata | `/workspace/artifacts/seis_ssl_cluster/clustering/nopims/pretrain_v1/<MODEL_TAG>/<SUBSET>/<EMBED_SPEC>/<CLUSTER_SPEC>` |
| `artifacts/visualizations` | Cluster figures, visualization summaries, and optional voxel labels | `/workspace/artifacts/seis_ssl_cluster/visualizations/clusters/nopims/pretrain_v1/<MODEL_TAG>/<SUBSET>/<EMBED_SPEC>/<CLUSTER_SPEC>/<VIZ_SPEC>` |
| `artifacts/lithology` | F3 token datasets, probes, predictions, visualizations, and reports | `/workspace/artifacts/seis_ssl_cluster/lithology/f3/facies_benchmark_v1/<MODEL_TAG>/<EMBED_SPEC>/<LABEL_SET>/probes/<PROBE_SPEC>` |

Checkpoint outputs must stay under `pretraining/`. Embedding outputs stop at
`<EMBED_SPEC>` and must not include clustering or visualization specs.
Clustering outputs stop at `<CLUSTER_SPEC>` and must not include visualization
specs. `runs/` is not a standard artifact path.

## Shared Results

`results/` contains selected lightweight review artifacts: Markdown reports,
metrics, CSV comparison tables, JSON summaries, and representative figures.
Do not publish checkpoints, embeddings, clustering models, `.npy`, `.npz`,
`.pt`, `.joblib`, `.pkl`, raw SEGY files, path lists, normalization statistics,
or full visualization dumps there.

Example:

```text
results/f3/facies_benchmark_v1/lithology_probe/<MODEL_TAG>/<EMBED_SPEC>/<LABEL_SET>/<PROBE_SPEC>/report.md
```

## Validation

Run the path contract validator before review:

```bash
python proc/seis_ssl_cluster/validate_artifact_paths.py \
  --root /workspace/artifacts/seis_ssl_cluster \
  --scan experiments proc docs README.md results \
  --fail-on-runs
```

This validator checks path strings in configs, docs, proc scripts, and
publish manifests. It complements `validate_results_artifacts.py`, which checks
the physical files already stored under `results/`.

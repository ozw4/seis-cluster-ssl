# Documentation

This directory contains runbooks and configuration contracts for
`seis_ssl_cluster` experiments.

## Artifacts And Results

`artifacts/` is the local generated-output area and is ignored by Git. Normal
experiment, training, embedding, clustering, and visualization outputs should
continue to use `/workspace/artifacts/seis_ssl_cluster/`.

`results/` is the repository-managed area for lightweight GitHub review
artifacts. Keep only selected reports, metrics, comparison tables, and
representative figures there. Do not commit checkpoints, embeddings, clustering
models, `.npy`, `.npz`, `.pt`, `.joblib`, `.pkl`, raw SEGY files, path lists,
normalization statistics, or full visualization dumps.

Validate shared results before review:

```bash
python proc/seis_ssl_cluster/validate_results_artifacts.py \
  --root results \
  --max-file-size-mb 10
```

See [results_sharing_policy.md](results_sharing_policy.md) for the review
workflow, required-file checks, and strict local-path validation.

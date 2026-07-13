# Documentation

This directory contains runbooks and configuration contracts for
`seis_ssl_cluster` experiments.

## Active research runbooks

- [F3 M3-V voxel lithology benchmark](f3_voxel_lithology_benchmark.md): V0/V1 contracts, exact original/six-split execution order, artifact identity checks, and release validation.
- [F3 strat-HMM M2-A boundary weighting](f3_strat_hmm_m2a_boundary_weighting.md): ordered export-to-summary workflow, artifact paths, failure handling, and M1 housekeeping.
- [F3 strat-HMM M2-A result decision](f3_strat_hmm_m2a_results.md): final Go/Stop/Hold evidence contract.

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

## Test Selection

Use pytest markers to select the local validation granularity:

```bash
pytest -q -m "not slow and not requires_segy and not requires_cuda"
pytest -q -m smoke
pytest -q -m integration
```

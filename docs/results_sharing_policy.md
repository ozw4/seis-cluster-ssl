# Results Sharing Policy

Use `artifacts/` for complete local outputs and `results/` for lightweight
GitHub review artifacts.

## Normal Runs

Experiment, training, embedding, clustering, and visualization commands should
continue to write full outputs under `/workspace/artifacts/seis_ssl_cluster/`.
That tree is local generated output and is not tracked by Git.

## Sharing

When a report builder supports publishing, use its publish configuration to copy
only selected lightweight outputs into `results/`. Keep normal experiment output
paths unchanged.

Files suitable for `results/` are selected Markdown reports, metrics files,
comparison tables, and representative figures. Do not commit checkpoints,
embeddings, clustering models, `.npy`, `.npz`, `.pt`, `.joblib`, `.pkl`, raw
SEGY files, path lists, normalization statistics, full visualization dumps, or
an `artifacts/` directory nested under `results/`.

## Validation

Run the lightweight validator before review:

```bash
python proc/seis_ssl_cluster/validate_results_artifacts.py \
  --root results \
  --max-file-size-mb 10
```

Use required-file checks for known review deliverables:

```bash
python proc/seis_ssl_cluster/validate_results_artifacts.py \
  --root results \
  --max-file-size-mb 10 \
  --required-file f3/facies_benchmark_v1/inspection/report.md \
  --required-file f3/facies_benchmark_v1/baseline_comparison/comparison_report.md
```

Publish manifest `target` entries are relative to the manifest file directory.
The validator resolves them under the caller's `--root`, so committed
`results/` artifacts remain valid after a checkout is relocated.

Local absolute paths such as `/home/dcuser/` and `/workspace/artifacts/` are
reported as warnings by default because some publish manifests record source
locations. For stricter CI checks, use:

```bash
python proc/seis_ssl_cluster/validate_results_artifacts.py \
  --root results \
  --max-file-size-mb 10 \
  --local-path-policy error
```

## Review

In GitHub review, inspect `results/` instead of `artifacts/`. For F3, start with
`results/f3/facies_benchmark_v1/inspection/report.md` and, after baseline
comparison has been published,
`results/f3/facies_benchmark_v1/baseline_comparison/comparison_report.md`.

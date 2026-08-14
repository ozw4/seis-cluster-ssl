# F3 Reports

This directory is for lightweight, human-readable F3 summaries tracked by Git.

Recommended layout:

```text
reports/
└── <survey>/
    └── <dataset-version>/
        └── <report-id>/
```

For F3, report IDs include `inspection`, `lithology_probe`, and
`baseline_comparison`; experiment-specific reports use the same level.

Keep full F3 execution output under `/workspace/artifacts/seis_ssl_cluster/`.
Commit only reproducible summaries such as selected Markdown reports, metrics,
comparison tables, and representative figures. Do not use this directory as a
pipeline input; experiment definitions and configuration stay in
`experiments/f3/`.

The retired `facies_benchmark_v1` reports are frozen under
`reports/f3/legacy/facies_benchmark_v1/`. They are historical references only,
not active pipeline inputs, and must not be compared directly with a future
section-count, single-seed benchmark.

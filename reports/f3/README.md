# F3 Reports

This directory is for lightweight, human-readable F3 summaries tracked by Git.

Recommended layout:

```text
reports/f3/
└── facies_benchmark_v1/
    ├── inspection/
    ├── lithology_probe/
    └── baseline_comparison/
```

Keep full F3 execution output under `/workspace/artifacts/seis_ssl_cluster/`.
Commit only reproducible summaries such as selected Markdown reports, metrics,
comparison tables, and representative figures. Do not use this directory as a
pipeline input; experiment definitions and configuration stay in
`experiments/f3/`.

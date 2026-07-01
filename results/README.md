# Results

`results/` はGitHub共有用の軽量成果物だけを置く。
完全な実行成果物は `/workspace/artifacts/seis_ssl_cluster/` に保存する。
`results/` 内のファイルは再生成可能なsummaryであり、生データやcheckpointではない。

Store only selected reports, metrics, comparison tables, and representative
figures here. Do not store checkpoints, embeddings, clustering models, `.npy`,
`.npz`, `.pt`, `.joblib`, `.pkl`, raw SEGY files, path lists, normalization
statistics, or full visualization dumps.

Recommended F3 layout:

```text
results/
└── f3/
    └── facies_benchmark_v1/
        ├── inspection/
        ├── lithology_probe/
        └── baseline_comparison/
```

# Reports

`reports/` はGit管理する軽量な人向け要約だけを置く。
完全な実行出力、中間生成物、後続処理の入力は、Git管理しない
`/workspace/artifacts/seis_ssl_cluster/` に保存する。実験定義と設定は
`experiments/` に置く。`reports/` 内のファイルをパイプラインの入力にしない。

Store only selected reports, metrics, comparison tables, and representative
figures here. Do not store checkpoints, embeddings, clustering models, `.npy`,
`.npz`, `.pt`, `.joblib`, `.pkl`, raw SEGY files, path lists, normalization
statistics, or full visualization dumps.

Recommended F3 layout:

```text
reports/
└── f3/
    └── facies_benchmark_v1/
        ├── inspection/
        ├── lithology_probe/
        └── baseline_comparison/
```

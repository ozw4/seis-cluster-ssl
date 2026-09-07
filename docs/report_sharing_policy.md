# Report sharing policy

Repository data has three distinct owners:

| Location | Purpose | Git |
|---|---|---|
| `experiments/` | Versioned experiment definitions and configuration | Tracked |
| `artifacts/` | Complete execution outputs, intermediate products, and every input consumed by a later stage | Ignored |
| `reports/` | Small, reviewable result summaries for people | Tracked |

Pipeline stages must never consume files from `reports/`. When results are
curated into `reports/`, include only the producer's explicitly owned review
files; this does not change the artifact lineage.

Reports may contain Markdown, JSON, CSV, and a small set of representative
figures. Do not publish raw data, checkpoints, embeddings, prediction volumes,
clustering models, path lists, normalization products, full visualization dumps,
or bulk NumPy, PyTorch, pickle, Joblib, or SEG-Y files.

Each report producer must define its exact published file set, and tests must
verify it. Review changes under `reports/` with `git diff` before committing.

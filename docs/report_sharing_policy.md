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

Each report producer must define its exact published file set. When adding or
changing a producer, verify that file set in focused tests. Review changes under
`reports/` with `git diff` before committing.

## Artifact layout

Within an artifact root, new experiment outputs use
`surveys/<survey>/<storage-series>/<artifact-kind>/<experiment-or-model>/...`.
Keep shared inputs with their producing storage series, even when a later
experiment consumes them. Cross-survey study outputs belong in `studies/`;
execution, recovery, and migration records belong in `operations/`.
Retained failed, interrupted, or superseded outputs belong in `archive/`.
Archiving does not imply that an output is unused or safe to delete.

Existing records may contain immutable historical paths. During relocation,
preserve their bytes and retain compatibility links at those paths. The hidden
`.legacy/` directory holds the old directory structure and links; it is not
a destination for new experiment definitions. Visible historical names at the
artifact root are compatibility aliases, not additional copies of the data.
Do not remove these aliases until the corresponding readers have an explicit,
validated replacement for resolving historical identities.

Storage migrations must record old/new locations, preserve payloads, reject
destination collisions, and provide a rollback journal. They must run without
concurrent artifact writers and verify the affected frozen identities. See the
[migration procedure](artifact_migration.md) for execution and recovery.

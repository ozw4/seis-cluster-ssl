# Results Sharing Policy

Use `artifacts/` for complete local outputs and `results/` for lightweight
GitHub review artifacts.

## Normal runs

Experiment, training, embedding, clustering, and visualization commands should
continue to write large generated outputs under the configured artifact root.
Do not move those outputs into `results/`.

## Sharing

Commit only the small file set owned explicitly by each producer: Markdown,
JSON, CSV, and representative figures needed for review.

Do not commit checkpoints, embeddings, raw `.npy`/`.npz` arrays, prediction
volumes, clustering models, or raw SEGY data under `results/`.

Each producer's focused tests should fix the expected review file set and check
that heavy artifacts are not emitted there.

## Review

Use `git diff` and normal code review as the final check for changes under
`results/`. For F3, start with the experiment-specific summary/report files and
representative figures committed by the producer.

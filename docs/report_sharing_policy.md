# Report Sharing Policy

The repository uses three distinct locations:

- `artifacts/`: complete execution outputs, intermediate products, and inputs
  to later processing. Git does not track this directory.
- `reports/`: lightweight, human-readable summaries tracked by Git. Pipeline
  stages must not consume files from this directory.
- `experiments/`: experiment definitions and configuration.

Normal experiment, training, embedding, clustering, and visualization commands
write under the configured artifact root. Publishing copies only the small,
producer-owned summary set into `reports/`; it does not change pipeline inputs.

Commit only the small file set owned explicitly by each producer: Markdown,
JSON, CSV, and representative figures needed for review.

Do not commit checkpoints, embeddings, raw `.npy`/`.npz` arrays, prediction
volumes, clustering models, or raw SEGY data under `reports/`.

Each producer's focused tests should fix the expected review file set and check
that heavy artifacts are not emitted there.

Use `git diff` and normal code review as the final check for changes under
`reports/`. For F3, start with the experiment-specific summary/report files and
representative figures committed by the producer.

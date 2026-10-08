# F3 masked/visible execution

Use the project Python environment and configured `SEIS_SSL_CLUSTER_WORKSPACE`,
`SEIS_SSL_CLUSTER_ARTIFACT_ROOT`, `F3_ROOT`; select placement with
`CUDA_VISIBLE_DEVICES`. `SEIS_SSL_CLUSTER_PYTHON` selects the wrapper Python.

```bash
bash experiments/f3/facies_benchmark_v2/134_hmm_v3_masked_visible_weight_sweep_v1/run_all.sh --plan
bash experiments/f3/facies_benchmark_v2/134_hmm_v3_masked_visible_weight_sweep_v1/run_all.sh --dry-run --from smoke --to full
```

Before live execution, `--freeze-control` audits the three completed 133
`3e-5` sources and their 45 decoder cells against the existing frozen 132
controls. It produces exactly `receipt.json` beneath `execution.yaml`'s
`control_root`. Its byte SHA-256 must match `control_receipt_sha256` in the
versioned definition. Re-freezing is idempotent only if contents and the exact
file set agree. This is a completed control subset, not a partial publication
of the six-arm LR sweep. Never resume its unfinished arms to produce this receipt.

Run all six full-batch one-step smokes, then validate schema-7 checkpoint
weights, optimizer groups and initial student/head/token hashes against the
matched controls before full training. Original parents and immutable targets
must match. Full runs are independent of both smoke and trained control weights.

```bash
bash experiments/f3/facies_benchmark_v2/134_hmm_v3_masked_visible_weight_sweep_v1/run_all.sh --execute --from smoke --to smoke
bash experiments/f3/facies_benchmark_v2/134_hmm_v3_masked_visible_weight_sweep_v1/run_all.sh --execute --from full --to summary
```

Stages: smoke, full, audit, embeddings, downstream, summary. A per-candidate
worker uses `--candidate <model-tag> --from full --to downstream`. Own-candidate
locks prevent duplicate writes; scientifically matching latest checkpoints
may resume, and completed sources, embeddings and decoders are audited before
reuse. Never resume a checkpoint from another loss ratio. On the current host,
allow at most two candidate pipelines, using GPU0/GPU1 only with at least
45 GiB free at admission. Do not kill other jobs or change the scientific batch.

For recovery from shared-GPU memory exhaustion, raise the machine-local
admission margin (the current recovery uses 65 GiB), retaining the two-pipeline
limit and unchanged training settings. Completed candidates may be excluded
from the queue only after source and all 15 decoder cells are audited against
the frozen control. Preserve the previous supervisor status in a recovery
record. A failed fresh run with no checkpoint must have its exact metadata-only
output set archived before retrying; never clear outputs containing a checkpoint.

The subsequent user-directed queue cancels unfinished `25:75` arms and runs
only the three `75:25` arms, completing MAE100 training and all 15 evaluations
before admitting LocalBT3/random. Keep the original six-arm scientific
definition and all saved outputs intact; cancellation is an operational record,
not a relabeling of incomplete runs. After the 45 requested `75:25` evaluations
are audited, mark the operational status `complete_requested_subset`. Do not
invoke or weaken the original strict 90-cell full-matrix summary publisher.

Logs and frozen controls live under
`operations/hmm_v3/masked_visible_weight_sweep_v1/f3/hmm_v3_masked_visible_weight_sweep_v1/`
in artifacts. Training, embeddings and decoder results use the new
`hmm_v3_masked_visible_weight_sweep_v1` survey namespace. No outputs overwrite
133 or 132. Reports are never pipeline inputs.

After all 90 cells finish, `--publish-summary` owns exactly `comparison.csv`,
`summary.json`, `summary.md` beneath `summary_root`; incomplete or conflicting
publication is rejected. The 18 comparison rows retain small/medium/large
separately, five paired layouts per row. Overall deltas average sizes within
each layout before using the five layout clusters. See the
[report policy](../../../../docs/report_sharing_policy.md).

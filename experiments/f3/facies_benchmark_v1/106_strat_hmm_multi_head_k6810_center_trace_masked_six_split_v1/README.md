# Experiment 106: center-trace masked six-split preflight

This experiment root contains the pre-registered six-split contract and its
start audit. It does not contain a dataset builder, decoder job, smoke job, or
result summarizer. Experiment 96 and the original-split handoff remain
read-only inputs.

Set the artifact root before loading the YAML:

```bash
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT=/path/to/artifacts/seis_ssl_cluster
```

Validate all live inputs without writing an artifact:

```bash
python proc/seis_ssl_cluster/audit_f3_center_trace_masked_six_split.py \
  --config experiments/f3/facies_benchmark_v1/106_strat_hmm_multi_head_k6810_center_trace_masked_six_split_v1/00_audit_center_trace_masked_six_split.yaml \
  --dry-run
```

The normal command writes only the candidate-owned preflight audit. An
existing output is an error. `--only-missing` reuses an existing audit only
when its complete JSON identity is byte-for-byte the live identity; it never
silently overwrites stale or partial output. Stale output requires the
explicit project quarantine flag:

```bash
python proc/seis_ssl_cluster/audit_f3_center_trace_masked_six_split.py \
  --config experiments/f3/facies_benchmark_v1/106_strat_hmm_multi_head_k6810_center_trace_masked_six_split_v1/00_audit_center_trace_masked_six_split.yaml \
  --only-missing --quarantine-invalid
```

The audit output is:

```text
$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/lithology/f3/facies_benchmark_v1/voxel_label_budget_center_trace_masked_k6810_six_split_v1/preflight/center_trace_masked_six_split_audit.json
```

This issue freezes a 36-row primary matrix, 24 future new scientific jobs,
and zero executed six-split scientific or smoke jobs. It does not calculate a
GO/HOLD/STOP result.

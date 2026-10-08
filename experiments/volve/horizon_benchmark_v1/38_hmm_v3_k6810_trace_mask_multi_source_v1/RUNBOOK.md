# Volve HMM v3 execution

Set `SEIS_SSL_CLUSTER_ARTIFACT_ROOT` to the artifact directory,
`SEIS_SSL_CLUSTER_WORKSPACE` to this checkout, the existing survey-root
environment variable, and `CUDA_VISIBLE_DEVICES` to the selected GPU.
Use the project's Python environment; `SEIS_SSL_CLUSTER_PYTHON` optionally
selects its Python executable for the shell wrapper.

```bash
bash experiments/volve/horizon_benchmark_v1/38_hmm_v3_k6810_trace_mask_multi_source_v1/run_all.sh --plan
bash experiments/volve/horizon_benchmark_v1/38_hmm_v3_k6810_trace_mask_multi_source_v1/run_all.sh --dry-run --from smoke --to full
bash experiments/volve/horizon_benchmark_v1/38_hmm_v3_k6810_trace_mask_multi_source_v1/run_all.sh --execute
```

The default is planning. The runner verifies the canonical v02 selection,
the pinned baseline definitions and the matched scientific configuration.
Live validation binds the preserved v02 pseudo-target manifests and completed
baseline artifacts. This experiment has no target-generation stage.

Execution order is `smoke`, `full`, `audit`, `embeddings`,
`downstream`, `summary`. Smoke is a single training step, with batch,
workers and all scientific parameters retained from full training. Each full
run starts from its original source checkpoint independently of smoke.

Use `--from STAGE --to STAGE` for a contiguous stage range and
`--candidate ID` for one source. Full execution resumes only that v3
run's own latest checkpoint when present; completed outputs are audited before
reuse. For example:

```bash
bash experiments/volve/horizon_benchmark_v1/38_hmm_v3_k6810_trace_mask_multi_source_v1/run_all.sh \
  --execute --from full --to full \
  --candidate mae100_hmm_v3_mh_k6810_ctmask010_distill020_cons010
```

A decoder run is identified by candidate, layout and size. The established
downstream resolver retains five layouts and three sizes. Its evaluation split,
decoder training, support masks and survey task remain those of v02.
The final paired summary requires all 45 v3 cells and their corresponding
45 completed v02 cells with successful source and decoder audits.

Execution logs and resolved launch configuration live under
`operations/hmm_v3/k6810_trace_mask_multi_source_v1/volve/` in the
artifact root. Checkpoints, embeddings, decoder runs and comparison products
are written beneath `surveys/volve/horizon_benchmark_v1/<kind>/hmm_v3_k6810_trace_mask_multi_source_v1/`.
The final summary owns exactly `comparison.csv`, `summary.json` and
`summary.md` in the `summary_root` declared by `execution.yaml`.
All later stages read artifacts, as required by the
[report policy](../../../../docs/report_sharing_policy.md).

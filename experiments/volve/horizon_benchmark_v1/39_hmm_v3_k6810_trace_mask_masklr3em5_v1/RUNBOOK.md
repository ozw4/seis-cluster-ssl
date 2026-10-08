# volve 3e-5/50:50 execution

Use the project Python environment, `SEIS_SSL_CLUSTER_WORKSPACE`,
`SEIS_SSL_CLUSTER_ARTIFACT_ROOT`, and the existing survey data-root variables.
Choose placement with `CUDA_VISIBLE_DEVICES`; `SEIS_SSL_CLUSTER_PYTHON`
selects the shell wrapper's Python interpreter. Planning is the default.

```bash
bash experiments/volve/horizon_benchmark_v1/39_hmm_v3_k6810_trace_mask_masklr3em5_v1/run_all.sh --plan
bash experiments/volve/horizon_benchmark_v1/39_hmm_v3_k6810_trace_mask_masklr3em5_v1/run_all.sh \
  --dry-run --from smoke --to full \
  --candidate mae100_hmm_v3_mh_k6810_ctmask010_distill020_cons010_masklr3em5
```

Execute requested candidates only. Run a full-batch one-step smoke, then verify
schema-7 identity, weights 0.5/0.5, head/encoder/mask optimizer rates,
and initial student/head/token hashes against the matched nonmasked v02 or
original trace-mask initial state before full training. Checkpoints and targets
remain source- and survey-bound. No pseudo-target refresh is performed.

```bash
bash experiments/volve/horizon_benchmark_v1/39_hmm_v3_k6810_trace_mask_masklr3em5_v1/run_all.sh \
  --execute --from smoke --to smoke \
  --candidate mae100_hmm_v3_mh_k6810_ctmask010_distill020_cons010_masklr3em5
bash experiments/volve/horizon_benchmark_v1/39_hmm_v3_k6810_trace_mask_masklr3em5_v1/run_all.sh \
  --execute --from full --to summary \
  --candidate mae100_hmm_v3_mh_k6810_ctmask010_distill020_cons010_masklr3em5
```

Stage order is smoke, full, audit, embeddings, downstream, summary. Fresh full
runs do not initialize from smoke or completed treatment checkpoints. Recovery
uses identity-validated own-candidate latest checkpoints; completed extraction
and decoder cells are audited before reuse. Candidate locks prevent duplicate
writers. Existing 1e-5 and stopped F3 experiments remain untouched.

Machine-local scheduling permits at most two pipelines on GPU0/GPU1, subject
to free memory. Preserve survey batch sizes (Parihaka 16, Volve 4); do not kill
other jobs or change scientific settings. Use conservative admission margins
for the shared host and record placement and any interruptions.

The artifact namespace is `hmm_v3_k6810_trace_mask_masklr3em5_v1` under
`surveys/volve/horizon_benchmark_v1/<kind>/`.
Logs live in the declared operations root. Each summary owns exactly
`comparison.csv`, `summary.json`, `summary.md`. Full-matrix output uses
`summary_root`; requested-candidate output uses its sibling
`candidates/<candidate-id>/` and requires all 15 audited cells. Publication is
idempotent only when the exact file set and contents match. Partial results
never become a complete matrix. Reports are never pipeline inputs.

The comparison keeps small/medium/large separate, five paired layouts per row.
Overall uncertainty uses five layout clusters, averaging sizes within each
layout first. See the [report policy](../../../../docs/report_sharing_policy.md).


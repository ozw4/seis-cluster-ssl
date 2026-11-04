# Parihaka MAE/no-HMM large: decoder seed 42001

User-requested repeat of MAE only, large only, all five original layouts.
The frozen encoder is the original MAE100 + MAE25 checkpoint; this is not
new pretraining and does not use HMM. Reuse its original embeddings unchanged.
Only decoder initialization and deterministic training tile order change:
seed 42000 becomes 42001, fixed in advance for all layouts.

## Canonical result designation

On 2026-10-13, after both decoder-seed results were reported, the user explicitly
designated seed 42001 as the canonical **Parihaka / no-HMM / MAE / large** result
over all five layouts. [`selection.yaml`](selection.yaml) binds this decision
to the execution definition, completed summary and each result's SHA-256.
Artifact paths in the selection are relative to `SEIS_SSL_CLUSTER_ARTIFACT_ROOT`.

Future result comparisons use these five cells instead of seed 42000 for this
exact scope. Small, medium, LocalBT, random, other surveys and all HMM results
remain unchanged. Preserve seed 42000 as a historical baseline and keep frozen
v1/v2 receipts and summaries byte-identical; they do not incorporate this later
designation. The selection does not change training defaults or rewrite old
aggregates. Comparisons using the new reference must disclose the different
decoder seed across versions, rather than claim a same-seed paired comparison.
This is an explicit post-result reference decision, not a seed-ranking rule.

Mean test Channel IoU is 0.438103 and Channel F1 is 0.609136 for the selected
reference; the original seed has mean IoU 0.455180 and F1 0.625512.

## Fixed execution contract

The canonical Channel resolver continues to reject other seeds. The separate
repeat resolver first validates the hash-bound canonical definition and then
replaces only the decoder seed and output root. Preserve 50 epochs, FP32,
batch 1, LR 0.001, balanced class weights, tile geometry, exact training
selection, validation sections and held-out test complement. Select the first
maximum validation Channel IoU checkpoint; test does not choose the seed.

## Execution and recovery

Set `SEIS_SSL_CLUSTER_WORKSPACE` to the checkout and
`SEIS_SSL_CLUSTER_ARTIFACT_ROOT` to its existing artifact root.

```bash
python proc/seis_ssl_cluster/run_parihaka_channel_seed_repeat.py \
  --config experiments/parihaka/facies_benchmark_v1/47_mae_no_hmm_large_decoder_seed42001_v1/execution.yaml --dry-run
CUDA_VISIBLE_DEVICES=1 python proc/seis_ssl_cluster/run_parihaka_channel_seed_repeat.py \
  --config experiments/parihaka/facies_benchmark_v1/47_mae_no_hmm_large_decoder_seed42001_v1/execution.yaml --execute
```

All five cells execute serially under one experiment lock. Admission waits for
at least 20 GiB free on the selected GPU; do not stop other jobs or change
scientific settings.
`--layout layout_000 --execute --max-steps 1` performs a one-step feasibility
check within that cell's fixed budget. Full execution resumes only each new
cell's own identity-matched `latest.pt`; no old decoder checkpoint is reused.
Completed cells are audited before reuse. Partial or conflicting outputs fail
closed; preserve them for diagnosis rather than overwriting them.

Runs use `surveys/parihaka/facies_benchmark_v1/benchmark/mae_no_hmm_large_decoder_seed42001_v1/`;
status and logs use `operations/parihaka/mae_no_hmm_large_decoder_seed42001_v1/`
inside artifacts. Neither the original results nor frozen v1/v2/v3 records
are replaced or republished. Small and medium are out of scope.

## Comparison

After all five cells pass full completion and source/supervision identity
audits, publish exactly `comparison.csv`, `summary.json`, `summary.md` under
the new `summary_root`. Compare the five public test metrics with seed 42000,
retaining layout-level values, mean, median, sample SD and the mean after
excluding the lowest/highest layout value independently for each metric.
The exact review-file set is tested; reports are never pipeline inputs.

This measures decoder-seed sensitivity of one fixed encoder, not pretraining
seed uncertainty. Keep both seed results. The later explicit canonical result
designation above does not make this a method gain or authorize choosing
whichever seed is preferable by test performance.

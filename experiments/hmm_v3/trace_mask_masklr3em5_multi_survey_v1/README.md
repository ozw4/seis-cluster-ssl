# Non-F3 trace-mask follow-up at mask-token LR 3e-5

Transfer the F3 3e-5/50:50 setting to Parihaka and Volve, with canonical
nonmasked v02 as the paired reference. This is a fresh 25-epoch treatment,
not a changed-LR continuation of the stopped 1e-5 trace-mask checkpoints.

Definitions: [Parihaka 46](../../parihaka/facies_benchmark_v1/46_hmm_v3_k6810_trace_mask_masklr3em5_v1/README.md),
[Volve 39](../../volve/horizon_benchmark_v1/39_hmm_v3_k6810_trace_mask_masklr3em5_v1/README.md).
They inherit their original trace-mask recipes, changing only the mask-token
optimizer rate, versioned tag and output namespace. The original parents,
immutable targets, canonical v02 consistency 0.1, encoder/head LR 1e-5,
10% column masking and equal normalized branch weights remain fixed.

The initial requested scope was MAE100 on both surveys: two training conditions
and 30 decoder cells. Those results are retained without rerunning them.
On 2026-10-21 the user explicitly requested the remaining non-F3 3e-5/50:50
conditions: LocalBT3 and random on both surveys, four fresh 25-epoch training
arms and 60 decoder cells. This completes the existing three-source matrix
without changing the frozen definitions. F3 and the old 1e-5 studies are not
part of this follow-up.

Use the existing decoder seed `42000` for the added conditions; F3 LocalBT's
separate final-result designation of seed `42001` does not transfer here.
Select the fixed epoch-25 encoder and use validation, not test, to select
decoder checkpoints. Preserve the existing survey splits and label budgets.

Use each survey runbook and an explicit candidate argument. Report the test
metrics separately per survey and small/medium/large, with five paired layouts
per size. Completed candidate-only summaries own the exact three review
files and are not labeled complete three-source matrices. No test-guided
choice of LR, target, model checkpoint or training duration is performed.

Execution and recovery records use artifacts/operations/hmm_v3/
trace_mask_masklr3em5_multi_survey_v1/. Preserve old outputs; later stages
consume only artifacts, never reports.

The 2026-10-21 missing-source execution has a separate operation record at
`artifacts/seis_ssl_cluster/operations/hmm_v3/trace_mask_masklr3em5_missing_sources_20261021_v1/`.
It runs all four full-batch one-step smokes and the initialization audit before
full training, then each arm's embeddings and 15 decoder cells. GPU admission
retains the existing 65 GiB (Parihaka) / 45 GiB (Volve) free-memory thresholds,
at most two owned pipelines and no changes to batch sizes. Candidate-level
locks prevent duplicate writers. After all added arms finish, publish each
survey's 45-cell paired comparison, reusing the completed MAE evidence and
preserving its original candidate summary and operation records.

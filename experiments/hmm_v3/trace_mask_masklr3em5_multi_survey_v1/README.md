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

The current default request scope is MAE100 on both surveys: two training
conditions and 30 decoder cells, unless the user explicitly selects all three
initialization families. Supported LocalBT3/random definitions are not an
authorization to run them. F3 and the old 1e-5 studies remain stopped.

Use each survey runbook and an explicit candidate argument. Report the test
metrics separately per survey and small/medium/large, with five paired layouts
per size. Completed candidate-only summaries own the exact three review
files and are not labeled complete three-source matrices. No test-guided
choice of LR, target, model checkpoint or training duration is performed.

Execution and recovery records use artifacts/operations/hmm_v3/
trace_mask_masklr3em5_multi_survey_v1/. Preserve old outputs; later stages
consume only artifacts, never reports.

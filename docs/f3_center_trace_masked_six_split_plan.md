# F3 center-trace masked six-split scientific contract

This document is the source of truth for the follow-up six-split evaluation of
the center-trace masked HMM path-reconstruction candidate. It freezes the
evaluation matrix and the decision rule before any six-split result is viewed.
This issue only establishes and audits the contract. It does not build a
dataset, run a decoder, run a CPU smoke, retrain a model, or publish a result.

## Scope and lineage

The candidate role is `mh_ctmask010_nocons` with model tag
`strat_hmm_pretext_mh_k6810_ctmask010_nocons_topblock1_distill_v1`.
The primary baseline role is `mh_nocons` with model tag
`strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1`.

The candidate must be the candidate admitted by the original-split handoff
`results/f3/facies_benchmark_v1/strat_hmm_multi_head_k6810_center_trace_masked_original_split_v1/center_trace_masked_original_handoff.json`.
The hard baseline and all six-split inputs are read-only lineage from
experiment 96. Experiment 96 and its published results are not rewritten by
this follow-up.

## Frozen primary matrix

The canonical split IDs are:

`split_000`, `split_001`, `split_002`, `split_003`, `split_004`, `split_005`.

The budgets are `cap25`, `cap50`, and `cap100`. The label-subset seed is `0`
and the decoder seed is `42000`. The primary metrics are `macro_f1` and
`mean_iou`.

The monitored classes are `3` and `5`. For each monitored class, retain
`f1`, `iou`, `boundary_recall_t2`, and `boundary_recall_t4` as diagnostic
metrics.

The full primary matrix is:

| rows | splits | budgets | model roles |
| ---: | ---: | ---: | ---: |
| 36 | 6 | 3 | 2 |

The 36 rows are composed of:

- 18 future candidate jobs: six splits by three budgets;
- 6 future new baseline `mh_nocons` cap100 jobs; and
- 12 historical baseline rows: experiment 96 `mh_nocons` cap25/cap50.

Thus the future new scientific job count is `24`. The experiment-96 matrix is
the fixed 36-row matrix of six splits by cap25/cap50 by `mae`,
`m1_current_k6`, and `mh_nocons`; it must not be extended or rewritten.

`cap25` is diagnostic only. It cannot establish GO by itself.

## Frozen six-split decision rule

For each budget, calculate candidate minus baseline deltas separately for both
primary metrics across the six splits. A budget is **positive** only when, for
both `macro_f1` and `mean_iou`, all of the following hold:

- mean delta `> 0`;
- median delta `> 0`; and
- wins `>= 4/6`.

A budget is **negative** only when, for both primary metrics, both of the
following hold:

- mean delta `< 0`; and
- wins `<= 2/6`.

Systematic major degradation is fixed before results are seen: for the same
class/metric, if `mean delta <= -0.05` occurs at two or more budgets, the
condition is present. The monitored class/metric family is the four metrics
for classes `3` and `5`: `f1`, `iou`, `boundary_recall_t2`, and
`boundary_recall_t4`.

The final labels are:

- `CTMASK_SIX_SPLIT_GO`: `cap50` is positive, `cap100` is positive, and
  there is no systematic major degradation;
- `CTMASK_SIX_SPLIT_STOP`: both `cap50` and `cap100` are negative, or
  systematic major degradation is present;
- `CTMASK_SIX_SPLIT_HOLD`: every other outcome.

This issue does not implement the result aggregation or any GO/HOLD/STOP
calculation. Those belong to a later evaluation issue and must use this
pre-registered rule.

## Start gate

The start audit requires the original-split handoff to have artifact type
`f3_center_trace_masked_original_screening_handoff`, `status: PASS`, and
`formal_status: CTMASK_ORIGINAL_GO`. Its `six_split_follow_up.ready` value
must be `true`; its candidate live validation must be exactly `15/15 PASS`;
and both `six_split_jobs_executed` and the six-split scientific execution
count must be zero.

The candidate lineage must have a passing pretraining handoff, checkpoint
schema `7`, objective `center_trace_masked_hmm_path_reconstruction_v1`, hard
Viterbi targets, K `[6, 8, 10]`, and unmasked encoder-token embedding input.
The live candidate embedding must be `[76, 113, 32, 384]`, `float16`, with a
`[76, 113, 32]` boolean valid-token mask. Candidate and hard-baseline valid
tokens must be bitwise identical. The checkpoint, embedding, metadata, and
handoff hashes must match their live files.

The read-only experiment-96 evidence must contain 12 canonical dataset rows,
36 completed scientific rows, and 12 completed `mh_nocons` cap25/cap50 rows,
all with decoder seed `42000` and label-subset seed `0`. The split inventory,
split-token dataset, full voxel split dataset, and original-split dataset
manifests are also checked for their declared identities and source hashes.

The audit records the current Git commit, `git status --short`, and the SHA-256
of the binary tracked diff from `HEAD`. It writes only its own candidate-owned
preflight artifact, and only after all checks pass.

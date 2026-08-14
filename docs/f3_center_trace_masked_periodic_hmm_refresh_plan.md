# F3 center-trace masked periodic HMM refresh

This document is the source-of-truth scientific contract for a new
center-trace masked pretraining treatment. Unlike the existing fixed-target
center-trace lineage, this treatment periodically re-extracts the full F3
survey embedding from the current student encoder, warm-starts the ordered
HMM centers from the previous generation, and replaces the hard targets for
the next generation.

Issue #313 adds this contract only. It does not run the treatment or modify
production code, experiment YAML, checkpoints, embeddings, pseudo-targets,
decoder jobs, or result artifacts. Existing experiment 104, 105, and 106
records and their documentation and artifacts remain historical read-only
evidence.

## Historical lineage and fixed baseline

The fixed center-trace lineage is the read-only scientific predecessor. Its
pretraining contract is recorded by [experiment 104](../experiments/f3/facies_benchmark_v1/104_strat_hmm_multi_head_k6810_center_trace_masked_v1/README.md),
its original-split screening by [experiment 105](../experiments/f3/facies_benchmark_v1/105_strat_hmm_multi_head_k6810_center_trace_masked_low_label_v1/README.md),
and its six-split preflight by [experiment 106](../experiments/f3/facies_benchmark_v1/106_strat_hmm_multi_head_k6810_center_trace_masked_six_split_v1/README.md).
Those records are not inputs to be regenerated or rewritten by this plan.

The fixed target and its initial ordered HMM centers are immutable inputs to
the initial generation. The target representation is hard
`hard_viterbi_labels_v1`; the K=6/8/10 heads, initial preprocessing, target
weights, model initialization, and training geometry are inherited from that
lineage. The new scientific variable is limited to periodic HMM-center and
hard pseudo-target refresh.

## Scientific identity

The following fields are fixed verbatim for this treatment:

```text
experiment_role:
  multi_head_center_trace_masked_periodic_hmm_refresh_hard_pretext

variant:
  ctmask010_refresh3ep_hmm2_nocons

model_role:
  mh_ctmask010_refresh3ep_hmm2_nocons

model_tag:
  strat_hmm_pretext_mh_k6810_ctmask010_refresh3ep_hmm2_nocons_topblock1_distill_v1

target_refresh_semantics:
  periodic_student_hmm_center_refresh_v1

refresh_schedule_semantics:
  after_epochs_2_5_8_11_14_17_20_v1

center_update_semantics:
  warm_start_full_mean_two_iterations_final_decode_v1

refresh_embedding_semantics:
  current_student_unmasked_eval_full_survey_v1

preprocessing_policy:
  freeze_initial_residualizer_pca_v1

target_activation_policy:
  atomic_next_epoch_activation_v1

checkpoint_selection_policy:
  final_completed_epoch_v1
```

The checkpoint identity is schema 8. The periodic refresh identity must not
be represented as the fixed-target `mh_ctmask010_nocons` identity.

## Common fixed conditions

All conditions shared with fixed center-trace restoration are frozen as
follows:

```text
K: [6, 8, 10]
target representation: hard_viterbi_labels_v1
center-trace column fraction: 0.10
masked/visible prototype weights: 0.50/0.50
prototype/usage/consistency/distillation: 1.0/0.005/0.0/0.2
student unfreeze: top encoder block 1
encoder learning rate: 1.0e-5
head learning rate: 3.0e-4
epochs: 25
samples per epoch: 4096
batch size: 4
seed: 42
teacher and student initialization: existing fixed center-trace lineage
```

The only changed scientific elements are the HMM centers and hard
pseudo-targets updated by the periodic refresh contract. The prototype head,
encoder, learned replacement token, and optimizer continue to update through
ordinary training across refreshes; they are not reset or copied from the
HMM centers.

## Exact refresh schedule

The initial hard target is active for epochs 1-2. A refresh is committed only
after the listed completed epoch, and its final target becomes active starting
with the next epoch:

```text
epochs 1-2: initial generation
refresh after epoch 2
refresh after epoch 5
refresh after epoch 8
refresh after epoch 11
refresh after epoch 14
refresh after epoch 17
refresh after epoch 20
epochs 21-25: final refreshed generation
refresh after epoch 25: none
```

There are exactly eight generations, with these immutable IDs and order:

```text
refresh_0000_initial
refresh_0001_epoch002
refresh_0002_epoch005
refresh_0003_epoch008
refresh_0004_epoch011
refresh_0005_epoch014
refresh_0006_epoch017
refresh_0007_epoch020
```

No additional initial, intermediate, end-of-epoch-25, or retry generation is
part of the scientific identity.

## Refresh computation contract

At each refresh, use the current student encoder in `eval()` and
`inference_mode()`. Apply no mask. Re-extract the full F3 survey embedding,
including its current valid-token mask, using the
`current_student_unmasked_eval_full_survey_v1` semantics.

The initial HMM artifact binds the preprocessing and path-scoring inputs for
all generations. The following are fixed and must be reused without drift:

- residualizer;
- PCA;
- transition cost;
- initial and terminal priors;
- expected-boundary prior;
- edge margin; and
- valid-token mask.

For each K, initialize the refresh from the previous generation's centers
with the same ordered state IDs. Do not run K-means reinitialization, state
permutation, or depth-based reordering. Perform exactly two iterations:

```text
decode with current centers
-> replace every non-empty center by the exact mean of assigned current features
```

After those two iterations, perform exactly one final decode. Center updates
are complete replacement. EMA, interpolation with the old center, and label
mixing are forbidden. An empty state is a hard failure even though the
replacement operation itself is defined for every non-empty state.

The final decode is the complete hard-target replacement for the next epoch;
there is no partial, blended, or within-epoch target activation. The following
conditions are hard failures:

- an empty state;
- a non-finite center or feature;
- a label outside the declared K range;
- valid-mask drift;
- no finite HMM path; or
- any other failure to validate the generation's declared input and output
  identities.

Record center movement, label-change rate, state counts, boundary count,
confidence distribution, and mean state depth as diagnostics. These are not
stopping gates unless one of the hard failures above occurs.

Confidence and boundary weight reuse the existing policy declared by the
initial hard-target lineage. No new confidence or boundary-weight formula is
introduced. If that policy is constant or absent, its existing meaning is
preserved exactly.

## Generation artifact contract

Each full output root owns the following refresh area:

```text
<full output root>/target_refresh/
  active_target_generation.json
  periodic_refresh_chain.json
  generations/
    refresh_0000_initial/
      refresh_generation.json
    refresh_0001_epoch002/
      refresh_generation.json
      ...
```

`refresh_0000_initial` is a descriptor binding the existing immutable hard
target manifest, initial ordered centers, and fixed preprocessing artifact
paths and hashes. It does not copy or rewrite historical arrays.

Each later generation binds generation-scoped embedding, K-specific centers,
final labels, pseudo-target arrays, the multi-head target manifest,
diagnostics, and the hash of every file in that generation. Every
`refresh_generation.json` has:

```text
artifact_type: strat_hmm_periodic_refresh_generation
schema_version: 1
status: COMPLETE
```

Generation output is built in staging and atomically published only after all
validation passes. An existing generation is never overwritten in place.
`active_target_generation.json` switches atomically and contains only the
path and hash of a complete generation manifest.

`periodic_refresh_chain.json` links the previous generation manifest hash,
source student state hash, refresh epoch, and fixed preprocessing identity.
Partial, foreign, or hash-drifted generations must never be silently reused.

## Checkpoint schema 8 contract

Schema 8 is a separate checkpoint identity and is resume-incompatible with
schemas 2-7 in both directions. Rolling resume uses `latest.pt`.

Step, epoch, and refresh-boundary checkpoints bind all of the following:

- active generation ID;
- active generation manifest path and hash;
- refresh-chain hash;
- last completed refresh epoch; and
- next refresh epoch.

Global minimum pretraining loss is not a model-selection criterion. After
epoch 25 completes normally, only that completed epoch's `latest.pt` is
atomically copied to `selected.pt`; it is the primary checkpoint for
downstream extraction. This identity does not generate, publish, or consume
`best.pt` as a scientific selected checkpoint.

Optimizer, head, encoder, learned replacement-token, AMP/RNG, and DataLoader
state continue across refreshes. A target refresh does not reset or
reinitialize any of those states.

## Experiment and review roots

The reserved experiment and review roots are:

```text
experiments/f3/facies_benchmark_v1/
  107_strat_hmm_multi_head_k6810_center_trace_masked_periodic_refresh_v1/

experiments/f3/facies_benchmark_v1/
  108_strat_hmm_multi_head_k6810_center_trace_masked_periodic_refresh_low_label_v1/

reports/f3/facies_benchmark_v1/
  strat_hmm_multi_head_k6810_center_trace_masked_periodic_refresh_v1/

reports/f3/facies_benchmark_v1/
  strat_hmm_multi_head_k6810_center_trace_masked_periodic_refresh_original_split_v1/
```

The pretraining handoff is fixed as:

```text
artifact_type:
  f3_center_trace_masked_periodic_refresh_pretraining_handoff
schema_version: 1
status: PASS
primary_checkpoint_role: completed_final_selected
```

This documentation issue does not create these roots or any files beneath
them.

## Original-split evaluation contract

The new candidate evaluation has exactly 15 jobs: three budgets by five
subsample seeds.

```text
new candidate jobs:
  3 budgets x 5 subsample seeds = 15

budgets:
  cap25, cap50, cap100

subsample seeds:
  0, 1, 2, 3, 4

decoder seed:
  42000 + subsample_seed
```

The primary comparison is:

```text
mh_ctmask010_refresh3ep_hmm2_nocons
  - mh_ctmask010_nocons
```

The diagnostic comparisons are:

```text
candidate - mh_nocons
candidate - m1_current_k6
candidate - mae
```

The primary budget decision uses the existing center-trace original-split
rule. A budget is positive only when both primary metrics meet their
condition:

- Macro F1 mean delta `> 0` and wins `>= 4/5`; and
- Mean IoU mean delta `> 0` and wins `>= 4/5`.

A budget is negative only when both primary metrics meet their condition:

- Macro F1 mean delta `< 0` and wins `<= 1/5`; and
- Mean IoU mean delta `< 0` and wins `<= 1/5`.

Systematic major degradation is present when, for classes 3 or 5 and one of
`f1`, `iou`, `boundary_recall_t2`, or `boundary_recall_t4`, the same
class/metric mean delta is `<= -0.05` at two or more budgets.

The formal status is:

```text
CTMASK_REFRESH_ORIGINAL_GO:
  at least 2 positive budgets and no systematic major degradation

CTMASK_REFRESH_ORIGINAL_STOP:
  at least 2 negative budgets or systematic major degradation

CTMASK_REFRESH_ORIGINAL_HOLD:
  otherwise
```

Only GO sets `six_split_follow_up.ready = true`. This series does not
implement or execute a six-split config, runner, job, or result. The six-split
job count is zero for every formal status.

## Explicit scope exclusions

This contract does not include:

- sweeps over K, mask ratio, loss weight, unfreeze depth, augmentation, HMM
  transition/path prior, PCA, or residualizer;
- per-batch refresh, EMA centers, soft targets, posterior targets, lateral
  smoothing, or XY consensus; or
- making the prototype head and HMM centers identical, or copying one into
  the other.

The fixed center-trace lineage, its historical artifacts, and all other
experiment identities remain separate evidence and are not rewritten by this
plan.

# F3 center-trace masked HMM path reconstruction

This document is the source-of-truth scientific contract for a future
center-trace masked pretraining treatment. It describes a new pretext identity
without changing the existing K=6/8/10 hard-HMM prototype or its published
evidence. This issue adds documentation only: it does not run the treatment or
modify production code, tests, experiment YAML, checkpoints, artifacts, or
published results.

## Fixed baseline and immutable evidence

The primary baseline is hard `mh_nocons`, with model tag
`strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1`. Its heads are K=6,
K=8, and K=10, and its target representation is
`hard_viterbi_labels_v1`. The baseline loss weights are:

| term | weight |
| --- | ---: |
| prototype | 1.0 |
| usage | 0.005 |
| consistency | 0.0 |
| distillation | 0.2 |

The fixed baseline fields are also recorded as:

```text
head_ks: [6, 8, 10]
target_representation: hard_viterbi_labels_v1
prototype_weight: 1.0
usage_weight: 0.005
consistency_weight: 0.0
distillation_weight: 0.2
```

The existing hard target manifest and its target arrays, confidence, boundary
weight, and valid mask are immutable inputs to this treatment. They are read
as published; they are not regenerated, smoothed, relabeled, or rewritten.

Soft-posterior results, lateral hard-target results, 3-of-4 XY consensus
results, and unanimous XY correction results are separate evidence. None of
those results or target semantics may be mixed into the center-trace masked
target. In particular, the center-trace treatment does not alter the HMM
labels, posterior, confidence, boundary weight, or valid mask.

The baseline identity and published decision context are recorded by the
[K=6/8/10 multi-head experiment](../experiments/f3/facies_benchmark_v1/94_strat_hmm_multi_head_k6810_v1/README.md)
and its frozen [results summary](../reports/f3/legacy/facies_benchmark_v1/strat_hmm_multi_head_k6810_v1/multi_head_results_summary.json).
Those records remain historical evidence; this plan makes no new scientific
claim.

## Scientific identifiers

The following identifiers are fixed and must be recorded verbatim in the
experiment and checkpoint identity:

```text
objective_semantics: center_trace_masked_hmm_path_reconstruction_v1
mask_semantics: xy_token_column_full_z_v1
replacement_semantics: learned_encoder_mask_token_v1
mask_rng_policy: stateless_step_seed_v1
```

The treatment identity is:

```text
model_tag: strat_hmm_pretext_mh_k6810_ctmask010_nocons_topblock1_distill_v1
experiment_role: multi_head_center_trace_masked_hard_pretext
variant: ctmask010_nocons
supervised_loss: structured_hmm_center_trace_masked_hard_v1
consistency_policy: disabled_for_center_trace_masked_v1
checkpoint schema: 7
```

## Mask construction contract

Masking operates on the patch/token grid. A mask unit is one XY token column
`(tx, ty, :)`; selecting that column masks all of its Z tokens.

- Raw amplitude and the voxel zero-mask are unchanged. After patch
  projection, and before position embedding is added and before encoder input,
  every selected token is replaced by the learned encoder mask token.
- Position embeddings remain present for selected tokens.
- For K=6/8/10, a column is eligible only when it contains at least one token
  in the common existing hard-target valid mask AND the student token-valid
  mask. Eligibility therefore uses the same target-valid semantics for every
  head.
- Let `N` be the number of eligible columns. A sample requires `N >= 2`.
- The number of selected columns is

  ```text
  min(N - 1, max(1, floor(0.10 * N + 0.5)))
  ```

  Columns are selected without replacement, so at least one eligible column
  remains visible.
- A selected column is replaced for all of its Z tokens, whether each token is
  valid or invalid. Supervised loss is applied only to valid tokens.
- Mask selection is stateless and is determined by stable integer mixing of
  the training seed, epoch, global step, batch index, and sample index. It
  uses the `stateless_step_seed_v1` policy: Python's process-randomized
  `hash()` and the global PyTorch RNG are not used.
- A sample without enough eligible columns is not a silent no-op. It must be
  rejected by existing crop resampling or by an explicit validation error.

The same selected-column decision is used for the K=6/8/10 supervised views
of a sample. No head-specific target regeneration or mask eligibility rule is
introduced.

## Student, teacher, and loss semantics

Only the student receives the masked token sequence. The teacher encodes the
same amplitude crop unmasked. Existing teacher behavior and the existing
hard-target inputs remain unchanged.

For each K, use the existing confidence multiplied by boundary-weighted hard
prototype cross-entropy (the per-token weight is `confidence ×
boundary_weight`). Normalize the masked valid-token contribution and the
visible valid-token contribution separately. Denote those per-K losses by
`L_masked,K` and `L_visible,K`:

```text
L_proto = 0.5 * mean_K(L_masked,K)
          + 0.5 * mean_K(L_visible,K)

L_total = 1.0 * L_proto
          + 0.005 * L_usage(all supervised valid tokens)
          + 0.2 * L_distill(visible valid tokens only)
```

The two prototype groups must not be allowed to bury one another through
unequal token counts. The current hard target's confidence, boundary weight,
and valid mask are retained exactly. Cross-head consistency has an exactly
zero contribution.

The learned mask token is a training-only parameter. Adding it does not
broaden the existing trainable scope beyond the student top encoder block;
existing head, optimizer, teacher, and initialization rules remain paired with
hard `mh_nocons`.

Embedding extraction and downstream inference do not generate masks. They use
the normal `AmplitudeMAE3D.encode_tokens` path and encode all tokens.

## Identity, checkpoint, and evaluation contract

Checkpoint schema 7 is a separate identity. It is not resume-compatible with
existing schemas 2-6 in either direction. A schema 7 checkpoint binds all of
the following:

- the immutable hard target manifest identity;
- the objective, mask, replacement, and RNG semantics;
- the mask/loss configuration and weights;
- the learned mask-token state;
- the initial student and head states; and
- the optimizer-group identity.

The initial student state and initial head state must have exact parity with
hard `mh_nocons`. The added mask token is new state and must be recorded under
its own identity rather than being presented as baseline parity.

Checkpoint selection keeps the existing `metrics.loss` criterion and
strictly-lower policy. Masked/visible valid fraction, per-K masked/visible
prototype loss, and per-K masked top-1 accuracy are diagnostic metrics. They
are recorded for inspection but are not pretraining-artifact GO thresholds.

An XY-shuffle masked-path accuracy decrease is not required by the
implementation, validation, or adoption gate. It must not be promoted into a
scientific acceptance criterion for this treatment.

## Explicit scope exclusions

The center-trace masked treatment does not include:

- changing HMM labels, posterior, confidence, or boundary weight;
- neighbour majority, posterior averaging, lateral smoothing, or any other
  target correction;
- contrastive positive or negative pairs;
- a simultaneous Z-only, short-interval, block, or random-token mask
  comparison;
- sweeps over mask ratio, loss weight, unfreeze depth, K, or augmentation;
- a raw-amplitude reconstruction decoder or a new HMM decoder;
- downstream prediction smoothing; or
- a six-split execution.

The fixed evaluation gate is the original-split gate. A six-split study can be
considered only as separate follow-up work if that original-split gate is GO;
it is not part of this contract.

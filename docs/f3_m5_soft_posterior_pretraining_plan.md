# M5-U — Posterior-aware soft multi-resolution HMM pretraining

Planning document only. No runnable M5 config or completed M5 scientific result
is implied.

M4's formal six-split status remains `M4_MH_SPLIT_HOLD`. Separately, the
project decision is `ADOPT_MH_NOCONS_FOR_M5`: use the selected hard-target
model as the starting baseline for the next method-development milestone. This
does not treat HOLD as CONFIRMED. Additional decoder seeds remain an optional
diagnostic, not a required gate for beginning M5-U.

## Scientific question

Does retaining the HMM state uncertainty discarded by hard Viterbi one-hot
targets as K=6/8/10 forward-backward posteriors improve low-label voxel
performance relative to hard-target `mh_nocons`?

## Fixed baseline and references

- Primary hard-target baseline: `mh_nocons`
  (`strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1`).
- Control: current K6.
- Reference: MAE.
- Not carried forward: `mh_cons010`.

## First M5-U change: target representation only

The initial M5-U experiment changes only the target representation:

| Stage | Target |
| --- | --- |
| Current hard-target baseline | Hard Viterbi one-hot state target |
| M5-U | Forward-backward posterior distribution |

The following remain fixed for the first M5-U experiment:

- K = [6, 8, 10]
- the same shared encoder and independent heads
- `consistency_weight = 0`
- the same teacher, student initialization, and unfreeze depth
- the same prototype, usage, and distillation weights
- the same crop, data order, and seed

Do not add any of the following to the first M5-U experiment: entropy
confidence weighting, posterior-temperature sweeps, hard/soft interpolation,
cross-head consistency, a boundary auxiliary head, lateral smoothing, EM target
refresh, extra K values, or best-K selection.

## Posterior artifact contract

For each K, the posterior artifact must satisfy:

- For valid tokens, `posterior` has shape `[valid token, K]`; values are
  finite and nonnegative, and every posterior row sums to 1.
- For invalid tokens, `posterior` is all zero.
- The common valid mask is bitwise exact across K6, K8, and K10.

Required diagnostics are:

- posterior-entropy quantiles
- top-1-probability quantiles
- top-1 minus top-2 margin
- Viterbi-state posterior probability
- expected normalized order
- effective posterior state usage
- boundary-versus-interior entropy
- per-trace monotonicity diagnostics

## Initial loss

The initial loss proposal is the mean soft categorical cross-entropy across the
K heads, plus the existing usage loss and existing distillation loss.
Consistency loss is not used.

## Evaluation plan

### Original-split screening

Run one new soft model across cap25, cap50, and cap100 with five paired decoder
seeds: 3 budgets × 5 seeds = 15 new decoder jobs.

Reuse the existing reference jobs for hard `mh_nocons`, current K6, and MAE.
Evaluate these paired comparisons:

- `mh_soft_nocons − mh_nocons`
- `mh_soft_nocons − current K6`
- `mh_soft_nocons − MAE`

The initial soft-versus-hard GO gate is positive only when at least two of the
three budgets each have all of the following:

- mean ΔMacro F1 > 0 and Macro F1 wins >= 4/5
- mean ΔMean IoU > 0 and Mean IoU wins >= 4/5

There must also be no systematic major degradation for Class 3 or Class 5.
This is a planned gate, not evidence that soft targets will improve results.

### Six-split follow-up

Perform the follow-up only if original-split screening is positive. It adds one
candidate across six splits and cap25/cap50: 6 splits × 2 budgets × 1 candidate
= 12 new decoder jobs. Reuse the existing 36 reference jobs and add only the
candidate jobs.

## Later candidates

Only after the soft-posterior change has been evaluated, consider these
candidates in order:

1. Boundary transition-probability auxiliary head.
2. Edge-aware lateral posterior smoothing.
3. One-step HMM/target refresh.

Do not mix these candidates into the first M5-U experiment.

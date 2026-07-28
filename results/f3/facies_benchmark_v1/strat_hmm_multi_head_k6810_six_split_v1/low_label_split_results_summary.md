# M4 six-split low-label summary

## Formal result

- Formal status: `M4_MH_SPLIT_HOLD`
- Systematic major degradation: `false`.

## Project decision

- Project decision: `ADOPT_MH_NOCONS_FOR_M5`.
- `mh_nocons` is adopted as the M5 hard-target baseline (`strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1`).
- Formal HOLD is retained; the project adoption does not reinterpret it as CONFIRMED.
- Additional decoder seeds are optional diagnostics, not a required gate.

## Primary evidence

| comparison | budget | mean ΔMacro F1 | median ΔMacro F1 | Macro F1 wins | mean ΔMean IoU | median ΔMean IoU | Mean IoU wins |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mh_nocons - current K6 | cap25 | +0.016611604394266899 | +0.016577729030455401 | 5/6 | +0.015077078634801086 | +0.016222674414893556 | 4/6 |
| mh_nocons - current K6 | cap50 | +0.014035314487712947 | +0.011124544861751906 | 4/6 | +0.016813738241475602 | +0.010289467789213858 | 3/6 |
| mh_nocons - MAE | cap25 | +0.037582555423096985 | +0.040544959586256424 | 6/6 | +0.046042863991780897 | +0.043366871565460097 | 6/6 |
| mh_nocons - MAE | cap50 | +0.022557693487874748 | +0.011110648156900049 | 4/6 | +0.031891207389166176 | +0.020459819392527168 | 6/6 |
| current K6 - MAE | cap25 | +0.020970951028830082 | +0.022681005511951458 | 5/6 | +0.030965785356979813 | +0.033427362003821709 | 5/6 |
| current K6 - MAE | cap50 | +0.0085223790001618029 | +0.012367860432553779 | 5/6 | +0.015077469147690573 | +0.023255665786706903 | 5/6 |

## Why the formal result is HOLD

- `cap50` Mean IoU for mh_nocons - current K6: mean Δ+0.016813738241475602; median Δ+0.010289467789213858; wins `3/6`.
- Preregistered requirement: wins ≥ `4/6`.
- The mean and median are positive, but the win-count requirement is not met.

## Original-split dependence

| metric | split_000 cap50 Δ | six-split cap50 mean Δ |
| --- | ---: | ---: |
| macro_f1 | +0.059690198730827149 | +0.014035314487712947 |
| mean_iou | +0.069676276961439743 | +0.016813738241475602 |
- The original split overestimated the multi-head incremental effect.

## MAE evidence

| comparison | budget | mean ΔMacro F1 | Macro F1 wins | mean ΔMean IoU | Mean IoU wins |
| --- | --- | ---: | ---: | ---: | ---: |
| mh_nocons - MAE | cap25 | +0.037582555423096985 | 6/6 | +0.046042863991780897 | 6/6 |
| mh_nocons - MAE | cap50 | +0.022557693487874748 | 4/6 | +0.031891207389166176 | 6/6 |
| current K6 - MAE | cap25 | +0.020970951028830082 | 5/6 | +0.030965785356979813 | 5/6 |
| current K6 - MAE | cap50 | +0.0085223790001618029 | 5/6 | +0.015077469147690573 | 5/6 |
- Structured HMM pretraining relative to MAE is the most robust conclusion in this evaluation.

## Class and boundary findings

- The systematic major degradation gate did not trigger.
- Boundary F1 means are positive across the reported budgets and tolerances.
| budget | mean Δboundary F1 t2 | mean Δboundary F1 t4 | mean Δvertical boundary-position MAE (oriented) |
| --- | ---: | ---: | ---: |
| cap25 | +0.0045853840259309402 | +0.014974487623771829 | -0.068567120555741504 |
| cap50 | +0.026374731167941895 | +0.03557461991898684 | +0.0040274906707428544 |
- Vertical boundary-position MAE worsens at cap25 (its oriented delta is negative; lower raw MAE is better).
- Class 3 / Class 5 and boundary recall are not uniformly improved across splits.
- Interpret overall Macro F1 / Mean IoU separately from boundary localization.

## Interpretation

- cap25 is robust in the preregistered primary evidence.
- cap50 is split-dependent in the preregistered primary evidence.
- The six-split evidence does not establish mh_nocons superiority as a formal confirmatory result.
- mh_nocons is adopted as the baseline for the next method-development stage.
- Proceed to soft-posterior target development; its effectiveness remains unverified.

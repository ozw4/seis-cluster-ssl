# F3 M5-LS target-only calibration

Target calibration: `M5_LS_TARGET_HOLD`

Selection policy: `target_only_smallest_eligible_beta_v1`

Selected beta: `HOLD`

Beta-zero parity: `PASS`

Candidate betas and the selection policy were fixed before diagnostics. No facies/lithology labels, decoder outputs, or downstream metrics are read.

## Candidate eligibility

| candidate | beta | eligible | reasons |
| --- | ---: | :---: | --- |
| beta010 | 0.10 | False | K=6: lateral transition count exceeds recomputed source transition count; K=8: lateral transition count exceeds recomputed source transition count; K=10: affinity-weighted XY disagreement is not reduced; K=10: highest-affinity XY disagreement is not reduced; K=10: lateral transition count exceeds recomputed source transition count |
| beta025 | 0.25 | False | K=6: lateral transition count exceeds recomputed source transition count; K=8: lateral transition count exceeds recomputed source transition count; K=10: affinity-weighted XY disagreement is not reduced; K=10: highest-affinity XY disagreement is not reduced; K=10: lateral transition count exceeds recomputed source transition count |
| beta050 | 0.50 | False | K=6: lateral transition count exceeds recomputed source transition count; K=8: lateral transition count exceeds recomputed source transition count; K=10: affinity-weighted XY disagreement is not reduced; K=10: highest-affinity XY disagreement is not reduced; K=10: lateral transition count exceeds recomputed source transition count |

## Execution status

Smoke: `NOT_READY_HOLD`

Full pretraining: `NOT_EXECUTED`

Embedding extraction: `NOT_EXECUTED`

Downstream screening: `NOT_EXECUTED`

This review records target-only technical diagnostics and makes no downstream performance conclusion.

# Canonical HMM v2 multi-source protocol

The canonical protocol is `k468_cons010_multi_source_evaluation_v1`:

- ordered heads K=[4, 6, 8]
- consistency weight 0.1
- distillation weight 0.2
- MAE, LocalBT and random source families
- F3, Parihaka and Volve evaluations
- small, medium and large data sizes over five layouts

[`selection.yaml`](selection.yaml) binds this decision to the tracked matrix and
aggregate definitions and to the completed 135-cell aggregate summary digest.
Use the selected experiment as the default reference for future HMM v2
multi-source work. Existing K6, K468 consistency-zero and K6810 results remain
immutable historical comparisons.

The selection is a protocol decision. It does not rewrite individual survey
metrics or assert that this protocol is the best condition in every row.

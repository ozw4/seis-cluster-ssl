# Canonical HMM v2 multi-source protocol

The canonical protocol is `k6810_cons010_multi_source_evaluation_v1`:

- ordered heads K=[6, 8, 10]
- consistency weight 0.1
- distillation weight 0.2
- MAE, LocalBT and random source families
- F3, Parihaka and Volve evaluations
- small, medium and large data sizes over five layouts

For future references, `v2_dis0` denotes experiment results obtained with only
the distillation weight changed from 0.2 to 0.0 relative to this canonical `v2`
protocol. All other settings and evaluation conditions remain those of the
matching v2 experiment, including K=[6, 8, 10] and consistency weight 0.1.
Existing zero-distillation results are documented in the
[MAE sensitivity study](../distillation_sensitivity_v1/README.md) and the
[Random ablation study](../random_distillation_ablation_v1/README.md).

`v2_dis0_con0` denotes the same v2 protocol with both distillation and
consistency weights set to 0.0. Other conditions remain fixed; see the
[zero-distillation/zero-consistency study](../v2_dis0_con0_v1/README.md).

[`selection.yaml`](selection.yaml) binds this decision to the tracked matrix and
aggregate definitions and to the completed 135-cell aggregate summary digest.
Use the selected experiment as the default reference for future HMM v2
multi-source work. Existing K6, K468 and K6810 consistency-zero results remain
immutable historical comparisons.

The selection is a protocol decision. It does not rewrite individual survey
metrics or assert that this protocol is the best condition in every row.

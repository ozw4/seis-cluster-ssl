# F3 single-head K6 without distillation

These twelve YAMLs define the MAE, Barlow Twins and Random zero-distillation
arms of the [shared v1 extension](../../../hmm_v1/zero_distillation_ablation_v1/README.md).
Training inherits each family's original v1 0.2 recipe; embeddings and decoder
supports use the established v2 geometry. The comparison uses original v1
validation metrics. See the shared matrix, runbook and validation record for
execution, complete-grid aggregation, and recovery.

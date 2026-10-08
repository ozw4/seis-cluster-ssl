# HMM v3 trace-mask pretraining

This study measures the effect of hiding complete XY token columns while
predicting their fixed ordered HMM labels. Its matched control is the
[canonical HMM v2 selection](../experiments/hmm_v2/canonical_multi_source_v1/selection.yaml):
K `[6, 8, 10]`, consistency weight 0.1 and distillation weight 0.2.

The fixed matrix covers F3, Parihaka and Volve, each with MAE100, asymmetric
LocalBT3 and random initialization. Each v3 arm starts independently from the
same parent checkpoint as its v2 control. The control's trained checkpoint is
used for comparison, while its original pseudo-target manifest is reused
without relabelling or refreshing.

For each crop, the student replaces approximately 10% of eligible XY token
columns with a learned mask token over the entire crop's Z axis. A column is
a patch-grid column, not a single raw seismic trace. Masked and visible
categorical losses are normalized separately and combined with equal weights.
The frozen teacher sees the same amplitude crop without this replacement;
feature distillation uses visible tokens only. Cross-head consistency retains
v2's normalized-order Smooth L1 objective, beta 0.1 and weight 0.1, evaluated
on the common supervised-valid support, including masked positions.

Parent initialization, target identity, preprocessing, valid masks, optimizer,
25-epoch budget and downstream supervision are inherited from the matched
v2 arm. Inference and embedding extraction use the ordinary unmasked encoder.
The older F3 center-trace experiment retains its separate zero-consistency
contract; it is not the control for v3.

Downstream evaluation pairs five layouts and three label-budget sizes per
arm, giving 135 candidate cells. F3 uses validation Mean IoU and Macro F1;
Parihaka uses test channel IoU and F1; Volve uses test horizon error and
within-two-samples accuracy. Results retain each survey's metric and direction.
Labels and downstream scores do not choose masks, targets or checkpoints.
The completed epoch-25 checkpoint is used for every candidate.

See the [study runbook](../experiments/hmm_v3/k6810_trace_mask_multi_source_v1/README.md)
for versioned definitions, execution, recovery and paired publication.

## F3-only mask-token LR follow-up

[Experiment 133](../experiments/f3/facies_benchmark_v2/133_hmm_v3_mask_token_lr_sweep_v1/README.md)
changes only the learned replacement-token optimizer group from `1e-5` to
`3e-5` or `1e-4`, for all three F3 initialization families. Encoder/head LR
remain `1e-5`. All six 25-epoch runs retain canonical v02 consistency `0.1`,
original parents, immutable targets and the completed v3 masking/loss contract.
Completed masked v3 is the paired control, not the unmasked v02 model.
The five-layout, three-size decoder matrix contributes 90 new evaluation cells.
This follow-up does not resume stopped Parihaka or Volve runs.

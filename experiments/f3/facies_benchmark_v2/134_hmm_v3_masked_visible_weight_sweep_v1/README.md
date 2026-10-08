# F3 masked/visible prototype-weight follow-up

MAE100, asymmetric LocalBT3 and random each receive two independent 25-epoch
runs: masked:visible weights `25:75` and `75:25`. Mask-token LR is `3e-5`;
encoder/head LR remain `1e-5`. K `[6,8,10]`, consistency `0.1`, distillation
`0.2`, seed, targets, preprocessing and 10% full-depth XY-column masking
are inherited from experiment 133's completed `3e-5`, `50:50` arms.

Each prototype branch is normalized on its own eligible valid tokens before
combination: `L_proto = w_masked * L_masked + w_visible * L_visible`.
The weights sum to one. They do not change the masking fraction, usage loss,
visible-only distillation, consistency support or the outer prototype weight.
Every run initializes from its original MAE100/LocalBT3/random parent, not
from the trained control. The scientific identity, checkpoint and resume
contract bind both weights to six distinct versioned model tags.

Five paired layouts times small/medium/large produce 15 decoder evaluations
per arm, 90 in total. Report validation Mean IoU and Macro F1 separately by
size against completed `50:50` at the same mask-token LR. Existing v1/v2
results remain contextual comparisons, not the isolated weighting control.
One pretraining seed does not estimate training-seed uncertainty. Reusing the
evaluation split is a controlled follow-up, not independent held-out tuning.

Experiment 133's unfinished `1e-4` arms and other stopped surveys stay stopped.
See [RUNBOOK.md](RUNBOOK.md) and the
[scientific note](../../../../docs/hmm_v3_trace_mask_pretraining.md).

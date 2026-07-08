# Stratigraphic HMM Pretraining Milestone 1

Milestone 1 defines the artifact contract needed to train a single ordered
prototype head from stratigraphic HMM pseudo-targets. The reusable input is one
token-grid pseudo-target artifact per survey and `k` value, containing HMM
labels, confidence weights, a valid-token mask, and deterministic JSON metadata.

The training scope for this milestone is single-head ordered prototype training
from these pseudo-targets. Later work may add top-block unfreeze and teacher
distillation after the pseudo-target I/O contract is stable.

Out of scope for milestone 1:

- Multi-resolution heads.
- Lateral smoothing.
- MAE/view-consistency continuation.

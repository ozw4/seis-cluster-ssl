# Parihaka HMM boundary-weight screening

This phase holds the selected HMM transition condition fixed and varies only
the prototype-loss weighting near HMM boundaries. The same comparison is
applied to MAE and trace-drop-free local-BT sources.

The zero-boundary-weight condition, non-HMM controls, and their downstream
artifacts are reused in place. New pseudo-targets are exported from the
selected HMM labels; no new HMM clustering is performed.

Experiment definition: [Channel config](40_channel_boundary_weight.yaml).
Study-wide context: [Parihaka benchmark overview](../README.md).

Export pseudo-targets only for the new candidates, then follow the overview's
HMM screening workflow with the
[validation summarizer](scripts/summarize_validation.py). Reused controls are
not execution targets. A boundary-range extension, if justified, is a
separately defined phase.

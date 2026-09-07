# F3 center-trace masked HMM-path reconstruction

This treatment tests whether an encoder benefits from reconstructing ordered
HMM pseudo-targets when complete lateral token columns are hidden from the
student. It keeps the established hard multi-head targets immutable and changes
the pretext objective rather than relabelling the data.

Only the student receives the masked representation; the teacher sees the same
amplitude crop without that mask. Inference and embedding extraction use the
ordinary unmasked encoder. This distinction prevents the training corruption
from becoming a downstream preprocessing requirement.

The comparison is against the matched hard-target baseline. Neighbour
correction, lateral smoothing, soft posteriors, target refresh, raw-amplitude
reconstruction, and downstream smoothing are separate questions. Facies labels
and downstream metrics do not select the mask or targets.

Exact mask sampling, loss normalization, checkpoint identity, trainable state,
and execution order belong to the implementation and YAML under
[`104_strat_hmm_multi_head_k6810_center_trace_masked_v1`](../experiments/f3/facies_benchmark_v1/104_strat_hmm_multi_head_k6810_center_trace_masked_v1/README.md).

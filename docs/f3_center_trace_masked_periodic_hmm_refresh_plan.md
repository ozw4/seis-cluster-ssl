# F3 center-trace masking with periodic HMM refresh

This treatment asks whether periodically rebuilding ordered hard pseudo-targets
from the current student improves the fixed-target center-trace objective. The
scientific variable is target refresh; the mask objective, downstream task, and
initial target lineage remain the matched predecessor.

Each refresh uses an unmasked full-survey student representation and preserves
the initial preprocessing and ordered-state interpretation. A newly generated
target becomes active only after complete validation, so a failed or partial
refresh cannot alter the running training state. Model and optimizer training
continue across refreshes rather than restarting.

The fixed-target predecessor is the primary control. Additional changes to K,
masking, loss weights, preprocessing, path priors, soft targets, lateral
smoothing, or consensus correction are outside this comparison. Diagnostic
target movement is evidence to inspect, not by itself a success criterion.

The schedule, centre update, atomic publication, resume identity, selected
checkpoint, and execution phases belong to the implementation and YAML under
[`107_strat_hmm_multi_head_k6810_center_trace_masked_periodic_refresh_v1`](../experiments/f3/facies_benchmark_v1/107_strat_hmm_multi_head_k6810_center_trace_masked_periodic_refresh_v1/README.md).

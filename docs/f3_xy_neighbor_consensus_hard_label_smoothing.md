# F3 XY-neighbour consensus correction

This experiment asks whether a conservative, single-pass lateral consensus can
remove isolated errors from frozen ordered HMM hard labels. It publishes a new
target identity rather than modifying the source target or the earlier
edge-aware smoothing treatment.

All proposals are computed from the frozen source labels and applied
synchronously. An ordered-trace guard prevents a local correction from reversing
the source stratigraphic order. Embeddings, amplitudes, posteriors, facies
labels, and downstream metrics are not correction inputs.

The matched source target is the primary control. Iterative smoothing,
re-decoding, affinity weighting, target refresh, and alternative neighbourhoods
are separate experiments; transition-count changes are diagnostics rather than
selection gates.

The exact voting rule, edge handling, artifact identity, and execution stages
belong to the implementation and YAML under
[`100_strat_hmm_multi_head_k6810_xy_neighbor_consensus_v1`](../experiments/f3/facies_benchmark_v1/100_strat_hmm_multi_head_k6810_xy_neighbor_consensus_v1/README.md).

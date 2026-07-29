# M5-LS — Edge-aware lateral smoothing with ordered hard reprojection

M5-LS follows source result `M5_U_ORIGINAL_STOP`.  It does not continue
soft-posterior categorical-CE training.  Its primary baseline is the selected
hard `mh_nocons` run, `strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1`:
K=6/8/10, consistency weight 0, prototype weight 1.0, usage weight 0.005, and
distillation weight 0.2.

The fixed semantic identifier is
`ordered_hmm_edge_aware_lateral_mean_field_hard_v1`.

## Fixed algorithm

For each head and center trace `(x, y, :)`, use only same-z XY four-neighbours,
in deterministic `(x-1, y)`, `(x+1, y)`, `(x, y-1)`, `(x, y+1)` order.  There
are no z, diagonal, or 26-neighbour edges; invalid endpoints remove an edge.
Neither hard-label agreement nor lithology/facies labels gate an edge.

Exact ordered-path posterior `q_j(k)` is used only to initialize this one
lateral message.  For valid endpoint embeddings,
`d_ij = clip(1 - cosine(e_i, e_j), 0, 2)` and
`a_ij = exp(-d_ij / affinity_scale)`.  Available neighbours are normalized:
`m_i(k) = sum_j a_ij q_j(k) / sum_j a_ij`.  No neighbour gives an all-zero
message and leaves emissions exactly unchanged.

With `beta = pairwise_strength_ratio`, update once:
`c_lat_i(k) = c_i(k) - beta * emission_gap_scale_K * m_i(k)`.  `beta` is
non-negative; `beta=0` is exact source-cost parity.  Then reproject to hard
labels with the existing ordered HMM `viterbi_decode_costs`, retaining its
transition, initial/terminal, expected-boundary, compacted-valid trace, and
tie-breaking semantics.  Invalid z positions are compacted in ascending order
and do not split a trace.

The exporter policy, not this core, fixes
`affinity_scale = max(median(valid undirected XY edge distance), 1e-6)` and
`emission_gap_scale_K = max(median(second_smallest_cost - smallest_cost over
valid tokens), 1e-6)`.  The core accepts both as explicit positive finite
arguments and uses float64 outputs (labels are int32).

## Deliberate exclusions

M5-LS does not change or require a boundary transition-probability auxiliary
head.  Do not mix in soft CE, posterior temperature, hard/soft interpolation,
cross-head consistency, z smoothing, EM refresh, extra K values, downstream
prediction smoothing, artifact export, providers, training dispatch,
checkpoints, F3 configs, or a scientific run.  The final supervision remains
the ordered-Viterbi hard label.

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

## Fixed scientific exclusions

M5-LS does not change or require a boundary transition-probability auxiliary
head. Do not mix in soft categorical cross-entropy training,
posterior-temperature sweeps, hard/soft interpolation, cross-head consistency,
z-direction or diagonal smoothing, iterative mean-field or convergence loops,
EM, HMM, centre, or target refresh, additional K values or best-K selection,
downstream prediction smoothing, or lithology/facies labels in target
generation or hyperparameter selection. These scientific exclusions remain
fixed throughout M5-LS. The final supervision remains the ordered-Viterbi hard
label.

## Staged implementation pipeline

M5-LS is implemented and evaluated in this order:

1. One-trace lateral-message and ordered-reprojection core.
2. The immutable lateral hard-target artifact/export.
3. Adapter into the existing hard-target provider/dataset/collate path.
4. The existing hard multi-head loss dispatch.
5. Representation-specific scientific/checkpoint/resume identity.
6. Separately versioned F3 config, smoke, full training, and embedding validation.
7. Paired original-split low-label evaluation and preregistered gate.

Stages 2-7 add no scientific loss terms or lateral operations. The smoothing
computation finishes offline in stage 2; the later stages are required
implementation plumbing, not a second scientific treatment.

## Artifact and training integration contract

The lateral target manifest is separate from the historical hard-target
manifest and the M5-U posterior manifest. Per-survey lateral arrays contain
hard labels (`int32`), unity confidence (`float32`), and the exact common valid
mask (`bool`); invalid labels are `-1` and invalid confidence is `0`.
Per-survey metadata remains compatible with the existing schema-v1 hard
pseudo-target reference/provider contract while retaining lateral provenance.

The training representation is `lateral_mean_field_hard_labels_v1`. It routes
to the existing hard multi-head dataset/collate/loss path; posterior arrays are
not placed in training batches. Training architecture, initialization,
trainable scope, optimizer groups, crop/AGC/zero-mask/data order/seed, and the
baseline loss weights remain paired with hard `mh_nocons`.

M5-LS checkpoints have a distinct representation/provenance identity and may
not resume from hard or soft checkpoints, while the student model state remains
embedding-extraction compatible.

# XY-neighbour consensus hard-label smoothing v1

`xy_neighbor_consensus_hard_label_smoothing_v1` is a successor to M5-LS, not
a modification of it. It creates a new immutable hard-target publication from
the frozen K=6/8/10 source hard-label manifest. Its training representation is
`xy_neighbor_consensus_hard_labels_v1`.

## Fixed algorithm

For every valid center token `(x, y, z)`, inspect only the same-`z` XY
four-neighbours in deterministic `(x-1, y)`, `(x+1, y)`, `(x, y-1)`,
`(x, y+1)` order. Invalid neighbours are excluded. All decisions are made
from the frozen source labels, then applied synchronously once:

- Four valid neighbours require one unique label to occur at least three times.
- Three valid neighbours require all three labels to agree.
- Two or fewer valid neighbours, ties, and a proposal equal to the center all
  leave the source label unchanged.

A non-identical proposal is eligible only at an internal valid token of its
trace: both a preceding and following valid token in that trace must exist.
Those valid tokens need not be physically adjacent in `z`. The proposal must
satisfy `previous_label <= proposal <= next_label`, where both bounds come from
the source trace. Endpoints and one-sided-valid positions remain unchanged.
This is a safety guard preserving the source trace's nondecreasing ordering;
it is not Viterbi decoding or an iterative smoothing operation.

## Immutable artifact and training contract

Each exported head writes `int32` labels, `float32` unity confidence, and the
unchanged `bool` valid-token mask. Invalid label values are copied from the
source unchanged. The manifest and per-survey metadata bind the source hard
manifest, source labels, semantic ID, and the complete fixed consensus policy.
Full validation recomputes labels and diagnostics from the source hard labels;
it detects altered arrays, invalid-mask changes, invalid-value changes, and any
ordered-trace violation.

The existing hard multi-head dataset, collate, and categorical loss are reused.
The representation has its own manifest/head hashes, source-hard-manifest hash,
fixed-policy identity, checkpoint schema, and resume compatibility boundary.
Hard baseline, M5-U posterior, and M5-LS lateral checkpoints cannot resume this
run or be resumed by it.

The final review rechecks the handoff's schema-v5 checkpoint and extraction
metadata identity against the selected target, and verifies the referenced
checkpoint and extraction-file digests. It does not inspect target or embedding
array values while doing so.

## Explicit exclusions

This method never reads embeddings, posterior tensors, affinities, emissions,
or HMM models. It does not update emissions, run Viterbi re-decoding, use a
target-smoothing beta parameter, perform beta calibration, use z or diagonal
neighbours, make more than one synchronous pass, refresh targets, or use
facies/lithology labels or downstream metrics to generate or choose targets.

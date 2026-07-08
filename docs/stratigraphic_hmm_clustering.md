# Stratigraphic HMM clustering

This page documents ordered stratigraphic unit clustering from frozen seismic
SSL embeddings. The goal is to discover vertically ordered units in an embedding
volume, not to assign supervised lithology or facies classes.

## Problem Definition

Input embeddings are fixed features from a pretrained encoder by default. The
clustering stage groups tokens into ordered units while encouraging each
vertical trace to progress monotonically through cluster IDs. F3 lithology labels
are not training inputs for this method; they can be used only after clustering
for sanity-check evaluation.

Lithology or facies clustering tries to recover rock or facies categories.
Stratigraphic HMM clustering instead treats cluster IDs as ordered units that may
track depositional or structural layering. A unit can contain mixed lithology,
and a lithology can recur in multiple stratigraphic units.

## Algorithm Summary

1. Run KMeans on sampled frozen embeddings.
2. Order centers by mean z.
3. Decode valid labels per vertical trace with Viterbi under the transition
   costs.
4. Update cluster centers from decoded assignments.
5. Repeat decode and update for the configured number of iterations.

Set `clustering.stratigraphic_hmm.emission_source` to `embedding` for the main
method, or to `z_coordinate` for the z-only guardrail. The z-only guardrail uses
normalized token z coordinates as one-dimensional emissions while still using
embedding artifacts for token grid shape and validity masks.

## Output Contract

The clustering entrypoint writes outputs under `clustering.output_dir`, grouped
by `k` value. Outputs follow the same contract as embedding clustering:

- fitted clustering model artifacts
- per-survey token label files
- resolved config and metadata
- method-specific metadata for stratigraphic HMM settings and iteration
  summaries

JSON metadata is written as strict JSON-safe text. Non-finite numerical values
such as forbidden reverse-transition infinities are represented as `null` in
metadata JSON, while `hmm_model.joblib` preserves the numerical transition-cost
array used for decoding.

Saved label grids are decoded from the saved final centers in
`cluster_centers.npy`, matching the centers stored in `hmm_model.joblib`.

Invalid tokens are omitted from the observed HMM sequence and remain `-1` in the
output label grid. They do not reset the vertical trace state sequence. With
`forbid_reverse: true`, monotonicity is enforced over consecutive valid
observations in each trace, so `labels[labels >= 0]` is non-decreasing in z
order for every decoded trace.

## Edge Margin Tokens

`clustering.stratigraphic_hmm.edge_margin_tokens` excludes otherwise valid
tokens near the volume edges from HMM training and decoding. The value is a
three-item token margin in x, y, and z order. For example, `[8, 8, 0]` removes
the outer eight token columns in x and y while keeping the full z range.

This is part of the clustering run itself. It is not post-hoc visualization
masking: excluded edge tokens are not sampled for center fitting, are not
decoded by Viterbi, and remain invalid in the output label grid.

## Path Prior

`clustering.stratigraphic_hmm.path_prior` adds soft sequence-level costs to the
Viterbi decode. The initial-state prior can softly favor a shallow starting
state, and the terminal-state prior can softly favor a deep ending state. These
anchors bias the decoded trace path without replacing the embedding emissions
or transition costs.

When enabled, `expected_boundaries` adds a soft prior on the number of state
transitions along a trace. `target: auto_k_minus_1` uses one fewer boundary than
the current number of states, while an integer target pins the preferred count
explicitly.

These priors are for stratigraphic unit discovery. They are not lithology or
facies classification targets, and they should not be interpreted as supervised
class evidence.

## Interpretation Caveats

The HMM can create plausible bands even when embeddings are weak. Run z-only and
random guardrails before making geological claims.

A path prior can create boundaries even when the embeddings provide weak
support. Always compare path-prior results against the z-only guardrail and the
matched no-path-prior run.

The z-only guardrail should produce ordered bands by construction. The embedding
HMM result is only scientifically stronger if it differs from z-only in
geologically meaningful ways, for example boundaries bending with reflectors or
respecting structural offsets rather than remaining flat depth bands.

Strict monotonicity can suppress repeated facies because this is a stratigraphic
unit method, not a repeated-facies classifier.

Useful diagnostics include reverse transition rate over consecutive valid trace
observations, boundary continuity, boundary z summaries, visual salt-and-pepper
reduction, and whether boundaries follow structure instead of collapsing into
flat depth bands. Inspect both XZ sections and XY slices: XZ sections reveal
vertical monotonicity and depth collapse, while XY slices reveal lateral striping
and edge artifacts.

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

## Interpretation Caveats

The HMM can create plausible bands even when embeddings are weak. Run z-only and
random guardrails before making geological claims.

The z-only guardrail should produce ordered bands by construction. The embedding
HMM result is only scientifically stronger if it differs from z-only in
geologically meaningful ways, for example boundaries bending with reflectors or
respecting structural offsets rather than remaining flat depth bands.

Strict monotonicity can suppress repeated facies because this is a stratigraphic
unit method, not a repeated-facies classifier.

Useful diagnostics include reverse transition rate over consecutive valid trace
observations, boundary continuity, boundary z summaries, visual salt-and-pepper
reduction, and whether boundaries follow structure instead of collapsing into
flat depth bands.

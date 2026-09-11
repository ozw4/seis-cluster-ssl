# Stratigraphic HMM clustering

Stratigraphic HMM clustering discovers vertically ordered units from frozen
seismic embeddings. It is not supervised lithology or facies classification:
labels may be used later for evaluation, but are not clustering inputs.

The method initializes units from embedding clusters, orders them vertically,
decodes each valid trace under a monotone transition model, and updates the
centres from those assignments. Invalid observations are skipped rather than
treated as sequence boundaries.

## Interpretation

An ordered prior can produce visually plausible bands even when the embeddings
carry little geological information. The current F3 definitions provide a
z-coordinate-only guardrail, which produces ordered bands by construction, and
a run without the optional path prior.

A Random-encoder guardrail is also needed before attributing a difference to
the learned embeddings, but no such F3 clustering YAML is currently defined.
Until that control is added, treat these runs as method and shape diagnostics,
not as matched evidence that the encoder carries geological information.

Evidence is stronger only when embedding-driven boundaries differ meaningfully
from these controls—for example, by following reflectors or structural offsets
instead of collapsing into flat depth bands. Inspect both vertical sections and
horizontal slices: they expose different failure modes.

Useful diagnostics include reverse transitions between consecutive valid
observations, boundary continuity and depth distribution, lateral striping, and
edge effects. Strict monotonicity may suppress recurring facies, so cluster IDs
must be interpreted as ordered units rather than facies classes.

Concrete settings, cache controls, output names, and execution commands belong
to the owning experiment YAML and
[`60_stratigraphic_clustering/README.md`](../experiments/f3/facies_benchmark_v1/60_stratigraphic_clustering/README.md).

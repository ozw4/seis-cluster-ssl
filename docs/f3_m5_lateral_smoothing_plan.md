# F3 edge-aware lateral smoothing

This treatment asks whether lateral continuity can improve ordered HMM
pseudo-targets without changing the downstream task or introducing supervised
facies information. It is a hard-target successor to the soft-posterior study,
not a continuation of posterior cross-entropy training.

The scientific comparison keeps the selected hard multi-head baseline and adds
one offline lateral message followed by ordered hard reprojection. This isolates
lateral evidence from changes to the student loss. Facies labels and downstream
metrics are excluded from target generation and candidate selection.

Interpretation must remain paired with the unsmoothed hard target. A visually
smoother result is not sufficient evidence: the treatment may erase real
boundaries or merely strengthen the ordering prior. Posterior training,
iterative smoothing, target refresh, and downstream prediction smoothing are
separate experiments.

The exact neighbourhood, affinity, reprojection, artifact identity, and
training settings belong to the implementation and YAML under
[`99_strat_hmm_multi_head_k6810_lateral_smoothing_v1`](../experiments/f3/facies_benchmark_v1/99_strat_hmm_multi_head_k6810_lateral_smoothing_v1/README.md).

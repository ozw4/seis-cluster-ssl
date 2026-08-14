# Strat-HMM pretraining milestone 1

Milestone 1 uses stratigraphic HMM clustering as a structured pretext task for
F3 encoder adaptation. HMM labels are pseudo-targets for representation
learning; they are not final lithology labels or evaluated task output.

The retained configs live in
`experiments/f3/facies_benchmark_v1/80_strat_hmm_pretraining_m1/` and cover:

1. bootstrap pseudo-target export;
2. isolated CPU smoke pretraining;
3. full single-head K=6 pretraining;
4. student embedding extraction; and
5. smoke-only pseudo-target refresh validation.

Run each command from the experiment
[README](../experiments/f3/facies_benchmark_v1/80_strat_hmm_pretraining_m1/README.md)
and keep complete outputs under `artifacts/seis_ssl_cluster/`.

The former token-level probe, voxel-count label-budget, split robustness,
result aggregation, and tracked report publication are retired. Reproduce
those historical results from the producer revision recorded in
`reports/f3/legacy/README.md`.

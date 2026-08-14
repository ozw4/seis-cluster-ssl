# F3 Strat-HMM M2-A boundary-weighted pretraining

M2-A changes only the pseudo-target boundary weight from the M1 pretraining
condition. The fixed candidate uses `alpha=0.5`, `tau=2.0`, `k=6`,
top-block unfreezing of one block, distillation weight `0.2`, and seed
`42`. No parameter sweep is active.

The retained workflow is:

1. export the alpha-zero parity target;
2. export the fixed boundary-weighted target;
3. run isolated smoke and full pretraining; and
4. extract the student embeddings.

Use the configs and command order in the experiment
[README](../experiments/f3/facies_benchmark_v1/84_strat_hmm_pretraining_m2a_boundary/README.md).
Complete checkpoints, pseudo-targets, and embeddings remain under
`artifacts/seis_ssl_cluster/`.

The former token probe, voxel-count label-budget, split robustness, result
summary, and tracked report publication are retired.

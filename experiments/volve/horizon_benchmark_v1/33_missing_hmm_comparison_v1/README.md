# Complete the Volve pretraining comparison

This experiment fills the four missing arms of the requested nine-arm comparison:
`rot90_asym_g060_3ep_hmm_k6_distill010`,
`rot90_asym_g060_3ep_hmm_k6_distill020`,
`random_init_hmm_k6_distill010`, and `random_init_hmm_k6_distill020`.
The MAE100→MAE25, MAE100→HMM25 at both weights, LocalBT3, and Random arms
already have completed outputs and are reused.

Each parent supplies its own embeddings, HMM pseudo targets, frozen teacher, and
student initialization. Random therefore means a self-target control: its targets
come from the canonical seed-42 random encoder. The LocalBT parent is the completed
asymmetric rot90 Gaussian-noise-0.6 three-epoch checkpoint from experiment 32.

Targets use the existing Volve ordered K=6 recipe (L2 normalization, local-token
position residualization, PCA64, ten HMM iterations, and edge margins `[8,8,0]`).
The existing parent embedding extraction has the same 128-cube windows, 64 overlap,
float16 storage, BF16 autocast, AGC65 preprocessing, and valid-token threshold 1.0
as the experiment-31 target extractor; it is reused directly.

HMM training preserves the Volve 25-epoch budget: 10,000 samples per epoch,
batch size 4, 62,500 optimizer updates, FP32, seed 42, top encoder block 1, and
learning rate 1e-5. Only the initialization/target source and distillation weight
(0.1 or 0.2) vary. Four data-loader workers limit shared-host resource pressure.

HMM resume preserves the requested budget, but does not guarantee the same
update sequence as uninterrupted training: recreating persistent data-loader
workers can change the next epoch's shuffle order. Record actual restarts and
do not interrupt healthy training merely to change GPUs.

From the repository root, set the environment described in the
[shared benchmark runbook](../README.md), then run:

```bash
bash experiments/volve/horizon_benchmark_v1/33_missing_hmm_comparison_v1/run_pretraining.sh
bash experiments/volve/horizon_benchmark_v1/33_missing_hmm_comparison_v1/run_downstream.sh
```

The pretraining driver runs dry-runs and separate two-step smoke outputs before
each full training run. The downstream driver audits the complete checkpoint,
parent SHA-256, live pseudo-target bindings, and declared clustering recipe before
training each frozen horizon decoder. Every arm runs all five layouts × three
data sizes with the existing 50-epoch decoder contract. Random baseline cells are
reused under `local_bt_recipe_arm_v1/runs`; each new model has its own directory.

If execution stops after the smoke run but before the first full checkpoint,
re-running the pretraining driver will refuse the existing smoke output instead
of overwriting it. Inspect that interrupted stage before resuming; do not delete
or overwrite its artifacts merely to bypass the safety check.

Each paired summary requires 30 cells: 15 new arm cells and the same 15 completed
Random cells. Summary JSON, Markdown, and per-cell CSV stay under
`artifacts/` in `horizon/volve/horizon_benchmark_v1/missing_hmm_comparison_v1/summary/`.
The drivers write no reports or bulk files to versioned directories. See the
[report sharing policy](../../../../docs/report_sharing_policy.md).

## Final completion evidence

The downstream driver's existing-metrics skip and the summary's `--check-only`
are not, by themselves, proof of a fully completed decoder. Before declaring the
comparison complete, additionally verify every cell's `latest.pt` has
`completed: true`, epoch 50, next position 0, and the full optimizer-step budget
for its freshly resolved training tiles. Check the complete ordered history,
validation-selected best checkpoint, held-out test results, and current source
and benchmark identities. Metrics can be written before the final completed
checkpoint, so retain these checks even when a metrics file already exists.

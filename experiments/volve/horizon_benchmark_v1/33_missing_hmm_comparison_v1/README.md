# Complete the Volve pretraining comparison

This experiment fills the four missing arms of the nine-arm Volve comparison:
`rot90_asym_g060_3ep_hmm_k6_distill010`,
`rot90_asym_g060_3ep_hmm_k6_distill020`,
`random_init_hmm_k6_distill010`, and `random_init_hmm_k6_distill020`.
The MAE100→MAE25, MAE100→HMM25 (both weights), LocalBT3, and Random arms have
completed outputs and are reused.

Each parent supplies its own embeddings, HMM pseudo targets, frozen teacher, and
student initialization. Random is a self-target control: its targets come from the
canonical seed-42 random encoder. The LocalBT parent is the completed asymmetric
rot90 Gaussian-noise-0.6 three-epoch checkpoint from experiment 32.

## Target recipe and HMM budget

- Targets: existing Volve ordered K=6 recipe (L2 normalization, local-token position
  residualization, PCA64, ten HMM iterations, edge margins `[8,8,0]`).
- Parent embedding extraction (reused directly; same as the experiment-31 target
  extractor): 128-cube windows, 64 overlap, float16 storage, BF16 autocast, AGC65
  preprocessing, valid-token threshold 1.0.
- HMM training: 25 epochs, 10,000 samples per epoch, batch size 4, 62,500 optimizer
  updates, FP32, seed 42, top encoder block 1, learning rate 1e-5, four data-loader
  workers. Only the initialization/target source and distillation weight (0.1 or
  0.2) vary across arms.

HMM resume preserves the requested budget, but does not guarantee the same
update sequence as uninterrupted training: recreating persistent data-loader
workers can change the next epoch's shuffle order. Record actual restarts and
do not interrupt healthy training merely to change GPUs.

## Execution

From the repository root, set the environment described in the
[shared benchmark runbook](../README.md), then run:

```bash
bash experiments/volve/horizon_benchmark_v1/33_missing_hmm_comparison_v1/run_pretraining.sh
bash experiments/volve/horizon_benchmark_v1/33_missing_hmm_comparison_v1/run_downstream.sh
```

- The pretraining driver runs a dry-run (`--dry-run`) and a separate two-step smoke
  run before each full run. The downstream driver audits the complete checkpoint,
  parent SHA-256, live pseudo-target bindings, and declared clustering recipe before
  training each frozen horizon decoder.
- Every arm runs all five layouts × three data sizes with the existing 50-epoch
  decoder contract. Random baseline cells are reused under
  `local_bt_recipe_arm_v1/runs`; each new model has its own directory.
- If a LocalBT execution stops after the smoke run but before the first full
  checkpoint, re-running the pretraining driver refuses the existing smoke output.
  Inspect that interrupted stage before resuming; do not delete or overwrite its
  artifacts to bypass the check.
- Each paired summary requires 30 cells: 15 new arm cells and the same 15 completed
  Random cells. Summary JSON, Markdown, and per-cell CSV stay under `artifacts/` in
  `horizon/volve/horizon_benchmark_v1/missing_hmm_comparison_v1/summary/`. The
  drivers write no reports or bulk files to versioned directories. See the
  [report sharing policy](../../../../docs/report_sharing_policy.md).

## Coordinated random HMM execution

- The shared HMM CLI coordinates only this experiment's canonical random
  `distill010` and `distill020` recipes; LocalBT and unrelated jobs are not
  coordinated and keep their existing behavior. For each canonical random recipe,
  the arm's `full_25ep` and `smoke_2step` commands share one POSIX lock under
  `.random_hmm_locks/`; different arms do not share a lock. Dry-runs create neither
  locks nor outputs.
- After acquiring the lock the CLI rechecks the recipe, input identities, and
  checkpoint state, even when `--resume` was supplied: a strictly completed
  checkpoint is reused without training; a valid partial checkpoint resumes;
  foreign, inconsistent, or aliased evidence fails closed.
- Resume and completed-run reuse require the arm's immutable
  `<arm>.<full_25ep|smoke_2step>.inputs.json` receipt and unchanged inputs; a
  missing receipt is never backfilled.
- A lock is not a GPU reservation: the operator must assign an available device and
  limit concurrent jobs. Do not stop a healthy producer to retrofit a lock. Before
  any auxiliary invocation (same CLI and canonical YAML, after a supported
  `--dry-run`, input validation, and an explicit GPU capacity check), verify that no
  random-arm producer started without this coordination is still running; do not
  rely on the main driver's current phase or an estimated time gap to prevent two
  writers.

## Final completion evidence

The downstream driver's existing-metrics skip and the summary's `--check-only` are
not, by themselves, proof of a fully completed decoder. Before declaring the
comparison complete, verify for every cell:

- `latest.pt` has `completed: true`, epoch 50, next position 0, and the full
  optimizer-step budget for its resolved training tiles.
- Complete ordered history, validation-selected best checkpoint, held-out test
  results, and current source and benchmark identities.
- Metrics can be written before the final completed checkpoint, so these checks
  apply even when a metrics file already exists.

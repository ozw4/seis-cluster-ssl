# Parihaka nine-arm pretraining comparison completion

Nine arms evaluated on the standard frozen Channel decoder benchmark (Parihaka
facies_benchmark_v1): MAE100 + MAE25, MAE100 + HMM25 at distillation weights 0.1
and 0.2, asymmetric LocalBT3, LocalBT3 + HMM25 at both weights, Random, and
Random + HMM25 at both weights.

Fixed conditions for the four new HMM arms (LocalBT-HMM0.1/0.2, Random-HMM0.1/0.2):

| Condition | Value |
| --- | --- |
| Epochs / samples per epoch / batch | 25 / 10,000 / 16 |
| Precision / seed | FP32 / 42 |
| Trainable encoder | one top encoder block, LR 1e-5 |
| CPU data-loader workers | 4 |
| Pseudo-targets | ordered K=6 targets, generated separately from the completed asymmetric LocalBT3 embedding and the canonical seed-42 Random embedding; both loss weights of a source share its targets |
| Teacher / student init | the same source checkpoint |
| Clustering inputs | existing overlap-64 FP32 embeddings reused |
| Budget per new HMM arm | 25 epochs, 15,625 optimizer steps |

## Execution

From the repository root, set `SEIS_SSL_CLUSTER_ARTIFACT_ROOT` to the shared
artifact root and select the approved GPU with `CUDA_VISIBLE_DEVICES`, then run:

```bash
bash experiments/parihaka/facies_benchmark_v1/40_pretraining_comparison_completion_v1/run_all.sh
```

- Exclusive survey lock; one GPU command at a time.
- Order per arm: dry-run, isolated one-step HMM smoke run, full training, embedding extraction, then all five layouts x three supervision sizes (15 cells per arm).
- On restart it resumes partial HMM/decoder checkpoints and audits the final HMM epoch, optimizer-step count, and scientific configuration (budget: 25 epochs / 15,625 optimizer steps per new HMM arm).

HMM resume preserves this budget, but does not guarantee the same update
sequence as uninterrupted training: recreating persistent data-loader workers
can change the next epoch's shuffle order. Record actual restarts and do not
interrupt healthy training merely to change GPUs.

Outputs remain under `artifacts/` using the namespace
`pretraining_comparison_completion_v1`. Stage logs are under
`channel_benchmark/pretraining_comparison_completion_v1/logs` within the artifact
root. Exact model, checkpoint, and result paths for all nine arms are listed in
[`50_summary/01_all_nine_arms.yaml`](50_summary/01_all_nine_arms.yaml).

## Existing MAE-HMM distillation 0.1

The existing MAE-HMM0.1 checkpoint completed 25 epochs and its embedding exists.
Its five medium-layout decoder runs completed training and validation selection
but were saved in `validation_only` mode. The driver evaluates the held-out test
set from each of those validation-selected best decoders with no optimizer steps.
It verifies the complete 50-epoch history, step budget, best epoch, and benchmark
identity, then copies the original checkpoints unchanged into the new result
root. The new metrics include checksummed evaluation provenance; original
validation-only artifacts remain intact. Small and large decoders are trained
fresh in the new result root with the same fixed 50-epoch benchmark contract.

## Completion audit and summaries

```bash
python -m seis_ssl_cluster.parihaka.pretraining_comparison \
  --summary-config experiments/parihaka/facies_benchmark_v1/40_pretraining_comparison_completion_v1/50_summary/01_all_nine_arms.yaml \
  --check-only
```

The driver requires all nine checkpoints and 135 test-result cells, checks exact
checkpoint hashes and cross-arm benchmark identity, then writes the exact
three-file set `comparison.csv`, `summary.json`, and `summary.md` into
`channel_benchmark/pretraining_comparison_completion_v1/summary` under the artifact
root. Existing baseline runs are read from their original artifact roots. The
comparison also audits each decoder's complete 50-epoch step history, numeric
CSV equality, checkpoint/result identity, first-maximum validation selection,
and finite held-out test metrics. Test-only exports must retain live checksummed
source provenance and byte-identical decoder checkpoints. No pipeline stage
consumes files from `reports/`.

## Isolated GPU 1 staging

`run_gpu1_staging.sh` runs the ten missing MAE-HMM0.1 small/large cells on GPU 1
with at most three decoder processes; it differs from the canonical configuration
only in its input/output run roots. All 50 training epochs, validation
selection, and held-out test evaluation are retained. Completed cells pass
`inspect_completed_channel_job` before being reported as successful; incomplete
cells resume their own `latest.pt`. Outputs stay under
`channel_benchmark/pretraining_comparison_completion_v1/gpu1_staging` within the
artifact root and are not automatically published into canonical results.
Adoption requires the complete-cell audit and a separate check that the main
driver is not writing the corresponding canonical cell. Medium cells are not part
of this staging driver.

## Optional shared GPU 0 HMM lane

```bash
bash experiments/parihaka/facies_benchmark_v1/40_pretraining_comparison_completion_v1/run_hmm_lane.sh --dry-run
bash experiments/parihaka/facies_benchmark_v1/40_pretraining_comparison_completion_v1/run_hmm_lane.sh
```

- Start only after the first LocalBT-HMM0.1 source has finished; check live GPU/process ownership first (this driver is not a GPU resource allocator).
- Limits: three simultaneous HMM trainings on GPU 0, three staging decoders on GPU 1.
- Uses GPU 0 and four CPU threads; runs LocalBT-HMM0.2, Random-HMM0.1, Random-HMM0.2 sequentially with the unchanged 25-epoch recipe; leaves embeddings, downstream cells, and the final comparison to `run_all.sh`.
- Each source retains its isolated one-step smoke run. Main and auxiliary HMM CLI processes cooperate only for these three canonical recipes: a strictly completed output is reused read-only, a valid owned partial resumes its latest checkpoint, and foreign or corrupt evidence or noncanonical overrides fail closed. The final full run must prove 25 epochs and 15,625 steps.
- Lock files under `pretraining/parihaka/facies_benchmark_v1/pretraining_comparison_completion_v1/.remaining_hmm_locks` (artifact root) must never be unlinked or replaced.
- `--dry-run` creates no locks, logs, or training outputs.

## Optional cooperative GPU 1 Channel lane

```bash
bash experiments/parihaka/facies_benchmark_v1/40_pretraining_comparison_completion_v1/run_channel_lane.sh \
  --model local_barlow_twins_rot90_asym_g060_3ep_hmm_k6_distill010 --dry-run
bash experiments/parihaka/facies_benchmark_v1/40_pretraining_comparison_completion_v1/run_channel_lane.sh \
  --model local_barlow_twins_rot90_asym_g060_3ep_hmm_k6_distill010
```

- Preconditions: all ten MAE-HMM0.1 staging cells finished, audited, and byte-identically published; every old main decoder allowed to finish naturally; the chosen arm's main embedding producer has exited successfully.
- Only the four new LocalBT/Random HMM model IDs are accepted; each invocation handles all fifteen cells of that one arm, in reverse layout order and large/medium/small order, with exactly three worker slots and four CPU threads per child; it cannot change seeds, budgets, precision, layouts, or output roots. The dry run validates configuration and CLI coordination without opening locks, loading GPU inputs, or writing artifacts. Repeat for another arm only after the main driver publishes its embeddings; the main queue remains responsible for all four arms and the summary.
- The lane rejects live staging workers, live embedding producers for the chosen arm, and any live new-arm decoder that does not hold its cell lock; ambiguous process records fail closed. Limit of three owned GPU-1 decoders.
- Complete cells are audited and skipped read-only; owned partials resume their latest checkpoint; foreign inputs, mismatched hashes, ambiguous partial outputs, or scientific overrides fail closed. An old partial cell without an input-hash receipt cannot be adopted. Lock files under `.channel_cell_locks` (comparison run root) must never be unlinked.
- On a child failure, queued work is suppressed, running children finish, and no trainer is killed.
- Older main scripts use metrics presence to skip cells, so a summary can hit the brief final-checkpoint publication window and fail its strict audit. After all writers finish, run the `--check-only` summary command above; generate the summary only if it passes and the summary output does not already exist. Never overwrite a failed audit or a partial result.

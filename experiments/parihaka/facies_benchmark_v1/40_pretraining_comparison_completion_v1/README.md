# Parihaka nine-arm pretraining comparison completion

This experiment completes the requested comparison before the standard frozen
Channel decoder benchmark: MAE100 + MAE25, MAE100 + HMM25 at distillation weights
0.1 and 0.2, asymmetric LocalBT3, LocalBT3 + HMM25 at both weights, Random, and
Random + HMM25 at both weights.

The four new HMM arms retain the established 25 epochs, 10,000 samples per epoch,
batch size 16, FP32 precision, seed 42, one trainable top encoder block, and
learning rates of 1e-5. CPU data loading uses four workers. Ordered K=6 targets
are generated separately from the completed asymmetric LocalBT3 embedding and
the canonical seed-42 Random embedding. Each source's two loss weights share its
targets; the teacher and student initialization are that same source checkpoint.
The existing overlap-64, FP32 embeddings are reused as clustering inputs.

## Execution

From the repository root, set `SEIS_SSL_CLUSTER_ARTIFACT_ROOT` to the shared
artifact root and select the approved GPU with `CUDA_VISIBLE_DEVICES`, then run:

```bash
bash experiments/parihaka/facies_benchmark_v1/40_pretraining_comparison_completion_v1/run_all.sh
```

The driver takes an exclusive survey lock and executes one GPU command at a time.
It dry-runs each stage, performs an isolated one-step HMM smoke run, completes
full training, extracts embeddings, and runs all five layouts with all three
supervision sizes. On restart it resumes partial HMM/decoder checkpoints and
audits the final HMM epoch, optimizer-step count, and scientific configuration.
The full budget is 25 epochs and 15,625 optimizer steps per new HMM arm.

HMM resume preserves this budget, but does not guarantee the same update
sequence as uninterrupted training: recreating persistent data-loader workers
can change the next epoch's shuffle order. Record actual restarts and do not
interrupt healthy training merely to change GPUs.

Complete outputs remain under `artifacts/` using the namespace
`pretraining_comparison_completion_v1`. Stage logs are under
`channel_benchmark/pretraining_comparison_completion_v1/logs` within the artifact
root. Exact model, checkpoint, and result paths for all nine arms are listed in
[`50_summary/01_all_nine_arms.yaml`](50_summary/01_all_nine_arms.yaml).

## Existing MAE-HMM distillation 0.1

The existing MAE-HMM0.1 checkpoint completed 25 epochs and its embedding exists.
Its five medium-layout decoder runs completed training and validation selection,
but were saved in `validation_only` mode. The driver evaluates the held-out test
set from each existing validation-selected best decoder with no optimizer steps.
It verifies the complete 50-epoch history, step budget, best epoch, and benchmark
identity, then copies the original checkpoints unchanged into the new result
root. The new metrics include checksummed evaluation provenance. Original
validation-only artifacts remain intact. Small and large decoders are trained
in the new result root with the same fixed 50-epoch benchmark contract.

## Completion audit and summaries

```bash
python -m seis_ssl_cluster.parihaka.pretraining_comparison \
  --summary-config experiments/parihaka/facies_benchmark_v1/40_pretraining_comparison_completion_v1/50_summary/01_all_nine_arms.yaml \
  --check-only
```

The driver ends by requiring all nine checkpoints and 135 test-result cells,
checking exact checkpoint hashes and cross-arm benchmark identity, and writing
the exact three-file set `comparison.csv`, `summary.json`, and `summary.md` into
`channel_benchmark/pretraining_comparison_completion_v1/summary` under the artifact
root. Existing baseline runs are read from their original artifact roots.
The comparison also audits each decoder's complete 50-epoch step history,
numeric CSV equality, checkpoint/result identity, first-maximum validation
selection, and finite held-out test metrics. Test-only exports must retain
live checksummed source provenance and byte-identical decoder checkpoints.
No pipeline stage consumes files from `reports/`.

## Isolated GPU 1 staging

When GPU 1 is available, `run_gpu1_staging.sh` runs the ten missing MAE-HMM0.1
small/large cells with at most three decoder processes. The staging configuration
differs from the canonical configuration only in its input/output run roots.
All 50 training epochs, validation selection, and held-out test evaluation are
retained. Completed cells pass `inspect_completed_channel_job` before being
reported as successful; incomplete cells resume their own `latest.pt`.

These runs remain below
`channel_benchmark/pretraining_comparison_completion_v1/gpu1_staging` within the
artifact root. They are not automatically published into canonical results.
Adoption requires the complete-cell audit and a separate check that the main
driver is not writing the corresponding canonical cell. Medium cells are not
part of this staging driver.

## Optional shared GPU 0 HMM lane

After the first LocalBT-HMM0.1 source has finished and its GPU 0 HMM slot is
available, the three remaining HMM sources can run ahead of the main driver's
decoder queue. Check live GPU/process ownership and capacity before starting;
this driver is not a GPU resource allocator. Keep the existing GPU 0 limit of
three simultaneous HMM trainings and the GPU 1 staging limit of three decoders.

```bash
bash experiments/parihaka/facies_benchmark_v1/40_pretraining_comparison_completion_v1/run_hmm_lane.sh --dry-run
bash experiments/parihaka/facies_benchmark_v1/40_pretraining_comparison_completion_v1/run_hmm_lane.sh
```

The auxiliary driver uses GPU 0 and four CPU threads, runs LocalBT-HMM0.2,
Random-HMM0.1, and Random-HMM0.2 sequentially, and leaves embedding generation,
all downstream cells, and the final comparison to `run_all.sh`. Each source
retains its isolated one-step smoke run and unchanged 25-epoch full recipe.
No live main script, running trainer, or scientific YAML needs to be modified
or restarted. New main and auxiliary HMM CLI processes cooperate automatically
only for these three canonical recipes and their exact smoke outputs.

Before entering the training runner or initializing CUDA, they acquire the same
Parihaka HMM lane lock and then the recipe lock. They re-read the exact canonical
configuration and checkpoint after waiting. A strictly completed output is
reused without training or modifying its files; a valid owned partial resumes
its latest checkpoint. Foreign or corrupt evidence and noncanonical overrides
fail closed. Full and smoke share the lane, so a main-process smoke cannot add
an extra HMM alongside an auxiliary full run. Smoke completion is verified as
one actual optimizer step, never inferred from a full-run checkpoint.

Completion/resume checks include the full resolved configuration and its saved
hash, model/scientific identity, exact current parent and pseudo-target file
sets and hashes, checkpoint stage/kind/boundary counters, and optimizer/RNG
state. The final full run must prove 25 epochs and 15,625 steps. Healthy training
must still not be interrupted to exploit resume: the shuffle caveat above
continues to apply.

Shared locks live outside training outputs under
`pretraining/parihaka/facies_benchmark_v1/pretraining_comparison_completion_v1/.remaining_hmm_locks`
within the artifact root. Do not unlink or replace lock files. The auxiliary
driver also has its own exclusive lock; lock paths reject symlinks, hardlinks,
and non-regular files without truncating their contents. Each invocation has a
fresh `logs/hmm_lane_*` directory, preserving earlier logs on re-entry.
The supported `--dry-run` does not create locks, logs, or training outputs.

## Optional cooperative GPU 1 Channel lane

Once all ten MAE-HMM0.1 staging cells have naturally finished and have been
audited and byte-identically published, GPU 1 can run three canonical decoders
for one ready new HMM arm. First let every main decoder started before the
cooperative CLI was installed finish naturally. Do not stop, migrate, or adopt
an incomplete old decoder. Also wait for the chosen arm's main embedding
producer to exit successfully; metadata presence alone is not sufficient.

```bash
bash experiments/parihaka/facies_benchmark_v1/40_pretraining_comparison_completion_v1/run_channel_lane.sh \
  --model local_barlow_twins_rot90_asym_g060_3ep_hmm_k6_distill010 --dry-run
bash experiments/parihaka/facies_benchmark_v1/40_pretraining_comparison_completion_v1/run_channel_lane.sh \
  --model local_barlow_twins_rot90_asym_g060_3ep_hmm_k6_distill010
```

Only the four new LocalBT/Random HMM model IDs are accepted. Each invocation
handles all fifteen cells of that one arm, in reverse layout order and
large/medium/small order, with exactly three worker slots and four CPU threads
per child. It cannot change seeds, budgets, precision, layouts, or output roots.
The dry run lists the exact commands and validates configuration and CLI
coordination without opening locks, loading GPU inputs, or writing artifacts.
Repeat for another arm only after the main driver publishes its embeddings.
The existing main queue remains responsible for all four arms and the summary.

Before execution, the auxiliary takes a nonblocking singleton lock and the
existing staging driver's lock. It requires strict completion of all ten old
staging cells, expected cell identities, and byte-identical canonical copies.
It rejects live staging workers, live embedding producers for the chosen arm,
and any live new-arm decoder that does not hold its exact shared cell lock.
An inaccessible or ambiguous process record fails closed. A failed gate
releases its locks, allowing legitimate old staging recovery. Both driver locks
are inherited by children, so a surviving child keeps the GPU-1 lane reserved
even if its parent exits unexpectedly. Lock files must never be unlinked.
This is not a general GPU allocator: continue checking other users' GPU capacity
and retain the existing limit of three owned GPU-1 decoders.

New main and auxiliary decoder CLI processes cooperate only for the sixty
canonical new-arm cells. They take the same cell lock before CUDA setup, then
re-read the exact recipe, audit the full HMM source, all embedding arrays and
metadata, and construct a fresh benchmark identity. Complete cells are audited
and skipped read-only; owned partial cells resume their own latest checkpoint.
Foreign inputs, mismatched hashes, ambiguous partial outputs, or scientific
overrides fail closed. Initial input hashes are atomically recorded outside the
canonical four-file cell, alongside its lock in `.channel_cell_locks` under
the comparison run root. These receipts pin checkpoint, embedding, label, and
configuration bytes across future resumes. An old partial without a receipt
cannot be adopted. An old complete cell may be checked and reused, but historical
embedding-body hashes cannot be retroactively proved if the old format did not
record them. No receipt is added to an old complete output.

Each invocation preserves logs in a new `logs/channel_lane_*` directory. On a
child failure, queued work is suppressed and already-running children are
allowed to finish; no trainer is killed. The main driver still owns embeddings
and the final nine-arm summary. Since older main scripts use metrics presence
to skip cells, a summary can encounter the brief final-checkpoint publication
window and fail its strict audit. After **all** writers finish, run the full
`--check-only` summary command above; generate the summary only if that passes
and the summary output does not already exist. Never overwrite a failed audit
or a partial result to make the queue continue.

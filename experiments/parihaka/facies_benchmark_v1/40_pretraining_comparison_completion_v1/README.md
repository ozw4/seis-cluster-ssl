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

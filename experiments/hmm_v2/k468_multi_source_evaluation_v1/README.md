# K=[4,6,8] multi-source evaluation

This is the K468 counterpart of `k6810_multi_source_evaluation_v1`:
F3, Parihaka and Volve × MAE100, asymmetric LocalBT3 and random initialization.
The head selection is fixed before execution. F3 uses validation metrics;
Parihaka and Volve use held-out test metrics without tuning or selection.

F3 MAE reuses the completed K468 arm from F3 experiment 127, including its
15 downstream cells. `f3_mae_k468_reuse_receipt.json` binds the checkpoint,
resolved training configuration, embeddings, mask, metadata, source summary
and every metrics file. The receipt was constructed from the completed source
summary and verified against live artifacts. Matching K6 controls use the
unchanged `../k6810_multi_source_evaluation_v1/hmm_v1_controls_receipt.json`.

The remaining eight arms each run 25 epochs and 15 downstream cells
(5 layouts × 3 data sizes): 120 new cells, 135 total. New heads K4 and K8 are
generated from the same source embeddings and clustering recipes as K6810;
historical K6 targets are frozen and combined in ordered `[4,6,8]` manifests.
New frozen-reference manifests resolve compatibility links before recording
source and target paths, matching receipt construction. Existing receipts and
input files are not rewritten.
Each source retains its parent checkpoint, preprocessing, masks, batch size,
workers and downstream recipe. The common recipe retains seed 42, 10,000
samples/epoch, top encoder block training, projection 128, temperature 0.1,
prototype loss 1.0, usage 0.005, distillation 0.2, consistency 0.0, AMP off,
and head/encoder learning rates 1e-5. Smoke and full training start independently.
Embedding extraction retains batch size 1 and prefetch queue depth 2.

## Start

Activate the repository Python environment. Set these absolute paths for the
current machine; machine-specific values belong only in ignored operation records:

- `SEIS_SSL_CLUSTER_ARTIFACT_ROOT`: the existing artifact root containing shared inputs.
- `F3_ROOT`: the directory containing `f3_labels.sgy`, using the same absolute
  path recorded in the frozen F3 voxel supervision metadata. A separate hard-link
  path to the same file does not satisfy the evaluation path identity contract.
- `SEIS_SSL_CLUSTER_VOLVE_ROOT`: the Volve dataset root.
- `CUDA_VISIBLE_DEVICES`: the available GPU to use for sequential execution.

From the repository root:

```bash
# Print the entire command plan, including the final aggregate command.
bash experiments/hmm_v2/k468_multi_source_evaluation_v1/run_all.sh --plan

# Verify reused F3 files and dry-run the eight target-stage configurations.
bash experiments/hmm_v2/k468_multi_source_evaluation_v1/run_all.sh --dry-run

# Execute F3, then Parihaka, then Volve, then the cross-survey aggregate.
bash experiments/hmm_v2/k468_multi_source_evaluation_v1/run_all.sh --execute
```

The study wrapper's dry-run covers existing target inputs. Training, embeddings,
downstream and summary dry-runs require the preceding stage outputs, so they
must be requested later through the survey drivers. The wrapper sets the
workspace and Python import path. It stops on the first failing command.

Survey definitions and stage-level restart instructions:

- [F3](../../f3/facies_benchmark_v2/129_hmm_v2_k468_multi_source_v1/RUNBOOK.md)
- [Parihaka](../../parihaka/facies_benchmark_v1/42_hmm_v2_k468_multi_source_v1/RUNBOOK.md)
- [Volve](../../volve/horizon_benchmark_v1/35_hmm_v2_k468_multi_source_v1/RUNBOOK.md)

## Outputs and restart

New survey outputs use
`$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/surveys/<survey>/<storage-series>/<kind>/hmm_v2_k468_multi_source_v1/`.
F3 training/targets retain the v1 storage series; F3 evaluation embeddings and
benchmark outputs use v2. Shared historical inputs keep their original paths.
Logs and locks use `operations/hmm_v2/k468_multi_source_evaluation_v1/<survey>/logs/`.

The final summary is
`$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/studies/hmm_v2/k468_multi_source_evaluation_v1/summary/`.
Both survey and aggregate producers publish exactly `comparison.csv`,
`summary.json` and `summary.md`; they do not write tracked reports. Columns use
`k468_mean` and `k468_secondary_mean`. There are 27 comparison rows; metrics
remain separate by survey. Positive improvement means K468 minus K6 for
classification and K6 minus K468 for Volve error.

Do not restart the whole wrapper after partial execution: target outputs are
never overwritten. Inspect the per-stage logs and use the owning survey driver
with `--from`/`--to`; resume incomplete training or decoder cells only with their
own `--resume` command. Completed models and cells are audited before reuse.
Existing summary directories are immutable. To aggregate after stage-level recovery:

```bash
export SEIS_SSL_CLUSTER_WORKSPACE="$PWD"
python -m seis_ssl_cluster.hmm.multi_source_aggregate \
  --config experiments/hmm_v2/k468_multi_source_evaluation_v1/aggregate.yaml --dry-run
python -m seis_ssl_cluster.hmm.multi_source_aggregate \
  --config experiments/hmm_v2/k468_multi_source_evaluation_v1/aggregate.yaml
```

See [VALIDATION.md](VALIDATION.md) for preparation evidence. Preparation does not
run clustering, target export, training, embedding extraction or downstream evaluation.

# k-means versus HMM with distillation and consistency disabled

Requested 2026-11-17. This study compares fixed unordered MiniBatchKMeans
pseudo-labels with the completed HMM [v2_dis0_con0](../../hmm_v2/v2_dis0_con0_v1/README.md)
experiment. Both methods use distillation weight **0**, consistency weight **0**,
K=[6,8,10], prototype weight 1 and usage weight 0.005.

## Fixed scientific contract

[matrix.yaml](matrix.yaml) binds all F3, Parihaka and Volve arms, each with MAE,
asymmetric LocalBT3 and Random initialization. Nine new encoder continuations
start from their original parent checkpoints for 25 epochs, with seed 42 and
only the original top encoder block trainable. They never start from a trained
k-means or HMM continuation. All original source-specific preprocessing, batch
sizes, crops, samples, learning rates, precision and teacher references remain
fixed. Diagnostic crops preserve RNG state; teacher construction preserves
initial student/head parameters, which the GPU smokes compare against HMM.
The teacher's weighted loss contribution is zero.

Reuse the existing k-means target manifests after validating their hashes,
original encoder checkpoint hashes, source features and valid-token support.
The latter must match the corresponding HMM targets exactly. Original k-means
fitting preprocessing and clustering settings are unchanged. Labels remain
unordered and fixed, with unit confidence on valid support. No ordered
cross-head loss is evaluated. The comparison measures pseudo-label generation
under matched zero auxiliary weights, not the isolated effect of HMM decoding.

Downstream covers **135 cells**: 3 surveys × 3 sources × 3 budgets × 5 layouts.
Decoder seed is 42000 on both sides; original 50-epoch budgets and validation
checkpoint selection are preserved. F3 primary comparison uses the shared
113,979,624-voxel common-test mask, separately from decoder validation metrics.
Parihaka uses test Channel IoU/F1; Volve uses test common-support macro MAE
[samples]/within-two-samples accuracy. Nested budgets and five layouts are not
independent encoder-seed repeats. The stopped Parihaka decoder-seed-42001
experiment is not resumed or substituted for the HMM seed-42000 reference.

## Execution and recovery

Set `SEIS_SSL_CLUSTER_WORKSPACE`, `SEIS_SSL_CLUSTER_ARTIFACT_ROOT`, `F3_ROOT`,
`SEIS_SSL_CLUSTER_VOLVE_ROOT`, and `PYTHONPATH=src` to existing inputs. Keep
machine paths only in ignored operational receipts. Use the repository Python
environment. To inspect without data, use `--plan` instead of `--dry-run`.

```bash
python proc/seis_ssl_cluster/run_kmeans_study.py \
  --matrix experiments/kmeans/kmeans_k6810_dis0_con0_v1/matrix.yaml \
  --dry-run --from targets --to full
bash experiments/kmeans/kmeans_k6810_dis0_con0_v1/run_all.sh
```

The wrapper selects GPU 1 by default; set `KMEANS_ZERO_GPU` before launch to
choose another available device. Execution is serial, with a study lock. Each
GPU stage waits for two idle utilization readings five seconds apart and
38 GiB free for encoder/extraction work or 12 GiB for decoders. Other workloads
are never stopped. Nine independent full-batch 1-step smokes must all pass
zero-weight and initialization checks before any full continuation is admitted.

Completed matching smokes, full encoder runs and decoder cells are audited
before reuse. Partial training requires explicit `--resume`, narrowed to one
survey/candidate/stage or one decoder cell, as in the original k-means CLI.
Only that job's `latest.pt` is accepted; encoder initialization checkpoints and
validation-best checkpoints are not resume state. Existing embeddings and final
publications are not overwritten; resume after completed extraction with
`--from downstream`. Any failure stops subsequent admissions and records the
failed command. No automatic retry or unrelated-process monitor is launched.

## Outputs and completion

New outputs follow `surveys/<survey>/<storage-series>/<kind>/kmeans_k6810_dis0_con0_v1/`
under the artifact root. Original targets stay at their producing study paths.
Operations are in `operations/kmeans/kmeans_k6810_dis0_con0_v1/`, with stage
logs, `status.json`, smoke receipts, detached launch and exit records.

The final summary in `studies/kmeans/kmeans_k6810_dis0_con0_v1/summary/` owns
exactly **`cells.csv`, `comparison.csv`, `summary.json`, `summary.md`**.
It requires all 135 audited pairs and all nine complete zero-weight histories.
The 27 survey/source/budget groups report five-layout means, sample SD and
paired differences. Raw differences are k-means minus HMM; positive
`primary_improvement` means k-means is better, reversing the sign for Volve MAE.
Both improvements and declines are retained.

The review publisher owns exactly **`README.md`, `cells.csv`, `comparison.csv`,
`provenance.json`** in `reports/kmeans_k6810_dis0_con0_v1/`. Reports never become
pipeline inputs. Checkpoints, arrays, prediction volumes and machine paths
remain in artifacts. After full completion, rebuild the full experiment
CSV/HTML/ZIP catalog, verify every previous evaluation and metric is retained,
then replace the current bundle while keeping a backup in operations.
A running or partially complete experiment is not a completed result.

# F3 M3-V voxel lithology benchmark

M2-A is complete with a **GO** decision. M3-V does not change its pretext task
or fine-tune an encoder: it is a downstream resolution benchmark over the
already frozen MAE, M1, and M2-A `overlap_x16` representations. The existing
`linear_balanced_v1` token benchmark remains the representation benchmark and
must not be replaced or reinterpreted by M3-V.

This runbook is intentionally artifact-driven. Commands that need F3 data,
embeddings, probes, checkpoints, or generated voxel artifacts are not suitable
for repository CI. If an input is absent, stop at the failing dry-run; do not
create metric, report, summary, or publish files that imply the job ran.

## Benchmark contract

The fixed source identities are:

| role | model tag |
|---|---|
| MAE reference | `amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1` |
| M1 baseline | `strat_hmm_pretext_m1_k6_topblock1_distill` |
| M2-A candidate | `strat_hmm_pretext_m2a_boundary_a050_t2_k6_topblock1_distill` |

All three models use `overlap_x16`, the existing `linear_balanced_v1` probe,
and one common original-split voxel-supervision artifact.

- **V0, `token_projection_nearest_v1`:** apply the fixed token probe and repeat
  each token prediction over its patch. There is no learned sub-token decoder.
- **V1, `frozen_embedding_decoder_nearest_voxel_ln_v1`:** train the same
  lightweight 3-D decoder independently on each model's precomputed frozen
  embeddings. It uses nearest-neighbor upsampling and voxelwise LayerNorm. The
  decoder can learn sub-token structure, but gradients never update the encoder.

Trilinear upsampling plus GroupNorm is not the M3-V benchmark implementation.
Voxelwise LayerNorm was selected because it does not depend on tile-level
spatial statistics and therefore preserves whole-grid/tiled consistency. The
experiment directory name `88_f3_voxel_decoder_v1` remains as the milestone
number; decoder checkpoint and artifact identity is the exact spec above.

Thus V1 minus V0 measures learned downstream decoding value for a fixed
representation, while M2-A V1 minus M1 V1 is the primary representation
comparison at voxel resolution. M1 V1 minus MAE V1 and M2-A V1 minus MAE V1
are secondary comparisons.

### Supervision, validation, and pairing

The common artifact root is:

```text
/workspace/artifacts/seis_ssl_cluster/lithology/f3/facies_benchmark_v1/
voxel_supervision/png_slices_segy_labels_v1/
```

It contains `supervision_split_grid.npy`, `voxel_dataset_metadata.json`,
`class_counts.csv`, `split_manifest.json`, and `voxel_dataset_summary.md`.
The PNG slice inventory defines train and validation slices; there is no random
voxel split. When train and validation slices intersect, validation precedence
applies and the voxel is validation exactly once. Invalid canonical tokens and
the configured z-border are unsupervised.

The metadata records SHA-256 identities for the label volume, inventory,
reference embedding metadata, and canonical valid-token mask. Each model's
embedding valid mask must match the recorded canonical valid-token hash. V0 and
V1 predictions also record their source identities and the supervision
split-grid identity. Evaluation first verifies those identities, then requires
every voxel marked validation in the split grid to be present in
`f3_valid_voxel_mask.npy`. A missing validation prediction is an error, not a
sample to drop.

For the six-split suite, `split_000` through `split_005` come unchanged from
the existing M1 inventory. Each split has one shared voxel dataset, split-grid
identity, class weights, and paired tile order for M1 and M2-A. The suite
manifests bind those identities before deltas are computed.

### Evaluation units and monitored metrics

Aggregate metrics count each unique supervised validation voxel once. The
evaluation also writes per-slice and per-trace diagnostic CSVs. The six-split
robustness summary uses the split as its statistical unit. Voxels within a
trace, slice, or split are spatially dependent, so M3-V makes no voxel-level
independence claim and computes no voxel-count p-value or confidence interval.

In addition to macro F1, mean IoU, balanced accuracy, and per-class metrics,
monitor classes 3 (Rijnland/Chalk) and 5 (Zechstein). Boundary evidence includes
boundary-region macro F1/mean IoU, vertical boundary F1 at registered
tolerances, boundary-position error, and class 3/5 boundary recall. Original
split configs evaluate radii/tolerances 1, 2, 4, and 8; cross-split comparison
uses the preregistered 2 and 4 values.

## Exact command order

Run from the repository root. Each dry-run precedes the corresponding write.
Later dry-runs may require complete outputs from earlier stages.

### 1. Build common voxel supervision

The first command to run in an environment with the external artifacts is:

```bash
python proc/seis_ssl_cluster/build_f3_lithology_voxel_dataset.py \
  --config experiments/f3/facies_benchmark_v1/87_f3_voxel_benchmark_v0/01_build_voxel_supervision.yaml \
  --dry-run
python proc/seis_ssl_cluster/build_f3_lithology_voxel_dataset.py \
  --config experiments/f3/facies_benchmark_v1/87_f3_voxel_benchmark_v0/01_build_voxel_supervision.yaml
```

### 2. Generate or verify token predictions

Run MAE, M1, then M2-A:

```bash
python proc/seis_ssl_cluster/predict_f3_lithology_tokens.py --config experiments/f3/facies_benchmark_v1/87_f3_voxel_benchmark_v0/02_predict_mae_tokens.yaml --dry-run
python proc/seis_ssl_cluster/predict_f3_lithology_tokens.py --config experiments/f3/facies_benchmark_v1/87_f3_voxel_benchmark_v0/02_predict_mae_tokens.yaml
python proc/seis_ssl_cluster/predict_f3_lithology_tokens.py --config experiments/f3/facies_benchmark_v1/87_f3_voxel_benchmark_v0/06_predict_m1_tokens.yaml --dry-run
python proc/seis_ssl_cluster/predict_f3_lithology_tokens.py --config experiments/f3/facies_benchmark_v1/87_f3_voxel_benchmark_v0/06_predict_m1_tokens.yaml
python proc/seis_ssl_cluster/predict_f3_lithology_tokens.py --config experiments/f3/facies_benchmark_v1/87_f3_voxel_benchmark_v0/10_predict_m2a_tokens.yaml --dry-run
python proc/seis_ssl_cluster/predict_f3_lithology_tokens.py --config experiments/f3/facies_benchmark_v1/87_f3_voxel_benchmark_v0/10_predict_m2a_tokens.yaml
```

If a complete prediction already exists, replace only its real-run command with
the same command plus `--skip-existing`. That option validates every required
array and the config, probe, scaler, model, checkpoint, embedding, and
valid-token identities. It never blesses a partial artifact.

### 3. V0 projection, evaluation, and report

Run each config in numeric order, first with `--dry-run` and then without it:

```bash
python proc/seis_ssl_cluster/project_f3_lithology_tokens_to_voxels.py --config experiments/f3/facies_benchmark_v1/87_f3_voxel_benchmark_v0/03_project_mae_nearest.yaml
python proc/seis_ssl_cluster/evaluate_f3_lithology_voxels.py --config experiments/f3/facies_benchmark_v1/87_f3_voxel_benchmark_v0/04_evaluate_mae_nearest.yaml
python proc/seis_ssl_cluster/build_f3_lithology_voxel_report.py --config experiments/f3/facies_benchmark_v1/87_f3_voxel_benchmark_v0/05_report_mae_nearest.yaml
python proc/seis_ssl_cluster/project_f3_lithology_tokens_to_voxels.py --config experiments/f3/facies_benchmark_v1/87_f3_voxel_benchmark_v0/07_project_m1_nearest.yaml
python proc/seis_ssl_cluster/evaluate_f3_lithology_voxels.py --config experiments/f3/facies_benchmark_v1/87_f3_voxel_benchmark_v0/08_evaluate_m1_nearest.yaml
python proc/seis_ssl_cluster/build_f3_lithology_voxel_report.py --config experiments/f3/facies_benchmark_v1/87_f3_voxel_benchmark_v0/09_report_m1_nearest.yaml
python proc/seis_ssl_cluster/project_f3_lithology_tokens_to_voxels.py --config experiments/f3/facies_benchmark_v1/87_f3_voxel_benchmark_v0/11_project_m2a_nearest.yaml
python proc/seis_ssl_cluster/evaluate_f3_lithology_voxels.py --config experiments/f3/facies_benchmark_v1/87_f3_voxel_benchmark_v0/12_evaluate_m2a_nearest.yaml
python proc/seis_ssl_cluster/build_f3_lithology_voxel_report.py --config experiments/f3/facies_benchmark_v1/87_f3_voxel_benchmark_v0/13_report_m2a_nearest.yaml
```

### 4. V1 smoke tests

The first V1 command is the required CPU two-step MAE dry-run:

```bash
python proc/seis_ssl_cluster/train_f3_lithology_voxel_decoder.py \
  --config experiments/f3/facies_benchmark_v1/88_f3_voxel_decoder_v1/01_train_mae_smoke.yaml \
  --dry-run --device cpu --max-steps 2
python proc/seis_ssl_cluster/train_f3_lithology_voxel_decoder.py --config experiments/f3/facies_benchmark_v1/88_f3_voxel_decoder_v1/01_train_mae_smoke.yaml --device cpu --max-steps 2
python proc/seis_ssl_cluster/train_f3_lithology_voxel_decoder.py --config experiments/f3/facies_benchmark_v1/88_f3_voxel_decoder_v1/06_train_m1_smoke.yaml --dry-run --device cpu --max-steps 2
python proc/seis_ssl_cluster/train_f3_lithology_voxel_decoder.py --config experiments/f3/facies_benchmark_v1/88_f3_voxel_decoder_v1/06_train_m1_smoke.yaml --device cpu --max-steps 2
python proc/seis_ssl_cluster/train_f3_lithology_voxel_decoder.py --config experiments/f3/facies_benchmark_v1/88_f3_voxel_decoder_v1/11_train_m2a_smoke.yaml --dry-run --device cpu --max-steps 2
python proc/seis_ssl_cluster/train_f3_lithology_voxel_decoder.py --config experiments/f3/facies_benchmark_v1/88_f3_voxel_decoder_v1/11_train_m2a_smoke.yaml --device cpu --max-steps 2
```

Smoke outputs use `frozen_embedding_decoder_nearest_voxel_ln_v1_smoke` and are
never scientific results.

### 5. V1 full training: MAE, M1, M2-A

```bash
python proc/seis_ssl_cluster/train_f3_lithology_voxel_decoder.py --config experiments/f3/facies_benchmark_v1/88_f3_voxel_decoder_v1/02_train_mae_full.yaml --dry-run
python proc/seis_ssl_cluster/train_f3_lithology_voxel_decoder.py --config experiments/f3/facies_benchmark_v1/88_f3_voxel_decoder_v1/02_train_mae_full.yaml --device auto
python proc/seis_ssl_cluster/train_f3_lithology_voxel_decoder.py --config experiments/f3/facies_benchmark_v1/88_f3_voxel_decoder_v1/07_train_m1_full.yaml --dry-run
python proc/seis_ssl_cluster/train_f3_lithology_voxel_decoder.py --config experiments/f3/facies_benchmark_v1/88_f3_voxel_decoder_v1/07_train_m1_full.yaml --device auto
python proc/seis_ssl_cluster/train_f3_lithology_voxel_decoder.py --config experiments/f3/facies_benchmark_v1/88_f3_voxel_decoder_v1/12_train_m2a_full.yaml --dry-run
python proc/seis_ssl_cluster/train_f3_lithology_voxel_decoder.py --config experiments/f3/facies_benchmark_v1/88_f3_voxel_decoder_v1/12_train_m2a_full.yaml --device auto
```

Full runs use identical hidden channels `[128, 64, 32]`, three `[2, 2, 2]`
upsampling stages, `[8, 8, 8]` token cores, `[1, 1, 1]` halos, balanced weights,
AdamW, 50 epochs, seed 42, and no augmentation or early stopping.

All new and resumed decoder jobs require checkpoint schema 5 or later. The old
`frozen_embedding_decoder_v1` checkpoint/artifact identity is incomplete and
is excluded from resume, inference, evaluation, and summaries. Do not manually
copy or rename old artifacts into the new path; regenerate the smoke stage and
then every downstream stage under the canonical spec.

### 6. V1 inference, evaluation, and report

Run each line first with `--dry-run`, then as shown:

```bash
python proc/seis_ssl_cluster/predict_f3_lithology_voxels.py --config experiments/f3/facies_benchmark_v1/88_f3_voxel_decoder_v1/03_predict_mae_voxels.yaml --device auto
python proc/seis_ssl_cluster/evaluate_f3_lithology_voxels.py --config experiments/f3/facies_benchmark_v1/88_f3_voxel_decoder_v1/04_evaluate_mae_voxels.yaml
python proc/seis_ssl_cluster/build_f3_lithology_voxel_report.py --config experiments/f3/facies_benchmark_v1/88_f3_voxel_decoder_v1/05_report_mae_voxels.yaml
python proc/seis_ssl_cluster/predict_f3_lithology_voxels.py --config experiments/f3/facies_benchmark_v1/88_f3_voxel_decoder_v1/08_predict_m1_voxels.yaml --device auto
python proc/seis_ssl_cluster/evaluate_f3_lithology_voxels.py --config experiments/f3/facies_benchmark_v1/88_f3_voxel_decoder_v1/09_evaluate_m1_voxels.yaml
python proc/seis_ssl_cluster/build_f3_lithology_voxel_report.py --config experiments/f3/facies_benchmark_v1/88_f3_voxel_decoder_v1/10_report_m1_voxels.yaml
python proc/seis_ssl_cluster/predict_f3_lithology_voxels.py --config experiments/f3/facies_benchmark_v1/88_f3_voxel_decoder_v1/13_predict_m2a_voxels.yaml --device auto
python proc/seis_ssl_cluster/evaluate_f3_lithology_voxels.py --config experiments/f3/facies_benchmark_v1/88_f3_voxel_decoder_v1/14_evaluate_m2a_voxels.yaml
python proc/seis_ssl_cluster/build_f3_lithology_voxel_report.py --config experiments/f3/facies_benchmark_v1/88_f3_voxel_decoder_v1/15_report_m2a_voxels.yaml
```

### 7. Original-split summary

```bash
python proc/seis_ssl_cluster/summarize_f3_lithology_voxel_results.py --config experiments/f3/facies_benchmark_v1/90_f3_voxel_results/01_summarize_original_split.yaml --dry-run
python proc/seis_ssl_cluster/summarize_f3_lithology_voxel_results.py --config experiments/f3/facies_benchmark_v1/90_f3_voxel_results/01_summarize_original_split.yaml
```

This writes the provisional original-split comparison to the artifact root.
Publication is deferred until the six-split evidence has also been summarized.

### 8. Six-split suite

```bash
EXP=experiments/f3/facies_benchmark_v1/89_f3_voxel_split_robustness
python proc/seis_ssl_cluster/build_f3_lithology_voxel_split_datasets.py --config "$EXP/01_build_voxel_split_datasets.yaml" --dry-run --only-missing
python proc/seis_ssl_cluster/build_f3_lithology_voxel_split_datasets.py --config "$EXP/01_build_voxel_split_datasets.yaml" --only-missing
python proc/seis_ssl_cluster/run_f3_lithology_voxel_v0_split_suite.py --config "$EXP/02_run_v0_split_projections.yaml" --dry-run --only-missing
python proc/seis_ssl_cluster/run_f3_lithology_voxel_v0_split_suite.py --config "$EXP/02_run_v0_split_projections.yaml" --only-missing
python proc/seis_ssl_cluster/run_f3_lithology_voxel_decoder_split_suite.py --config "$EXP/03_run_v1_split_decoders.yaml" --dry-run --only-missing --device auto
python proc/seis_ssl_cluster/run_f3_lithology_voxel_decoder_split_suite.py --config "$EXP/03_run_v1_split_decoders.yaml" --only-missing --device auto
```

`--only-missing` skips a job only after its complete terminal artifact and
identities validate. The V1 run manifest records `latest.pt` for interrupted
jobs; invoke the same suite command again with `--only-missing` to resume.

### 9. Final summary, publish, and checks

After all split jobs are complete, build the split summary and publish it
together with the already-written original-split summary. This is the only
publication step:

```bash
EXP=experiments/f3/facies_benchmark_v1/89_f3_voxel_split_robustness
python proc/seis_ssl_cluster/summarize_f3_lithology_voxel_split_robustness.py --config "$EXP/04_summarize_voxel_split_robustness.yaml" --dry-run
python proc/seis_ssl_cluster/summarize_f3_lithology_voxel_split_robustness.py --config "$EXP/04_summarize_voxel_split_robustness.yaml"
```

The final lightweight result tree contains both original-split products and the
six-split summary/tables. Validate that repository-managed publication:

```bash
python proc/seis_ssl_cluster/validate_results_artifacts.py --root results --max-file-size-mb 10
find results -type f \( -name '*.npy' -o -name '*.pt' -o -name '*.joblib' \) -print
rg -n '(^|[/:])runs/' experiments/f3/facies_benchmark_v1/{87_f3_voxel_benchmark_v0,88_f3_voxel_decoder_v1,89_f3_voxel_split_robustness,90_f3_voxel_results}
```

The validator checks the repository-managed lightweight result tree. The two
discovery commands must print nothing. Active
config regression tests additionally resolve every path and assert the MAE,
M1, and M2-A source tags and checkpoint semantics.

## Artifact paths and resuming

Let `$ROOT=/workspace/artifacts/seis_ssl_cluster` and
`$MODEL_ROOT=$ROOT/lithology/f3/facies_benchmark_v1/<MODEL_TAG>/overlap_x16/png_slices_segy_labels_v1`.
The paths below are recommended examples from the checked-in configs, not a
hierarchy enforced by a repository validator. Each config's explicit paths are
the source of truth.

```text
$MODEL_ROOT/predictions/linear_balanced_v1/                 token predictions
$MODEL_ROOT/voxel_predictions/token_projection_nearest_v1/ V0 predictions
$MODEL_ROOT/voxel_evaluations/token_projection_nearest_v1/ V0 metrics
$MODEL_ROOT/voxel_reports/token_projection_nearest_v1/     V0 report
$MODEL_ROOT/voxel_decoders/frozen_embedding_decoder_nearest_voxel_ln_v1/    V1 checkpoints
$MODEL_ROOT/voxel_predictions/frozen_embedding_decoder_nearest_voxel_ln_v1/ V1 predictions
$MODEL_ROOT/voxel_evaluations/frozen_embedding_decoder_nearest_voxel_ln_v1/ V1 metrics
$MODEL_ROOT/voxel_reports/frozen_embedding_decoder_nearest_voxel_ln_v1/     V1 report
$ROOT/lithology/f3/facies_benchmark_v1/reports/voxel_benchmark_v1/
$ROOT/lithology/f3/facies_benchmark_v1/voxel_robustness/m2a_vs_m1_v1/
  v1/frozen_embedding_decoder_nearest_voxel_ln_v1/split=<SPLIT>/model=<MODEL_TAG>/
results/f3/facies_benchmark_v1/voxel_lithology_benchmark_v1/
```

Prediction artifacts contain labels, confidence, a valid mask, metadata, and
optionally probabilities. Evaluation artifacts include overall, boundary,
per-slice, and per-trace metrics. Complete local arrays and checkpoints stay in
`artifacts/`; only selected review files enter `results/`.

Training writes `latest.pt` after each epoch for exact continuation and
`best.pt` for inference. Resume a standalone full run with:

```bash
python proc/seis_ssl_cluster/train_f3_lithology_voxel_decoder.py \
  --config <FULL_CONFIG> --device auto \
  --resume <MODEL_ROOT>/voxel_decoders/frozen_embedding_decoder_nearest_voxel_ln_v1/latest.pt
```

Never resume training from `best.pt`, and never infer from `latest.pt`. All
non-training stages refuse existing outputs by default. Keep complete stages;
after inspection, remove only the incomplete stage directory and restart at the
first missing stage. Do not replace shared supervision while downstream
metadata refers to its hash.

### Why full probability volumes are off by default

V0 and V1 configs set `write_probabilities: false`. Labels (`int16`), confidence
(`float16`), and the valid mask (`bool`) are sufficient for the registered
metrics and cost about 5 bytes per voxel. A six-class `float16` probability
volume adds `N_voxels * 6 * 2` bytes: about 1.12 GiB per 100 million voxels,
6.71 GiB for the six original V0/V1 artifacts, and 26.82 GiB for the 24 M1/M2-A
six-split V0/V1 artifacts, before filesystem overhead. Enable it only for a
preregistered analysis that requires calibrated class probabilities and has
the corresponding storage budget.

## Failure handling

- **Valid-token hash mismatch:** stop. Confirm the config points to the correct
  model's `overlap_x16` embeddings and rebuild the affected upstream embedding
  only if its provenance is wrong. Never substitute a mask or bypass the hash.
- **Split-grid or tile identity mismatch:** stop. Confirm both model jobs use the
  same split-specific voxel dataset and inventory. Do not redraw a split or
  hand-edit a manifest; rebuild only the affected derived split stage.
- **Partial artifact:** do not use `--skip-existing` or count it as complete for
  `--only-missing`. Inspect logs, remove only that stage directory, and rerun.
- **Checkpoint source mismatch:** standalone training resumes only from its own
  `latest.pt`; inference uses that run's `best.pt`. Verify the model tag,
  resolved config, supervision identities, and tile hashes before retrying.
- **Missing validation voxel prediction:** treat it as a producer failure.
  Regenerate the prediction after fixing coverage/valid-mask provenance; never
  shrink the validation mask or silently omit the voxel.

## Out of scope

M3-V does not include encoder fine-tuning, a raw-amplitude skip, coordinate or
depth features, boundary-weighted downstream loss, CRF/Potts/lateral smoothing,
a label-budget voxel experiment, patch-size changes, or multi-resolution
pretext. Any of these requires a separate preregistered experiment.

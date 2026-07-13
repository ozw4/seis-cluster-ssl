# F3 original-split V0 benchmark

V0 is the fixed token-probe baseline: the existing `linear_balanced_v1` token
predictions are expanded to voxels by nearest patch repetition. It does not train
a voxel decoder. All three encoders use `overlap_x16` and the one shared
`voxel_supervision/png_slices_segy_labels_v1` artifact. M2-A's preregistered
primary comparison is M1.

Run from the repository root. On a fresh workflow, validate and execute each
stage before moving to the next one: later dry-runs require the real outputs of
the preceding stages. Each dry-run writes nothing. For MAE, the exact order is:

```bash
python proc/seis_ssl_cluster/build_f3_lithology_voxel_dataset.py --config experiments/f3/facies_benchmark_v1/87_f3_voxel_benchmark_v0/01_build_voxel_supervision.yaml --dry-run
python proc/seis_ssl_cluster/build_f3_lithology_voxel_dataset.py --config experiments/f3/facies_benchmark_v1/87_f3_voxel_benchmark_v0/01_build_voxel_supervision.yaml
python proc/seis_ssl_cluster/predict_f3_lithology_tokens.py --config experiments/f3/facies_benchmark_v1/87_f3_voxel_benchmark_v0/02_predict_mae_tokens.yaml --dry-run
python proc/seis_ssl_cluster/predict_f3_lithology_tokens.py --config experiments/f3/facies_benchmark_v1/87_f3_voxel_benchmark_v0/02_predict_mae_tokens.yaml
python proc/seis_ssl_cluster/project_f3_lithology_tokens_to_voxels.py --config experiments/f3/facies_benchmark_v1/87_f3_voxel_benchmark_v0/03_project_mae_nearest.yaml --dry-run
python proc/seis_ssl_cluster/project_f3_lithology_tokens_to_voxels.py --config experiments/f3/facies_benchmark_v1/87_f3_voxel_benchmark_v0/03_project_mae_nearest.yaml
python proc/seis_ssl_cluster/evaluate_f3_lithology_voxels.py --config experiments/f3/facies_benchmark_v1/87_f3_voxel_benchmark_v0/04_evaluate_mae_nearest.yaml --dry-run
python proc/seis_ssl_cluster/evaluate_f3_lithology_voxels.py --config experiments/f3/facies_benchmark_v1/87_f3_voxel_benchmark_v0/04_evaluate_mae_nearest.yaml
python proc/seis_ssl_cluster/build_f3_lithology_voxel_report.py --config experiments/f3/facies_benchmark_v1/87_f3_voxel_benchmark_v0/05_report_mae_nearest.yaml --dry-run
python proc/seis_ssl_cluster/build_f3_lithology_voxel_report.py --config experiments/f3/facies_benchmark_v1/87_f3_voxel_benchmark_v0/05_report_mae_nearest.yaml
```

Repeat the same dry-run/real-run pairing for the prediction, projection,
evaluation, and report configs for M1 (`06`–`09`) and then M2-A (`10`–`13`).
The MAE prediction config intentionally repeats the established MAE source
identity and conditions; it may be omitted only after complete artifact
validation succeeds. To validate and reuse a complete token prediction, rerun
its prediction command with `--skip-existing`; the command rejects partial
artifacts and config/source-identity mismatches. Do not use `--skip-existing`
for a partial or unvalidated directory.

Outputs live below each model's
`lithology/f3/facies_benchmark_v1/<MODEL>/overlap_x16/png_slices_segy_labels_v1/`
root. Token predictions are under `predictions/linear_balanced_v1`; V0 voxel
predictions, evaluations, and reports use `token_projection_nearest_v1` in
their respective directories.

The stages are idempotent by refusal: an existing output is not overwritten.
To resume, retain a complete stage and restart at the first missing stage. For
an incomplete stage, remove only that incomplete output directory after
inspection, then rerun it; never replace the shared supervision artifact while
other model results refer to its recorded hash.

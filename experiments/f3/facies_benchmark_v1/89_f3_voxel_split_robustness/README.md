# F3 voxel split robustness (M1 versus M2-A)

The complete M3-V workflow and release checks are in
[`docs/f3_voxel_lithology_benchmark.md`](../../../../docs/f3_voxel_lithology_benchmark.md).

This suite reuses the six M1 `split_000` through `split_005` inventories. It
does not generate or redraw split conditions. V0 and V1 use the same
split-specific voxel supervision, and every model comparison is paired by
split.

```bash
EXP=experiments/f3/facies_benchmark_v1/89_f3_voxel_split_robustness
python proc/seis_ssl_cluster/build_f3_lithology_voxel_split_datasets.py --config "$EXP/01_build_voxel_split_datasets.yaml" --dry-run --only-missing
python proc/seis_ssl_cluster/build_f3_lithology_voxel_split_datasets.py --config "$EXP/01_build_voxel_split_datasets.yaml" --only-missing
python proc/seis_ssl_cluster/run_f3_lithology_voxel_v0_split_suite.py --config "$EXP/02_run_v0_split_projections.yaml" --dry-run --only-missing
python proc/seis_ssl_cluster/run_f3_lithology_voxel_v0_split_suite.py --config "$EXP/02_run_v0_split_projections.yaml" --only-missing
python proc/seis_ssl_cluster/run_f3_lithology_voxel_decoder_split_suite.py --config "$EXP/03_run_v1_split_decoders.yaml" --dry-run --only-missing --device auto
python proc/seis_ssl_cluster/run_f3_lithology_voxel_decoder_split_suite.py --config "$EXP/03_run_v1_split_decoders.yaml" --only-missing --device auto
```

After every split job is complete, run the final summary/publish step:

```bash
python proc/seis_ssl_cluster/summarize_f3_lithology_voxel_split_robustness.py --config "$EXP/04_summarize_voxel_split_robustness.yaml" --dry-run
python proc/seis_ssl_cluster/summarize_f3_lithology_voxel_split_robustness.py --config "$EXP/04_summarize_voxel_split_robustness.yaml"
```

The V1 manifest records `latest.pt` as the resume path after every attempted
job. `--only-missing` skips only jobs with a complete evaluation artifact.
The summary reports raw split rows, paired deltas, win rates, and a provisional
`positive`, `negative`, or `hold` status. Its statistical unit is the split;
it does not compute voxel-level p-values or confidence intervals. Its final
publish manifest includes both this six-split evidence and the original-split
summary; no result is published before this step.

Do not redraw the six splits. A valid pair has the same canonical valid-token
hash, split-grid identity, class weights, and tile order. Any mismatch stops the
suite rather than producing an unpaired comparison.

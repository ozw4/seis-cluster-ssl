# M3-V-LB original-split low-label voxel benchmark

This experiment compares the frozen-embedding voxel decoder for MAE, M1, and
M2-A under three token-label budgets (`cap25`, `cap50`, and `cap100`) and five
paired subsample seeds. A budget is a per-class cap on selected **token rows**;
it is not a voxel-count cap. Duplicate selected rows are audited separately,
and unique selected token coordinates define the shared supervision mask.

Each selected token coordinate expands to its clipped `8 x 8 x 8` voxel block.
The block is intersected with the canonical full training mask, and the dense
voxel labels already present inside that intersection are used unchanged. The
token majority label is never copied across the block. Canonical full
validation is preserved bitwise and remains identical for every budget, seed,
and encoder.

Training uses `uniform_tiles_with_replacement`, batch size 1, 50 epochs, and a
preregistered `steps_per_epoch: 440`. The value 440 is the common canonical
train-tile count in all three completed original full-label V1 runs. This fixes
optimizer-step compute across label budgets. For subsample seed `s`, the
decoder/data-order seed is `42000 + s`; the three models in one paired
condition therefore share decoder initialization, tile sampling order, class
weights, and supervision geometry.

`latest.pt` is resume-only. Full-volume inference must use the selected
`best.pt`, with `write_probabilities: false`. Existing full-label V1 runs are
read only as single seed-42 anchors and are never retrained or mixed into the
five-seed paired means, medians, or win counts. Raw checkpoints, arrays,
embeddings, label volumes, and predictions remain under the artifact root and
are not published to `results/`.

The scientific claim is limited to F3's original split and this fixed decoder.
Six-split low-label evaluation is outside this milestone and may be selected
only after review of these results.

The preregistered scientific decision uses paired seed aggregates only. A
monitored-class major degradation check covers class 3 and class 5 F1, IoU,
and boundary recall at tolerances 2 and 4; mixed primary-metric directions or
positive means supported by fewer than 4/5 wins are reported as `HOLD`.

## Execution order

```bash
export PYTHONPATH="/workspace/src${PYTHONPATH:+:$PYTHONPATH}"
export EXP_LB=experiments/f3/facies_benchmark_v1/91_f3_voxel_label_budget_v1

python proc/seis_ssl_cluster/build_f3_lithology_voxel_label_budget_datasets.py \
  --config "$EXP_LB/01_build_voxel_label_budget_datasets.yaml" --dry-run
python proc/seis_ssl_cluster/build_f3_lithology_voxel_label_budget_datasets.py \
  --config "$EXP_LB/01_build_voxel_label_budget_datasets.yaml" --only-missing

python proc/seis_ssl_cluster/run_f3_lithology_voxel_label_budget_suite.py \
  --config "$EXP_LB/02_run_voxel_label_budget_suite.yaml" --dry-run --device auto
python proc/seis_ssl_cluster/run_f3_lithology_voxel_label_budget_suite.py \
  --config "$EXP_LB/02_run_voxel_label_budget_suite.yaml" \
  --smoke-only --budget cap25 --subsample-seed 0 --device cpu
python proc/seis_ssl_cluster/run_f3_lithology_voxel_label_budget_suite.py \
  --config "$EXP_LB/02_run_voxel_label_budget_suite.yaml" \
  --only-missing --device auto

python proc/seis_ssl_cluster/summarize_f3_lithology_voxel_label_budget.py \
  --config "$EXP_LB/03_summarize_voxel_label_budget.yaml" --dry-run
python proc/seis_ssl_cluster/summarize_f3_lithology_voxel_label_budget.py \
  --config "$EXP_LB/03_summarize_voxel_label_budget.yaml"
```

Only a complete 15-dataset / 45-job summary may publish the Markdown, JSON,
CSV, and PNG review products to
`results/f3/facies_benchmark_v1/voxel_lithology_label_budget_v1/`.

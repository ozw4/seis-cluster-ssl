# F3 Strat-HMM M1 Robustness

Run these configs after the M1 baseline and candidate token datasets are in place.
They cover only B: label-budget robustness and C: split/index robustness.

```bash
export ROOT=/workspace/artifacts/seis_ssl_cluster
export EXP=experiments/f3/facies_benchmark_v1/81_strat_hmm_m1_robustness

# B: label budget
python proc/seis_ssl_cluster/build_f3_lithology_label_budget_datasets.py --config "$EXP/01_build_label_budget_datasets.yaml" --dry-run
python proc/seis_ssl_cluster/build_f3_lithology_label_budget_datasets.py --config "$EXP/01_build_label_budget_datasets.yaml"
python proc/seis_ssl_cluster/run_f3_lithology_label_budget_probes.py --config "$EXP/02_run_label_budget_probes.yaml" --only-missing
python proc/seis_ssl_cluster/summarize_f3_lithology_label_budget_robustness.py --suite-root "$ROOT/lithology/f3/facies_benchmark_v1/robustness/label_budget_m1_v1"

# C: split/index
python proc/seis_ssl_cluster/generate_f3_lithology_split_inventories.py --config "$EXP/04_generate_split_inventories.yaml" --dry-run
python proc/seis_ssl_cluster/generate_f3_lithology_split_inventories.py --config "$EXP/04_generate_split_inventories.yaml"
python proc/seis_ssl_cluster/build_f3_lithology_split_sweep_datasets.py --config "$EXP/05_build_split_sweep_datasets.yaml" --only-missing
python proc/seis_ssl_cluster/run_f3_lithology_split_sweep_probes.py --config "$EXP/06_run_split_sweep_probes.yaml" --only-missing
python proc/seis_ssl_cluster/summarize_f3_lithology_split_robustness.py --suite-root "$ROOT/lithology/f3/facies_benchmark_v1/robustness/split_index_m1_v1"
```

If split inventory generation fails because the validation constraints are too
strict for F3, relax the class 5 minimum gradually in
`04_generate_split_inventories.yaml`. Keep `require_validation_all_classes: true`
so class-missing validation splits are rejected.

## Decision

Go:
  label-budget and split-index mean delta_macro_f1 and delta_mean_iou are positive,
  and win_rate >= 0.7.

Hold:
  full-budget wins but low-budget or split-index results are mixed,
  or balanced_accuracy consistently degrades.

Stop:
  improvements disappear under label-budget or split-index perturbations.

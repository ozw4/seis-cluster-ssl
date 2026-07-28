# M5-U soft-posterior original-split screening

Run the frozen soft-posterior candidate against the paired original-split references:

```bash
export SEIS_SSL_CLUSTER_WORKSPACE="$(pwd)"
export F3_ROOT="/path/to/F3"

python proc/seis_ssl_cluster/run_f3_lithology_soft_posterior_voxel_label_budget.py --config experiments/f3/facies_benchmark_v1/98_strat_hmm_multi_head_k6810_soft_posterior_low_label_v1/01_run_soft_voxel_label_budget.yaml --dry-run
python proc/seis_ssl_cluster/run_f3_lithology_soft_posterior_voxel_label_budget.py --config experiments/f3/facies_benchmark_v1/98_strat_hmm_multi_head_k6810_soft_posterior_low_label_v1/01_run_soft_voxel_label_budget.yaml --only-missing
python proc/seis_ssl_cluster/summarize_f3_lithology_voxel_label_budget_soft_posterior.py --config experiments/f3/facies_benchmark_v1/98_strat_hmm_multi_head_k6810_soft_posterior_low_label_v1/02_summarize_soft_voxel_label_budget.yaml --dry-run
python proc/seis_ssl_cluster/summarize_f3_lithology_voxel_label_budget_soft_posterior.py --config experiments/f3/facies_benchmark_v1/98_strat_hmm_multi_head_k6810_soft_posterior_low_label_v1/02_summarize_soft_voxel_label_budget.yaml
```

Only `M5_U_ORIGINAL_GO` makes the six-split follow-up ready; this stage never runs six-split jobs.

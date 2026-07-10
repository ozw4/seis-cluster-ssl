# F3 Strat-HMM M2-A versus M1 robustness

Run this paired comparison after the M1 and M2-A full-split token datasets
exist. The generic robustness builders fail if baseline and candidate token
identities differ. The split suite reuses the M1 `split_000` through
`split_005` inventory rather than generating new comparison conditions.

```bash
export ROOT=/workspace/artifacts/seis_ssl_cluster
export EXP=experiments/f3/facies_benchmark_v1/85_strat_hmm_m2a_robustness

# Label budget: dry-run, build, probe, summarize.
python proc/seis_ssl_cluster/build_f3_lithology_label_budget_datasets.py --config "$EXP/01_build_label_budget_datasets.yaml" --dry-run
python proc/seis_ssl_cluster/build_f3_lithology_label_budget_datasets.py --config "$EXP/01_build_label_budget_datasets.yaml"
python proc/seis_ssl_cluster/run_f3_lithology_label_budget_probes.py --config "$EXP/02_run_label_budget_probes.yaml" --dry-run
python proc/seis_ssl_cluster/run_f3_lithology_label_budget_probes.py --config "$EXP/02_run_label_budget_probes.yaml" --only-missing
python proc/seis_ssl_cluster/summarize_f3_lithology_label_budget_robustness.py --config "$EXP/03_summarize_label_budget.yaml" --dry-run
python proc/seis_ssl_cluster/summarize_f3_lithology_label_budget_robustness.py --config "$EXP/03_summarize_label_budget.yaml"

# Split/index: dry-run, build from the existing M1 inventory, probe, summarize.
python proc/seis_ssl_cluster/build_f3_lithology_split_sweep_datasets.py --config "$EXP/04_build_split_sweep_datasets.yaml" --dry-run
python proc/seis_ssl_cluster/build_f3_lithology_split_sweep_datasets.py --config "$EXP/04_build_split_sweep_datasets.yaml" --only-missing
python proc/seis_ssl_cluster/run_f3_lithology_split_sweep_probes.py --config "$EXP/05_run_split_sweep_probes.yaml" --dry-run
python proc/seis_ssl_cluster/run_f3_lithology_split_sweep_probes.py --config "$EXP/05_run_split_sweep_probes.yaml" --only-missing
python proc/seis_ssl_cluster/summarize_f3_lithology_split_robustness.py --config "$EXP/06_summarize_split_sweep.yaml" --dry-run
python proc/seis_ssl_cluster/summarize_f3_lithology_split_robustness.py --config "$EXP/06_summarize_split_sweep.yaml"
```

Expected manifests and reports are under:

- `$ROOT/lithology/f3/facies_benchmark_v1/robustness/label_budget_m2a_boundary_vs_m1_v1/{suite_manifest.json,probe_run_manifest.json,reports/}`
- `$ROOT/lithology/f3/facies_benchmark_v1/robustness/split_index_m2a_boundary_vs_m1_v1/{split_dataset_manifest.json,split_probe_run_manifest.json,reports/}`

The original-full-split metrics are produced by M2-A configs 07 and 08. After
those jobs complete, rerun config 08 to regenerate the shared baseline table:

```bash
python proc/seis_ssl_cluster/build_f3_lithology_report.py \
  --config experiments/f3/facies_benchmark_v1/84_strat_hmm_pretraining_m2a_boundary/08_build_lithology_report.yaml

python - <<'PY'
import csv
from pathlib import Path

path = Path('/workspace/artifacts/seis_ssl_cluster/lithology/f3/facies_benchmark_v1/reports/baseline_comparison/comparison_table.csv')
rows = list(csv.DictReader(path.open()))
tags = {row['MODEL_TAG'] for row in rows}
assert 'strat_hmm_pretext_m1_k6_topblock1_distill' in tags
assert 'strat_hmm_pretext_m2a_boundary_a050_t2_k6_topblock1_distill' in tags
print('full-split M1/M2-A rows: OK')
PY
```

# Current-code single-head K=6 control

Status: `CONTROL_READY_POSITIVE`

## Scope

F3 original split; cap25/cap50/cap100; five paired subsample seeds; fixed frozen voxel decoder. Results are descriptive paired-seed summaries.

## Readiness

- Current K6 vs MAE positive budgets: ['cap25', 'cap50', 'cap100']
- Historical-M1 drift triggers: []
- Monitored-class major degradations: []

## Files

- `control_paired_deltas.csv` contains all per-seed paired deltas.
- `control_summary_by_budget.csv` contains means, medians, sample SD, ranges, wins/losses/ties, and worst seeds.
- `control_monitored_class_summary.csv` contains class 3/5 F1, IoU, and boundary-recall summaries.

## Final repository provenance

- HEAD: `371ec7ae4156fdf551a11dd1a9d7ae31ec2ed3ec`
- `git diff --binary HEAD` SHA-256: `cbe069e5b564daf1a4d7e0024cbc502f391c590de3e40cfb3d167c6d02a54af4`
- Changed files: `32`
  - ` M src/seis_ssl_cluster/config/base.py`
  - ` M src/seis_ssl_cluster/config/pretraining.py`
  - ` M src/seis_ssl_cluster/data/f3_voxel_decoder_dataset.py`
  - ` M src/seis_ssl_cluster/embedding/extractor.py`
  - ` M src/seis_ssl_cluster/f3/lithology/voxel_label_budget_results.py`
  - ` M src/seis_ssl_cluster/f3/lithology/voxel_label_budget_runner.py`
  - ` M src/seis_ssl_cluster/training/strat_hmm/runner.py`
  - ` M src/seis_ssl_cluster/training/strat_hmm/runtime.py`
  - ` M src/seis_ssl_cluster/training/strat_hmm_checkpoint.py`
  - ` M tests/seis_ssl_cluster/test_active_experiment_configs.py`
  - ` M tests/seis_ssl_cluster/test_f3_voxel_decoder_dataset.py`
  - ` M tests/seis_ssl_cluster/test_proc_entrypoints.py`
  - ` M tests/seis_ssl_cluster/test_strat_checkpoint_extraction.py`
  - ` M tests/seis_ssl_cluster/test_strat_hmm_pretraining_head_only.py`
  - `?? experiments/f3/facies_benchmark_v1/93_strat_hmm_m1_current_k6_control/01_train_current_k6_smoke.yaml`
  - `?? experiments/f3/facies_benchmark_v1/93_strat_hmm_m1_current_k6_control/02_train_current_k6_full.yaml`
  - `?? experiments/f3/facies_benchmark_v1/93_strat_hmm_m1_current_k6_control/03_extract_current_k6_embeddings.yaml`
  - `?? experiments/f3/facies_benchmark_v1/93_strat_hmm_m1_current_k6_control/04_build_current_k6_token_dataset.yaml`
  - `?? experiments/f3/facies_benchmark_v1/93_strat_hmm_m1_current_k6_control/05_train_current_k6_token_probe.yaml`
  - `?? experiments/f3/facies_benchmark_v1/93_strat_hmm_m1_current_k6_control/06_build_current_k6_token_report.yaml`
  - `?? experiments/f3/facies_benchmark_v1/93_strat_hmm_m1_current_k6_control/07_run_current_k6_voxel_label_budget.yaml`
  - `?? experiments/f3/facies_benchmark_v1/93_strat_hmm_m1_current_k6_control/08_summarize_current_k6_control.yaml`
  - `?? experiments/f3/facies_benchmark_v1/93_strat_hmm_m1_current_k6_control/README.md`
  - `?? proc/seis_ssl_cluster/run_f3_lithology_voxel_label_budget_control.py`
  - `?? proc/seis_ssl_cluster/summarize_f3_lithology_voxel_label_budget_control.py`
  - `?? proc/seis_ssl_cluster/validate_f3_current_k6_control.py`
  - `?? src/seis_ssl_cluster/config/f3_lithology_voxel_label_budget_control.py`
  - `?? src/seis_ssl_cluster/f3/current_k6_control.py`
  - `?? src/seis_ssl_cluster/f3/lithology/voxel_label_budget_control.py`
  - `?? tests/seis_ssl_cluster/test_config_f3_lithology_voxel_label_budget_control.py`
  - `?? tests/seis_ssl_cluster/test_f3_current_k6_control.py`
  - `?? tests/seis_ssl_cluster/test_f3_lithology_voxel_label_budget_control.py`

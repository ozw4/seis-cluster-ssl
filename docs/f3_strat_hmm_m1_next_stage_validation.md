# F3 Strat-HMM Milestone-1 Next-Stage Validation

Validation for issue #195 completed on 2026-07-10.

## Tests run

- `PYTHONPATH=src python -m compileall -q src proc tests`
- `PYTHONPATH=src pytest -q` for
  `test_config_module_imports.py`, `test_proc_entrypoints.py`,
  `test_f3_strat_hmm_m1_results_summary.py`,
  `test_stratigraphy_shuffle_targets.py`,
  `test_f3_strat_hmm_m1_guardrails.py`,
  `test_strat_hmm_pretraining_head_only.py`, `test_strat_pseudo_dataset.py`,
  `test_stratigraphy_pseudo_targets.py`,
  `test_f3_lithology_token_dataset.py`, and `test_f3_lithology_report.py`:
  **201 passed**.
- `PYTHONPATH=src pytest -q` for
  `test_active_experiment_configs.py`,
  `test_strat_hmm_pretraining_distill.py`, and `test_results_publish.py`:
  **177 passed**.

The repository has no separate
`test_f3_strat_hmm_m1_visualizations.py`. Its milestone-1 visualization checks
are part of `test_f3_strat_hmm_m1_results_summary.py`, including non-empty
figure generation, the expected figure set, publishing, and the single-split
plot regression.

## Tests not run

- The full repository test suite was not run because this issue requests the
  focused milestone-1 reporting and guardrail checks. The directly affected
  suites and named non-regression suites passed.
- Full training and downstream experiment runs were not executed. They require
  the external F3 data and milestone artifacts and are outside this test/CI
  cleanup issue.

## Known caveats

- The tests use synthetic fixtures and config resolution; they do not establish
  guardrail experiment outcomes on F3 data.
- A final guardrail decision still requires complete full-budget and paired
  `cap25`, `cap100`, `cap500`, and `full` results with `suite.strict: true`.

## Next recommended experiment command

Validate the first distillation-only guardrail run before writing artifacts:

```bash
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config experiments/f3/facies_benchmark_v1/83_strat_hmm_m1_guardrails/01_train_distillation_only_smoke.yaml \
  --dry-run --device cpu --max-steps 2
```

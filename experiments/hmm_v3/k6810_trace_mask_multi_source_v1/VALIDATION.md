# HMM v3 execution readiness: 2026-10-01

The authorized matrix is F3, Parihaka and Volve × MAE100, asymmetric
LocalBT3 and random initialization, with K `[6, 8, 10]`. The matched
control is the canonical v02 selection with consistency weight 0.1.
The new runs retain that weight, reuse the fixed target manifests and
start from the matched original parent checkpoints.

## Live input and GPU checks

- All nine completed v02 sources passed the multi-head source audit.
- All 48 survey-definition YAML files parsed. The nine downstream
  configurations passed their survey-specific resolvers.
- All 18 smoke/full training commands passed live `--dry-run` validation.
- All nine one-step GPU smoke runs completed and wrote schema-7
  checkpoints. Identity-bound checkpoint validation passed for each run.
- The detached full-matrix launcher revalidated and reused all nine
  completed smoke checkpoints before starting full training.

## Regression checks

The implementation, execution definitions, publication contract and thin CLI
were checked with:

```bash
python -m pytest -q \
  tests/seis_ssl_cluster/test_hmm_v3_trace_mask_driver.py \
  tests/seis_ssl_cluster/test_hmm_v3_trace_mask_summary.py \
  tests/seis_ssl_cluster/test_f3_facies_benchmark_v2_configs.py \
  tests/seis_ssl_cluster/test_f3_local_bt_noise_rotation_search_configs.py \
  tests/seis_ssl_cluster/test_volve_local_bt_recipe_arm_configs.py \
  tests/seis_ssl_cluster/test_proc_entrypoints.py
```

Result: **171 passed**.

The portable repository sweep was:

```bash
python -m pytest -q -m 'not slow and not requires_segy and not requires_cuda'
```

Result: 6,415 passed, 145 deselected and one failure in a newly added
gradient test's fixture: its prototype-head constructor omitted the required
`normalize` argument. Only that test fixture was corrected. The affected
masking, consistency and checkpoint contracts were then revalidated:

```bash
python -m pytest -q \
  tests/seis_ssl_cluster/test_strat_center_trace_masked.py \
  tests/seis_ssl_cluster/test_strat_center_trace_schema7.py \
  tests/seis_ssl_cluster/test_strat_multi_head_losses.py
```

Result: **38 passed**. The successful unaffected checks were reused; the
portable sweep was not repeated after the test-only correction.

Changed implementation/test paths passed focused Ruff lint checks. New
modules, the loss implementation and changed tests passed formatting checks;
the two existing large configuration/checkpoint modules retained their
unrelated pre-existing formatting. The package-isolation check and
`git diff --check` also passed.

## Execution handoff

The nine-arm background pipeline was launched at
`2026-10-01T03:48:16Z`. It executes each survey's 25-epoch training, source
audit, embedding extraction, 45 decoder cells and paired v02 comparison.
The complete matrix has 135 candidate decoder cells.

Machine-local launch records are in the artifact root under
`operations/hmm_v3/k6810_trace_mask_multi_source_v1/`:

- `launch.json`: launch time, process identity and hashes of checkout inputs.
- `status.json`: coarse survey/job status and completed job ranges.
- `launcher.log`: transitions and survey-specific invocation log locations.
- `<survey>/logs/invocation-*/`: individual stage commands and logs.

The launcher stops and records a failure if a stage exits unsuccessfully.
Survey runbooks describe identity-bound restart using the existing driver.
Post-launch verification confirmed that the F3 MAE100 and LocalBT3 arms
completed all 25 epochs and passed their source audits. The remaining matrix
continues in the background; use the live records above for its current stage.
The complete nine-arm training/evaluation matrix and scientific results are
**not yet complete**.

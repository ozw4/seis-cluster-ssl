# Documentation

This directory contains runbooks and configuration contracts for
`seis_ssl_cluster` experiments.

## Core contracts

- [Configuration](configuration.md): YAML ownership, explicit path policy,
  stage schemas, and runtime settings.
- [Configuration validation](config_validation.md): resolver boundaries,
  stage dispatch, path validation, and failure policy.
- [Proc CLI contract](proc_cli_contract.md): thin entrypoint boundary, shared
  CLI helpers, overrides, dry-run behavior, and stdout rules.
- [F3 module structure](f3_module_structure.md): package boundaries, public
  imports, token dataset schema, and report responsibilities.

## Active research runbooks

- [Parihaka survey-specific 3D amplitude MAE pretraining](parihaka_mae_pretraining.md): amplitude provenance, ZXY-to-XYZ preparation, direct data paths, fixed training and checkpoint contracts, label exclusion, and transductive claim boundary.
- [F3 inspection](f3_facies_benchmark_inspection.md): raw inventory, geometry, label checks, and quicklook evidence.
- [F3 strat-HMM milestone-1 pretraining](strat_hmm_pretraining_milestone1.md): pseudo-target export, pretraining, extraction, and smoke validation.
- [F3 strat-HMM guardrail producers](f3_strat_hmm_m1_guardrails.md): distillation-only and shuffled-target pretraining artifacts.
- [F3 strat-HMM M2-A boundary weighting](f3_strat_hmm_m2a_boundary_weighting.md): boundary-weighted target export, pretraining, and embedding extraction.
- [F3 M5-LS lateral smoothing](f3_m5_lateral_smoothing_plan.md): fixed edge-aware lateral-message and ordered hard-reprojection artifact contract.
- [F3 center-trace masked HMM path reconstruction](f3_center_trace_masked_pretraining_plan.md): fixed center-trace mask, hard-target loss, trainability, and checkpoint identity.
- [F3 center-trace masked periodic HMM refresh](f3_center_trace_masked_periodic_hmm_refresh_plan.md): periodic student-embedding refresh, generation artifacts, and schema-8 checkpoint identity.
- [F3 XY-neighbour consensus hard-label smoothing](f3_xy_neighbor_consensus_hard_label_smoothing.md): single synchronous source-label consensus correction with an ordered-trace guard and no posterior, affinity, re-decoding, or beta calibration.
- [F3 unanimous XY-neighbour correction](f3_xy_neighbor_unanimous_outlier_correction.md): unanimous-neighbour target correction and producer validation.

## Repository output policy

- `artifacts/`: complete execution outputs, intermediate products, and inputs
  to later processing. It is not tracked by Git; local runs normally use
  `/workspace/artifacts/seis_ssl_cluster/`.
- `reports/`: lightweight, human-readable summaries tracked by Git. Do not use
  files here as pipeline inputs.
- `experiments/`: experiment definitions and configuration.

Each report producer owns its explicit lightweight file set. Verify that set
in focused tests and inspect `git diff` during review. See
[report_sharing_policy.md](report_sharing_policy.md) for the repository policy.

## Test Selection

Use pytest markers to select the local validation granularity:

```bash
pytest -q -m "not slow and not requires_segy and not requires_cuda"
pytest -q -m smoke
pytest -q -m integration
```

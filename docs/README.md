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

- [F3 M3-V voxel lithology benchmark](f3_voxel_lithology_benchmark.md): V0/V1 contracts, canonical `frozen_embedding_decoder_nearest_voxel_ln_v1` identity, exact original/six-split execution order, input checks, and release validation.
- [F3 strat-HMM M2-A boundary weighting](f3_strat_hmm_m2a_boundary_weighting.md): ordered export-to-summary workflow, artifact paths, failure handling, and fixed evaluation inputs.
- [F3 strat-HMM M2-A result decision](f3_strat_hmm_m2a_results.md): final Go/Stop/Hold evidence contract.
- [F3 M5-LS lateral smoothing](f3_m5_lateral_smoothing_plan.md): fixed edge-aware lateral-message and ordered hard-reprojection algorithm plus staged artifact, training, and evaluation contract.
- [F3 center-trace masked HMM path reconstruction](f3_center_trace_masked_pretraining_plan.md): fixed center-trace mask, hard-target loss, trainability, checkpoint identity, and original-split evaluation contract.
- [F3 center-trace masked periodic HMM refresh](f3_center_trace_masked_periodic_hmm_refresh_plan.md): periodic student-embedding refresh, warm-start ordered-center update, generation artifacts, schema-8 checkpoint identity, and original-split evaluation contract.
- [F3 XY-neighbour consensus hard-label smoothing](f3_xy_neighbor_consensus_hard_label_smoothing.md): single synchronous source-label consensus correction with an ordered-trace guard and no posterior, affinity, re-decoding, or beta calibration.

## Artifacts And Results

`artifacts/` is the local generated-output area and is ignored by Git. Normal
experiment, training, embedding, clustering, and visualization outputs should
continue to use `/workspace/artifacts/seis_ssl_cluster/`.

`results/` is the repository-managed area for lightweight GitHub review
artifacts. Keep only selected Markdown, JSON, CSV, and representative figures
there. Do not commit checkpoints, embeddings, clustering models, raw arrays,
prediction volumes, or raw SEGY files.

Each producer owns its explicit lightweight result file set. Verify that set in
focused tests and inspect `git diff` during review. See
[results_sharing_policy.md](results_sharing_policy.md) for the repository policy.

## Test Selection

Use pytest markers to select the local validation granularity:

```bash
pytest -q -m "not slow and not requires_segy and not requires_cuda"
pytest -q -m smoke
pytest -q -m integration
```

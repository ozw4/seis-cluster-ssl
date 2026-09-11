# Task 08 integration review

The F3 experiment preparation is internally consistent. No blocking
implementation findings remain from this review. This conclusion covers
configuration, portable synthetic evidence, and orchestration; it does not
certify any live target, checkpoint, embedding, decoder, or screening result.

## Reviewed contracts

| Boundary | Evidence checked |
| --- | --- |
| Historical inputs → targets | K=4/8/10 and separate K6 replay preserve the complete baseline clustering mapping except K/output. Existing MAE100 embeddings are referenced directly. |
| Targets → manifests | Four ascending head sets; shared roots agree; historical K6 is the training target and replay is parity-only. Generic schema, array alignment, valid-mask and replay rejection tests pass. |
| Manifests → continuation | Every full/smoke recipe binds its own manifest/hash and candidate ID. Full recipes preserve Condition 2b except the specified Multi-Head contract. |
| Scientific budget | MAE100 initialization, 25 epochs, 15,625 steps, distillation 0.2, top-one encoder block, seed 42, FP32, equal per-head losses and consistency 0 remain fixed. Smoke is separate and never initializes full training. |
| Completed continuation → embeddings | Task 04 validates completed full latest checkpoints, manifest semantics/replay, parent identity, and frozen state. Task 05 selects precisely those checkpoints. |
| Embeddings → downstream | Canonical extraction/decoder conditions are preserved; four candidates each contribute the exact five-layout × three-size matrix. |
| Downstream → summary | The same 15 historical K6 cells are read directly from the canonical runs root. Sources, decoder completion and evaluation provenance remain mandatory. Primary mean IoU and secondary macro F1 match the HMM_v1 F3 reporting contract. Statistical replication is by layout. |
| Driver → all configurations | New integration test checks the complete 81-command plan against manifest, training, source-audit, embedding, downstream and summary references. The export wrapper expands its one planned shell command into four exports. |
| Recovery and writes | Explicit stage/candidate/cell selection, own-checkpoint resume, failed-command propagation, unique logs and store lock are covered. Completed outputs are not silently skipped or replaced. |

No soft posterior, lateral smoothing, XY-neighbor processing, center-trace
masking, periodic refresh, consistency sweep, LocalBT/Random candidate, or
Volve/Parihaka execution was added. The existing Random embedding geometry/mask
reference required by the canonical candidate auditor is read-only.

## Implementation scope

Generic changes across Tasks 01–07 are limited to accepting experiment-owned
hard Multi-Head model IDs while preserving legacy variant bindings, adding a
read-only completed Multi-Head audit, and composing existing paired evidence
readers/statistics for screening with a separate historical runs root and mean
IoU primary metric. Manifest schema/building and Single-Head training branches
are unchanged. The older LocalBT paired-summary CLI keeps its existing defaults.

Task 08 adds a cross-stage integration test and this review record; it requires
no further production-code change. HMM_v1 experiment definitions and
`reports/hmm_v1/` have no diff. Experiment files contain only YAML, Markdown,
and shell assets, with environment-based artifact paths rather than local
machine paths. Pipeline inputs do not come from `reports/`.

## Validation

The reviewed portable suite passed 317 tests; the added cross-stage integration
test passed separately (318 total). The combined selection is reproducible as:

```bash
pytest -q \
  tests/seis_ssl_cluster/test_f3_hmm_v2_*.py \
  tests/seis_ssl_cluster/test_strat_hmm_source_audit.py \
  tests/seis_ssl_cluster/test_strat_multi_head_target_manifest.py \
  tests/seis_ssl_cluster/test_config_strat_hmm_multi_head.py \
  tests/seis_ssl_cluster/test_config_strat_hmm_pretext.py \
  tests/seis_ssl_cluster/test_strat_multi_head_losses.py \
  tests/seis_ssl_cluster/test_strat_checkpoint_extraction.py \
  tests/seis_ssl_cluster/test_f3_lithology_candidate_benchmark.py \
  tests/seis_ssl_cluster/test_f3_lithology_paired_candidate_results.py \
  tests/seis_ssl_cluster/test_f3_completed_decoder_contract.py \
  tests/seis_ssl_cluster/test_f3_completed_evaluation_provenance.py
python -m compileall -q src proc tests
python tools/check_seis_ssl_cluster_isolation.py
```

Ruff passes on all changed Python/test files. Both shell scripts and documented
Bash blocks pass `bash -n`; `git diff --check` and the experiment file inventory
check pass. Driver tests use stubbed processes for execution/failure paths and
real generic manifest dry-runs over synthetic temporary targets. No long
clustering, training, embedding extraction, export, or downstream run was started.

## Live handoff

Use [RUNBOOK.md](RUNBOOK.md) for execution and restart boundaries. A whole-pipeline
dry-run requires the later stages' artifacts already to exist; `--plan` is the
artifact-free preparation check. Direct entry to embeddings/downstream still
requires all four completed sources to pass Task 04. Summary requires all 60
candidate cells and the 15 historical cells to pass live evidence checks.

Live artifact availability, GPU feasibility, runtime, memory use, target/replay
parity on the real survey, and screening quality remain unverified here.
Only a subsequent scheduled run can establish those results. This preparation
review does not select a winning candidate or authorize expansion to other surveys.

# HMM v2 MAE feature-distillation sensitivity

This is a prespecified fixed-seed sensitivity analysis, not a coefficient or
checkpoint selection. The canonical selection remains unchanged.

Execution is complete: nine 25-epoch continuations, nine extractions and all
135 new downstream cells finished by 2026-11-06 11:26:54 UTC. The 45 medium
cells preceded all 90 small/large cells, without filtering conditions. On
2026-11-11 the 60 F3 common-test prediction scores and final 180-cell comparison
were completed. See the [results and sensitivity figure](../../../reports/hmm_v2_distillation_sensitivity_v1/README.md).
The existing 0.2 sources/45 downstream cells were reused; there was no extra
0.2 training. Final aggregation uses recorded lineage and runner outputs,
without additional retrospective experiment audits, following user steering.

The [matrix](matrix.yaml) freezes F3, Parihaka and Volve × distillation
`0.0, 0.1, 0.2, 0.4`, K `[6,8,10]`, prototype 1.0, usage 0.005 and consistency
0.1 (beta 0.1). Nine new independent continuations start from each survey's
original MAE100 checkpoint, also used as the frozen teacher. The completed
0.2 HMM models and their 45 downstream cells are reused only after source and
cell audits. Target manifests, labels, confidence and valid masks are reused.
There is no MAE pretraining, HMM fitting or target regeneration.

All inherited settings remain fixed: top encoder block, 25 epochs,
10,000 samples/epoch, seed 42, optimizer/LR/batch/workers/preprocessing, embedding
geometry/precision, decoder/seed/supports/validation selection and five layouts
with nested small/medium/large budgets. New training definitions add only the
coefficient, new output/model identities, and 16 fixed diagnostic crops.
The validator checks all other training, embedding and decoder values against
the SHA-bound baseline definitions. Smoke runs use separate outputs and one
step; they are never counted as full training.

## Preparation and execution

Use the repository Python environment. Set `SEIS_SSL_CLUSTER_ARTIFACT_ROOT`,
`SEIS_SSL_CLUSTER_WORKSPACE`, `F3_ROOT`, `SEIS_SSL_CLUSTER_VOLVE_ROOT` and
`PYTHONPATH` to the existing inputs on the execution machine. `F3_ROOT` must
retain the frozen label path spelling. Machine-specific values belong in ignored
`operations/hmm_v2/distillation_sensitivity_v1/` records, not Git.

```bash
python -m seis_ssl_cluster.hmm.distillation_sensitivity --phase validate
python -m seis_ssl_cluster.hmm.distillation_inputs --with-baseline-cells
python -m seis_ssl_cluster.hmm.distillation_sensitivity --phase full --dry-run
CUDA_VISIBLE_DEVICES=1 python -m seis_ssl_cluster.hmm.distillation_sensitivity --phase smoke
CUDA_VISIBLE_DEVICES=1 python -m seis_ssl_cluster.hmm.distillation_inputs --diagnose-baselines
SENSITIVITY_GPU=1 bash experiments/hmm_v2/distillation_sensitivity_v1/run_all.sh
```

Check GPU memory/utilization and existing jobs before choosing GPUs.
`run_all.sh` uses at most two jobs on each selected GPU. For two available GPUs,
use `SENSITIVITY_GPUS="0 1"`; the single-GPU `SENSITIVITY_GPU=1` form remains
supported. It completes all nine full
continuations, then all nine embeddings, then **all 45 medium cells**, then
**all 90 small/large cells**. The remaining phase refuses to start before all
45 medium results exist. No coefficient or survey is filtered by medium results.
The wrapper locks the study; individual commands and decoder cells also lock
their own outputs. It does not stop existing jobs.

The first execution recorded the preflight and initial alignment using the
same audit operations in `preflight.py` / `initial_alignment.py` under the
ignored operation directory; the package audit command above reproduces them.
The recorded `preflight.json`, `initial_alignment.json` and
`historical_initial_alignment.json` bind actual input hashes, initial
student/head/teacher states, and the first two batches from the original
8-worker dataloaders. Same seed alone is not treated as proof of equivalence.

For source-only rechecking, omit `--with-baseline-cells`. Audit commands write
new `*_recheck.json` records and refuse to overwrite an existing record.

## Recovery

Use the exact same condition's checkpoint. Never use a completed HMM parent as
an initialization for a fresh coefficient. For example:

```bash
CUDA_VISIBLE_DEVICES=1 python -m seis_ssl_cluster.hmm.distillation_sensitivity \
  --phase full --survey f3 --weight 0.0 --resume
CUDA_VISIBLE_DEVICES=1 python -m seis_ssl_cluster.hmm.distillation_sensitivity \
  --phase medium --survey f3 --weight 0.0 --resume
CUDA_VISIBLE_DEVICES=1 python -m seis_ssl_cluster.hmm.distillation_sensitivity \
  --phase remaining --survey f3 --weight 0.0 --resume
python -m seis_ssl_cluster.hmm.distillation_sensitivity --phase test --resume
python -m seis_ssl_cluster.hmm.distillation_sensitivity --phase summary
python -m seis_ssl_cluster.hmm.distillation_review
```

The wrapper audits completed full runs/cells instead of overwriting them and
resumes partial full runs/cells from their own `latest.pt`. Event records contain
commands, visible GPU assignments, UTC times, exit status and resumed-checkpoint
hashes. Extraction uses
the existing `--skip-existing` metadata validation. Preserve a failed output and
its log before attempting any recovery that needs a different destination.
Completed summary directories are immutable; a successful summary is not rerun.
The summary references the retained `preflight.json` and
`experiments_complete.json` rather than repeating source/cell audits. The
original input/checkpoint/target bindings remain in those operation records.

## Evaluation and reporting

F3 uses exactly the established common-test mask, 113,979,624 voxels, with
validation-selected decoders. The explicit common-test evaluator has a separate
fixed four-coefficient namespace and requires the original reference mask hash.
It audits all 60 F3 source/decoder/prediction identities before scoring. Existing
0.2 predictions may be scored again for parity; no extra decoder is trained.
Validation values are retained separately in `f3_validation.csv` and summary JSON and are never added
to the test curve. F3 is same-survey transductive label holdout, not a complete
spatial amplitude holdout. Parihaka uses test Channel IoU; Volve uses test macro
MAE in samples. Secondary metrics are retained.

Each survey/budget/coefficient has five individual layout values, mean and
sample SD (`ddof=1`). Improvement is candidate minus 0.2 for IoU and 0.2 minus
candidate for Volve MAE. Five layouts are not five encoder seeds. Nested budgets
are not independent replicates and are not pooled for hypothesis testing.
Results describe continuation seed 42; no canonical coefficient is changed.

Each new run records raw and weighted losses, per-head usage entropy and pairwise
head consistency in the existing epoch diagnostics. The frozen teacher is
constructed in the original positive-weight position even at weight zero, so
initialization RNG consumption stays aligned. Only the distillation gradient
contribution becomes zero; prototype/consistency learning still updates the
same top block. Fixed evaluation crops (dataset epoch 0, sample indices 0–15)
are cached without consuming the training dataloader or changing any RNG state.
They are evaluated in eval mode before training and after each completed epoch.
The token-weighted mean cosine distance remains observable at coefficient zero.

The reused 0.2 checkpoints retain only initial/final fixed-crop evaluations;
intermediate fixed-crop values cannot be reconstructed from the retained
checkpoints. Their original changing-training-crop loss curves are saved
separately. Feature distance alone does not establish downstream performance
or preservation of geological information.

Full outputs remain under the policy's
`surveys/<survey>/<series>/<kind>/hmm_v2_k6810_distillation_sensitivity_v1/`,
with F3 training in v1 storage and embeddings/downstream in v2. Cross-survey
outputs use `studies/hmm_v2/distillation_sensitivity_v1/`; execution evidence uses
`operations/hmm_v2/distillation_sensitivity_v1/`.
The primary summary producer owns exactly `cells.csv`, `comparison.csv`,
`feature_diagnostics.csv`, `f3_validation.csv`, `summary.json`, `summary.md`,
`sensitivity.png`.
The 3×3 sensitivity figure shows coefficients including zero, with separate
survey and budget panels and each metric's original units.

## Existing auxiliary analyses

```bash
python tools/freeze_hmm_v1.py --artifact-root "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT" \
  --output-dir "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/studies/hmm_v2/distillation_sensitivity_v1/v1_recovered" --write
python tools/freeze_hmm_v1.py --artifact-root "$SEIS_SSL_CLUSTER_ARTIFACT_ROOT" --check-source
python -m seis_ssl_cluster.hmm.distillation_auxiliary
```

The v1 freeze tool reconstructs live original cells and verifies them against
the report/manifest. Analysis consumes these artifact originals; reports do not
become pipeline inputs. F3 uses the existing common-test artifacts matching
`reports/hmm_f3_common_test_v1/`. Parihaka/Volve v1 compares 0.1 versus 0.2 by
source, layout and budget. F3 v1 has execution differences including workers;
paired layouts do not imply identical continuation RNG streams.

The v2 auxiliary analysis compares consistency 0.1 versus 0 for matching heads,
source, layout and budget, with K6810 and K468 in separate rows. It follows the
existing aggregate configurations to historical compatibility paths and verifies
completed source checkpoints and metrics. Old incomplete runbook text does not
override newer completed artifacts. The auxiliary producer owns exactly
`v1_pairs.csv`, `v2_consistency_pairs.csv`, `comparison.csv`, `summary.json`,
`summary.md`. Missing original files, if any, are recorded without selecting
or stopping the main experiment.

See [validation](VALIDATION.md) for execution and implementation evidence.

## CPU scheduling recovery

The first four medium cells exposed CPU oversubscription (64 intra-op threads
per process). A three-step probe on an actual F3 training tile produced identical
initial/final parameter hashes and identical metrics with 64, 1 and 4 threads.
Decoder subprocesses therefore use `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`;
DataLoader worker counts, seeds, optimizer, batch sizes, data and deterministic
CUDA settings are unchanged. Four owned cells resumed from their own rolling
checkpoints; no encoder was interrupted or retrained. The source logs, checkpoint
hashes and retained positions are in `decoder_thread_recovery.json`,
`thread_probe.json` and the execution event log.

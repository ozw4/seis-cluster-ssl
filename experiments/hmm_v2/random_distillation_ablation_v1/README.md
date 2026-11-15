# Random HMM v2: zero versus existing 0.2 distillation

This follow-up prespecifies **0.0 versus 0.2** for Random initialization on F3,
Parihaka and Volve. The 0.0 arms are three new independent 25-epoch runs; the
three existing 0.2 models and all 45 baseline decoder cells are reused. New
downstream training is 45 cells (five layouts × three budgets × three surveys),
and the final comparison has 90 cells. No coefficient selection is performed.

All stages completed on **2026-11-11 11:26 UTC**: three full trainings, three
extractions, 45 new decoder cells, 30 F3 common-test prediction scores, the
90-cell comparison and the seven-file report. The three existing 0.2 models
and all 45 baseline decoder results were reused without retraining. See the
[completed results](../../../reports/hmm_v2_random_distillation_ablation_v1/README.md).
Zero distillation improved all nine survey/budget means and 42 of 45 individual
layout values in this fixed-seed comparison. All conditions are reported.

The [manifest](matrix.yaml) binds the original baseline and all 12 new YAML
definitions. Every new definition inherits its survey's **Random** 0.2 recipe,
including its initial checkpoint, fixed teacher, fixed Random HMM targets,
confidence/masks, K=[6,8,10], projection/temperature/normalization, prototype 1.0,
usage 0.005, consistency 0.1, top encoder block, 10,000 samples/epoch, seed 42,
optimizer, LR, batch, workers, preprocessing, extraction and decoder rules.
Only distillation weight, new model/output identities, and 16 fixed diagnostic
crops differ. There is no MAE/HMM fitting or pseudo-label regeneration.

Baseline training/extraction/downstream definitions are in:

- F3: `experiments/f3/facies_benchmark_v2/131_hmm_v2_k6810_cons010_multi_source_v1/`
- Parihaka: `experiments/parihaka/facies_benchmark_v1/44_hmm_v2_k6810_cons010_multi_source_v1/`
- Volve: `experiments/volve/horizon_benchmark_v1/37_hmm_v2_k6810_cons010_multi_source_v1/`

The baseline candidate is `random_init_hmm_v2_mh_k6810_distill020_cons010`;
the new candidate is `random_init_hmm_v2_mh_k6810_distill000_cons010`.
Student and fixed teacher start from the original Random initialization, never
the completed 0.2 HMM. Baseline inputs, run-metadata identities and the 45 raw
result file references are recorded in ignored operation records. The user
requested no extra retrospective experiment audits; the existing runners retain
their normal input/resume checks.

## Run and resume

Use the existing repository Python environment and set artifact/data environment
variables as described in the [MAE runbook](../distillation_sensitivity_v1/README.md).
Machine-specific values stay in ignored operation records. Check GPU availability
before choosing devices; never stop unrelated jobs. On the initial launch both
H100 NVL GPUs were idle. Three new runs were assigned F3/Volve to GPU0 and
Parihaka to GPU1.

```bash
python -m seis_ssl_cluster.hmm.distillation_sensitivity \
  --matrix experiments/hmm_v2/random_distillation_ablation_v1/matrix.yaml --phase validate
python -m seis_ssl_cluster.hmm.distillation_sensitivity \
  --matrix experiments/hmm_v2/random_distillation_ablation_v1/matrix.yaml --phase full --dry-run
CUDA_VISIBLE_DEVICES=0 python -m seis_ssl_cluster.hmm.distillation_sensitivity \
  --matrix experiments/hmm_v2/random_distillation_ablation_v1/matrix.yaml --phase smoke
SENSITIVITY_GPUS="0 1" bash experiments/hmm_v2/random_distillation_ablation_v1/run_all.sh
```

The script runs all three full continuations, then all three extractions, retained
baseline feature diagnostics, **all 15 medium cells**, and **all 30 small/large
cells**. No condition is filtered using medium results. It then scores the 30 F3
common-test prediction cells, aggregates all 90 cells, draws the figure and
curates the report. Decoder subprocesses use the already validated single CPU
intra-op thread setting; declared DataLoader workers remain inherited.

For a partial condition, resume only its own rolling checkpoint. For example:

```bash
CUDA_VISIBLE_DEVICES=0 python -m seis_ssl_cluster.hmm.distillation_sensitivity \
  --matrix experiments/hmm_v2/random_distillation_ablation_v1/matrix.yaml \
  --phase full --survey f3 --weight 0.0 --resume
CUDA_VISIBLE_DEVICES=0 python -m seis_ssl_cluster.hmm.distillation_sensitivity \
  --matrix experiments/hmm_v2/random_distillation_ablation_v1/matrix.yaml \
  --phase medium --survey f3 --weight 0.0 --resume
CUDA_VISIBLE_DEVICES=0 python -m seis_ssl_cluster.hmm.distillation_sensitivity \
  --matrix experiments/hmm_v2/random_distillation_ablation_v1/matrix.yaml \
  --phase remaining --survey f3 --weight 0.0 --resume
python -m seis_ssl_cluster.hmm.distillation_sensitivity \
  --matrix experiments/hmm_v2/random_distillation_ablation_v1/matrix.yaml --phase test --resume
python -m seis_ssl_cluster.hmm.distillation_sensitivity \
  --matrix experiments/hmm_v2/random_distillation_ablation_v1/matrix.yaml --phase summary
python -m seis_ssl_cluster.hmm.distillation_review --random
```

Completed Random training/cells are skipped without a new audit. Extraction uses
the existing metadata-aware `--skip-existing`. Commands, UTC times, devices,
return codes and resumed-checkpoint identities are logged in `execution/events.jsonl`.
The study lock prevents duplicate orchestration. Completed summaries/reports are
immutable; never overwrite them. `experiments_complete.json` records completion
of training/extraction/45 decoders; `analysis_complete.json` is written only after
successful final curation. Both live under
`operations/hmm_v2/random_distillation_ablation_v1/` in the artifact root.

## Evaluation and interpretation

F3 uses the established 113,979,624-voxel common-test mask with validation-selected
decoders. The separate fixed Random namespace requires the original mask hash;
validation values remain a separate 30-row table. Parihaka uses test Channel IoU,
and Volve uses test macro MAE in samples. Secondary metrics remain available.
The five-layout sample SD has ddof=1. Paired improvement is candidate minus 0.2
for IoU and 0.2 minus candidate for MAE. Layouts are not encoder-seed repetitions;
nested budgets are not independent replicates. This is a seed-42 sensitivity
comparison. Canonical coefficients and selected checkpoints are preserved.

Fixed-crop teacher cosine distance is measured before and after every new epoch,
including at zero distillation. Raw/weighted losses, head usage entropies and
pairwise consistency are retained. Existing 0.2 weights permit initial/final
fixed-crop measurement; intermediate checkpoints were not retained. Original
per-epoch changing-training-crop distances remain separate. A smaller feature
distance alone does not establish better downstream or geological information
retention. Random versus MAE also differs in teacher and HMM targets, so their
performance difference is not an isolated initialization effect.

All pipeline inputs/outputs stay in artifacts. The study summary producer owns
exactly seven files under `studies/hmm_v2/random_distillation_ablation_v1/summary/`:
`cells.csv`, `comparison.csv`, `feature_diagnostics.csv`, `f3_validation.csv`,
`summary.json`, `summary.md`, `sensitivity.png`. The figure has nine separate
dataset/budget panels and original metric units, with both 0 and 0.2 shown.

The curated report at `reports/hmm_v2_random_distillation_ablation_v1/` owns
exactly seven files: `README.md`, `provenance.json`, `cells.csv`, `comparison.csv`,
`feature_diagnostics.csv`, `f3_validation.csv`, `sensitivity.png`. It is never
a pipeline input. See [validation](VALIDATION.md) for completed preparation and
execution evidence. Full-resolution arrays and checkpoints are not published.

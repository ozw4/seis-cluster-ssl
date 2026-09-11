# HMM_v1 freeze

- **Status:** FROZEN
- **Freeze date:** 2026-09-11
- **Result snapshot:** 2026-09-10 06:52 UTC
- **Source commit:** `01da413293b7fdcd138b7ad3a9171a76c5f62f0f`
- **F3 completion-definition revision:** `b4d6dececf2802808bd0c340f83101414db5646d`
- **Representative condition:** Condition 2b

HMM_v1 is the complete nine-condition comparison of the existing
**Single-Head** ordered K=6 HMM Prompt Clustering continuation. It covers F3
lithology classification, Parihaka channel classification, and Volve
five-horizon estimation. The machine-readable [manifest](manifest.json), frozen
[405-cell table](cells.csv), and derived [summary](summary.json) are the
authoritative freeze record. Result values were generated from the completed
source aggregates and their referenced metrics by the
[freeze tool](../../tools/freeze_hmm_v1.py); they were not transcribed by hand.

## Conditions

| ID | Frozen definition |
|---|---|
| 1 | MAE 100 epochs + MAE 25 epochs |
| 2a | MAE 100 epochs + HMM 25 epochs, distill 0.1 |
| 2b | MAE 100 epochs + HMM 25 epochs, distill 0.2 |
| 3 | asymmetric LocalBT 3 epochs |
| 4a | asymmetric LocalBT 3 epochs + HMM 25 epochs, distill 0.1 |
| 4b | asymmetric LocalBT 3 epochs + HMM 25 epochs, distill 0.2 |
| 5 | random weight |
| 6a | random weight + HMM 25 epochs, distill 0.1 |
| 6b | random weight + HMM 25 epochs, distill 0.2 |

Conceptually these are the MAE, LocalBT, and Random parents before and after
HMM continuation; the two distillation weights make nine reported IDs.

## Canonical implementation, configuration, and execution

The Single-Head branch is the `head` configuration without `head.spec`. It is
resolved in [pretraining.py](../../src/seis_ssl_cluster/config/pretraining.py),
built as `OrderedPrototypeHead` by
[components.py](../../src/seis_ssl_cluster/training/strat_hmm/components.py), and
trained by [runner.py](../../src/seis_ssl_cluster/training/strat_hmm/runner.py)
and [epoch.py](../../src/seis_ssl_cluster/training/strat_hmm/epoch.py). The CLI
entrypoint is
[train_strat_hmm_pretext.py](../../proc/seis_ssl_cluster/train_strat_hmm_pretext.py).

The manifest lists every frozen training and downstream config. The main
condition bindings are:

- F3: canonical v3 [downstream matrix](../../experiments/f3/facies_benchmark_v2/110_lithology_mae_local_bt_five_way_v3/60_five_way.yaml), asymmetric LocalBT/HMM [definition](../../experiments/f3/facies_benchmark_v2/123_local_bt_noise_rotation_search_v1/README.md), Random distill-0.1 [definition](../../experiments/f3/facies_benchmark_v2/125_lithology_random_init_self_target_hmm_k6_distill010_v1/README.md), and missing-arm [completion runbook](../../experiments/f3/facies_benchmark_v2/126_pretraining_comparison_completion_v1/README.md).
- Parihaka: [all nine arms](../../experiments/parihaka/facies_benchmark_v1/40_pretraining_comparison_completion_v1/50_summary/01_all_nine_arms.yaml) and its [runbook](../../experiments/parihaka/facies_benchmark_v1/40_pretraining_comparison_completion_v1/README.md).
- Volve: MAE comparison [runbook](../../experiments/volve/horizon_benchmark_v1/31_mae_local_bt_hmm_five_way_v1/README.md), asymmetric LocalBT [runbook](../../experiments/volve/horizon_benchmark_v1/32_local_bt_recipe_arm_v1/README.md), and missing HMM arms [runbook](../../experiments/volve/horizon_benchmark_v1/33_missing_hmm_comparison_v1/README.md).

Checkpoints remain outside Git under `SEIS_SSL_CLUSTER_ARTIFACT_ROOT`. The
manifest records the exact path, Condition ID, byte size, and SHA-256 for all 27
survey/condition checkpoints, together with the hashes of the source aggregate
files used for this snapshot.

## Evaluation contract and completeness

Each survey/condition has five layouts by three data sizes, or 15 cells. The
snapshot loaded all 405 expected cells with zero missing cells, duplicate cell
identities, unreadable results, or non-finite selected metrics.

- F3 primary metric: mean IoU on unique validation voxels; higher is better.
- Parihaka primary metric: test channel IoU; higher is better.
- Volve primary metric: test macro MAE in samples; lower is better. Paired
  improvement is `parent MAE - HMM MAE`, so positive means improvement.

## Official results

| Task | Best | Primary mean | Population SD | Secondary mean | Paired improvement over Condition 1 | Better cells |
|---|---|---:|---:|---:|---:|---:|
| F3 | 2b | mean IoU 0.582562 | 0.069338 | macro F1 0.698094 | +0.029489 | 14/15 |
| Parihaka | 2b | channel IoU 0.364694 | 0.089054 | channel F1 0.527928 | +0.020879 | 12/15 |
| Volve | 2b | macro MAE 8.502419 | 2.653019 | within ±2 samples 0.464422 | +0.439231 | 11/15 |

HMM25 improves the 15-cell mean primary metric over its matching MAE,
asymmetric LocalBT, or Random parent for every task/parent/weight combination.
MAE100 + HMM25 with distill 0.2 is best by the mean primary metric on all three
tasks. In the MAE series, both continuations start from the same MAE100
checkpoint, and switching to HMM is better than continuing MAE for 25 epochs.

The asymmetric LocalBT parent is the 3-epoch checkpoint: a preliminary check
found that continuing LocalBT for another 25 epochs degraded performance. The
Random + HMM arms are an ablation indicating that HMM alone can form useful
representations from random initialization. Distill 0.2 is consistently better
within the MAE series; the best weight for LocalBT and Random is task-dependent.

## Limits and version boundary

These are descriptive statistics from one training seed. They do not establish
statistical significance or multi-seed training uncertainty. F3 is a validation
voxel evaluation, while Parihaka and Volve are test evaluations; absolute values
must not be compared between surveys. Resume preserves the fixed epoch/step
budget but can change later shuffle order after persistent workers are recreated.

HMM_v1 configs, argument meanings, defaults, and the no-`head.spec`
Single-Head factory branch are frozen. MultiHead work must use a separate
experiment version and new configs with an explicit head specification. Do not
edit an HMM_v1 config, add MultiHead-only keys to it, or reinterpret its defaults.
No MultiHead implementation or experiment definition is part of this freeze.

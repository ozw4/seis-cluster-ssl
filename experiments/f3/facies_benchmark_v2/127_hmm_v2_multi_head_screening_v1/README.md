# F3 HMM_v2 Multi-Head screening

Compare four Multi-Head continuations against the frozen HMM_v1 F3
Condition 2b (`mae_hmm_k6`) under the same training and downstream conditions.
The experimental factor is the set of ordered prototype heads. This screening
covers F3; selection and subsequent Volve/Parihaka experiments require separate
decisions. Use [RUNBOOK.md](RUNBOOK.md) for execution and recovery.

| Candidate ID | Head K values | Manifest directory |
| --- | --- | --- |
| `mae100_hmm_v2_mh_k46_distill020` | [4, 6] | `mh_k46` |
| `mae100_hmm_v2_mh_k68_distill020` | [6, 8] | `mh_k68` |
| `mae100_hmm_v2_mh_k468_distill020` | [4, 6, 8] | `mh_k468` |
| `mae100_hmm_v2_mh_k6810_distill020` | [6, 8, 10] | `mh_k6810` |

The screening resolver requires exactly these four ID/head-set combinations
and this control. A partial candidate set cannot produce a complete summary.
Each candidate has an independent `multi_head_target_manifest.json`; shared
heads refer to the same targets and all manifests share one embedding identity.

## Fixed scientific conditions

| Component | HMM_v1 Condition 2b conditions retained |
| --- | --- |
| Initialization | F3 survey-specific MAE 100-epoch checkpoint for student and frozen teacher. |
| Continuation | 25 epochs, 10,000 samples/epoch, batch size 16, 15,625 optimizer steps; top encoder block only; seed 42; FP32. |
| Data and optimizer | Existing F3 split, preprocessing, zero mask, model shape and optimizer recipe from experiment 21's MAE100 HMM K6 continuation. |
| Losses | Distillation weight 0.2; prototype and usage losses use the existing equal per-head means; consistency weight 0.0. |
| Targets | Hard Viterbi labels from the existing HMM_v1 MAE100 `overlap_x64` embeddings; no re-extraction. |
| Extraction | Canonical Condition 2b settings: 128-cube windows, overlap 64, batch size 1, minimum valid-token fraction 0.5, AMP disabled. Float16 is the existing storage dtype. |
| Downstream | Experiment 110 v3's decoder, optimizer, supervision, validation mask and evaluation policy; layouts `layout_000`–`layout_004`, sizes `small`, `medium`, `large`, decoder seed 42000. |

Except for K and output paths, clustering retains the entire
[historical K6 configuration](../../facies_benchmark_v1/21_ssl_hmm_continuation_v1/20_hmm_targets/mae100/k6/02_cluster_hmm_k6.yaml):
embedding normalization and token-position residualization, PCA 64 without
whitening, token sample and batch sizes, seed 42, ten HMM iterations, z axis
and direction, edge margin, transition costs, reverse prohibition, max jump 1,
anchors, mean-z initialization and empty-cluster policy. Expected boundaries
remain disabled. Export retains schema v2, confidence 1.0, boundary alpha 0.0
and tau 1.0.

Historical K6 targets are immutable and are consumed directly from
`${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/pseudo_targets/f3/facies_benchmark_v1/ssl_hmm_continuation_v1/mae100/k6`.
The separate K6 replay is solely exact-parity evidence for every manifest;
it never replaces the training target. No copy, rename or symlink is needed.
Manifest construction rejects mismatched token grids, valid masks,
source-target alignment, hashes or K6 replay arrays.

Soft posterior targets, lateral smoothing, XY-neighbor processing,
center-trace masking, periodic refresh and consistency-loss exploration are
excluded. LocalBT and Random are not screening candidates. The canonical
Random embedding metadata and valid mask are read only as the existing
downstream auditor's geometry reference. HMM_v1 definitions, Single-Head
branches and frozen reports remain unchanged.

## Definitions and artifact locations

The numbered directories contain configs for targets, manifests, pretraining,
embeddings, downstream and summary. One-step smoke recipes are feasibility
checks; each full recipe starts independently from MAE100. The source selected
for downstream is the audited full-run `latest.pt` at epoch 25/step 15,625.

All generated targets, manifests, checkpoints, embeddings and results stay
outside Git under `${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}`:

| Artifact | Relative path |
| --- | --- |
| K4/8/10 clustering; K6 replay clustering | `clustering/f3/facies_benchmark_v1/hmm_v2_multi_head_screening_v1/mae100/{k4810,k6_replay}/` |
| Shared targets; parity-only K6 replay | `pseudo_targets/f3/facies_benchmark_v1/hmm_v2_multi_head_screening_v1/mae100/{shared,k6_replay}/k<K>/` |
| Four manifests | `pseudo_targets/f3/facies_benchmark_v1/hmm_v2_multi_head_screening_v1/mh_k<46,68,468,6810>/multi_head_target_manifest.json` |
| Training | `pretraining/f3/facies_benchmark_v1/hmm_v2_multi_head_screening_v1/<candidate>/{smoke_1step,full_25ep}/` |
| Embeddings | `embeddings/f3/facies_benchmark_v2/hmm_v2_multi_head_screening_v1/<candidate>/overlap_x64/` |
| Downstream cells | `f3_lithology_benchmark/hmm_v2_multi_head_screening_v1/runs/model=<candidate>/layout=<layout>/size=<size>/` |
| Candidate runner summaries | `f3_lithology_benchmark/hmm_v2_multi_head_screening_v1/summary/<candidate>/` |
| Paired screening results | `f3_lithology_benchmark/hmm_v2_multi_head_screening_v1/paired_summary/` |
| Execution logs | `f3_lithology_benchmark/hmm_v2_multi_head_screening_v1/logs/` |

## Frozen control and completion criteria

`60_summary/01_paired_screening.yaml` requires the versioned
`control_freeze_receipt`, [hmm_v1_condition_2b_receipt.json](60_summary/hmm_v1_condition_2b_receipt.json).
This audit-only receipt is generated once from the authoritative HMM_v1 freeze
records (`manifest.json`, `cells.csv`, `summary.json`) dated 2026-09-11.
It binds the historical checkpoint SHA-256 and the canonical serialization
of all 15 F3 Condition 2b cells' layout, data size, Mean IoU and Macro F1.
Generation verifies the complete 405-cell freeze and records its file hashes.
Hashes and results must not be entered manually or regenerated to accommodate
a changed live control.

Runtime summary reads the receipt and live artifacts; it does not read the
frozen reports. All metric calculations use the audited live metrics. The
live checkpoint and 15 control cells must match the receipt, in addition to
passing source, decoder and evaluation provenance checks. Canonical historical
cells are read directly from their existing runs root.

Completion requires all four full-source audits, all 60 candidate cells
(four candidates × five layouts × three sizes), and the same 15 frozen
historical cells. Missing or duplicate cells, changed control metrics or
checkpoint bytes, or inconsistent source, training or evaluation evidence
abort summary publication. A successful plan, smoke run or partial stage is
not screening completion.

Mean IoU on unique validation voxels is primary; Macro F1 is secondary.
Positive deltas favor the candidate. By-size statistics use five paired
layouts; across sizes, inference first averages within each layout (n=5).
The pooled 15-cell statistics are descriptive. Comparisons are exploratory,
without automatic winner selection or multiplicity-adjusted significance.

The paired summary atomically publishes exactly:

- `comparison.csv`: 60 audited pairs with `candidate_mean_iou`,
  `control_mean_iou`, `mean_iou_candidate_minus_control`, `candidate_macro_f1`,
  `control_macro_f1`, `macro_f1_candidate_minus_control`, and evidence hashes.
- `summary.json`: paired evidence, source and freeze audits, and statistics.
- `summary.md`: primary by-size screening table and interpretation contract.

The candidate runner's summaries compare against its canonical reference;
use `paired_summary/` for this frozen K6 screening. Existing paired summaries
are never overwritten.

# F3 independent-test audit and proposal v1

- Audit date: 2026-10-20 UTC.
- Status: **complete: 345/345 test cells, 69 five-layout groups, completed 2026-10-20; no training performed**.
- Scope: F3 HMMなし / HMM v1 / HMM v2 only. HMM v3, Parihaka and Volve are excluded from reruns.
- Existing evaluations remain immutable exploratory validation results.
- The initial audit is retained below; [evaluation.yaml](evaluation.yaml) now defines the authorized evaluation-only run.
- Audit and test evaluation are separate phases: no new task scores were calculated before mask/selection agreement.
- Machine-specific absolute paths are deliberately omitted.

## Agreed evaluation criterion

### User clarification on 2026-10-20: current criterion

The user explicitly confirmed that neither figures nor observed results were
used to adjust parameters, and requested the criterion that test data must not
be used to select a model's best epoch. Accept this first-party history
clarification; do not continue to require a completely unexposed spatial block.

The current proposal retains the ordinary additional requirement that the
reported test labels are not decoder-training labels. Unlabelled-amplitude
pretraining, shared encoder context, prior figure generation, and labelled
data use by other historical experiments are not additional exclusion gates
under this clarified criterion. Historical provenance below is still recorded,
but the earlier inability to certify a pristine spatial test is **not a blocker
for this revised evaluation**.

**Existing checkpoints and predictions support the requested revised test
evaluation without new learning.** After confirmation that the split has the
same train/validation/test roles as Parihaka and Volve, the user explicitly
requested execution. The agreed mask is:

`test = reference_valid_voxels AND (canonical_supervision_split_grid == 0)`

Use the current dense-label-section canonical grid, not a method-specific
subsampled training grid. This excludes the complete common training-label
pool as well as validation, so the same test mask is used for all 23 methods,
three budgets and five layouts. The excluded training pool is a conservative
superset of each decoder's actual training labels. The validation masks used
for every best-epoch selection are also excluded.

Read-only mask-geometry counts, not task-performance measurements:

| Mask | Voxels |
|---|---:|
| Reference-valid full-volume voxels | 121,070,400 |
| Canonical train-label pool | 6,620,640 |
| Validation used for best-epoch selection | 470,136 |
| Proposed common test | **113,979,624** |

No test confusion matrix, IoU, F1, prediction-versus-label comparison, or
test-driven checkpoint/model selection was calculated during the initial
audit. The authorized execution writes the fixed mask and source-selection
receipt before scoring and stores new test results separately.

The mask is a within-survey label holdout, not a contiguous spatial block.
It has no new spatial buffer, can be adjacent to supervised sections, and
shares encoder context with them. Full-survey amplitudes and normalization
were used in learned pretraining: report **same-survey transductive evaluation
on labels excluded from decoder training and checkpoint selection**, not
complete spatial holdout or unseen-survey generalization.

### Authorized execution scope

- Encoder training: **0**; embedding extraction: **0**; decoder training: **0**.
- Full-volume inference: **0** if the existing prediction artifacts pass the
  final reuse checks; 345 existing full-volume prediction/valid-mask artifacts
  were found, each naming its own selected decoder best.pt, with exact-once
  coverage and 121,070,400 valid voxels. All 345 actual best.pt SHA-256 values
  match the prediction source identities. Existing evaluation records bind
  the same valid-mask SHA-256 for all 345 cells:
  a1e4ad9b2ea27d10dffa1d91a2c7a64251b198201af6b5167910f977f4d93088.
- New numeric test evaluations: **345**, under a new output namespace.
- Main table: test mean IoU and macro-F1, arithmetic mean plus sample SD
  (ddof=1) across five layouts, separately for each condition and budget.
- Supplement: all 345 rows, confusion matrices, classwise metrics and all
  mask/checkpoint/source identities. Freeze the six-class metric policy and
  all selected checkpoint hashes before scoring. No model, head set,
  consistency setting, epoch or mask changes based on test results.

All old validation scores and artifacts remain exploratory records and are
not overwritten. Agreement authorizes only additional scoring of the common
test, not training, parameter adjustment or test-driven checkpoint selection.

## Evaluation-only runbook

The completed [paper comparison](../../../../reports/hmm_f3_common_test_v1/README.md)
contains mean-IoU and macro-F1 tables with sample SD, and links to all 23
conditions and 345 individual results. It is a review output only, not a
pipeline input. All 345 test cells use 113,979,624 voxels. Summary JSON SHA-256:
`a22b2fb563d9c7ac53446b8168ad2c5c50261559554c6ccc61b2ad9de43610f7`.

The [thin CLI](../../../../proc/seis_ssl_cluster/evaluate_f3_common_test.py)
uses the dedicated [evaluator](../../../../src/seis_ssl_cluster/f3/lithology/common_test.py),
not the legacy validation evaluator with renamed masks. The owning config
pins the inventory, labels, canonical split, reference geometry and class-info
hashes. CPU processing is serial and does not load model weights for inference.

```bash
export SEIS_SSL_CLUSTER_WORKSPACE="$PWD"
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT="$PWD/artifacts/seis_ssl_cluster"
export PYTHONPATH=src

.venv/bin/python proc/seis_ssl_cluster/evaluate_f3_common_test.py \
  --config experiments/f3/facies_benchmark_v2/135_lithology_independent_test_audit_v1/evaluation.yaml \
  --dry-run

.venv/bin/python proc/seis_ssl_cluster/evaluate_f3_common_test.py \
  --config experiments/f3/facies_benchmark_v2/135_lithology_independent_test_audit_v1/evaluation.yaml \
  --execute
```

An existing output requires explicit `--execute --resume`. Resume must retain
the same selection receipt, test-mask hash, inputs and cached metrics. It does
not choose a different epoch or overwrite completed cells/summary.

The new artifact namespace is
`surveys/f3/facies_benchmark_v2/evaluation/lithology_common_test_v1/`:

- `selection_receipt.json`: all 345 preselected decoder/source identities,
  bound before scoring, and freeze time.
- `inputs/test_mask.npy`, `inputs/test_definition.json`: common test mask,
  rule, count, input hashes and zero train/validation-overlap assertion.
- `cells/<condition>/<layout>/<size>/metrics.json` and `evidence.json`: exactly
  two files per cell; complete six-class confusion matrix and derived metrics,
  test-mask hash and source identities.
- `status.json` and `.run.lock`: execution state and duplicate-run protection.
- `summary/`: exactly **comparison.csv, per_layout.csv, class_metrics.csv,
  confusion_matrices.csv, summary.json, summary.md**. These are the producer's
  six published review files; no prediction volumes/checkpoints are published.

Every prediction and valid-mask NPY is hashed during its scoring pass and
checked against the historical source evidence. Missing test coverage or
unknown classes fail rather than shrinking the test mask. Fixed six-class
IoU/F1 preserve the existing zero-denominator policy (zero for absent classes).

Validation before the first live score: read-only 345-cell preflight; 54
focused/new-and-existing regression tests; Ruff lint and formatting;
package-isolation check. The first execution used evaluator source SHA-256
`7deb25049ae950fe2b3cfe6c479965e0031b91ce8969091a2f106cdb42b41c8d`.
Regression coverage includes mask exclusion, unknown labels, missing coverage,
input/checkpoint drift, sample-SD aggregation, duplicate/missing layouts,
no-overwrite/resume behavior, and exact cell/summary publication file sets.

## Existing condition inventory

[condition_inventory.csv](condition_inventory.csv) enumerates all 23 existing
representation conditions, with checkpoint SHA-256, embedding-metadata SHA-256,
artifact-relative source locations, and downstream run roots. These are source
identity bindings, not selected results. Every condition has five layouts times
three label budgets.

| Version | Heads | Distillation | Consistency | Existing sources | Conditions | Cells |
|---|---|---:|---:|---|---:|---:|
| HMMなし | none | n/a | n/a | MAE / LocalBT / random | 3 | 45 |
| v1 | K6 | 0.1 | 0 | MAE / LocalBT / random | 3 | 45 |
| v1 | K6 | 0.2 | 0 | MAE / LocalBT / random | 3 | 45 |
| v2 | K4,6 | 0.2 | 0 | MAE only | 1 | 15 |
| v2 | K6,8 | 0.2 | 0 | MAE only | 1 | 15 |
| v2 | K4,6,8 | 0.2 | 0 | MAE / LocalBT / random | 3 | 45 |
| v2 | K4,6,8 | 0.2 | 0.1 | MAE / LocalBT / random | 3 | 45 |
| v2 | K6,8,10 | 0.2 | 0 | MAE / LocalBT / random | 3 | 45 |
| v2 canonical | K6,8,10 | 0.2 | 0.1 | MAE / LocalBT / random | 3 | 45 |
| Total | | | | | **23** | **345** |

MAE has 9 conditions / 135 cells; LocalBT and random each have 7 / 105.
The common three-source matrix is 21 conditions / 315 cells. MAE-only
two-head screenings add 2 / 30. MAE K468 and K6810 zero-consistency cells
reused by multi-source summaries are counted once, not as new experiments.

LocalBT/random K46 and K68 are not existing completed conditions. Requiring
all three sources for those two head configurations would add four encoder
conditions and 60 downstream cells: 27 conditions / 405 cells total.
That is a scope extension requiring agreement, not an automatic completion
of this audit. No two-head consistency-0.1 arms are implied.

Definitions and completed evidence:

- [v1 frozen definitions](../../../../reports/hmm_v1/README.md).
- [MAE four-head-set screening](../127_hmm_v2_multi_head_screening_v1/README.md).
- [K6810 multi-source](../128_hmm_v2_k6810_multi_source_v1/RUNBOOK.md).
- [K468 multi-source](../129_hmm_v2_k468_multi_source_v1/RUNBOOK.md).
- [K468 consistency 0.1](../130_hmm_v2_k468_cons010_multi_source_v1/RUNBOOK.md).
- [K6810 consistency 0.1](../131_hmm_v2_k6810_cons010_multi_source_v1/RUNBOOK.md).

HMMなし MAE means **MAE100 + MAE25**, not MAE100 alone. LocalBT means
the chosen asymmetric rotation/noise LocalBT3 source, not the historical
LocalBT100 + LocalBT25 source. Random is the fixed seed-42 initialization.

## Data and label-use history

### Current benchmark

The prepared cube is [601, 901, 255] in inline/crossline/sample order:
inline 100–700 and crossline 300–1200, inclusive; sampling interval is 5 ms.
The token grid is [76, 113, 32] at patch size [8, 8, 8].

The current [section inventory](../10_prepare/section_inventory_v2.csv) has:

- Train: 20 inlines, 250–630 in steps of 20.
- Train: 20 crosslines, 450–625 and 850–1125 in steps of 25.
- Validation: inline 150, crosslines 350 and 750.
- Intersecting train/validation voxels use validation precedence.

The canonical train pool has 6,620,640 valid train voxels and validation has
470,136 unique valid voxels. Individual layout/budget training sets subsample
this pool, not the complete pool. The [class-balanced layout definition](../109_f3_voxel_section_layout_v3/README.md)
uses fixed seeds 0–4, nested 25/50/100 token-row caps per class and nominal
9,600/19,200/38,400 supervised-voxel budgets (actual counts have a 5% tolerance).
Label availability/class composition was inspected for this training-pool
calibration. Train line names alone are not a complete history of label access.

### Earlier work matters too

A read-only search found **83 split_manifest.json records, seven distinct
line-assignment patterns**, in the F3 survey artifact tree. Some are repeated
prepared-data definitions; existence alone does not prove each was trained.
The six-split label-budget experiment additionally has **36 completed metrics
artifacts**, so its rotated train/validation usage cannot be ignored.

The conservative union of inventoried supervised line roles is:

- Inlines: 150; 250–630 in steps of 20; 650.
- Crosslines: 350; 450–625 in steps of 25; 650; 750;
  850–1125 in steps of 25; 1150.

The six-split history assigns inline 150 and crosslines 350/750 to training
in some splits. Those lines cannot be considered untouched simply because
the current benchmark calls them validation.

Earlier inventories use five train inlines and seven train crosslines.
The older voxel and token/probe workflows, class-balanced training-pool
calibration, and qualitative comparison products must all remain part of
a future candidate-region exclusion audit; the union above is a conservative
line inventory, not an exact per-voxel gradient-use mask.

### Qualitative exposure

An existing HMM-v1 paper comparison, generated 2026-09-11, contains amplitude,
ground truth, and predictions for six methods on **30 sections**:

- Inline numbers: 130, 190, 249, 310, 370, 430, 490, 549, 610, 670.
- Crossline numbers: 345, 435, 525, 615, 705, 795, 885, 975, 1065, 1155.
- Horizontal sample indices: 16, 38, 63, 89, 114, 140, 165, 190, 216, 239.

Horizontal comparison figures cover inline 100–699 and crossline 300–1195.
Thus a full-depth XY block can intersect prior comparative figures even
when it avoids all named training/validation vertical lines. Orthogonal
ground-truth quicklooks also exist, including sample 127.

Generating a figure, validating label values, or computing class availability
is **not automatically** performance-informed tuning. The initial audit could
not resolve human use from artifacts alone. The user's subsequent explicit
confirmation that figures/results were not used to tune parameters resolves
that question for the current criterion; figure generation is not treated as
disqualifying exposure. No claim of a wholly unexposed spatial test is made.

The raw F3 root inventory contains one seismic SEG-Y, one label SEG-Y, the
15 labelled PNG sections, and reference_osv attribute files. No separate
untouched labelled test cube was established by this audit. This is scoped
to the configured F3 assets; it does not assert that additional F3 data
cannot exist elsewhere.

## Checkpoint selection audit

All **345** best.pt files were loaded on CPU and checked against their complete
50-epoch history. The existing deterministic rule is:

1. Higher validation macro-F1.
2. If equal within epsilon 1e-12, higher validation mean IoU.
3. If still equal, lower validation weighted cross entropy.
4. If still tied, retain the earlier checkpoint.

Selected zero-based epochs range from 1 to 49. All histories have 50 rows,
finish at epoch 49 / global step 22,000, and use 440 steps/epoch.
Do not describe the old rule as simply "maximum validation IoU."

All 345 final evaluation metadata records report unique_validation_voxel,
split code 2, 470,136 voxels. Training-time selection and final reporting
therefore use the same labelled validation region, even though inference
tiling can make training-time and final metric values differ.

The train/decoder/tile configuration is identical across all 345 cells:
frozen encoder; 384-dimensional embeddings; decoder hidden channels
[128,64,32]; nearest upsampling; voxelwise LayerNorm; 50 epochs; batch 1;
LR 1e-3; weight decay 1e-4; balanced CE from train counts; AMP on;
gradient clip 1; 440 steps/epoch; decoder seed 42000.
For each of the 15 layout/budget pairs, train and validation tile-manifest
identities agree across all 23 methods.

## Unlabelled amplitude and spatial context

Learned source encoders use a manifest covering the complete [601,901,255]
F3 amplitude cube, without a spatial ROI exclusion. HMM fitting and
pseudo-target extraction likewise use full-survey embeddings. Source
checkpoints are fixed-budget latest.pt endpoints; representation recipes
have historical downstream validation measurements. The user has confirmed
that observed results were not used to adjust parameters; the existence of
those measurements is not treated as evidence of result-driven tuning.

If a test region is chosen inside this cube, those encoders have used its
unlabelled amplitudes in pretraining. This supports **within-survey
transductive evaluation with held-out downstream labels**, not an inductive
spatial holdout. The no-HMM random encoder has no learned pretraining weights,
but its extraction still uses the same full-volume contextual geometry.

Full-volume amplitude-normalization statistics also include this cube.
They may be explicitly retained for a transductive protocol, but cannot
be described as train-only statistics in a complete spatial-holdout claim.

Existing embeddings use 128-cube encoder windows, overlap 64 and count-averaged
overlapping token features. Decoder cores span 8 tokens / 64 voxels per axis
with one-token / 8-voxel input halos. Separating voxel labels or slice numbers
does not establish disjoint contextual inputs.

A conservative geometry-only calculation using the 15 canonical MAE decoder
train/validation tile manifests finds:

- 274,816 token-grid positions in the padded grid.
- 236,322 positions in the train decoder-input union.
- 73,280 in the validation decoder-input union.
- 258,414 in their combined union.
- All 378 encoder extraction windows intersect that combined input union;
  their contextual envelope covers all 274,816 positions.

The same manifests apply to the other 22 methods. This calculation is a
potential-context envelope, not evidence that every amplitude affected
a gradient or that every location's performance was inspected.

See the [same-survey claim boundary](../../../../docs/same_survey_pretraining_claims.md).

## Earlier strict spatial proposal (superseded; retained as audit history)

The following stricter design and the conditional retraining counts in the
next section were prepared before the user's criterion clarification. They
are **not** the active execution requirements. The active no-retraining
protocol is specified under "Agreed evaluation criterion" above. They remain here only
to explain what would be needed if a complete spatial-holdout claim were
requested later.

1. Establish a candidate-specific exposure ledger, including historical
   supervised masks, token-row support, evaluation masks, qualitative figure
   coordinates and any human performance-informed use. Record uncertainties.
   A region's name or an unused PNG filename is not sufficient evidence.
2. Designate contiguous full-depth XY test block(s), preferably separate from
   the development area. Reject previously used labels/comparative regions
   under the agreed history criterion. Do not choose blocks using model scores.
   If no adequate block is eligible, stop and request additional unseen
   labelled F3 data; do not relabel existing validation.
3. Keep old train/validation definitions only if their labels and contextual
   supports satisfy the new exclusion contract. Otherwise create new versioned
   development masks and five layouts. Restrict class-balanced budget selection
   to train-only pools; keep 25/50/100 row caps per class and the 5% voxel-count
   tolerance if feasible. Infeasible pools require agreement, not test-driven
   repairs. Use the same masks for every method.
4. Use an explicit spatial exclusion/guard region. Its extent is derived from
   actual encoder-window support, overlap merging, decoder input halos and
   preprocessing, not an arbitrary one-token or one-line gap.
   For context-isolated inference, crop/re-extract per partition, pad from
   that partition only, and reject any window/tile crossing partition boundaries.
   Score only fixed eligible cores. Verify train/validation/test label masks
   and the agreed contextual footprints are disjoint. Freeze buffer coordinates
   and hashes with the split; no physical-distance claim without coordinate
   spacing information.
5. Select decoder best checkpoints on validation only using the existing
   F1/IoU/CE rule and fixed budget. Freeze all source/model settings and all
   selected checkpoint hashes before opening the test scoreboard.
6. Evaluate each selected checkpoint once on the same test mask. No
   test-dependent epochs, postprocessing, class weights, model/head/consistency
   selection, or selection among different test partitions.
7. Report test mean IoU and macro-F1 over unique test voxels under a frozen
   six-class/undefined-class policy. The main table is the arithmetic mean
   plus **sample SD (ddof=1)** across five layouts for each model/budget.
   Save all 345 individual rows, confusion matrices, classwise metrics, split
   and source hashes, and coverage checks. Five layouts are not five training
   seeds or five independent spatial test regions. Nested budgets are not
   independent replicates.

This proposal concerns label and context isolation. A complete amplitude
holdout additionally needs train-only preprocessing fitting, SSL crop
sampling, clustering/PCA/HMM fitting and pseudo-target generation. Reusing
old full-cube learned encoders does not satisfy that stronger protocol.

## Source audit and earlier conditional work counts

All 23 source checkpoint bytes were SHA-256 checked against their embedding
metadata; the nine v1/no-HMM source hashes also match the frozen v1 record.
All 23 embedding arrays and valid-token arrays are present with the shared
recorded extraction geometry. Bulk embedding array hashes were not recomputed
during this audit; verify them and the new partition compatibility before reuse.

| Agreed situation | Encoder training stages | Decoder fits | New test cell evaluations |
|---|---:|---:|---:|
| Genuinely unseen additional F3 test region; existing development and checkpoint selection unchanged and eligible | 0 | 0 | 345 |
| Eligible within-cube label test; new purged development/context partition, explicitly transductive | 0 | 345 | 345 |
| Complete spatial amplitude holdout requiring fresh encoder pipeline | 23 learned training stages, plus random initialization | 345 | 345 |

For the first row, new test embeddings for up to 23 representations and
345 inference/evaluation runs are needed; old full-cube predictions do
not cover additional data. Old development embeddings/decoders can be reused
only after source, coordinate, label-history and context checks pass.

For the second row, old encoder weights may be retained, but changed
context geometry requires up to 23 new partition-aware embedding sets and
all 345 decoders must start fresh rather than resume old checkpoints.

The complete-holdout training-stage count decomposes into MAE100 foundation
(1), final MAE continuation/HMM variants (9), LocalBT3 and its HMM variants
(7), and random-initialized HMM variants (6): 23 learned stages.
There are 23 final representation conditions including the untrained
random condition; this does not mean 23 independent MAE100 foundations.
Targets and normalization also need regeneration. Retraining alone does
not retroactively erase prior test-region performance exposure.

These are conditional execution counts, not authorization to run.
**Actual new training runs in this audit: 0; new test metric evaluations: 0.**

## Preservation and implementation gate

Existing reports, frozen receipts, source checkpoints, embeddings, decoder
checkpoints, predictions and summaries were not changed. New independent-test
outputs must use a separate experiment namespace and distinct split identifiers.

The current split builder exposes only unsupervised=0, train=1, validation=2;
the legacy evaluator hardcodes validation and its filenames/aggregation.
The active proposal requires a separate explicit-test-mask evaluation route
with focused regression coverage before scoring. It need not rebuild or
retrain the existing datasets/decoders. Never pass a test mask under the name
"validation" to reuse the old evaluator. Check decoder-training and selected
validation masks for zero overlap with test and validate prediction coverage.

A later producer must define and test its exact published file set under the
[report policy](../../../../docs/report_sharing_policy.md). This audit adds
only an experiment README and the condition inventory; it adds no producer.

Recorded agreement for execution: the canonical-split-zero common test mask;
23 existing conditions / 345 cells; reuse of fixed best checkpoints and
existing full-volume predictions; and the explicitly transductive,
non-spatial-block interpretation. The optional 27-condition expansion is not
part of this execution.

## Audit evidence locations

Paths in this table are relative to the artifact root, except the explicitly
marked repository input. They describe historical evidence, not new pipeline
inputs. Decoder evidence is located below every run root in the inventory:
`layout=<id>/size=<budget>/decoder/{best.pt,history.csv,resolved_config.json,
run_metadata.json,train_tile_manifest.json,validation_tile_manifest.json}`.

| Evidence | Location | SHA-256 |
|---|---|---|
| Current inventory, repository input | `experiments/f3/facies_benchmark_v2/10_prepare/section_inventory_v2.csv` | `dd218dd94db06440c5e3a59a59167e9c9c8db327970cb5fc411070065f738c20` |
| Current split | `surveys/f3/facies_benchmark_v2/data/lithology/voxel_supervision/dense_label_sections_v2/split_manifest.json` | `7f1bf536cee1e4b6efbc95ab0b8a6fd8f6357b387df2086b984ea5d25adfa924` |
| Example rotated split assigning current validation lines to train | `surveys/f3/facies_benchmark_v1/evaluation/lithology/voxel_label_budget_multi_head_k6810_six_split_v1/datasets/split_004/cap50/voxel_supervision/split_manifest.json` | `db692ea671e091caeec2d6e53013cb85f932677dd3051f74b98f4d7cc8ac60fc` |
| Comparative figure coordinates | `paper_figures/hmm_v1/f3/layout_000/large/selected_slices.csv` | `db178c924b7c2bb578330d1b6ea1dff682f303f99198bff50c2d25824c78e4c6` |
| Comparative figure provenance and display extents | `paper_figures/hmm_v1/f3/layout_000/large/manifest.json` | `a60d02723669c12c05a8e14b63d3a1db52738f4fe37eb27f67357455ca001624` |

Selection implementation: [checkpoint.py](../../../../src/seis_ssl_cluster/training/voxel_decoder/checkpoint.py).
Legacy split implementation: [voxel_split.py](../../../../src/seis_ssl_cluster/f3/lithology/voxel_split.py).
Legacy final evaluation: [voxel_evaluation.py](../../../../src/seis_ssl_cluster/f3/lithology/voxel_evaluation.py).
Embedding context: [sliding_window.py](../../../../src/seis_ssl_cluster/embedding/sliding_window.py)
and [merge.py](../../../../src/seis_ssl_cluster/embedding/merge.py).

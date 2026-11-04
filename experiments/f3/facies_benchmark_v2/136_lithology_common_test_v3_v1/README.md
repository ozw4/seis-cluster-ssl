# F3 v3: additional common-test evaluation

Authorized on 2026-10-21 after completion of the
[HMMなし/v1/v2 test evaluation](../135_lithology_independent_test_audit_v1/README.md).
This extension scores existing validation-selected decoders only. It does not
resume stopped training, select a new checkpoint, or replace exploratory
validation results or the previous common-test outputs.

**Completed 2026-10-21: 135/135 cells, 27 five-layout groups.**
The [curated comparison](../../../../reports/hmm_f3_common_test_v3_v1/README.md)
retains all nine conditions and all individual values; it is not a pipeline input.
Run identity: `c273a0077fd9d56c9991ba8247d3b038292a29fdb1af15865e4d7ff7c1eb1dee`.
Summary JSON SHA-256:
`a331533437544353fd6ff8ecb26116462d197438e418319156894d9cdabcef1e`.

**Later final-result designation (2026-10-21):** the user selected decoder
seed 42001 for LocalBT / v3 / mask-token LR=3e-5 / 50:50, all 15 layout/budget
cells. Use [the explicit selection record](../137_local_bt_v3_masklr3em5_decoder_seed42001_v1/selection.yaml)
for future comparisons of that exact condition. This completed 135-cell,
seed-42000 source aggregate remains unchanged as a historical evaluation;
all other conditions keep their existing result bindings.

## Fixed scope

The [inventory](condition_inventory.csv) pins all nine completed F3 v3
representation conditions, their encoder and embedding-metadata hashes,
original training recipes, and existing downstream run roots.

| Source | Mask-token LR | Masked:visible | Conditions | Test cells |
|---|---|---|---:|---:|
| MAE / LocalBT / random | 1e-5 | 50:50 | 3 | 45 |
| MAE / LocalBT / random | 3e-5 | 50:50 | 3 | 45 |
| MAE only | 1e-4 | 50:50 | 1 | 15 |
| MAE only | 3e-5 | 25:75 | 1 | 15 |
| MAE only | 3e-5 | 75:25 | 1 | 15 |
| Total | | | **9** | **135** |

Each condition has small/medium/large budgets and five paired layouts.
K=[6,8,10], distillation=0.2, consistency=0.1, trace-column masking=10%,
and encoder/head LR=1e-5 throughout. Each continuation was trained for 25
epochs from its original source parent. The frozen epoch-25 encoder and
the original decoder best.pt are reused; test scores never select epochs.

Historical definitions:
[base trace masking](../132_hmm_v3_k6810_trace_mask_multi_source_v1/README.md),
[mask-token LR](../133_hmm_v3_mask_token_lr_sweep_v1/README.md), and
[branch-weight sweep](../134_hmm_v3_masked_visible_weight_sweep_v1/README.md).
LocalBT/random LR=1e-4 and both branch-weight arms have no completed
downstream cells. They are explicitly excluded, not restarted or represented
as zero-valued results. Parihaka and Volve are outside this run.

## Identical test definition and controls

[evaluation.yaml](evaluation.yaml) retains the agreed mask:

`test = reference_valid_voxels AND (canonical_supervision_split_grid == 0)`

All **113,979,624** test voxels must have prediction coverage for every cell.
The complete canonical training-label pool and validation used to select
decoder best epochs are excluded. The 15 layout/budget supervision grids
are checked for zero test overlap before scoring. The reproduced mask must
match the prior test-mask SHA-256 exactly:
`2bac34f4ebf8612d120673ae3dbda1a8720624e27a1701491059abf1c60cfe1d`.

Decoder conditions remain seed=42000, 50 epochs, 22,000 steps and the
existing validation-selected best checkpoint, paired with the same label
layouts/budgets as HMMなし/v1/v2. Six-class metrics use a single confusion
matrix per entire test volume, including the existing absent-class policy.
Summaries report arithmetic mean and sample SD (ddof=1) across five layouts;
all 135 individual results are retained. No test-driven ranking or recipe
selection is performed.

Unlabelled full-survey amplitudes and shared encoder context are retained.
This remains **same-survey transductive downstream-label holdout**, without a
spatial buffer, not complete spatial holdout or unseen-survey transfer.

## Execution

The existing [CLI](../../../../proc/seis_ssl_cluster/evaluate_f3_common_test.py)
and [evaluator](../../../../src/seis_ssl_cluster/f3/lithology/common_test.py)
support an explicit, separately validated nine-condition v3 profile. The
original 23-condition profile and its output namespace remain unchanged.

```bash
export SEIS_SSL_CLUSTER_WORKSPACE="$PWD"
export SEIS_SSL_CLUSTER_ARTIFACT_ROOT="$PWD/artifacts/seis_ssl_cluster"
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

.venv/bin/python proc/seis_ssl_cluster/evaluate_f3_common_test.py \
  --config experiments/f3/facies_benchmark_v2/136_lithology_common_test_v3_v1/evaluation.yaml \
  --dry-run

.venv/bin/python -u proc/seis_ssl_cluster/evaluate_f3_common_test.py \
  --config experiments/f3/facies_benchmark_v2/136_lithology_common_test_v3_v1/evaluation.yaml \
  --execute
```

The run is serial on CPU: **zero training, zero embedding extraction,
zero new inference; 135 additional numeric evaluations**. Before scoring,
all selected source hashes and test inputs are frozen in the selection receipt.
Prediction/coverage hashes are checked while scoring; mismatches and missing
coverage fail instead of changing the mask. Existing outputs require explicit
identity-checked `--execute --resume`; completed cells are not overwritten.

The separate artifact namespace is
`surveys/f3/facies_benchmark_v2/evaluation/lithology_common_test_v3_v1/`.
It owns `selection_receipt.json`, `inputs/`, `cells/`, `status.json`, and a
lock. Each cell publishes exactly `metrics.json` and `evidence.json`.
The summary publishes exactly six review files: **comparison.csv,
per_layout.csv, class_metrics.csv, confusion_matrices.csv, summary.json,
summary.md**. CSV/JSON retain mask-token LR and branch weights explicitly.
Reports are review-only copies, never execution inputs.

Focused regression coverage includes both fixed condition profiles,
v3 parameter identity, exact prior-mask equality, v3 parameter preservation
in summaries, original no-overwrite/resume guarantees and exact published
file sets. Existing confusion/split regression coverage is reused.

Before execution: 135-cell dry-run, **58 passing related regression tests**,
Ruff lint/format and package-isolation validation. The old 23-condition dry-run
still produces run identity
`22e45c6629d606c3eb9885ea2be7471ba7dd8dcf333899e7f2621feec0234519`.
Execution used evaluator SHA-256
`79564130e43cd314d1ec718596511425b850c63b1f2892aeaa69c75a1a68629e`.
After execution, all 27 mean/sample-SD groups were independently recomputed
from 135 individual rows, all fixed test counts and exact publication file
sets were checked, and the original HMMなし/v1/v2 summary hash was unchanged.
All 135 supervision-grid hashes and decoder/train/tiles configurations match
the earlier comparison. An independent existing generic confusion kernel
recomputed MAE/base-v3/layout_000/large on all test voxels and matched exactly.

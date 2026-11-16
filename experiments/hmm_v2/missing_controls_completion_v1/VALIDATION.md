# Completion and recovery

All 42 missing decoder cells completed on 2026-11-15, with the three existing
Barlow layout_000/large cells reused. F3 common-test evaluation also completed:
45 v1 zero-distillation cells and 30 Barlow v2 cells (including 15 rescored
0.2 controls). The original 0.2 control values matched exactly.

The detached worker stopped at final aggregation at 21:30 UTC: a manifest path
string was passed to the Path-only CSV reader. This was a postprocessing error;
training, selected checkpoints and evaluations had already completed. The
collector now converts the path explicitly. Its regression fixture enforces
the real reader's Path contract so the same mismatch cannot be masked again.
All 25 focused completion/common-test tests pass, together with Ruff lint and
format checks. Only summary generation was rerun; no model or score was rerun.

The exact ten-file summary is now published. Independent numerical verification
checked 90 Barlow cells / 18 means, 405 aligned v1 cells / 81 means, and 540
factorial cells / 108 means. Every group has five unique layouts; all values are
finite. Means and sample SDs were checked for both principal/secondary metrics
(414 group-metric checks), as were all 135 paired contrasts, to 1e-12 tolerance.
The recovery receipt records the output hashes and corrected module hash.

Evidence remains under
`artifacts/seis_ssl_cluster/operations/hmm_v2/missing_controls_completion_v1/`:
`complete.json`, `recovery_complete.json`, `summary_recovery.log`, and the
original failed `exit.json` / `summary_failure_status.json`. The historical
failure records are retained; the current `status.json` records completion.
Results are under the matching artifact study's `summary/` directory.

Barlow Twins' descriptive 15-cell mean favors zero distillation on each survey:

| Survey | Metric | Weight 0 | Weight 0.2 |
| --- | --- | ---: | ---: |
| F3 | Common-test mean IoU | 0.453518 | 0.447653 |
| Parihaka | Test channel IoU | 0.171067 | 0.166576 |
| Volve | Test horizon MAE [samples] | 11.555585 | 12.498038 |

These means span five layouts and three nested budgets. They are descriptive,
not 15 independent encoder repetitions. The fixed continuation seed is 42.

## Initial paired HMM/no-HMM CSV update (2026-11-16; superseded below)

The 540-row review now includes the matched no-HMM score and signed improvement
for both metrics. The independent 135 no-HMM evaluations are reused four times,
with decoder seed 42000 throughout. No checkpoint, training, or evaluation was
rerun. The original 540 HMM scores match bit for bit; the derived outputs are
separate from the immutable original artifact summary.

Verification checked all 675 input file hashes, all 540 paired differences,
and 648 group/metric mean-and-SD pairs across 108 groups to 1e-12. Volve input
valid-token/lateral-mask hashes and output-validity policies also matched in
every pair. Seed, split and full decoder/support identities are checked before
joining. Focused tests cover improvement directions, paired SDs, missing or
mismatched controls, recorded model identities, and exact five/seven-file
artifact/review publication. The numerical receipt is in the operation's
`no_hmm_comparison_20261116/numerical_validation.json`.


## Retain all baseline seeds (2026-11-16; expanded below)

The initial paired table incorrectly omitted the five Parihaka MAE/large
seed-42001 results. The current CSVs overwrite that derived table and retain
both seeds: 560 comparison rows, 140 distinct no-HMM evaluations and 112
five-layout summaries. All original 540 comparisons and their numerical values
remain present. The additional 20 comparisons use the five seed-42001 baselines
with each of the four HMM loss settings. Both seeds, same/cross-seed status and
canonical/historical reference status are explicit; seeds are never pooled.
The existing experiment-47 selection designates 42001 as canonical and
preserves 42000 as historical for Parihaka MAE/large.

Independent verification checked all 680 input hashes, all 560 signed paired
differences and 672 mean/SD checks across 112 groups. The five seed-42001
source hashes match the canonical selection record. Original HMM-only summary
bytes and the three other review comparison tables remain unchanged. No model
training or evaluation was rerun.

The full saved-experiment CSV/HTML catalog was also refreshed at its existing
artifact path. It contains 6,898 split/support evaluation rows and 436,242
scalar endpoint values. All previous 6,739 rows and 423,970 scalar values remain
present with unchanged metrics. Both decoder seeds include all five layouts
and both validation/test splits. Plain MAE under a directory whose name
contains `hmm` is correctly identified from its actual model metadata as
no-HMM. Manifest membership, published hashes and ZIP integrity are checked.

Validation: 15 comparison/catalog tests and 77 related completion, common-test
and namespace tests passed, with Ruff lint/format and package isolation checks.
Playwright verified all 6,898 HTML rows, both seeds across all five test layouts,
seed filtering, visible seed columns and filtered CSV downloads, with no
JavaScript errors.
Numerical receipts are under the operation's
`all_seeds_20261116/numerical_validation.json` and `catalog_validation.json`.
The refreshed full catalog is
`studies/all_experiment_metrics_20261115_v1/catalog/` within the artifact root;
its sibling `catalog.zip` is overwritten as well.


## All HMM versions in the main CSV (2026-11-16)

The earlier 560-row table still restricted the main comparison to v2. The
current file at the same path contains 4,754 rows covering all 4,689 saved HMM
evaluation identities in the complete catalog. Those identities comprise
1,696 v1, 2,205 v2, 525 v3 and 263 historical evaluations. Repeated comparison
rows retain every compatible baseline seed. No version, seed, split, support,
checkpoint selection or archived HMM endpoint is dropped from this inventory.

All 405 aligned v1 reference cells are present with bit-identical principal and
secondary scores and a verified no-HMM comparison, including distillation
weights 0, 0.1 and 0.2 on all three surveys. The original 560 v2 comparisons also
retain their original scores and improvements. Actual recorded HMM decoder
seeds include 42 and 42000 through 42004; unknown seeds remain empty.

There are 1,785 verified paired rows and 2,969 unpaired rows. Unpaired results
retain their measured HMM scores with empty baseline/delta fields and an
explicit reason; validation is not compared with test. The 369 aggregate rows
separate version, K, model, cohort, selection, split, support and seeds. Partial
layout coverage and duplicate-layout conflicts are explicit; n=1 SDs are empty.

Early v1 tags without `hmm_v1` now use their recorded single-head training
schema or evaluation manifest for version identification. Volve v1 zero-loss
exports omit HMM flags, so their actual encoder training configs are checked.
The list of checkpoint policies available in a Volve exporter is descriptive;
it does not change the selected score or scientific decoder/support identity.

Independent checks cover every saved HMM catalog identity, all 405 v1 reference
cells, all 560 original comparisons, all 1,785 signed paired differences,
2,214 aggregate metric checks and 3,589 original-source hashes. Immutable HMM
summary inputs and the other three comparison tables remain unchanged.
The five-file artifact and seven-file review publication contracts are retained.

The full CSV/HTML catalog retains all 6,898 evaluation rows and metric values;
HTML now exposes 1,696 rows under HMM v1, including the nonzero distillation
settings previously shown as legacy. Playwright checked that filter and the
Parihaka v1 test CSV download without JavaScript errors. The overwritten ZIP
also includes the updated 4,754-row comparison as a copied source table.

Validation: 22 comparison/catalog tests and 77 related completion, common-test
and namespace tests passed, plus Ruff lint/format and package isolation.
Evidence is under `operations/hmm_v2/missing_controls_completion_v1/` in
`all_versions_20261116/`: `verify.py`, `numerical_validation.json`,
`browser_validation.json`, and `publication.json`. No training, prediction or
scientific evaluation was rerun.

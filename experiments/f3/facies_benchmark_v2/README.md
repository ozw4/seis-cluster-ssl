# F3 facies benchmark v2

This namespace evaluates F3 lithology representations with supervision budgets
defined by nested inline and crossline sections. The prepared dataset identity
remains `facies_benchmark_v2`; layout and comparison version names identify
later experiment generations rather than new prepared datasets.

The current canonical five-way workflow uses the class-balanced nested v3
layout:

1. [Build the v3 section layout](109_f3_voxel_section_layout_v3/README.md).
2. [Run and summarize the v3 five-way comparison](110_lithology_mae_local_bt_five_way_v3/README.md).

The positional layout and downstream comparison under the `v2`-named `109` and
`110` directories are historical. The `110` directory still owns prepared-data
and encoder-source phases reused by v3; its README marks that shared subset
explicitly. New downstream runs use the v3 workflow.

The `111`–`114` Local Barlow Twins view screens are also historical; their
retained YAML and linked reports preserve the record. Use the current candidate
workflow for new screens.

Experiments `119`–`123` share one
[Local Barlow Twins candidate runbook](LOCAL_BT_CANDIDATE_RUNBOOK.md).

The retained [Random self-target HMM condition](125_lithology_random_init_self_target_hmm_k6_distill010_v1/README.md)
and [nine-condition completion definitions](126_pretraining_comparison_completion_v1/README.md)
are frozen as part of [HMM_v1](../../../reports/hmm_v1/README.md).

Concrete paths, model settings, and producer commands belong to the linked
experiment YAML and runbooks rather than this index.

## Scoped final-result designation

On 2026-10-21 the user designated **decoder seed 42001** as the final result
for **F3 / LocalBT / HMM v3 / mask-token LR=3e-5 / masked:visible=50:50**,
covering small/medium/large and all five layouts (15 cells).
The [decision and immutable result bindings](137_local_bt_v3_masklr3em5_decoder_seed42001_v1/selection.yaml)
and [runbook](137_local_bt_v3_masklr3em5_decoder_seed42001_v1/README.md)
are authoritative for future comparisons of that condition. Seed 42000 remains
historical; other models, v3 settings, HMMなし/v1/v2 and other surveys are
unaffected. This later user designation does not change the canonical
five-way training defaults or rewrite existing aggregates.

All active v2-namespace runbooks use the same repository-root environment:

```bash
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?export artifact root first}"
: "${F3_ROOT:?export F3 data root first}"
: "${SEIS_SSL_CLUSTER_WORKSPACE:?export repository root first}"
```

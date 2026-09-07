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

The `111`–`114` Local Barlow Twins view screens are also historical. Their
experiment-specific training and reporting programs were removed; the retained
YAML and linked reports preserve the record. Use the current candidate workflow
for new screens.

Experiments `119`–`123` share one
[Local Barlow Twins candidate runbook](LOCAL_BT_CANDIDATE_RUNBOOK.md).

Experiment [`124`](124_lithology_random_init_hmm_k6_distill010_v1/README.md) is a
random-init HMM-K6 Stage 2 control evaluated on the v3 five-way cells through
the candidate path; it is not a member of the fixed five-way model set.

Concrete paths, model settings, and producer commands belong to the linked
experiment YAML and runbooks rather than this index.

All active v2-namespace runbooks use the same repository-root environment:

```bash
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?export artifact root first}"
: "${F3_ROOT:?export F3 data root first}"
: "${SEIS_SSL_CLUSTER_WORKSPACE:?export repository root first}"
```

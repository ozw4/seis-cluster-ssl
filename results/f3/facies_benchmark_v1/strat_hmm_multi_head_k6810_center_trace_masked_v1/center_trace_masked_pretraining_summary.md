# F3 center-trace masked pretraining review

- Status: `PASS`
- Model tag: `strat_hmm_pretext_mh_k6810_ctmask010_nocons_topblock1_distill_v1`
- Variant: `ctmask010_nocons`
- Execution Git SHA: `9e5371de9944026b7e1e4a1e0f78490508773cec`
- Execution dirty status: `['M src/seis_ssl_cluster/embedding/extractor.py', ' M src/seis_ssl_cluster/f3/center_trace_masked_pretraining_validation.py', ' M src/seis_ssl_cluster/training/strat_hmm/runner.py', ' M src/seis_ssl_cluster/training/strat_hmm/runtime.py', ' M tests/seis_ssl_cluster/test_f3_center_trace_masked_pretraining_validation.py', ' M tests/seis_ssl_cluster/test_proc_entrypoints.py', '?? proc/seis_ssl_cluster/publish_f3_center_trace_masked_pretraining_results.py', '?? results/f3/facies_benchmark_v1/strat_hmm_multi_head_k6810_center_trace_masked_v1/', '?? src/seis_ssl_cluster/f3/center_trace_masked_pretraining_results.py', '?? tests/seis_ssl_cluster/test_f3_center_trace_masked_pretraining_results.py']`
- Full run: epoch `25` / global step `25600`
- Selected checkpoint: `step` epoch `19` step `19000` loss `0.47312438`
- Selected checkpoint SHA-256: `3562467846253aad44dce9520e784debe992dec733e627fcbe0e88f18a817285`
- Embedding shape/dtype: `[76, 113, 32, 384]` / `float16`
- Valid-token count: `237225`
- Execution counts: training fresh `1` / resume `0`; embedding fresh `1` / reuse `0`
- PASS handoff SHA-256: `27a31a98483ef3ce6d5f59d80c94602ab82dc562d4b345b1555f5fd655ccbbe6`

## Fixed scientific identity

The hard K=6/8/10 target identity, center-trace mask semantics, 0.50/0.50 masked-visible objective, visible-only distillation, learned replacement token, and disabled consistency policy were validated from the live PASS handoff.

## Training diagnostics

| Metric | Minimum | Maximum |
| --- | ---: | ---: |
| `loss` | 0.55876677 | 0.81871803 |
| `loss_prototype_masked_k6` | 0.53625273 | 0.74093171 |
| `loss_prototype_visible_k6` | 0.38821921 | 0.60215471 |
| `loss_prototype_masked_k8` | 0.74587458 | 1.0181586 |
| `loss_prototype_visible_k8` | 0.56480262 | 0.83630957 |
| `loss_prototype_masked_k10` | 0.63346144 | 0.90803707 |
| `loss_prototype_visible_k10` | 0.46939861 | 0.80392076 |
| `masked_top1_accuracy_k6` | 0.71038302 | 0.78475503 |
| `masked_top1_accuracy_k8` | 0.6152714 | 0.70105424 |
| `masked_top1_accuracy_k10` | 0.66082531 | 0.75024876 |
| `masked_supervised_token_fraction` | 0.086889863 | 0.087771118 |
| `visible_supervised_token_fraction` | 0.77291095 | 0.78063554 |
| `eligible_xy_column_count` | 228.05908 | 230.13867 |
| `selected_xy_column_count` | 23.047119 | 23.263672 |

## Downstream status

Downstream decoder evaluation, the original-split gate, and six-split screening were not executed here. The validated PASS handoff is ready for that authorized downstream screening.

Scientific superiority is not concluded by this pretraining review.

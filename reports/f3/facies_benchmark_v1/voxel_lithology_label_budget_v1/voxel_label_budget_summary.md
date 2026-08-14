# F3 original-split low-label voxel benchmark

Status: **COMPLETE**

Budgets are per-class selected token-row caps. Dense voxel labels inside selected token blocks are retained, and full validation is fixed.

## Scientific decisions

- M1 vs MAE: **POSITIVE**
- M2-A vs MAE: **HOLD**
- M2-A vs M1: **HOLD**

## Paired primary metrics

| budget | comparison | mean Δ macro F1 | median Δ macro F1 | wins | mean Δ mean IoU | median Δ mean IoU | wins |
|---|---|---:|---:|---:|---:|---:|---:|
| cap25 | M1 - MAE | 0.024365 | 0.024273 | 4/5 | 0.039930 | 0.039301 | 5/5 |
| cap25 | M2-A - MAE | 0.011726 | 0.006805 | 4/5 | 0.021919 | 0.015679 | 5/5 |
| cap25 | M2-A - M1 | -0.012639 | -0.010666 | 1/5 | -0.018011 | -0.023417 | 1/5 |
| cap50 | M1 - MAE | 0.026638 | 0.027867 | 5/5 | 0.036034 | 0.031944 | 5/5 |
| cap50 | M2-A - MAE | 0.015795 | 0.005767 | 4/5 | 0.019234 | 0.013167 | 5/5 |
| cap50 | M2-A - M1 | -0.010843 | -0.018403 | 1/5 | -0.016800 | -0.021219 | 1/5 |
| cap100 | M1 - MAE | 0.010492 | 0.012807 | 4/5 | 0.012947 | 0.019137 | 4/5 |
| cap100 | M2-A - MAE | 0.017742 | 0.015898 | 5/5 | 0.023316 | 0.025352 | 5/5 |
| cap100 | M2-A - M1 | 0.007251 | 0.006227 | 4/5 | 0.010370 | 0.008567 | 5/5 |

## Interpretation limits

- The aggregation unit is five paired subsample seeds, not voxels.
- No p-values or confidence intervals were computed.
- `full` is one reused seed-42 run and is excluded from paired wins.
- Any sample-efficiency statement is limited to this F3 original split and fixed frozen-embedding decoder.
- Six-split low-label robustness is outside this milestone.

# Parihaka Channel transferred local-BT recipe

Primary metric: `test.channel_iou`; higher is better.

Deltas are candidate minus comparator. Positive values favor `local_barlow_twins_rot90_asym_g060_3ep`.

## Mean Channel IoU over 15 cells

| model | mean |
|---|---:|
| local_barlow_twins_rot90_asym_g060_3ep | 0.154521 |
| local_barlow_twins | 0.132287 |
| random | 0.076323 |

## Paired deltas

| comparison | scope | n | mean | median | sample std | wins/ties/losses | paired t |
|---|---|---:|---:|---:|---:|---:|---:|
| candidate_minus_reference | small | 5 | 0.031904 | 0.023508 | 0.022400 | 5/0/0 | 3.1848 |
| candidate_minus_reference | medium | 5 | 0.020763 | 0.007390 | 0.024592 | 4/0/1 | 1.8879 |
| candidate_minus_reference | large | 5 | 0.014036 | 0.014012 | 0.011616 | 5/0/0 | 2.7020 |
| candidate_minus_reference | all_15 | 15 | 0.022235 | 0.019093 | 0.020319 | 14/0/1 | 4.2381 |
| candidate_minus_random | small | 5 | 0.100672 | 0.091857 | 0.025463 | 5/0/0 | 8.8405 |
| candidate_minus_random | medium | 5 | 0.106199 | 0.142784 | 0.068313 | 5/0/0 | 3.4762 |
| candidate_minus_random | large | 5 | 0.027723 | 0.032312 | 0.021744 | 4/0/1 | 2.8508 |
| candidate_minus_random | all_15 | 15 | 0.078198 | 0.077221 | 0.054991 | 14/0/1 | 5.5075 |

## Per-cell Channel IoU

| size | layout | candidate | reference | random | candidate-reference | candidate-random |
|---|---|---:|---:|---:|---:|---:|
| small | layout_000 | 0.100328 | 0.071354 | 0.000000 | 0.028974 | 0.100328 |
| small | layout_001 | 0.091857 | 0.075829 | 0.000000 | 0.016028 | 0.091857 |
| small | layout_002 | 0.143755 | 0.072696 | 0.000000 | 0.071059 | 0.143755 |
| small | layout_003 | 0.077221 | 0.057268 | 0.000000 | 0.019953 | 0.077221 |
| small | layout_004 | 0.090201 | 0.066693 | 0.000000 | 0.023508 | 0.090201 |
| medium | layout_000 | 0.124597 | 0.117207 | 0.076382 | 0.007390 | 0.048215 |
| medium | layout_001 | 0.142784 | 0.139480 | 0.000000 | 0.003304 | 0.142784 |
| medium | layout_002 | 0.170396 | 0.132931 | 0.000000 | 0.037466 | 0.170396 |
| medium | layout_003 | 0.139357 | 0.083631 | 0.121372 | 0.055726 | 0.017985 |
| medium | layout_004 | 0.151617 | 0.151687 | 0.000000 | -0.000070 | 0.151617 |
| large | layout_000 | 0.217911 | 0.198818 | 0.189954 | 0.019093 | 0.027957 |
| large | layout_001 | 0.210483 | 0.207610 | 0.178171 | 0.002874 | 0.032312 |
| large | layout_002 | 0.204630 | 0.201145 | 0.166911 | 0.003486 | 0.037719 |
| large | layout_003 | 0.225935 | 0.211922 | 0.176809 | 0.014012 | 0.049126 |
| large | layout_004 | 0.226747 | 0.196031 | 0.235248 | 0.030716 | -0.008501 |

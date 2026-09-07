# F3 lithology candidate versus five-way summary

Candidate `random_init_self_target_hmm_k6_distill010` versus the canonical five-way models `mae`, `mae_hmm_k6`, `local_barlow_twins`, `local_barlow_twins_hmm_k6`, `random`. Primary metric: `macro_f1` on unique validation voxels; paired unit is `layout_id`, aggregated per size.

| size | comparison | candidate mean | reference mean | delta mean | median | sample std | +/0/- |
|---|---|---:|---:|---:|---:|---:|---|
| small | candidate_minus_random | 0.532539 | 0.466787 | 0.065753 | 0.054673 | 0.040248 | 5/0/0 |
| small | candidate_minus_mae | 0.532539 | 0.596543 | -0.064004 | -0.071270 | 0.033632 | 0/0/5 |
| small | candidate_minus_mae_hmm_k6 | 0.532539 | 0.635257 | -0.102717 | -0.112335 | 0.035987 | 0/0/5 |
| small | candidate_minus_local_bt | 0.532539 | 0.405799 | 0.126740 | 0.118351 | 0.045556 | 5/0/0 |
| small | candidate_minus_local_bt_hmm_k6 | 0.532539 | 0.433279 | 0.099261 | 0.090061 | 0.036940 | 5/0/0 |
| medium | candidate_minus_random | 0.595871 | 0.512144 | 0.083727 | 0.075214 | 0.034128 | 5/0/0 |
| medium | candidate_minus_mae | 0.595871 | 0.656427 | -0.060556 | -0.058958 | 0.038844 | 0/0/5 |
| medium | candidate_minus_mae_hmm_k6 | 0.595871 | 0.681300 | -0.085429 | -0.083387 | 0.026104 | 0/0/5 |
| medium | candidate_minus_local_bt | 0.595871 | 0.460533 | 0.135338 | 0.133508 | 0.036167 | 5/0/0 |
| medium | candidate_minus_local_bt_hmm_k6 | 0.595871 | 0.487187 | 0.108684 | 0.103905 | 0.034716 | 5/0/0 |
| large | candidate_minus_random | 0.680907 | 0.578663 | 0.102244 | 0.104017 | 0.019983 | 5/0/0 |
| large | candidate_minus_mae | 0.680907 | 0.768400 | -0.087493 | -0.077043 | 0.027800 | 0/0/5 |
| large | candidate_minus_mae_hmm_k6 | 0.680907 | 0.777726 | -0.096819 | -0.082665 | 0.036663 | 0/0/5 |
| large | candidate_minus_local_bt | 0.680907 | 0.532079 | 0.148828 | 0.160122 | 0.035298 | 5/0/0 |
| large | candidate_minus_local_bt_hmm_k6 | 0.680907 | 0.554500 | 0.126407 | 0.129384 | 0.032100 | 5/0/0 |

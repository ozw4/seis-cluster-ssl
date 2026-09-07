# F3 lithology candidate versus five-way summary

Candidate `random_init_hmm_k6_distill010` versus the canonical five-way models `mae`, `mae_hmm_k6`, `local_barlow_twins`, `local_barlow_twins_hmm_k6`, `random`. Primary metric: `macro_f1` on unique validation voxels; paired unit is `layout_id`, aggregated per size.

| size | comparison | candidate mean | reference mean | delta mean | median | sample std | +/0/- |
|---|---|---:|---:|---:|---:|---:|---|
| small | candidate_minus_random | 0.641326 | 0.466787 | 0.174539 | 0.158943 | 0.045093 | 5/0/0 |
| small | candidate_minus_mae | 0.641326 | 0.596543 | 0.044782 | 0.033008 | 0.027367 | 5/0/0 |
| small | candidate_minus_mae_hmm_k6 | 0.641326 | 0.635257 | 0.006069 | -0.008065 | 0.039093 | 2/0/3 |
| small | candidate_minus_local_bt | 0.641326 | 0.405799 | 0.235526 | 0.223981 | 0.058956 | 5/0/0 |
| small | candidate_minus_local_bt_hmm_k6 | 0.641326 | 0.433279 | 0.208047 | 0.194835 | 0.049365 | 5/0/0 |
| medium | candidate_minus_random | 0.697715 | 0.512144 | 0.185571 | 0.195903 | 0.038722 | 5/0/0 |
| medium | candidate_minus_mae | 0.697715 | 0.656427 | 0.041288 | 0.055978 | 0.043638 | 4/0/1 |
| medium | candidate_minus_mae_hmm_k6 | 0.697715 | 0.681300 | 0.016415 | 0.026929 | 0.032158 | 4/0/1 |
| medium | candidate_minus_local_bt | 0.697715 | 0.460533 | 0.237182 | 0.247649 | 0.046150 | 5/0/0 |
| medium | candidate_minus_local_bt_hmm_k6 | 0.697715 | 0.487187 | 0.210528 | 0.214642 | 0.039433 | 5/0/0 |
| large | candidate_minus_random | 0.780574 | 0.578663 | 0.201911 | 0.202411 | 0.011159 | 5/0/0 |
| large | candidate_minus_mae | 0.780574 | 0.768400 | 0.012174 | 0.016361 | 0.021960 | 3/0/2 |
| large | candidate_minus_mae_hmm_k6 | 0.780574 | 0.777726 | 0.002848 | -0.004513 | 0.030116 | 2/0/3 |
| large | candidate_minus_local_bt | 0.780574 | 0.532079 | 0.248495 | 0.245242 | 0.026045 | 5/0/0 |
| large | candidate_minus_local_bt_hmm_k6 | 0.780574 | 0.554500 | 0.226074 | 0.233001 | 0.018985 | 5/0/0 |

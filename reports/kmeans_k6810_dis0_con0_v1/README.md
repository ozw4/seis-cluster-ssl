# Unordered k-means versus HMM v2: kmeans_k6810_dis0_con0_v1

Fixed unordered k-means versus HMM target generation, with distillation=0 and consistency=0 on both sides. Raw differences are kmeans minus HMM; lower horizon MAE is better.

| Survey | Source | Size | k-means | HMM v2 | Difference ± sample SD |
| --- | --- | --- | ---: | ---: | ---: |
| f3 | mae | small | 0.499676 | 0.531850 | -0.032173 ± 0.016333 |
| f3 | mae | medium | 0.565524 | 0.594844 | -0.029320 ± 0.022109 |
| f3 | mae | large | 0.685026 | 0.693742 | -0.008716 ± 0.027093 |
| f3 | local_bt | small | 0.349352 | 0.391923 | -0.042571 ± 0.010951 |
| f3 | local_bt | medium | 0.418070 | 0.436628 | -0.018558 ± 0.009748 |
| f3 | local_bt | large | 0.512957 | 0.524520 | -0.011563 ± 0.017676 |
| f3 | random | small | 0.343810 | 0.465669 | -0.121859 ± 0.034607 |
| f3 | random | medium | 0.408624 | 0.535686 | -0.127062 ± 0.042078 |
| f3 | random | large | 0.492627 | 0.633638 | -0.141011 ± 0.007733 |
| parihaka | mae | small | 0.241846 | 0.264075 | -0.022229 ± 0.034941 |
| parihaka | mae | medium | 0.350242 | 0.362105 | -0.011863 ± 0.022321 |
| parihaka | mae | large | 0.452981 | 0.445807 | 0.007173 ± 0.019380 |
| parihaka | local_bt | small | 0.098052 | 0.104106 | -0.006054 ± 0.012876 |
| parihaka | local_bt | medium | 0.144290 | 0.162945 | -0.018655 ± 0.014399 |
| parihaka | local_bt | large | 0.210134 | 0.244455 | -0.034322 ± 0.021992 |
| parihaka | random | small | 0.129051 | 0.164400 | -0.035349 ± 0.022175 |
| parihaka | random | medium | 0.190806 | 0.237122 | -0.046316 ± 0.036247 |
| parihaka | random | large | 0.257064 | 0.292712 | -0.035648 ± 0.023688 |
| volve | mae | small | 12.541267 | 11.435393 | 1.105874 ± 0.502316 |
| volve | mae | medium | 9.072344 | 7.876137 | 1.196208 ± 0.614421 |
| volve | mae | large | 6.000498 | 4.685829 | 1.314669 ± 0.467960 |
| volve | local_bt | small | 17.152348 | 13.935149 | 3.217199 ± 2.341363 |
| volve | local_bt | medium | 14.132020 | 12.144287 | 1.987733 ± 0.825052 |
| volve | local_bt | large | 10.416765 | 9.213950 | 1.202815 ± 0.953712 |
| volve | random | small | 17.114345 | 11.740045 | 5.374299 ± 0.935040 |
| volve | random | medium | 13.413901 | 9.452107 | 3.961795 ± 1.207678 |
| volve | random | large | 9.525378 | 5.403001 | 4.122377 ± 0.739972 |

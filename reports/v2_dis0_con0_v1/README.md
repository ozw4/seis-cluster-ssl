# v2_dis0_con0

K=[6,8,10]; distillation=0; consistency=0.

Five-layout means and sample SD; continuation seed 42; one encoder run per source.
F3 common-test Mean IoU; Parihaka test Channel IoU; Volve test MAE [sample].

| Survey | Source | Budget | Primary mean | Sample SD |
|---|---|---|---:|---:|
| f3 | mae | small | 0.531850 | 0.031730 |
| f3 | mae | medium | 0.594844 | 0.044435 |
| f3 | mae | large | 0.693742 | 0.025820 |
| f3 | local_bt | small | 0.391923 | 0.023498 |
| f3 | local_bt | medium | 0.436628 | 0.029469 |
| f3 | local_bt | large | 0.524520 | 0.019768 |
| f3 | random | small | 0.465669 | 0.049330 |
| f3 | random | medium | 0.535686 | 0.055780 |
| f3 | random | large | 0.633638 | 0.023723 |
| parihaka | mae | small | 0.264075 | 0.092913 |
| parihaka | mae | medium | 0.362105 | 0.031577 |
| parihaka | mae | large | 0.445807 | 0.015432 |
| parihaka | local_bt | small | 0.104106 | 0.022051 |
| parihaka | local_bt | medium | 0.162945 | 0.024259 |
| parihaka | local_bt | large | 0.244455 | 0.019696 |
| parihaka | random | small | 0.164400 | 0.033116 |
| parihaka | random | medium | 0.237122 | 0.042892 |
| parihaka | random | large | 0.292712 | 0.017496 |
| volve | mae | small | 11.435393 | 1.233294 |
| volve | mae | medium | 7.876137 | 0.475818 |
| volve | mae | large | 4.685829 | 0.491978 |
| volve | local_bt | small | 13.935149 | 0.883999 |
| volve | local_bt | medium | 12.144287 | 0.477840 |
| volve | local_bt | large | 9.213950 | 0.660718 |
| volve | random | small | 11.740045 | 0.732604 |
| volve | random | medium | 9.452107 | 0.530533 |
| volve | random | large | 5.403001 | 0.213959 |

All nine 25-epoch continuations and 135 decoder cells are complete.
No statistical-significance claim; layouts are not encoder-seed repetitions.
Raw predictions, embeddings and checkpoints remain in artifacts.

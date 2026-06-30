# F3 lithology token dataset

- train tokens: 28724
- validation tokens: 7003
- all labeled tokens: 35727
- supervised slices: 15
- split strategy: png_label_inventory slice split; no random token split
- cross-split token overlap: validation precedence; removed 449 train rows for 449 token_xyz values

## Per-slice tokenization

| split | slice | total | retained | dropped | ambiguous | empty | invalid_embedding |
|---|---|---:|---:|---:|---:|---:|---:|
| train | inline 250 | 3616 | 2974 | 642 | 199 | 0 | 443 |
| train | inline 350 | 3616 | 2998 | 618 | 171 | 0 | 447 |
| train | inline 450 | 3616 | 3023 | 593 | 150 | 0 | 443 |
| train | inline 550 | 3616 | 3050 | 566 | 118 | 0 | 448 |
| train | inline 650 | 3616 | 3062 | 554 | 105 | 0 | 449 |
| train | crossline 450 | 2432 | 2010 | 422 | 93 | 0 | 329 |
| train | crossline 550 | 2432 | 2000 | 432 | 105 | 0 | 327 |
| train | crossline 650 | 2432 | 2020 | 412 | 90 | 0 | 322 |
| train | crossline 850 | 2432 | 1992 | 440 | 116 | 0 | 324 |
| train | crossline 950 | 2432 | 2002 | 430 | 99 | 0 | 331 |
| train | crossline 1050 | 2432 | 2014 | 418 | 90 | 0 | 328 |
| train | crossline 1150 | 2432 | 2028 | 404 | 75 | 0 | 329 |
| validation | inline 150 | 3616 | 2999 | 617 | 169 | 0 | 448 |
| validation | crossline 350 | 2432 | 2003 | 429 | 99 | 0 | 330 |
| validation | crossline 750 | 2432 | 2001 | 431 | 109 | 0 | 322 |

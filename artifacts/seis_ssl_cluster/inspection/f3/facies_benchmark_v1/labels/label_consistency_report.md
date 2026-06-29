# F3 label consistency report

- status: FAIL
- F3 root: `/home/dcuser/data/public_data/field/F3`
- class_info: `/home/dcuser/data/public_data/field/F3/interpretation/class_info.json`
- SEGY label volume: `/home/dcuser/data/public_data/field/F3/f3_labels.sgy`
- PNG labels: 15
- max mismatch threshold: 0.001
- max observed mismatch rate: 0.003925921089856586
- total mismatch pixels: 10816

## Per-slice results

| split | slice | orientation | png_shape | segy_shape | matched | mismatched | mismatch_rate | unknown_png | unexpected_segy | threshold |
|---|---|---|---|---|---:|---:|---:|---:|---|---|
| train | inline 250 | transpose_png_to_segy | [255, 901] | [901, 255] | 228854 | 901 | 0.00392157 | 0 | `[]` | FAIL |
| train | inline 350 | transpose_png_to_segy | [255, 901] | [901, 255] | 228853 | 902 | 0.00392592 | 0 | `[]` | FAIL |
| train | inline 450 | transpose_png_to_segy | [255, 901] | [901, 255] | 228854 | 901 | 0.00392157 | 0 | `[]` | FAIL |
| train | inline 550 | transpose_png_to_segy | [255, 901] | [901, 255] | 228854 | 901 | 0.00392157 | 0 | `[]` | FAIL |
| train | inline 650 | transpose_png_to_segy | [255, 901] | [901, 255] | 228854 | 901 | 0.00392157 | 0 | `[]` | FAIL |
| train | crossline 450 | transpose_png_to_segy | [255, 601] | [601, 255] | 152654 | 601 | 0.00392157 | 0 | `[]` | FAIL |
| train | crossline 550 | transpose_png_to_segy | [255, 601] | [601, 255] | 152654 | 601 | 0.00392157 | 0 | `[]` | FAIL |
| train | crossline 650 | transpose_png_to_segy | [255, 601] | [601, 255] | 152654 | 601 | 0.00392157 | 0 | `[]` | FAIL |
| train | crossline 850 | transpose_png_to_segy | [255, 601] | [601, 255] | 152654 | 601 | 0.00392157 | 0 | `[]` | FAIL |
| train | crossline 950 | transpose_png_to_segy | [255, 601] | [601, 255] | 152654 | 601 | 0.00392157 | 0 | `[]` | FAIL |
| train | crossline 1050 | transpose_png_to_segy | [255, 601] | [601, 255] | 152654 | 601 | 0.00392157 | 0 | `[]` | FAIL |
| train | crossline 1150 | transpose_png_to_segy | [255, 601] | [601, 255] | 152654 | 601 | 0.00392157 | 0 | `[]` | FAIL |
| validation | inline 150 | transpose_png_to_segy | [255, 901] | [901, 255] | 228854 | 901 | 0.00392157 | 0 | `[]` | FAIL |
| validation | crossline 350 | transpose_png_to_segy | [255, 601] | [601, 255] | 152654 | 601 | 0.00392157 | 0 | `[]` | FAIL |
| validation | crossline 750 | transpose_png_to_segy | [255, 601] | [601, 255] | 152654 | 601 | 0.00392157 | 0 | `[]` | FAIL |

## Warnings

- one or more PNG labels required non-default orientation; see per-slice JSON metadata
- one or more slices exceed max_mismatch_rate

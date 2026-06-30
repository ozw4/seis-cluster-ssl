# F3 label consistency report

- status: PASS
- F3 root: `/home/dcuser/data/public_data/field/F3`
- class_info: `/home/dcuser/data/public_data/field/F3/interpretation/class_info.json`
- SEGY label volume: `/home/dcuser/data/public_data/field/F3/f3_labels.sgy`
- PNG labels: 15
- max mismatch threshold: 0.001
- ignored z-border samples: 1
- max observed mismatch rate: 0.003925921089856586
- max observed effective mismatch rate: 4.386869223041592e-06
- total mismatch pixels: 10816

## Per-slice results

| split | slice | orientation | png_shape | segy_shape | matched | mismatched | raw_mismatch_rate | effective_mismatch_rate | border_only_mismatch | border_mismatch_pixel_count | interior_mismatch_pixel_count | unknown_png | unexpected_segy | threshold |
|---|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|
| train | inline 250 | transpose_png_to_segy | [255, 901] | [901, 255] | 228854 | 901 | 0.00392157 | 0 | yes | 901 | 0 | 0 | `[]` | PASS |
| train | inline 350 | transpose_png_to_segy | [255, 901] | [901, 255] | 228853 | 902 | 0.00392592 | 4.38687e-06 | no | 901 | 1 | 0 | `[]` | PASS |
| train | inline 450 | transpose_png_to_segy | [255, 901] | [901, 255] | 228854 | 901 | 0.00392157 | 0 | yes | 901 | 0 | 0 | `[]` | PASS |
| train | inline 550 | transpose_png_to_segy | [255, 901] | [901, 255] | 228854 | 901 | 0.00392157 | 0 | yes | 901 | 0 | 0 | `[]` | PASS |
| train | inline 650 | transpose_png_to_segy | [255, 901] | [901, 255] | 228854 | 901 | 0.00392157 | 0 | yes | 901 | 0 | 0 | `[]` | PASS |
| train | crossline 450 | transpose_png_to_segy | [255, 601] | [601, 255] | 152654 | 601 | 0.00392157 | 0 | yes | 601 | 0 | 0 | `[]` | PASS |
| train | crossline 550 | transpose_png_to_segy | [255, 601] | [601, 255] | 152654 | 601 | 0.00392157 | 0 | yes | 601 | 0 | 0 | `[]` | PASS |
| train | crossline 650 | transpose_png_to_segy | [255, 601] | [601, 255] | 152654 | 601 | 0.00392157 | 0 | yes | 601 | 0 | 0 | `[]` | PASS |
| train | crossline 850 | transpose_png_to_segy | [255, 601] | [601, 255] | 152654 | 601 | 0.00392157 | 0 | yes | 601 | 0 | 0 | `[]` | PASS |
| train | crossline 950 | transpose_png_to_segy | [255, 601] | [601, 255] | 152654 | 601 | 0.00392157 | 0 | yes | 601 | 0 | 0 | `[]` | PASS |
| train | crossline 1050 | transpose_png_to_segy | [255, 601] | [601, 255] | 152654 | 601 | 0.00392157 | 0 | yes | 601 | 0 | 0 | `[]` | PASS |
| train | crossline 1150 | transpose_png_to_segy | [255, 601] | [601, 255] | 152654 | 601 | 0.00392157 | 0 | yes | 601 | 0 | 0 | `[]` | PASS |
| validation | inline 150 | transpose_png_to_segy | [255, 901] | [901, 255] | 228854 | 901 | 0.00392157 | 0 | yes | 901 | 0 | 0 | `[]` | PASS |
| validation | crossline 350 | transpose_png_to_segy | [255, 601] | [601, 255] | 152654 | 601 | 0.00392157 | 0 | yes | 601 | 0 | 0 | `[]` | PASS |
| validation | crossline 750 | transpose_png_to_segy | [255, 601] | [601, 255] | 152654 | 601 | 0.00392157 | 0 | yes | 601 | 0 | 0 | `[]` | PASS |

## Warnings

- one or more PNG labels required non-default orientation; see per-slice JSON metadata
- one or more PNG/SEGY label mismatches are confined to ignored z-border samples

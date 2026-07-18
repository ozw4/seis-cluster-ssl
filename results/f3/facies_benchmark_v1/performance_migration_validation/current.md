# seis-ssl-cluster performance benchmark

## Input conditions

- Schema version: 2
- Seed: 248
- Warm-up iterations: 3
- Measured iterations: 20
- Environment: `{"cpu_count":128,"device":"cpu","machine":"x86_64","numpy_version":"1.24.4","platform":"Linux-6.8.0-106-generic-x86_64-with-glibc2.35","processor":"x86_64","python_version":"3.10.12","torch_num_threads":1,"torch_version":"2.13.0+cu130"}`

| Case | Shape and settings |
|---|---|
| memmap_repeated_open_crop | `{"crop_xyz":[128,128,128],"open_crop_count":4,"volume_xyz":[160,160,160]}` |
| spatial_mask_16_cubed_m075_block1 | `{"block_size_tokens_xyz":[1,1,1],"mask_ratio":0.75,"token_grid_xyz":[16,16,16]}` |
| amplitude_preprocessing | `{"agc_window_z":65,"crop_xyz":[128,128,128],"patch_size_xyz":[8,8,8]}` |
| position_embedding_visible_selection | `{"batch":1,"embedding_dim":128,"token_grid_xyz":[16,16,16],"visible_tokens":1024}` |
| embedding_merge_token_to_voxel | `{"embedding_dim":128,"merge_window_count":2,"token_grid_xyz":[16,16,16],"voxel_shape_xyz":[128,128,128]}` |
| token_phase_residualization | `{"embedding_dim":128,"token_phase_groups":512,"tokens":4096}` |
| hmm_squared_euclidean_emission | `{"dtype":"float32","feature_dim":128,"states":12,"tokens":4096}` |

## Results

| Case | Version | Input fingerprint | Median (s) | P25-P75 (s) | Comparable | Baseline median (s) | Speedup | Note |
|---|---:|---|---:|---:|---|---:|---:|---|
| memmap_repeated_open_crop | 1 | `b743bd5ae1735fb5` | 0.003009 | 0.002980-0.003040 | not requested | — | — | No baseline report supplied. |
| spatial_mask_16_cubed_m075_block1 | 1 | `890eb0425654af76` | 0.000053 | 0.000053-0.000054 | not requested | — | — | No baseline report supplied. |
| amplitude_preprocessing | 1 | `cf6645bf3351a8e9` | 0.053956 | 0.053331-0.054220 | not requested | — | — | No baseline report supplied. |
| position_embedding_visible_selection | 1 | `2eb04d39fba43f54` | 0.001183 | 0.001179-0.001204 | not requested | — | — | No baseline report supplied. |
| embedding_merge_token_to_voxel | 1 | `b6f4bc0efa804e20` | 0.003326 | 0.003298-0.003365 | not requested | — | — | No baseline report supplied. |
| token_phase_residualization | 1 | `d1a6120f921289ae` | 0.031550 | 0.031496-0.031666 | not requested | — | — | No baseline report supplied. |
| hmm_squared_euclidean_emission | 2 | `026be7ec5111fdc7` | 0.001963 | 0.001958-0.001971 | not requested | — | — | No baseline report supplied. |

## Cautions

- Speedup is baseline median divided by current median.
- A multiplier is omitted when the case name, version, or input fingerprint differs, or when the current median is zero.
- Compare runs from the same machine and software environment; interquartile overlap can indicate timing noise.
- AMP and lower-precision paths may be numerically close rather than bitwise identical.

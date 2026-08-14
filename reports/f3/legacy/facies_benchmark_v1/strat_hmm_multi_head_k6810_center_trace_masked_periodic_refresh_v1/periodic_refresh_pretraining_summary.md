# F3 periodic center-trace masked pretraining review

- Status: `PASS`
- Model tag: `strat_hmm_pretext_mh_k6810_ctmask010_refresh3ep_hmm2_nocons_topblock1_distill_v1`
- Variant: `ctmask010_refresh3ep_hmm2_nocons`
- Execution Git SHA: `3ee1e5485408f3d179675afddc6136897135aec8`
- Execution dirty status: `['M src/seis_ssl_cluster/embedding/extractor.py', ' M src/seis_ssl_cluster/f3/center_trace_masked_periodic_refresh_validation.py', ' M tests/seis_ssl_cluster/test_proc_entrypoints.py', '?? experiments/f3/facies_benchmark_v1/107_strat_hmm_multi_head_k6810_center_trace_masked_periodic_refresh_v1/05_review_periodic_refresh_results.yaml', '?? proc/seis_ssl_cluster/publish_f3_center_trace_masked_periodic_refresh_results.py', '?? src/seis_ssl_cluster/f3/center_trace_masked_periodic_refresh_results.py', '?? tests/seis_ssl_cluster/test_f3_center_trace_masked_periodic_refresh_results.py']`
- Refreshes/generations: `7` / `8`
- Final checkpoint: epoch `25`, global step `25600`
- Selected checkpoint SHA-256: `46c084c82c5c4cdb70375b432bd54aab7f2c68477b8027d01e1773bdfffea663`
- Final embedding shape/dtype: `[76, 113, 32, 384]` / `float16`
- Valid-token count: `237225`
- PASS handoff: `fc93d547a6bab07623bf8c77b9e297d490c47dc3b0d0027b5a4b0455f10969bf`

## Fixed center-trace parity

Parity status is `PASS`; the allowed differences are recorded in the portable JSON summary. Initial target, HMM, preprocessing, and model initialization hashes were revalidated from live bytes.

## Refresh generations

| Generation | Epoch | Source student hash | Manifest hash |
| --- | ---: | --- | --- |
| `refresh_0000_initial` | 0 | `None` | `6f9738585ac43b248e5a065f4dc0e6a1cd3a35b8f6153bd2d28e4a0d9856c20a` |
| `refresh_0001_epoch002` | 2 | `9fc020188263873c3d20dc64fb0a1aba54474b75f3a054ae816168a32c800fc2` | `8e382cfb4608a25536b0f87b8b997c27d642e9cac5110e1bd837d585cd2f83f6` |
| `refresh_0002_epoch005` | 5 | `bd3b1ff3133f2da4669ff9195cd0710c57df1db5c4bd2c8e6f45f629faa5eb99` | `dffd26bd86d82c26510fdafc82b78ece9ae6a719cb2981ab6b73855908b19863` |
| `refresh_0003_epoch008` | 8 | `490573dbdde09fd9f4c5878465ecb035248d6ff0345673b240df8ecdf705ffa7` | `9463ebb114b8f74a5a183d4d3230119250602e84eb40cb298f904788ec499059` |
| `refresh_0004_epoch011` | 11 | `ac357ff13516f3707d54040a8e69d83548b683c96ab4f3a703d4d58bfbdc8ebd` | `6fbcbe34ef09a7ba9486c949948c04b0e8ecfbbb980e2a2e960b7cd122d21fff` |
| `refresh_0005_epoch014` | 14 | `3bb9f69f321a5691ebce929d2732bb02451ab24c1e16254a1382099a8448f5dd` | `63868a1eef7c42c8a8a027252c3b8ca850e8a8ebf20469af49b0ce6724d9ce39` |
| `refresh_0006_epoch017` | 17 | `75f39f919d1531b361cd5809b725b244f3d8de1f517aa25e41ab49e80dc755f2` | `a04574f720c02dcbc0060a81e12169ef1629919917722180ccc734c52cbe21c9` |
| `refresh_0007_epoch020` | 20 | `9e4b41fa61b02b418f934a2785f227ee940adf942d0012c4add183ded55bc451` | `7bfe852037e44de31e8f9adee5237bff1482295c745f44efbe0711b34857d48a` |

## Downstream status

- Original-split ready: `True`
- Decoder jobs executed: `0`
- Six-split jobs executed: `0`

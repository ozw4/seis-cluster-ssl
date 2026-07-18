# M1 embedding A/B/C parity

- Current git SHA: `332478be21a021e46ee6c1d9423f14859b0cd819`
- Historical baseline SHA: `7731f341a293ea0c5cb5c5dfabba574148861e3a`

| Pair | Status | Mask exact | Array exact | Max abs error | Mean cosine |
| --- | --- | --- | --- | --- | --- |
| A_historical_vs_B_current_cache_off | NUMERIC_DRIFT | True | False | 0.00390625 | 0.9999999572 |
| A_historical_vs_C_current_memmap_cache | NUMERIC_DRIFT | True | False | 0.00390625 | 0.9999999572 |
| B_current_cache_off_vs_C_current_memmap_cache | EXACT | True | True | 0 | 0.9999999741 |

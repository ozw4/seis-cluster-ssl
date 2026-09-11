# F3 Local Barlow Twins Gaussian-view validation report

This is a validation-only, human-reviewable report. It is never a pipeline input, and no test data or test metric was used.

The actual-data view diagnostic found inverse-aligned legacy views identical (correlation 1.000000, RMS 0.000000); Gaussian std 0.05 and 0.10 reduced correlation to 0.997322 and 0.989375, with paired RMS 0.070698 and 0.141396. This diagnostic does not cover the identity policy or rule out its fixed-position shortcut.

## View selection

The immutable medium-layout lock selected `local_barlow_twins_gaussian_noise_std010` using unrounded mean validation macro-F1 over five layouts.

| candidate | medium mean macro-F1 |
|---|---:|
| `local_barlow_twins_gaussian_noise_std005` | 0.456541231 |
| `local_barlow_twins_gaussian_noise_std010` | 0.461918010 |
| `local_barlow_twins_identity_gaussian_noise_std010` | 0.456916651 |

## Attempt ledger

| candidate | base + continuation epochs | medium mean | medium wins | evaluated cells | wins vs random | outcome |
|---|---:|---:|---:|---:|---:|---|
| `local_barlow_twins_gaussian_noise_std005` | 25 + 25 | 0.456541231 | 0/5 | 5 | 0/5 | not_fully_evaluated |
| `local_barlow_twins_gaussian_noise_std010` | 25 + 25 | 0.461918010 | 0/5 | 5 | 0/5 | failed_medium_gate |
| `local_barlow_twins_identity_gaussian_noise_std010` | 25 + 25 | 0.456916651 | 0/5 | 5 | 0/5 | not_fully_evaluated |
| `local_barlow_twins_legacy_flip_25ep` | 25 + 25 | 0.470530646 | 0/5 | 5 | 0/5 | failed_medium_gate |

## Final validation result

FAIL for this reached duration: no arm met the preregistered final criterion. Failure stage: `medium_5of5`.

Gaussian-minus-legacy attribution and identity-minus-forced-flip geometry contrasts are recorded separately in `paired_deltas.csv` and `summary.json`; they are not conflated with random-baseline success.

All paths are repository-relative or rooted at `${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}`. Exact live-file SHA-256 identities are recorded in the CSV and JSON outputs.
Base runs were operator-observed fresh launches after output-root absence checks and without `--resume`; the 25-epoch base checkpoint schema has no invocation resume counter, so this particular fact is not checkpoint-authenticated.

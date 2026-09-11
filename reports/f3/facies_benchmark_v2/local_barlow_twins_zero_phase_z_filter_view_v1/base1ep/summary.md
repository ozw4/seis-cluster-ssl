# F3 Barlow Twins zero-phase Z-filter w=.25 validation

This is a validation-only, human-readable projection. It is not a pipeline input.

- Decision: **FAIL**
- Candidate: `local_barlow_twins_zero_phase_z_filter_w025_base1ep`
- Medium gate open: `False`
- Strict medium wins: `2/5`
- Strict wins over random: `2/5`
- Validation rows: `15`
- Frozen parent control rows: `5`

## Attempt

| Z-filter mean | Random mean | Delta vs random | p=.02 mean | Delta vs p=.02 | 15/15 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.504512030 | 0.512144227 | -0.007632197 | 0.510727766 | -0.006215736 | False |

The p=.02 rows are frozen parent medium controls for direct attribution only. They do not affect the gate or pass decision.

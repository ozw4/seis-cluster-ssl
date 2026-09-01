# F3 Barlow Twins trace-drop p=.02 validation

This is a validation-only, human-readable projection. It is not a pipeline input.

- Decision: **FAIL**
- Candidate: `local_barlow_twins_horizontal_trace_drop_p002_base1ep`
- Medium gate open: `False`
- Strict medium wins: `2/5`
- Strict wins over random: `2/5`
- Validation rows: `15`
- Frozen parent control rows: `5`

## Attempt

| p=.02 mean | Random mean | Delta vs random | p=.01 mean | Delta vs p=.01 | 15/15 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.510727766 | 0.512144227 | -0.001416461 | 0.515091213 | -0.004363447 | False |

The p=.01 rows are frozen parent medium controls for direct attribution only. They do not affect the gate or pass decision.

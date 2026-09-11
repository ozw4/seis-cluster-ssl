# F3 Barlow Twins Gaussian-view base5 validation

This is a validation-only, human-readable projection. It is not a pipeline input.

- Decision: **FAIL**
- Winner: `none`
- Medium gate open: `False`
- Base pretraining epochs: `5`
- Fixed continuation epochs: `25`
- Candidate validation cells: `10`
- Random baseline cells: `5`

## Attempts

| Arm | Medium mean | Mean delta vs random | Positive cells | 15/15 |
| --- | ---: | ---: | ---: | --- |
| `local_barlow_twins_gaussian_noise_std010_base5ep` | 0.489563919 | -0.022580308 | 0/5 | False |
| `local_barlow_twins_legacy_flip_base5ep` | 0.467797662 | -0.044346565 | 0/5 | False |

Configuration selection was inherited from the failed, pinned 25-epoch validation result; no base5 metric was used to choose the view.

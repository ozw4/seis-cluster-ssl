# F3 Barlow Twins Gaussian-view base1 validation

This is a validation-only, human-readable projection. It is not a pipeline input.

- Decision: **FAIL**
- Winner: `none`
- Medium gate open: `False`
- Base pretraining epochs: `1`
- Fixed continuation epochs: `25`
- Candidate validation cells: `10`
- Random baseline cells: `5`

## Attempts

| Arm | Medium mean | Mean delta vs random | Positive cells | 15/15 |
| --- | ---: | ---: | ---: | --- |
| `local_barlow_twins_gaussian_noise_std010_base1ep` | 0.512369394 | +0.000225167 | 3/5 | False |
| `local_barlow_twins_legacy_flip_base1ep` | 0.513610498 | +0.001466271 | 3/5 | False |

Configuration selection was inherited from the failed, pinned base-5 validation result; no base1 metric was used to choose the view.

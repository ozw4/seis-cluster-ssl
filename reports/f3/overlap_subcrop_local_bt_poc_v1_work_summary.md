# F3 overlapping-subcrop Local Barlow Twins PoC v1 — work summary

Date: 2026-09-02

## Scope completed

- Added the exact `overlapping_subcrop_xy_v1` Local Barlow Twins policy,
  configuration validation, parent-crop derivation, deterministic paired
  subcrops, distinct horizontal flips, and physical-token correspondence.
- Integrated the policy into Local Barlow Twins training without changing the
  model, loss, collate, or bare-encoder embedding contract.
- Added the F3 PoC experiment at
  `experiments/f3/facies_benchmark_v1/112_local_bt_overlap_subcrop_poc_v1`.
- Applied the requested downstream layout from
  `experiments/f3/facies_benchmark_v2/109_f3_voxel_section_layout_v3`.
- Added a thin downstream runner that fixes `medium`, infers the model arm from
  the config filename, and skips only the PoC source audit.
- Added strict screen/final decision logic: `layout_001` must strictly beat
  Random before the other layouts run; adoption requires at least four strict
  wins among five layouts.
- Added deterministic representation diagnostics for Random and every
  candidate. The fixed contract samples 8,192 C-order valid tokens by midpoint
  systematic sampling, calculates in float64, and applies affine-free
  per-token LayerNorm with `eps=1e-5`.

## Validation completed

- Overlap, embedding, PoC, and five-way focused suite: 322 passed.
- Representation-diagnostic, extractor, and PoC suite: 101 passed.
- Phase-A config contract suite after adding shift02/shift06: 23 passed.
- Final combined focused suite after the commits: 341 passed.
- Python compilation, focused Ruff lint, and `git diff --check` passed.
- Informational `ruff format --check` reported mechanical reformatting for 11
  touched shared/PoC files. No broad formatter-only churn was applied during
  this stop handoff.
- One-step H100 feasibility completed with finite metrics and peak CUDA memory
  27,610 MiB.

## Fixed representation-diagnostic definitions

For sampled token features `X`:

- `raw_feature_norm`: mean row L2 norm.
- `token_wise_feature_std`: mean population standard deviation per feature
  across sampled tokens.
- `raw_feature_effective_rank`: exponential spectral entropy of the centered
  population covariance eigenvalues.
- `layer_norm_feature_std`: the same feature standard deviation after
  affine-free per-token LayerNorm.
- `layer_norm_effective_rank`: the same effective rank after LayerNorm.

The canonical valid-token mask has shape `76 x 113 x 32`, 237,225 valid
tokens, SHA-256
`3bfeb8db8a47420ae7671db90a7e4d6e5a07fceba27648ec76213df3c2b38fd7`,
and sampled-index SHA-256
`44efdbe4e7f7a50caf7c6d1658a5764b4126ac1f0da7ec7d3cc796febfe90de1`.

## Random baseline

Representation diagnostics:

| Metric | Value |
| --- | ---: |
| raw feature norm | 26.4523632615 |
| token-wise feature std | 0.6585601699 |
| raw feature effective rank | 23.9998007592 |
| LayerNorm feature std | 0.5220072650 |
| LayerNorm effective rank | 25.4812122319 |

The fresh PoC `layout_001 / medium` decoder reproduced the canonical Random
macro-F1 exactly: `0.4965410800214613` over 470,136 unique validation voxels.

## Candidates completed

| Candidate | Shift | Screen macro-F1 | Delta vs Random | Screen | Current role |
| --- | --- | ---: | ---: | --- | --- |
| `shift04_proj384_pairs128_lambda005` | `[4,4,0]` | 0.4115269085 | -0.0850141715 | fail | initial |
| `shift02_proj384_pairs128_lambda005` | `[2,2,0]` | 0.4476944851 | -0.0488465949 | fail | leader |

Both were fresh seed-42 10-epoch runs with epoch 10 / global step 6,250,
finite histories, full-volume float16 embeddings, required representation
diagnostics, and fresh `layout_001` decoder/evaluation artifacts. Because each
failed the screen, layouts 000/002/003/004 were not run.

Candidate representation diagnostics:

| Candidate | Raw norm | Token std | Raw eff. rank | LN std | LN eff. rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| `shift04_proj384_pairs128_lambda005` | 95.7963889138 | 4.7375605914 | 21.8244936985 | 0.9725560302 | 21.9308540078 |
| `shift02_proj384_pairs128_lambda005` | 95.6020204637 | 4.7402329837 | 21.8931642113 | 0.9752670668 | 22.0107362799 |

Important artifacts:

- Initial checkpoint SHA-256:
  `175a1999b1606ca1319358271b72087675e48c1da0e66ef78fc3101b3c7080de`.
- Shift02 checkpoint SHA-256:
  `e26f30ba84752421e8845a431c59f3fd1813becf968d23b12532e0b049f15868`.
- Diagnostics root:
  `${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/diagnostics/f3/local_bt_overlap_subcrop_poc_v1/representation`.
- Downstream root:
  `${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}/f3_lithology_benchmark/local_bt_overlap_subcrop_poc_v1`.
- Search source of truth:
  `experiments/f3/facies_benchmark_v1/112_local_bt_overlap_subcrop_poc_v1/search_results.csv`.

## Stop state and next step

Work was stopped on request immediately after starting
`shift06_proj384_pairs128_lambda005`. No training process remains. Its artifact
directory contains only `resolved_config.json`; it has no epoch history or
checkpoint and is not yet a result row in `search_results.csv`.

The next search action is Phase A `shift06_proj384_pairs128_lambda005`. Before
restarting it as a fresh run, explicitly handle the incomplete output directory
so that the runner's no-overwrite contract is preserved. Do not resume from a
different candidate, and do not treat the current directory as a completed
attempt. After completion, extract embeddings, write the same five diagnostics,
run only the v3 `layout_001 / medium` screen, append the CSV row, and recompute
the leader before creating any later candidate.

The working tree also contains unrelated concurrent Volve changes and the
input ZIP. They are outside this work and are intentionally excluded from the
commits for this PoC.

# F3 voxel section-layout benchmark contract

## Scope

The v1 benchmark replaces subsample seed as the statistical unit with five
explicit teacher-section layouts: `layout_000` through `layout_004`. Every
layout has nested `small`, `medium`, and `large` train masks. They contain,
respectively, 1 inline plus 1 crossline, 2 plus 2, and 4 plus 4 teacher
sections. Small is a strict subset of medium, and medium is a strict subset of
large. The statistical unit is exactly `layout_id`.

The validation mask is one shared, read-only mask for every layout, data size,
and model. When synthetic inventory rows are supplied to the resolver, any
training line marked `split: validation` is rejected. The resolver never opens
an inventory path.

## Calibration handoff

Concrete line numbers and target voxel counts are intentionally not fixed in
this repository change. A later calibration tool will select the lines and set
one positive integer target for each data size. The same size target must be
used by all five layouts. Those targets are calculated from the median
`actual_train_voxel_count` of the corresponding historical cap datasets.

The resulting handoff has schema version
`f3_voxel_section_layout_contract_v1`. Its closed top-level fields are:

```yaml
schema_version: f3_voxel_section_layout_contract_v1
statistical_unit: layout_id
nesting_semantics: strict_small_medium_large
validation_mask_semantics: shared_across_all_layouts_sizes_and_models
stable_selection_semantics: stable_sha256_voxel_rank_v1
patch_size: [8, 8, 8]
allowed_relative_error: 0.05
decoder_seed: 42000
layouts: []  # exactly layout_000..layout_004; populated by calibration
decoder: {}  # exact fixed mapping shown below
```

`stable_sha256_voxel_rank_v1` means that eligible labeled voxels are ordered by
a SHA-256 rank of their canonical integer voxel coordinate and selected without
replacement. Implementations must not use process hash state, directory order,
or model identity in this ordering. The allowed relative error must be in
`(0, 0.1]`; it is an acceptance tolerance and not a tuning parameter.

After calibration, builders and runners consume this handoff directly. They do
not read the old cap manifests. Unknown fields, Boolean values supplied where
integers are required, duplicate teacher lines, validation-line overlap, and
non-nested selections fail closed.

## Fixed decoder

All roster members use the same `frozen_embedding_decoder_nearest_voxel_ln_v1`
voxel decoder. Its closed mapping is:

```yaml
spec: frozen_embedding_decoder_nearest_voxel_ln_v1
embedding_dim: 384
class_count: 6
hidden_channels: [128, 64, 32]
upsample_factors: [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
upsample_mode: nearest
normalization: voxelwise_layer_norm
epochs: 50
batch_size: 1
learning_rate: 0.001
weight_decay: 0.0001
class_weight: balanced
sampling_mode: uniform_tiles_with_replacement
steps_per_epoch: 440
amp: true
gradient_clip_norm: 1.0
write_probabilities: false
seed: 42000
```

Any field addition, omission, type change, or value drift is rejected. In
particular, every job uses decoder seed 42000; the layout ID does not perturb
the decoder seed.

## Model roster and selection

The model roster is
[`00_model_roster.yaml`](../experiments/f3/facies_benchmark_v1/109_f3_voxel_section_layout_v1/00_model_roster.yaml).
It names exactly 14 existing embedding roots under the configured artifact
root. Paths are relative, may not escape that root, and are checked against the
canonical model-tag path. There is no runtime model discovery.

`mae` is the parentless baseline. Candidate models are eligible for formal
adoption. `m1_distill_only` and `m1_shuffled_hmm` have the explicit
`diagnostic` role: their metrics are aggregated, but they are never selection
eligible.

## Historical evidence

The earlier `cap25/cap50/cap100` by subsample seed `0..4` experiments and the
six-split downstream experiment are historical evidence only. Their configs,
statuses, and results remain unchanged. They are inputs to scientific context
and calibration decisions, not jobs in the new section-layout design. This
contract does not delete, regenerate, overwrite, or reinterpret those files.

This change defines config, schema, documentation, and synthetic unit tests
only. It does not run the real F3 dataset, decoder training, inference, or
summary; it also does not change or regenerate pretraining, pseudo-targets, HMM
clustering, checkpoints, or embeddings.

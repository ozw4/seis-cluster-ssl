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

Concrete line numbers and target voxel counts are not checked into the example
layout file. The user runs
`prepare_f3_lithology_voxel_section_layout_contract.py` in `inspect` mode,
reviews the candidate report, supplies five ordered 4+4 layouts, then runs
`finalize`. The same size target is used by all five layouts. Targets are the
integer medians of the five canonical `actual_train_voxel_count` values for
`cap25`, `cap50`, and `cap100`; the exact 15-row budget/seed matrix is required.

The resulting handoff has schema version
`f3_voxel_section_layout_contract_v1`. Its closed top-level fields are:

```yaml
schema_version: f3_voxel_section_layout_contract_v1
artifact_type: f3_lithology_voxel_section_layout_contract
statistical_unit: layout_id
nesting_semantics: strict_small_medium_large
validation_mask_semantics: shared_across_all_layouts_sizes_and_models
selection_semantics: stable_hash_partial_section_token_footprints_v1
stable_selection_semantics: stable_hash_partial_section_token_footprints_v1
patch_size: [8, 8, 8]
patch_size_xyz: [8, 8, 8]
allowed_relative_error: 0.05
target_train_voxel_counts: {small: 0, medium: 0, large: 0} # generated integers
active_prefix_counts:
  small: {inline: 1, crossline: 1}
  medium: {inline: 2, crossline: 2}
  large: {inline: 4, crossline: 4}
decoder_seed: 42000
layouts: []  # exactly layout_000..layout_004; populated by calibration
decoder: {}  # exact fixed mapping shown below
validation_identity: {}             # generated count/hash/source identity
source_file_identities: {}          # generated path/SHA-256 mappings
legacy_budget_source_identity: {}   # generated path/SHA-256 mapping
```

`stable_hash_partial_section_token_footprints_v1` uses an intersecting 8³ token
coordinate as the candidate unit. Only voxels in the intersection of that
block, the active line-plane union, the canonical train mask, and valid
annotations become teachers. Inline/crossline intersections are counted once
and validation voxels are excluded. Token order is SHA-256 of layout ID, token
coordinate, and semantics version. A coverage pass selects at least one
candidate for every active line before target filling. Small tokens are kept in
medium, and medium tokens are kept in large; the closer of the prefixes just
before and after the target is used. Python `hash()`, global RNG, and subsample
seed do not participate. The allowed relative error is in `(0, 0.1]`.

`inspect` writes only candidate CSV/JSON statistics. `finalize` writes only the
canonical contract, and only if all three sizes in all layouts contain six
classes, classes 3 and 5 are nonzero, all active lines contribute, relative
count error is within tolerance, selections are nested, no validation line is
active, and the validation mask identity is unchanged. `--dry-run` performs
the same reads and validation but writes nothing.

After calibration, builders and runners consume this handoff directly. They do
not read the old cap manifests. Unknown fields, Boolean values supplied where
integers are required, duplicate teacher lines, validation-line overlap, and
non-nested selections fail closed.

## Common voxel datasets

`build_f3_lithology_voxel_section_layout_datasets.py` replays the canonical
selection against the source grid instead of trusting preview counts. It writes
the exact 15 rows in layout-ID order and `small`, `medium`, `large` order under
`datasets/layout=<id>/size=<size>/voxel_supervision/`. Each condition contains
only the split grid, selected token coordinates, two metadata files, class
counts, the canonical split manifest, and a short summary.

For each selected token, train supervision is the intersection of its clipped
8³ block, the active inline/crossline plane union, canonical train, and known
dense labels. The builder never supervises the remainder of a token block or a
whole selected section. It rechecks live-mask nesting, all six classes,
nonzero classes 3 and 5, positive contributions from every active line,
count-error tolerance, and bitwise-identical validation across all 15
conditions. Source shape, dtype, geometry, hashes, class order, reference-valid
tokens, and validation identity are fail-closed inputs.

The default build stages and reloads the complete suite before replacing the
final directory. Existing output is refused. `--only-missing` reuses only exact
complete conditions; stale or partial output is rejected unless
`--quarantine-invalid` explicitly moves it to a timestamped sibling. Dry-run
performs source validation and prints the 15-condition plan without writes or
quarantines. These datasets are model-independent and shared by all 14 roster
members.

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

## Generic decoder runner

`run_f3_lithology_voxel_section_layout_suite.py` accepts exactly one
`--model-id` from the closed 14-model roster. It validates the selected
embedding array, valid-token mask, embedding metadata, checkpoint evidence,
and the complete 15-row common-dataset manifest before classifying any job.
Missing or drifted embeddings are errors; extraction is never invoked.

Each scientific model has exactly 15 canonical output locations under
`benchmark_v1/runs/model=<model_id>/layout=<layout_id>/size=<data_size>`. Every
job constructs the same frozen nearest-voxel decoder: seed 42000, 50 epochs,
440 replacement-sampling steps per epoch, batch size 1, balanced class weights,
best-checkpoint inference, no probability volume, and the common evaluation
schema.

The per-model `section_layout_run_manifest.json` is atomically replaced after
each job. Completed rows retain dataset/mask/token identities, embedding
evidence, initial decoder state, class weights, sampling sequence, tile
identities, exact-once inference checks, best checkpoint identity, metric
schema identity, and canonical metric paths. A later failure retains earlier
complete rows. Only a matching latest checkpoint can resume; foreign identity
drift is an error. Model-owned partial outputs are moved only when explicit
quarantine is requested.

Dry-run performs validation and planning with zero writes or stage execution.
The explicit one-condition smoke mode stops at two optimizer steps and uses a
disjoint non-scientific root and manifest.

## Paired-layout result gate

The generic summarizer treats `layout_id` as the paired statistical unit. It
requires the exact five-layout by three-size complete matrix for every loaded
model and rejects any mismatch in supervision, decoder initialization,
sampling, tile, or metric-schema identity before computing a delta. For every
comparison and size it reports the mean, median, sample standard deviation,
wins, ties, losses, and all five per-layout deltas. It does not calculate
p-values or confidence intervals and does not assert voxel independence.

Formal parent status uses both Macro F1 and Mean IoU. `medium` and `large` must
both satisfy strict positive evidence for `SECTION_LAYOUT_GO`; `small` is
diagnostic only. Strict negative evidence at both sizes, or the same monitored
class 3/5 metric degrading by at least 0.05 at two or more sizes, produces
`SECTION_LAYOUT_STOP`; every other case is `SECTION_LAYOUT_HOLD`. Diagnostic
roster models receive metrics and a formal status but remain selection
ineligible. Project adoption remains `PENDING_REVIEW`.

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

# Config Validation Split Plan

This is an inventory and target module map for splitting
`src/seis_ssl_cluster/config/validate.py`. It is not an implementation plan for
changing validation behavior. The existing YAML contracts, `ArtifactPaths`
path layout, checkpoint/resume compatibility, F3 workflows, `artifacts/`, and
`results/` policies remain unchanged.

## Current Public API

The public import path must remain compatible:

```python
from seis_ssl_cluster.config.validate import resolve_mae_training_config
from seis_ssl_cluster.config.validate import validate_config
```

Current public resolver functions:

- `resolve_manifest_build_config`
- `resolve_normalization_stats_config`
- `resolve_normalization_qc_config`
- `resolve_mae_training_config`
- `resolve_embedding_extraction_config`
- `resolve_clustering_config`
- `resolve_cluster_visualization_config`
- `resolve_f3_facies_inspection_config`

Current public helper or dispatcher functions:

- `validate_config`

After the split, `validate.py` should become a thin compatibility layer that
imports these names from stage modules and re-exports the same `__all__`.
Callers must not need to change imports.

## Current Inventory

Fixed contract and routing constants:

- Common stage routing: `_ALLOWED_TOP_LEVEL`, `_REQUIRED_TOP_LEVEL`,
  `_STAGE_RESOLVERS`
- Fixed amplitude-only/checkpoint contracts: `_FIXED_RAW_KEYS`,
  `_FIXED_DISABLED_NORMALIZATION_KEYS`,
  `_CHECKPOINT_OWNED_EXTRACTION_SECTIONS`
- MAE data and visualization contracts: `_AMPLITUDE_AGC_KEYS`,
  `_AMPLITUDE_AGC_ENABLED_REQUIRED_KEYS`,
  `_MAE_TRAINING_VISUALIZATION_KEYS`
- Clustering contracts: `_CLUSTERING_EMBEDDINGS_KEYS`, `_CLUSTERING_KEYS`,
  `_CLUSTERING_REQUIRED_KEYS`, `_CLUSTERING_RESIDUALIZATION_KEYS`,
  `_CLUSTERING_RESIDUALIZATION_ENABLED_REQUIRED_KEYS`, `_CLUSTERING_PCA_KEYS`
- Cluster visualization contracts: `_VISUALIZATION_CLUSTERING_KEYS`,
  `_VISUALIZATION_KEYS`, `_VISUALIZATION_REQUIRED_KEYS`,
  `_VISUALIZATION_UNDERLAY_KEYS`, `_VISUALIZATION_COMPARISON_KEYS`,
  `_VISUALIZATION_SUMMARY_KEYS`
- F3 inspection contracts: `_F3_FACIES_INSPECTION_TOP_LEVEL`,
  `_F3_FACIES_INSPECTION_REQUIRED_TOP_LEVEL`,
  `_F3_FACIES_INSPECTION_PATH_KEYS`,
  `_F3_FACIES_INSPECTION_OUTPUT_KEYS`,
  `_F3_FACIES_INSPECTION_DATASET_KEYS`,
  `_F3_FACIES_INSPECTION_PUBLISH_KEYS`,
  `_F3_FACIES_INSPECTION_PATH_KEY_SUFFIXES`
- Artifact path constants: `_NOPIMS_DATASET`, `_NOPIMS_PRETRAIN_VERSION`

Stage-aware base helpers:

- `_resolve_base`
- `_ResolvedPaths`
- `_reject_stage_key`
- `_validate_top_level_sections`
- `_reject_legacy_attribute_config`
- `_validate_paths`

Common private helpers:

- `_validate_mapping`
- `_validate_allowed_keys`
- `_validate_required_keys`
- `_validate_required_key`
- `_merge_section_defaults`
- `_required_mapping`
- `_required_child_mapping`
- `_iter_mapping_keys`
- Primitive scalar/list validators: `_validate_non_empty_str`,
  `_validate_positive_int_triplet`, `_validate_nonnegative_int_triplet`,
  `_validate_positive_int_list`, `_validate_unique_positive_int_list`,
  `_validate_nonnegative_int_list`, `_validate_positive_int`,
  `_validate_nonnegative_int`, `_validate_optional_positive_int`,
  `_validate_optional_nonnegative_int`, `_validate_positive_number`,
  `_validate_nonnegative_number`, `_validate_positive_finite_number`,
  `_validate_nonnegative_finite_number`, `_validate_optional_fraction`,
  `_validate_fraction`, `_validate_bool`, `_is_int`, `_is_number`

Path validation helpers:

- `_validate_absolute_path`
- `_validate_non_empty_path`
- `_validate_path`
- `_validate_optional_output_path_under_root`
- `_validate_path_under_root`
- `_is_relative_to`

Artifact path validation helpers:

- `_validate_artifact_output_path`
- `_validate_nopims_checkpoint_path`
- `_validate_nopims_pretraining_path`
- `_validate_nopims_embedding_path`
- `_validate_nopims_clustering_path`
- `_validate_nopims_cluster_visualization_path`
- `_artifact_relative_path`
- `_is_nopims_artifact_path`
- `_validate_artifact_path_matches`
- `_raise_nopims_artifact_path_error`

Checkpoint/resume compatibility helpers:

- `_reject_checkpoint_owned_extraction_sections`
- `_validate_nopims_checkpoint_path`
- `_validate_nopims_pretraining_path`

Stage-specific validation helpers:

- Manifest build: `resolve_manifest_build_config`
- Normalization stats/QC: `resolve_normalization_stats_config`,
  `resolve_normalization_qc_config`, `_validate_normalization`
- MAE training: `resolve_mae_training_config`,
  `_reject_fixed_contract_keys`, `_validate_model`, `_validate_masking`,
  `_validate_train`, `_validate_optional_train_numbers`,
  `_validate_optional_train_seed`, `_validate_optional_train_device`,
  `_validate_loss`, `_validate_loss_target_normalization`,
  `_validate_zero_mask`, `_validate_amplitude_agc`,
  `_validate_mae_training_visualization`,
  `_validate_mae_debug_general_fields`, `_validate_mae_debug_triggers`,
  `_validate_mae_debug_rendering_fields`, `_mae_debug_enabled`,
  `_mae_debug_has_trigger`, `_validate_divisible_crop_patch`,
  `_validate_mae_debug_clip_percentiles`, `_validate_mae_debug_columns`
- Embedding extraction: `resolve_embedding_extraction_config`,
  `_validate_overlap_less_than_window`, `_validate_embedding_output_dtype`
- Clustering: `resolve_clustering_config`,
  `_validate_clustering_residualization`,
  `_validate_clustering_normalization`, `_validate_clustering_method`
- Cluster visualization: `resolve_cluster_visualization_config`,
  `_validate_survey_id_list`, `_validate_visualization_modes`,
  `_validate_slice_coordinate_space`
- F3 inspection: `resolve_f3_facies_inspection_config`,
  `_validate_f3_facies_inspection_top_level_sections`,
  `_validate_f3_facies_inspection_paths`,
  `_validate_f3_facies_inspection_outputs`,
  `_validate_f3_facies_inspection_artifact_paths`,
  `_is_f3_facies_inspection_path_key`, `_validate_f3_facies_dataset`,
  `_validate_f3_facies_inspection_publish`

There are no F3 lithology probe, F3 lithology baseline, or results publish
resolver functions in `validate.py` today. Keep those workflows behaviorally
unchanged; only introduce new config modules for them if a later prompt adds
actual resolver functions.

## Target Module Map

Use names that match existing stage concepts and keep movement incremental.

| Target module | Move from `validate.py` |
| --- | --- |
| `src/seis_ssl_cluster/config/common.py` | Mapping/key/default helpers, primitive scalar/list validators, generic path parsing/root helpers, and generic normalization validator if shared by multiple stages |
| `src/seis_ssl_cluster/config/base.py` | `Config`, `_resolve_base`, `_ResolvedPaths`, top-level section validation, legacy-key rejection, and stage path-key validation |
| `src/seis_ssl_cluster/config/artifact_paths.py` | `_validate_artifact_output_path`, NOPIMS artifact constants and all `_validate_nopims_*`, `_artifact_relative_path`, `_is_nopims_artifact_path`, `_validate_artifact_path_matches`, `_raise_nopims_artifact_path_error`, plus artifact-registry path helpers |
| `src/seis_ssl_cluster/config/manifest.py` | `resolve_manifest_build_config` and manifest-build-only checks |
| `src/seis_ssl_cluster/config/normalization.py` | `resolve_normalization_stats_config`, `resolve_normalization_qc_config`, `_validate_normalization` if it is not kept in `common.py` |
| `src/seis_ssl_cluster/config/pretraining.py` | `resolve_mae_training_config`, fixed raw/default contract checks, model/masking/train/loss/zero-mask/amplitude-AGC validation, MAE debug visualization helpers |
| `src/seis_ssl_cluster/config/embedding.py` | `resolve_embedding_extraction_config`, checkpoint-owned section rejection, embedding window/overlap/output dtype validation |
| `src/seis_ssl_cluster/config/clustering.py` | `resolve_clustering_config`, clustering key constants, residualization, PCA, method, and embedding-normalization validation |
| `src/seis_ssl_cluster/config/cluster_visualization.py` | `resolve_cluster_visualization_config`, visualization key constants, survey ID/mode/slice-coordinate validation |
| `src/seis_ssl_cluster/config/f3_inspection.py` | `resolve_f3_facies_inspection_config`, F3 inspection key constants, F3 path/output/dataset/publish/artifact-path checks |
| `src/seis_ssl_cluster/config/f3_lithology.py` | Reserved for future F3 lithology probe config validation; no current `validate.py` symbols to move |
| `src/seis_ssl_cluster/config/f3_baselines.py` | Reserved for future F3 lithology baseline config validation; no current `validate.py` symbols to move |
| `src/seis_ssl_cluster/config/results.py` | Reserved for future results publish/validation config validation; no current `validate.py` symbols to move |
| `src/seis_ssl_cluster/config/validate.py` | Compatibility imports, `_STAGE_RESOLVERS`, `validate_config`, and `__all__` only after all stage modules exist |

## Compatibility Layer Shape

Once stage modules exist, `validate.py` should keep the current public names:

```python
from .cluster_visualization import resolve_cluster_visualization_config
from .clustering import resolve_clustering_config
from .embedding import resolve_embedding_extraction_config
from .f3_inspection import resolve_f3_facies_inspection_config
from .manifest import resolve_manifest_build_config
from .normalization import (
    resolve_normalization_qc_config,
    resolve_normalization_stats_config,
)
from .pretraining import resolve_mae_training_config
```

`validate_config` may stay in `validate.py` with `_STAGE_RESOLVERS`, or move to a
small dispatcher module and be re-exported from `validate.py`. In either case,
`from seis_ssl_cluster.config.validate import ...` must continue to work.

## Dependency Rules

To avoid circular imports:

- `common.py` contains only primitive validators, common mapping helpers,
  common default merging, and generic path parsing/root helpers. It must not
  import stage modules, stage contract constants, or `ArtifactPaths`.
- `base.py` contains stage-aware top-level routing helpers and may import
  `config.schema` plus primitive helpers from `common.py`. It must not import
  stage modules.
- `artifact_paths.py` contains path and `ArtifactPaths` contract validation. It
  may import `seis_ssl_cluster.paths` and common primitive helpers, but it must
  not import stage modules.
- Stage modules may import `common.py`, `base.py`, `artifact_paths.py`,
  `config.schema`, and `seis_ssl_cluster.paths`.
- Stage modules must not import `validate.py`.
- `validate.py` imports public symbols from stage modules and re-exports them
  for compatibility.

## Non-Goals For The Split

- Do not change resolver behavior, YAML schemas, or default values.
- Do not reintroduce `runs/` as a valid standard path.
- Do not alter `artifacts/` or `results/` operating policy.
- Do not change F3 inspection, F3 lithology, F3 baseline, or results publish
  workflow behavior.
- Do not add fallback import layers beyond the final `validate.py` compatibility
  re-export.

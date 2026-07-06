# Config Validation Split Summary

## 分割前の問題

`src/seis_ssl_cluster/config/validate.py` に、NOPIMS の manifest /
normalization / pretraining / embedding / clustering / visualization と F3
inspection、results publish まわりの validation entrypoint が集中していた。
そのため、stage 固有の validation 変更でも大きな単一 module を読む必要があり、
import cycle のリスクや public import 互換性の確認範囲が分かりにくかった。

今回の確認対象は責務分割後の回帰確認であり、YAML contract、
`ArtifactPaths` の path layout、checkpoint/resume 互換性、F3 downstream
workflow、`artifacts/` と `results/` の運用方針は変更していない。

## 新しいmodule構成

- `config/validate.py`: 後方互換 re-export と `validate_config` dispatcher。
- `config/common.py`: mapping/key/default、primitive 型、数値、list、path root
  validation の共通 helper。
- `config/base.py`: stage 共通の top-level section、paths section、legacy key
  rejection。
- `config/artifact_paths.py`: NOPIMS artifact path contract の stage 別検証。
- `config/artifact_path_validation.py`: artifact path scan CLI 側の validation。
- `config/manifest.py`: NOPIMS manifest build。
- `config/normalization.py`: normalization stats / normalization QC。
- `config/pretraining.py`: MAE pretraining。
- `config/embedding.py`: embedding extraction。
- `config/clustering.py`: clustering。
- `config/cluster_visualization.py`: cluster visualization。
- `config/f3_inspection.py`: F3 inspection。
- `config/f3_lithology.py`: F3 lithology 系の config entrypoint re-export。
- `config/f3_baselines.py`: F3 lithology baseline 系の config entrypoint re-export。
- `config/results.py`: results publish / results artifact validation。

## validate.pyの互換層

`validate.py` は従来の import path を維持する互換層として残っている。
主要 resolver は stage module から直接 import して re-export し、F3
lithology / baseline の proc-owned entrypoint は `__getattr__` で lazy に
re-export する。これにより、従来どおり以下の import が動く。

```python
from seis_ssl_cluster.config.validate import resolve_mae_training_config
from seis_ssl_cluster.config.validate import resolve_embedding_extraction_config
from seis_ssl_cluster.config.validate import resolve_clustering_config
from seis_ssl_cluster.config.validate import resolve_cluster_visualization_config
```

F3 inspection、F3 lithology、F3 baseline、results validation 系の既存 public
entrypoint も `validate.py` から import 可能であることを確認した。

## stage別resolver一覧

- `resolve_manifest_build_config`: `config/manifest.py`
- `resolve_normalization_stats_config`: `config/normalization.py`
- `resolve_normalization_qc_config`: `config/normalization.py`
- `resolve_mae_training_config`: `config/pretraining.py`
- `resolve_embedding_extraction_config`: `config/embedding.py`
- `resolve_clustering_config`: `config/clustering.py`
- `resolve_cluster_visualization_config`: `config/cluster_visualization.py`
- `resolve_f3_facies_inspection_config`: `config/f3_inspection.py`
- `f3_prepare_volume_config_from_mapping`: `config/f3_lithology.py`
- `f3_lithology_token_dataset_config_from_mapping`: `config/f3_lithology.py`
- `f3_lithology_probe_config_from_mapping`: `config/f3_lithology.py`
- `f3_lithology_prediction_config_from_mapping`: `config/f3_lithology.py`
- `f3_lithology_visualization_config_from_mapping`: `config/f3_lithology.py`
- `f3_lithology_report_config_from_mapping`: `config/f3_lithology.py`
- `f3_lithology_publish_config_from_mapping`: `config/f3_lithology.py`
- `f3_lithology_baseline_token_dataset_config_from_mapping`:
  `config/f3_baselines.py`
- `random_mae_checkpoint_config_from_mapping`: `config/f3_baselines.py`
- `f3_lithology_comparison_report_config_from_mapping`:
  `config/f3_baselines.py`
- `f3_lithology_comparison_publish_config_from_mapping`:
  `config/f3_baselines.py`
- `validate_results_artifacts`: `config/results.py`
- `validate_config`: `config/validate.py`

## common helper一覧

- Mapping / key helpers: `_validate_mapping`, `_validate_allowed_keys`,
  `_validate_required_keys`, `_validate_required_key`, `_required_mapping`,
  `_required_child_mapping`, `_iter_mapping_keys`, `_merge_section_defaults`
- Path helpers: `_validate_absolute_path`, `_validate_non_empty_path`,
  `_validate_path`, `_validate_optional_output_path_under_root`,
  `_validate_path_under_root`, `_is_relative_to`
- String / bool / number helpers: `_validate_non_empty_str`,
  `_validate_bool`, `_validate_positive_int`, `_validate_nonnegative_int`,
  `_validate_optional_positive_int`, `_validate_optional_nonnegative_int`,
  `_validate_positive_number`, `_validate_nonnegative_number`,
  `_validate_positive_finite_number`, `_validate_nonnegative_finite_number`,
  `_validate_optional_fraction`, `_validate_fraction`, `_is_int`,
  `_is_number`
- List / tuple helpers: `_validate_positive_int_triplet`,
  `_validate_nonnegative_int_triplet`, `_validate_positive_int_list`,
  `_validate_unique_positive_int_list`, `_validate_nonnegative_int_list`
- Stage base helpers: `_resolve_base`, `_ResolvedPaths`, `_reject_stage_key`,
  `_validate_top_level_sections`, `_reject_legacy_attribute_config`,
  `_validate_paths`
- Artifact path helpers: `_validate_artifact_output_path`,
  `_validate_nopims_checkpoint_path`, `_validate_nopims_pretraining_path`,
  `_validate_nopims_embedding_path`, `_validate_nopims_clustering_path`,
  `_validate_nopims_cluster_visualization_path`, `_artifact_relative_path`,
  `_is_nopims_artifact_path`, `_validate_artifact_path_matches`,
  `_raise_nopims_artifact_path_error`

## path contractとの関係

標準 artifact root は `/workspace/artifacts/seis_ssl_cluster` のままである。
MAE checkpoint は `pretraining/`、embedding は `embeddings/`、clustering は
`clustering/`、cluster visualization は `visualizations/clusters/`、F3
lithology は `lithology/` 配下を使う。`runs` は標準 path として復活させていない。

`results/` は lightweight review artifact 用であり、checkpoint、embedding、
model artifact、raw data、large binary は publish 対象外のまま。今回の変更では
`ArtifactPaths`、path validation、results validation の contract は変更していない。

## 後方互換性

既存の public resolver 名は維持されている。`validate.py` は互換 re-export 層として
残っており、外部 caller は従来どおり
`from seis_ssl_cluster.config.validate import ...` を使用できる。

active config regression として、NOPIMS の `10_pretrain`、`20_embedding`、
`30_clustering`、`40_visualization` と、F3 `facies_benchmark_v1` 配下の
inspection / prepare / embedding / lithology / baseline / random encoder /
comparison config を resolver で解決する test coverage を追加した。

## 実行したtestsと結果

- `python -m compileall -q src proc tests`: pass
- `pytest -q tests/seis_ssl_cluster/test_config.py tests/seis_ssl_cluster/test_config_module_imports.py tests/seis_ssl_cluster/test_active_experiment_configs.py`:
  325 passed
- `pytest -q tests/seis_ssl_cluster/test_config_*.py`: 110 passed
- `PYTHONPATH=src python proc/seis_ssl_cluster/validate_artifact_paths.py --root /workspace/artifacts/seis_ssl_cluster --scan experiments proc docs README.md results --fail-on-runs`:
  pass, `error_count: 0`, `warning_count: 41`
- `PYTHONPATH=src python proc/seis_ssl_cluster/validate_results_artifacts.py --root results --max-file-size-mb 10`:
  pass, `error_count: 0`, `warning_count: 5`
- `pytest -q tests/seis_ssl_cluster/test_artifact_paths.py tests/seis_ssl_cluster/test_artifact_path_validation_cli.py tests/seis_ssl_cluster/test_results_publish.py tests/seis_ssl_cluster/test_results_validation.py`:
  49 passed
- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_NUM_THREADS=1 pytest -q tests/seis_ssl_cluster/test_f3_lithology_token_dataset.py tests/seis_ssl_cluster/test_f3_lithology_probe.py tests/seis_ssl_cluster/test_f3_lithology_baseline_features.py tests/seis_ssl_cluster/test_f3_lithology_baseline_comparison.py`:
  44 passed
- Explicit `seis_ssl_cluster.config.validate` compatibility import check: pass
- Requested grep check for forbidden `runs` active artifact path patterns:
  hits are negative/rejection tests only; active config path としての該当なし

`tests/test_proc_dry_run.py` は実行していない。

## 未解決事項

- Artifact path validation は `error_count: 0` だが、既存の F3 embedding path
  shape、docs 内の legacy `runs` 言及、publish manifest の local source path
  について warning が残っている。いずれも今回の scope では挙動変更しない。
- Results validation は `error_count: 0` だが、既存 README / publish manifest の
  local absolute path marker warning が残っている。これも今回の scope では変更しない。

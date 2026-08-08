# F3 Module Refactor Plan

This document inventories the current F3 inspection, lithology probe, and
baseline comparison modules and records the module map for the staged refactor.
The implemented batches keep CLI names, YAML keys, artifact paths, result
publishing, and metric definitions stable while adding shared IO helpers, a
shared lithology token dataset schema module, and a lithology report subpackage
behind compatibility imports.

The existing F3 label source policy remains unchanged. Teacher labels come from
`f3_labels.sgy` / `f3_facies_labels.npy`, and class `0` remains a valid class,
not an unlabeled sentinel.

## Current F3 Module Structure

Most legacy public F3 library modules remain directly under
`src/seis_ssl_cluster/f3/`. The current implementation also includes
`src/seis_ssl_cluster/f3/io/` for shared label/SEGY/prepare-volume helpers,
`src/seis_ssl_cluster/f3/lithology/token_dataset.py` for the shared token NPZ
schema, and `src/seis_ssl_cluster/f3/lithology/report/` for report internals.
Legacy direct imports remain compatibility wrappers where code has moved.
The proc entrypoints in `proc/seis_ssl_cluster/*f3*.py` load existing YAML
contracts and call the library API, mostly through the aggregate
`seis_ssl_cluster.f3` import surface.

### Source Modules

| Module | Responsibility | Main public API | Local dependencies | Target destination | Compat wrapper |
| --- | --- | --- | --- | --- | --- |
| `src/seis_ssl_cluster/f3/__init__.py` | Aggregate public F3 import surface. | Re-exports dataclasses, constants, and functions from all F3 modules. | All sibling F3 modules. | Keep as aggregate package surface. | Preserve exports; no extra wrapper if submodules move. |
| `src/seis_ssl_cluster/f3/baseline_features.py` | Build baseline feature token datasets from a reference token dataset. Covers `z_only`, `amplitude_stats`, and `xyz_coordinates`. | `F3BaselineReferenceTokenDataset`, `F3BaselineTokenDatasetOutputs`, `F3BaselineFeatureConfig`, `F3LithologyBaselineTokenDatasetConfig`, `F3LithologyBaselineTokenDatasetResult`, `build_f3_lithology_baseline_token_dataset`, `f3_lithology_baseline_token_dataset_config_from_mapping`. | None under `seis_ssl_cluster`. | `src/seis_ssl_cluster/f3/lithology/baselines.py`. | Yes, if moved. |
| `src/seis_ssl_cluster/f3/consistency.py` | Compare PNG label slices with SEGY/NPY teacher label slices and write consistency outputs. | `F3LabelConsistencyFigureConfig`, `F3LabelConsistencyOutputConfig`, `F3LabelConsistencyOutputResult`, `F3LabelSlice`, `F3LabelConsistencyAlignment`, `F3LabelConsistencyRecord`, `F3LabelConsistencyReport`, `extract_teacher_label_slice`, `align_png_class_ids_to_segy_slice`, `check_f3_label_consistency`, `write_f3_label_consistency_outputs`. | `f3.png_labels`, `f3.segy`, `f3.visualization`. | `src/seis_ssl_cluster/f3/inspection/consistency.py`. | Yes, if moved. |
| `src/seis_ssl_cluster/f3/inspection.py` | Inventory raw F3 files and class-info metadata. | `F3FileRecord`, `F3FileInventory`, `F3InventoryOutputConfig`, `scan_f3_file_inventory`, `find_class_info_file`, `write_f3_file_inventory_outputs`, `render_file_inventory_markdown`. | `f3.labels`. | `src/seis_ssl_cluster/f3/inspection/inventory.py`. | Yes, if moved. |
| `src/seis_ssl_cluster/f3/labels.py` | Parse `class_info` payloads and label PNG filenames; provide RGB helpers. | `F3ClassInfo`, `LabelPngNameParts`, `read_class_info`, `parse_class_info_payload`, `parse_label_png_name`, `extract_label_split`, `rgb_to_hex`. | None under `seis_ssl_cluster`. | `src/seis_ssl_cluster/f3/io/labels.py`. | Yes, if moved. |
| `src/seis_ssl_cluster/f3/lithology_prediction.py` | Apply trained lithology probes to token grids, write predictions, and compute validation-slice metrics. | `F3LithologyPredictionInputs`, `F3LithologyPredictionOutputs`, `F3LithologyPredictionConfig`, `F3LithologyPredictionResult`, `predict_f3_lithology_tokens`, `read_f3_lithology_prediction_classes`. | `embedding.sliding_window`, `f3.lithology_tokens`, `f3.metrics`, `f3.splits`. | `src/seis_ssl_cluster/f3/lithology/prediction.py`. | Yes, if moved. |
| `src/seis_ssl_cluster/f3/lithology_probe.py` | Train and evaluate logistic-regression and MLP lithology probes. | `F3LithologyProbeInputs`, `F3LithologyProbeOutputs`, `F3LithologyProbeSettings`, `F3LithologyProbeConfig`, `F3LithologyProbeResult`, `F3IdentityScaler`, `F3TorchMLPClassifier`, `train_and_evaluate_f3_lithology_probe`, `load_token_dataset`. | `f3.metrics`. | `src/seis_ssl_cluster/f3/lithology/probe.py`. | Yes, if moved. |
| `src/seis_ssl_cluster/f3/lithology_report.py` | Build/publish lithology reports and baseline comparison reports; render comparison figures and Markdown. | `F3LithologyReportConfig`, `F3LithologyPublishConfig`, `F3LithologyReportResult`, `F3LithologyComparisonReportConfig`, `F3LithologyComparisonPublishConfig`, `F3LithologyComparisonReportResult`, `build_f3_lithology_report`, `publish_f3_lithology_report`, `build_f3_lithology_comparison_report`, `publish_f3_lithology_comparison_report`, `render_f3_lithology_report_markdown`. | `results`. | Split under `src/seis_ssl_cluster/f3/lithology/report/`: `metrics_loader.py`, `figures.py`, `markdown.py`, `comparison.py`, `publish.py`. Keep a small package-level facade. | Yes, highest risk if moved. |
| `src/seis_ssl_cluster/f3/lithology_tokens.py` | Build token-level lithology datasets from embeddings and labels; define token policy and token arrays. | `F3LithologyTokenPolicy`, `F3LithologyTokenDatasetInputs`, `F3LithologyTokenDatasetOutputs`, `F3ReferenceTokenDataset`, `F3LithologyTokenDatasetConfig`, `F3EmbeddingArtifact`, `F3SliceTokenization`, `F3TokenArrays`, `F3LithologyTokenDatasetResult`, `build_f3_lithology_token_dataset`, `tokenize_f3_lithology_slice`, `load_f3_embedding_artifacts`, `read_f3_lithology_class_info`, `write_f3_lithology_token_quicklooks`. | `embedding.sliding_window`, `f3.labels`, `f3.splits`, `f3.tokenization`, `f3.visualization`. | `src/seis_ssl_cluster/f3/lithology/token_dataset.py` plus shared token helpers in `src/seis_ssl_cluster/f3/lithology/tokens.py`. | Yes, if moved. |
| `src/seis_ssl_cluster/f3/lithology_visualization.py` | Render lithology prediction slices and figure sidecars. | `F3LithologyVisualizationInputs`, `F3LithologyVisualizationOutputs`, `F3LithologyVisualizationFigureConfig`, `F3LithologyVisualizationConfig`, `F3LithologyVisualizationResult`, `F3LithologySliceFigure`, `visualize_f3_lithology_predictions`, `read_f3_lithology_visualization_classes`. | `f3.lithology_tokens`, `f3.metrics`, `f3.splits`, `f3.visualization`. | `src/seis_ssl_cluster/f3/lithology/visualization.py`. | Yes, if moved. |
| `src/seis_ssl_cluster/f3/metrics.py` | Compute lithology metrics and write metric CSV/Markdown artifacts. | `compute_lithology_metrics`, `write_metrics_csv`, `write_confusion_matrix_csv`, `render_classification_report_markdown`. | None under `seis_ssl_cluster`. | `src/seis_ssl_cluster/f3/lithology/metrics.py`. | Yes, if moved. |
| `src/seis_ssl_cluster/f3/png_labels.py` | Inspect PNG label images, map RGB colors to class IDs, and write PNG label summaries. | `PngLabelUnknownColor`, `PngLabelClassCount`, `PngLabelMap`, `PngLabelFileInspection`, `F3PngLabelInspection`, `F3PngLabelOutputConfig`, `rgb_to_class_id_map`, `normalize_png_rgb`, `count_class_pixels`, `read_png_rgb`, `inspect_f3_png_labels`, `write_f3_png_label_inspection_outputs`, `save_png_label_distribution_figures`. | `f3.inspection`, `f3.labels`. | `src/seis_ssl_cluster/f3/inspection/png_labels.py`. | Yes, if moved. |
| `src/seis_ssl_cluster/f3/prepare_volume.py` | Prepare F3 seismic/label NPY volumes, manifests, metadata, and normalization statistics. | `F3PrepareRootPaths`, `F3PrepareInputPaths`, `F3PrepareOutputPaths`, `F3PrepareDatasetConfig`, `F3PrepareNormalizationConfig`, `F3PrepareVolumeConfig`, `F3PrepareVolumeResult`, `prepare_f3_facies_volume`, `f3_prepare_volume_config_from_mapping`. | `config.schema`, `data.normalization`, `data.schema`, `data.volume_store`, `f3.labels`, `f3.segy`. | `src/seis_ssl_cluster/f3/io/prepare_volume.py`. | Yes, if moved. |
| `src/seis_ssl_cluster/f3/report.py` | Build/publish the inspection report from inspection artifacts. | `F3InspectionReportConfig`, `F3InspectionPublishConfig`, `F3InspectionReportResult`, `build_f3_inspection_report`, `publish_f3_inspection_report`, `render_f3_inspection_report_markdown`. | `results`. | `src/seis_ssl_cluster/f3/inspection/report.py`. | Yes, if moved. |
| `src/seis_ssl_cluster/f3/segy.py` | Locate/read F3 SEGY files and inspect seismic/label geometry and value ranges. | `F3SegyPaths`, `F3SegyGeometry`, `F3SegyFileInspection`, `F3SegyInspection`, `F3SegyInspectionOutputConfig`, `axis_assumption_metadata`, `calculate_seismic_amplitude_stats`, `calculate_label_unique_values`, `find_f3_segy_paths`, `read_f3_segy_file`, `inspect_f3_segy_files`, `write_f3_segy_inspection_outputs`. | `f3.inspection`, `f3.labels`. | `src/seis_ssl_cluster/f3/io/segy.py`. | Yes, if moved. |
| `src/seis_ssl_cluster/f3/splits.py` | Load slice split manifests and resolve inline/crossline geometry to array indices. | `F3SliceSplitRecord`, `F3LineGeometry`, `load_f3_slice_split_records`, `read_f3_line_geometry`, `f3_line_geometry_from_mapping`, `resolve_f3_slice_array_index`, `f3_slice_split_manifest`. | `f3.labels`. | `src/seis_ssl_cluster/f3/lithology/tokens.py` or `src/seis_ssl_cluster/f3/io/labels.py`; keep separate if both inspection and lithology continue to share it. | Yes, if moved. |
| `src/seis_ssl_cluster/f3/tokenization.py` | Preview and apply tokenization of label slices. | `F3TokenizationConfig`, `F3TokenizationFigureConfig`, `F3TokenizationOutputConfig`, `F3TokenizationOutputResult`, `F3TokenPlaneSpec`, `F3TokenizationAlignment`, `F3TokenizationSliceResult`, `F3TokenizationPreviewRecord`, `token_plane_spec`, `tokenize_label_slice`, `load_f3_label_consistency_alignments`, `write_f3_tokenization_preview_outputs`, `render_tokenization_summary_markdown`, `apply_tokenization_alignment`. | `f3.png_labels`, `f3.visualization`. | `src/seis_ssl_cluster/f3/inspection/tokenization_preview.py` for preview/reporting and `src/seis_ssl_cluster/f3/lithology/tokens.py` for shared token helpers. | Yes, if moved. |
| `src/seis_ssl_cluster/f3/visualization.py` | Shared F3 visualization helpers for quicklook, label alignment, RGB conversion, and legends. | `F3QuicklookFigureConfig`, `F3QuicklookOutputConfig`, `F3DisplaySlice`, `F3ResolvedLineIndex`, `F3PngLabelAlignment`, `F3QuicklookResult`, `make_orthogonal_display_slice`, `make_teacher_seismic_display_slice`, `resolve_teacher_line_index`, `align_png_label_to_seismic_slice`, `class_id_image_to_rgb`, `facies_legend_labels`, `write_f3_quicklook_outputs`. | `f3.labels`, `f3.png_labels`, `f3.segy`. | `src/seis_ssl_cluster/f3/inspection/visualization.py`; shared palette helpers may stay package-level if lithology still imports them. | Yes, if moved. |

### Shared Modules Added By The Refactor

| Module | Responsibility | Main public API | Compatibility |
| --- | --- | --- | --- |
| `src/seis_ssl_cluster/f3/io/labels.py` | Shared class-info and PNG label-name parsing helpers. | `F3ClassInfo`, `LabelPngNameParts`, `read_class_info`, `parse_class_info_payload`, `parse_label_png_name`, `extract_label_split`, `rgb_to_hex`. | Re-exported by `src/seis_ssl_cluster/f3/labels.py`. |
| `src/seis_ssl_cluster/f3/io/prepare_volume.py` | Shared F3 volume preparation implementation. | `prepare_f3_facies_volume`, `f3_prepare_volume_config_from_mapping`, and prepare-volume dataclasses. | Re-exported by `src/seis_ssl_cluster/f3/prepare_volume.py`. |
| `src/seis_ssl_cluster/f3/io/segy.py` | Shared SEGY path, cube-read, axis, and stats helpers plus the current SEGY inspection facade. | `find_f3_segy_paths`, `read_f3_segy_file`, `calculate_seismic_amplitude_stats`, `calculate_label_unique_values`, `inspect_f3_segy_files`, `write_f3_segy_inspection_outputs`. | Re-exported by `src/seis_ssl_cluster/f3/segy.py`; a later inspection/io split should move inspection-only rendering and output writing out of `io`. |
| `src/seis_ssl_cluster/f3/lithology/token_dataset.py` | Shared token NPZ schema load/save/validation and feature replacement helpers. | `F3LithologyTokenDataset`, `load_f3_lithology_token_dataset`, `save_f3_lithology_token_dataset`, `validate_f3_lithology_token_dataset`, `replace_token_features`. | Re-exported through `seis_ssl_cluster.f3` and `seis_ssl_cluster.f3.lithology`. |
| `src/seis_ssl_cluster/f3/lithology/report/` | Lithology report internals split by metrics loading, figure generation, Markdown rendering, comparison aggregation, and publishing. | Package facade exports `F3LithologyReportConfig`, `F3LithologyComparisonReportConfig`, `build_f3_lithology_report`, `build_f3_lithology_comparison_report`, publish configs/functions, figure style helpers, and Markdown rendering. | Re-exported by `src/seis_ssl_cluster/f3/lithology_report.py`; the aggregate `seis_ssl_cluster.f3` surface remains unchanged. |

### Proc Entrypoints

All existing proc entrypoints and argument surfaces must remain stable. Future
module moves should only update imports behind these entrypoints, not CLI names,
YAML keys, output paths, or dry-run behavior.

| Entrypoint | Responsibility | Main public API | Library/config dependencies | Target destination | Compat wrapper |
| --- | --- | --- | --- | --- | --- |
| `proc/seis_ssl_cluster/build_f3_inspection_report.py` | Build and optionally publish the F3 inspection report. | `build_parser`, `main`. | `cli`, `config`, `config.schema`, `f3`. | Keep proc path. | Not applicable. |
| `proc/seis_ssl_cluster/build_f3_lithology_baseline_features.py` | Backward-facing CLI alias for baseline feature token dataset builds. | `build_parser`; delegates `main`. | Delegates to `build_f3_lithology_baseline_token_dataset.py`. | Keep proc path. | Keep alias. |
| `proc/seis_ssl_cluster/build_f3_lithology_baseline_token_dataset.py` | Build baseline token datasets from reference token datasets. | `build_parser`, `main`. | `cli`, `config`, `f3`. | Keep proc path. | Not applicable. |
| `proc/seis_ssl_cluster/build_f3_lithology_comparison_report.py` | Build and publish baseline comparison report artifacts. | `build_parser`, `main`. | `cli`, `config`, `config.f3_baselines`, `f3`, `paths`. | Keep proc path. | Not applicable. |
| `proc/seis_ssl_cluster/build_f3_lithology_report.py` | Build/publish one lithology run report. | `build_parser`, `main`. | `cli`, `config`, `config.f3_lithology`, `f3`. | Keep proc path. | Not applicable. |
| `proc/seis_ssl_cluster/build_f3_lithology_token_dataset.py` | Build token dataset from embedding artifacts and F3 labels. | `build_parser`, `main`. | `cli`, `config`, `config.f3_lithology`, `f3`. | Keep proc path. | Not applicable. |
| `proc/seis_ssl_cluster/check_f3_label_consistency.py` | Run PNG-vs-teacher label consistency checks. | `build_parser`, `main`. | `cli`, `config`, `config.schema`, `f3`. | Keep proc path. | Not applicable. |
| `proc/seis_ssl_cluster/inspect_f3_files.py` | Inventory F3 input files. | `build_parser`, `main`. | `cli`, `config`, `config.schema`, `f3`. | Keep proc path. | Not applicable. |
| `proc/seis_ssl_cluster/inspect_f3_png_labels.py` | Inspect PNG label colors and summaries. | `build_parser`, `main`. | `cli`, `config`, `config.schema`, `f3`. | Keep proc path. | Not applicable. |
| `proc/seis_ssl_cluster/inspect_f3_segy_geometry.py` | Inspect SEGY geometry and label values. | `build_parser`, `main`. | `cli`, `config`, `config.schema`, `f3`. | Keep proc path. | Not applicable. |
| `proc/seis_ssl_cluster/predict_f3_lithology_tokens.py` | Predict lithology token classes over volume grids. | `build_parser`, `main`. | `cli`, `config`, `config.f3_lithology`, `f3`. | Keep proc path. | Not applicable. |
| `proc/seis_ssl_cluster/prepare_f3_facies_volume.py` | Prepare F3 facies benchmark volume artifacts. | `build_parser`, `main`. | `cli`, `config`, `f3`. | Keep proc path. | Not applicable. |
| `proc/seis_ssl_cluster/preview_f3_tokenization.py` | Generate tokenization preview outputs. | `build_parser`, `main`. | `cli`, `config`, `config.schema`, `f3`. | Keep proc path. | Not applicable. |
| `proc/seis_ssl_cluster/train_f3_lithology_probe.py` | Train lithology probe models. | `build_parser`, `main`. | `cli`, `config`, `config.f3_lithology`, `f3`. | Keep proc path. | Not applicable. |
| `proc/seis_ssl_cluster/visualize_f3_lithology_predictions.py` | Visualize lithology predictions. | `build_parser`, `main`. | `cli`, `config`, `config.f3_lithology`, `f3`. | Keep proc path. | Not applicable. |
| `proc/seis_ssl_cluster/visualize_f3_quicklook.py` | Render inspection quicklook figures. | `build_parser`, `main`. | `cli`, `config`, `config.schema`, `f3`. | Keep proc path. | Not applicable. |

### Tests

The existing focused tests cover the current import surface and behavior. Future
module moves should update only internal imports in tests where necessary; tests
that intentionally assert public imports from `seis_ssl_cluster.f3` should keep
that expectation.

| Test file | Coverage area | Current imports | Target after refactor |
| --- | --- | --- | --- |
| `tests/seis_ssl_cluster/test_f3_file_inventory.py` | File inventory, class-info parsing, proc output writing. | `seis_ssl_cluster.f3`. | Inspection inventory facade. |
| `tests/seis_ssl_cluster/test_f3_inspection_report.py` | Inspection report JSON/Markdown, readiness, publishing, proc dry run. | `seis_ssl_cluster.f3`. | Inspection report facade. |
| `tests/seis_ssl_cluster/test_f3_label_consistency.py` | PNG/SEGY alignment, mismatch handling, class `0` behavior, consistency output. | `seis_ssl_cluster.f3`. | Inspection consistency facade. |
| `tests/seis_ssl_cluster/test_f3_lithology_baseline_comparison.py` | Comparison report tables, figures, publishing, and explicit paths. | `seis_ssl_cluster.f3`, `seis_ssl_cluster.f3.lithology_report`. | Lithology report/comparison facade plus old wrapper. |
| `tests/seis_ssl_cluster/test_f3_lithology_baseline_contract.py` | Baseline output layout and metadata contract. | None from package. | No change expected. |
| `tests/seis_ssl_cluster/test_f3_lithology_baseline_features.py` | Baseline feature builders and config validation. | `seis_ssl_cluster.f3`. | Lithology baselines facade. |
| `tests/seis_ssl_cluster/test_f3_lithology_prediction.py` | Prediction grids, metadata, metrics, class loading validation. | `seis_ssl_cluster.f3`. | Lithology prediction facade. |
| `tests/seis_ssl_cluster/test_f3_lithology_probe.py` | Probe training, MLP class weighting, metrics, invalid configs. | `seis_ssl_cluster.f3`. | Lithology probe facade. |
| `tests/seis_ssl_cluster/test_f3_lithology_report.py` | Lithology report Markdown/JSON, publishing, comparison aggregation, proc dry run. | `seis_ssl_cluster.f3`. | Lithology report facade. |
| `tests/seis_ssl_cluster/test_f3_lithology_token_dataset.py` | Token dataset schema, split behavior, class `0`, reference dataset reuse. | `seis_ssl_cluster.f3`. | Lithology token dataset facade. |
| `tests/seis_ssl_cluster/test_f3_lithology_visualization.py` | Prediction visualization outputs and class loading validation. | `seis_ssl_cluster.f3`. | Lithology visualization facade. |
| `tests/seis_ssl_cluster/test_f3_png_labels.py` | RGB-to-class mapping, PNG filename parsing, PNG summary outputs. | `seis_ssl_cluster.f3`. | Inspection PNG label facade. |
| `tests/seis_ssl_cluster/test_f3_prepare_volume.py` | Prepare-volume proc, explicit path handling, missing SEGY errors. | `seis_ssl_cluster.config`, `seis_ssl_cluster.data`, `seis_ssl_cluster.f3`. | IO prepare-volume facade. |
| `tests/seis_ssl_cluster/test_f3_quicklook_visualization.py` | Quicklook slices, label alignment, legends, proc dry run. | `seis_ssl_cluster.f3`. | Inspection visualization facade. |
| `tests/seis_ssl_cluster/test_f3_segy_inspection.py` | SEGY stats, axis assumptions, missing files, proc outputs. | `seis_ssl_cluster.f3`. | IO SEGY facade. |
| `tests/seis_ssl_cluster/test_f3_tokenization_preview.py` | Tokenization preview, class `0`, output summaries, proc dry run. | `seis_ssl_cluster.f3`. | Inspection preview facade plus shared lithology token helpers. |

### Experiment YAML

Existing YAML files are part of the public workflow contract and should not be
renamed or reshaped by the refactor. The current F3 YAML inventory is:

| Area | YAML files | Primary config sections |
| --- | --- | --- |
| Inspection | `experiments/f3/facies_benchmark_v1/00_inspection/01_inspect_files.yaml`, `02_inspect_segy_geometry.yaml`, `03_inspect_png_labels.yaml`, `04_make_quicklook_figures.yaml`, `05_check_label_consistency.yaml`, `06_make_tokenization_preview.yaml`, `07_build_inspection_report.yaml`. | `paths`, `outputs`, `dataset`, `inspection`, and `publish` for the report step. |
| Prepare | `experiments/f3/facies_benchmark_v1/10_prepare/01_prepare_f3_volume.yaml`. | `paths`, `inputs`, `outputs`, `dataset`, `normalization`. |
| F3 embedding | `experiments/f3/facies_benchmark_v1/20_embedding/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/overlap_x16.yaml`. | `paths`, `manifests`, `embeddings`, `embedding`. |
| Lithology probe | `experiments/f3/facies_benchmark_v1/50_lithology/amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1/overlap_x16/png_slices_segy_labels_v1/01_build_token_dataset.yaml` through `06_build_lithology_report.yaml`. | `paths`, `dataset`, `model`, `embeddings`, `labels`, `registry`, `lithology`, `token_dataset`, `probe`, `evaluation`, `predictions`, `visualizations`, `reports`, `publish`. |
| Baseline comparison | `experiments/f3/facies_benchmark_v1/50_lithology_baselines/05_build_baseline_comparison_report.yaml`. | `paths`, `dataset`, `comparison`, `publish`. |
| Baseline feature runs | `experiments/f3/facies_benchmark_v1/50_lithology_baselines/{amplitude_stats_v1,xyz_coordinates_v1,z_only_v1}/01_build_baseline_token_dataset.yaml`, `02_train_linear_probe.yaml`, `03_build_report.yaml`. | Reference token dataset, baseline feature, probe, report, and comparison sections. |
| Random encoder baseline | `experiments/f3/facies_benchmark_v1/50_lithology_baselines/random_encoder_amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_seed42_v1/01_create_random_checkpoint.yaml` through `05_build_report.yaml`. | Random checkpoint, embedding extraction, token dataset, probe, report sections. |

## Target Module Structure

The target structure should group responsibilities without over-splitting the
current code. The intended destination map is:

```text
src/seis_ssl_cluster/f3/
├── __init__.py
├── io/
│   ├── __init__.py
│   ├── segy.py
│   ├── labels.py
│   └── prepare_volume.py
├── inspection/
│   ├── __init__.py
│   ├── inventory.py
│   ├── png_labels.py
│   ├── consistency.py
│   ├── tokenization_preview.py
│   ├── visualization.py
│   └── report.py
└── lithology/
    ├── __init__.py
    ├── token_dataset.py
    ├── tokens.py
    ├── baselines.py
    ├── probe.py
    ├── prediction.py
    ├── visualization.py
    ├── metrics.py
    └── report/
        ├── __init__.py
        ├── metrics_loader.py
        ├── figures.py
        ├── markdown.py
        ├── comparison.py
        └── publish.py
```

If a future implementation shows that a proposed split only moves complexity
without reducing coupling, keep the smaller number of modules. In particular,
`lithology/report/` should be split only around existing seams: metric loading,
figure generation, Markdown rendering, comparison aggregation, and publishing.

## Move Order

1. **Token dataset schema commonality.** Stabilize the shared schema used by
   pretrained token datasets and baseline token datasets before moving files.
   This includes token arrays, split manifest handling, metadata fields, feature
   source metadata, and class-count outputs. The move target is
   `f3/lithology/token_dataset.py`, `f3/lithology/tokens.py`, and
   `f3/lithology/baselines.py`.
2. **Lithology report split.** Extract report internals after the token dataset
   schema is stable. Keep generated Markdown, JSON, CSV, figure names, publish
   manifests, and metric definitions identical. The move target is
   `f3/lithology/report/`.
3. **Inspection and IO module organization.** Move lower-level F3 file, SEGY,
   label, PNG, quicklook, consistency, tokenization preview, and inspection
   report modules into `f3/io/` and `f3/inspection/`. Keep the proc entrypoints
   and experiment YAML paths unchanged.
4. **Old import compatibility wrappers.** After each move, leave the old module
   path as a small re-export wrapper until downstream imports have been migrated
   intentionally. Remove wrappers only in a separately planned breaking change.

## Compatibility Policy

- Preserve `import seis_ssl_cluster.f3` and every name exported through
  `seis_ssl_cluster.f3.__all__`.
- Preserve direct public module imports such as
  `seis_ssl_cluster.f3.lithology_report`,
  `seis_ssl_cluster.f3.lithology_tokens`, and
  `seis_ssl_cluster.f3.segy` by leaving one-hop re-export wrappers when modules
  move.
- Wrappers should not implement alternate behavior, fallback path discovery, or
  compatibility data transformations. They should import from the new module and
  expose the same public names.
- Existing proc paths under `proc/seis_ssl_cluster/` remain the CLI contract.
  Module moves may update internal imports only.
- Existing YAML keys, artifact paths, result paths, figure
  names, and report filenames remain unchanged.
- Do not restore `runs/` as a standard path. `artifacts/` remains the generated
  local-output area, and `results/` remains the lightweight review-output area.

## Test Policy

For the original documentation-only inventory step, run:

```bash
python -m compileall -q src proc tests

PYTHONPATH=src pytest -q \
  tests/seis_ssl_cluster/test_config.py \
  tests/seis_ssl_cluster/test_active_experiment_configs.py
```

Do not run `tests/test_proc_dry_run.py`.

For future module moves, add or update focused tests around the moved module and
then run the relevant F3 tests for that workflow. Keep at least one import test
for the old direct module path and one import/export test for
`seis_ssl_cluster.f3`. The current compatibility test covers old direct imports
for SEGY, PNG labels, consistency, lithology tokens, baseline features,
lithology probe, and lithology report, plus the implemented new imports under
`f3.io`, `f3.lithology.token_dataset`, and `f3.lithology.report`.

## Non-Goals

- No token dataset artifact format migration.
- No changes to existing CLI names, proc entrypoints, YAML files, artifact
  layout, result publishing, metric definitions, or report contents.
- No change to F3 label source policy.
- No change to class `0` handling.
- No Hydra, Pydantic, `console_scripts`, or broad framework migration.

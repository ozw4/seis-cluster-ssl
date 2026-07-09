from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from proc.seis_ssl_cluster.run_f3_lithology_label_budget_probes import (
	run_f3_lithology_label_budget_probes,
)
from proc.seis_ssl_cluster.run_f3_lithology_split_sweep_probes import (
	f3_lithology_split_sweep_probe_config_from_mapping,
	run_f3_lithology_split_sweep_probes,
)
from proc.seis_ssl_cluster.summarize_f3_lithology_label_budget_robustness import (
	summarize_label_budget_robustness,
)
from proc.seis_ssl_cluster.summarize_f3_lithology_split_robustness import (
	summarize_split_robustness,
)
from seis_ssl_cluster.config.f3_lithology_robustness import (
	f3_lithology_label_budget_config_from_mapping,
	f3_lithology_label_budget_probe_config_from_mapping,
	f3_lithology_split_inventory_config_from_mapping,
	f3_lithology_split_sweep_dataset_config_from_mapping,
)
from seis_ssl_cluster.f3.lithology.robustness import (
	build_f3_lithology_label_budget_datasets,
	build_f3_lithology_split_inventories,
	build_f3_lithology_split_sweep_datasets,
	load_token_dataset_npz,
	paired_token_identity_hash,
	save_token_dataset_npz,
)
from seis_ssl_cluster.f3.lithology.token_dataset import F3LithologyTokenDataset
from seis_ssl_cluster.f3.splits import load_f3_slice_split_records

REPO_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = REPO_ROOT / 'proc' / 'seis_ssl_cluster'


def test_label_budget_flow_runs_end_to_end_on_synthetic_tokens(
	tmp_path: Path,
) -> None:
	baseline, candidate = _write_label_budget_sources(tmp_path)
	output_root = tmp_path / 'label_budget_out'
	build_config = _label_budget_config(baseline, candidate, output_root)

	build_result = build_f3_lithology_label_budget_datasets(build_config)
	assert len(build_result.dataset_roots) == build_config.expected_dataset_count
	assert all(root.is_dir() for root in build_result.dataset_roots)

	suite_rows = _read_json(output_root / 'suite_manifest.json')['rows']
	_hashes_match_by_condition(
		suite_rows,
		key_fields=('budget_id', 'subsample_seed'),
	)

	probe_config = f3_lithology_label_budget_probe_config_from_mapping(
		_label_budget_probe_mapping(output_root, _write_class_info(tmp_path)),
	)
	run_f3_lithology_label_budget_probes(probe_config)

	summary = summarize_label_budget_robustness(output_root)
	assert summary.pair_count == 2
	assert summary.paired_deltas_csv.is_file()

	paired_rows = _read_csv(summary.paired_metrics_csv)
	for field in ('accuracy', 'balanced_accuracy', 'macro_f1', 'mean_iou'):
		assert all(np.isfinite(float(row[field])) for row in paired_rows)

	delta_rows = _read_csv(summary.paired_deltas_csv)
	mean_delta = np.mean([float(row['delta_macro_f1']) for row in delta_rows])
	assert mean_delta > 0.0


def test_split_index_flow_runs_end_to_end_on_synthetic_artifacts(
	tmp_path: Path,
) -> None:
	paths = _write_split_fixture(tmp_path)
	inventory_config = _split_inventory_config(tmp_path, paths)

	inventory_result = build_f3_lithology_split_inventories(inventory_config)
	assert len(inventory_result.inventory_paths) == 2
	for inventory in inventory_result.inventory_paths:
		assert load_f3_slice_split_records(inventory)

	dataset_config = _split_dataset_config(
		tmp_path,
		paths,
		inventory_result.manifest_json,
	)
	dataset_result = build_f3_lithology_split_sweep_datasets(dataset_config)
	assert len(dataset_result.dataset_roots) == 4

	dataset_rows = _read_json(dataset_result.manifest_json)['rows']
	_hashes_match_by_condition(dataset_rows, key_fields=('split_id',))

	probe_config = f3_lithology_split_sweep_probe_config_from_mapping(
		_split_probe_mapping(dataset_config.output_root, _write_class_info(tmp_path)),
	)
	run_f3_lithology_split_sweep_probes(probe_config)

	summary = summarize_split_robustness(dataset_config.output_root)
	assert summary.pair_count == 2
	assert summary.paired_deltas_csv.is_file()
	assert {
		row['split_id'] for row in _read_csv(summary.paired_deltas_csv)
	} == {'split_000', 'split_001'}


def test_label_budget_builder_is_deterministic_for_selected_rows_and_hashes(
	tmp_path: Path,
) -> None:
	baseline, candidate = _write_label_budget_sources(tmp_path)
	outputs = []
	for index in range(2):
		config = _label_budget_config(
			baseline,
			candidate,
			tmp_path / f'label_budget_out_{index}',
			overwrite=True,
		)
		build_f3_lithology_label_budget_datasets(config)
		outputs.append(_selected_rows_and_hashes(config.output_root))

	assert outputs[0] == outputs[1]


def test_split_inventory_generator_is_deterministic_for_validation_slices(
	tmp_path: Path,
) -> None:
	paths = _write_split_fixture(tmp_path)
	results = []
	for index in range(2):
		config = _split_inventory_config(tmp_path, paths, output_name=f'splits_{index}')
		build_f3_lithology_split_inventories(config)
		results.append(_validation_slices_by_split(config.output_root))

	assert results[0] == results[1]


def test_missing_split_pair_fails_before_probe_outputs_are_written(
	tmp_path: Path,
) -> None:
	paths = _write_split_fixture(tmp_path)
	inventory_result = build_f3_lithology_split_inventories(
		_split_inventory_config(tmp_path, paths),
	)
	dataset_config = _split_dataset_config(
		tmp_path,
		paths,
		inventory_result.manifest_json,
	)
	build_f3_lithology_split_sweep_datasets(dataset_config)
	manifest_path = dataset_config.output_root / 'split_dataset_manifest.json'
	manifest = _read_json(manifest_path)
	manifest['rows'] = [
		row
		for row in manifest['rows']
		if not (
			row['split_id'] == 'split_001'
			and row['model_role'] == 'candidate'
		)
	]
	_write_json(manifest_path, manifest)
	config = f3_lithology_split_sweep_probe_config_from_mapping(
		_split_probe_mapping(dataset_config.output_root, _write_class_info(tmp_path)),
	)

	with pytest.raises(ValueError, match='requires baseline and candidate rows'):
		run_f3_lithology_split_sweep_probes(config)

	assert not (dataset_config.output_root / 'probes').exists()


def test_mismatched_split_pair_hash_fails_before_probe_outputs_are_written(
	tmp_path: Path,
) -> None:
	paths = _write_split_fixture(tmp_path)
	inventory_result = build_f3_lithology_split_inventories(
		_split_inventory_config(tmp_path, paths),
	)
	dataset_config = _split_dataset_config(
		tmp_path,
		paths,
		inventory_result.manifest_json,
	)
	build_f3_lithology_split_sweep_datasets(dataset_config)
	manifest_path = dataset_config.output_root / 'split_dataset_manifest.json'
	manifest = _read_json(manifest_path)
	for row in manifest['rows']:
		if row['split_id'] == 'split_001' and row['model_role'] == 'candidate':
			row['paired_identity_hash'] = 'mismatched'
			break
	_write_json(manifest_path, manifest)
	config = f3_lithology_split_sweep_probe_config_from_mapping(
		_split_probe_mapping(dataset_config.output_root, _write_class_info(tmp_path)),
	)

	with pytest.raises(ValueError, match='paired_identity_hash mismatch'):
		run_f3_lithology_split_sweep_probes(config)

	assert not (dataset_config.output_root / 'probes').exists()


@pytest.mark.parametrize(
	'script_name',
	[
		'build_f3_lithology_label_budget_datasets.py',
		'run_f3_lithology_label_budget_probes.py',
		'generate_f3_lithology_split_inventories.py',
		'build_f3_lithology_split_sweep_datasets.py',
		'run_f3_lithology_split_sweep_probes.py',
	],
)
def test_config_cli_missing_config_path_is_helpful(script_name: str) -> None:
	completed = _run_cli(PROC_DIR / script_name)

	assert completed.returncode != 0
	assert '--config' in completed.stderr
	assert 'required' in completed.stderr


@pytest.mark.parametrize(
	'script_name',
	[
		'summarize_f3_lithology_label_budget_robustness.py',
		'summarize_f3_lithology_split_robustness.py',
	],
)
def test_summary_cli_missing_suite_root_is_helpful(script_name: str) -> None:
	completed = _run_cli(PROC_DIR / script_name)

	assert completed.returncode != 0
	assert '--suite-root' in completed.stderr
	assert 'required' in completed.stderr


def test_robustness_clis_dry_run_without_writing_outputs(tmp_path: Path) -> None:
	specs = _write_dry_run_cli_specs(tmp_path)

	for script_name, args, expected_stdout, unwritten_paths in specs:
		completed = _run_cli(PROC_DIR / script_name, *args)

		assert completed.returncode == 0, (
			script_name,
			completed.stdout,
			completed.stderr,
		)
		assert expected_stdout in completed.stdout
		for path in unwritten_paths:
			assert not path.exists(), f'{script_name} wrote {path}'


@pytest.mark.parametrize(
	'script_name,config_writer',
	[
		(
			'build_f3_lithology_label_budget_datasets.py',
			'_write_invalid_label_budget_build_config',
		),
		(
			'run_f3_lithology_label_budget_probes.py',
			'_write_invalid_label_budget_probe_config',
		),
		(
			'generate_f3_lithology_split_inventories.py',
			'_write_invalid_split_inventory_config',
		),
		(
			'build_f3_lithology_split_sweep_datasets.py',
			'_write_invalid_split_dataset_config',
		),
		(
			'run_f3_lithology_split_sweep_probes.py',
			'_write_invalid_split_probe_config',
		),
	],
)
def test_config_clis_reject_invalid_output_root(
	tmp_path: Path,
	script_name: str,
	config_writer: str,
) -> None:
	config_path = globals()[config_writer](tmp_path)

	completed = _run_cli(
		PROC_DIR / script_name,
		'--config',
		str(config_path),
		'--dry-run',
	)

	assert completed.returncode != 0
	assert 'output_root must be an absolute path' in completed.stderr
	assert not (tmp_path / 'relative').exists()


@pytest.mark.parametrize(
	'script_name,missing_manifest',
	[
		(
			'summarize_f3_lithology_label_budget_robustness.py',
			'suite_manifest.json',
		),
		(
			'summarize_f3_lithology_split_robustness.py',
			'split_dataset_manifest.json',
		),
	],
)
def test_summary_clis_reject_invalid_suite_root(
	tmp_path: Path,
	script_name: str,
	missing_manifest: str,
) -> None:
	suite_root = tmp_path / 'missing_suite_root'

	completed = _run_cli(
		PROC_DIR / script_name,
		'--suite-root',
		str(suite_root),
		'--dry-run',
	)

	assert completed.returncode != 0
	assert missing_manifest in completed.stderr


def _write_dry_run_cli_specs(
	tmp_path: Path,
) -> list[tuple[str, tuple[str, ...], str, tuple[Path, ...]]]:
	baseline, candidate = _write_label_budget_sources(tmp_path)
	class_info = _write_class_info(tmp_path)
	label_build_config = tmp_path / 'label_budget_build.yaml'
	label_build_root = tmp_path / 'label_budget_build_out'
	_write_label_budget_build_config(
		label_build_config,
		baseline=baseline,
		candidate=candidate,
		output_root=label_build_root,
	)

	label_probe_root = tmp_path / 'label_budget_probe_out'
	build_f3_lithology_label_budget_datasets(
		_label_budget_config(baseline, candidate, label_probe_root),
	)
	label_probe_config = tmp_path / 'label_budget_probe.yaml'
	_write_label_budget_probe_config(
		label_probe_config,
		output_root=label_probe_root,
		class_info=class_info,
	)

	label_summary_root = tmp_path / 'label_budget_summary_out'
	build_f3_lithology_label_budget_datasets(
		_label_budget_config(baseline, candidate, label_summary_root),
	)
	run_f3_lithology_label_budget_probes(
		f3_lithology_label_budget_probe_config_from_mapping(
			_label_budget_probe_mapping(label_summary_root, class_info),
		),
	)

	paths = _write_split_fixture(tmp_path)
	split_inventory_config = tmp_path / 'split_inventory.yaml'
	split_inventory_root = tmp_path / 'split_inventory_dry_run_out'
	_write_split_inventory_config(
		split_inventory_config,
		tmp_path=tmp_path,
		paths=paths,
		output_root=split_inventory_root,
	)

	split_dataset_inventory = build_f3_lithology_split_inventories(
		_split_inventory_config(
			tmp_path,
			paths,
			output_name='split_dataset_inventory_out',
		),
	)
	split_dataset_config = tmp_path / 'split_dataset.yaml'
	split_dataset_root = tmp_path / 'split_dataset_dry_run_out'
	_write_split_dataset_config(
		split_dataset_config,
		tmp_path=tmp_path,
		paths=paths,
		inventory_manifest=split_dataset_inventory.manifest_json,
		output_root=split_dataset_root,
	)

	split_probe_inventory = build_f3_lithology_split_inventories(
		_split_inventory_config(
			tmp_path,
			paths,
			output_name='split_probe_inventory_out',
		),
	)
	split_probe_root = tmp_path / 'split_probe_out'
	build_f3_lithology_split_sweep_datasets(
		_split_dataset_config(
			tmp_path,
			paths,
			split_probe_inventory.manifest_json,
			output_root=split_probe_root,
		),
	)
	split_probe_config = tmp_path / 'split_probe.yaml'
	_write_split_probe_config(
		split_probe_config,
		output_root=split_probe_root,
		class_info=class_info,
	)

	split_summary_inventory = build_f3_lithology_split_inventories(
		_split_inventory_config(
			tmp_path,
			paths,
			output_name='split_summary_inventory_out',
		),
	)
	split_summary_root = tmp_path / 'split_summary_out'
	build_f3_lithology_split_sweep_datasets(
		_split_dataset_config(
			tmp_path,
			paths,
			split_summary_inventory.manifest_json,
			output_root=split_summary_root,
		),
	)
	run_f3_lithology_split_sweep_probes(
		f3_lithology_split_sweep_probe_config_from_mapping(
			_split_probe_mapping(split_summary_root, class_info),
		),
	)

	return [
		(
			'build_f3_lithology_label_budget_datasets.py',
			('--config', str(label_build_config), '--dry-run'),
			'execution: dry-run; label-budget datasets skipped',
			(label_build_root,),
		),
		(
			'run_f3_lithology_label_budget_probes.py',
			('--config', str(label_probe_config), '--dry-run'),
			'execution: dry-run; probe training skipped',
			(
				label_probe_root / 'probes',
				label_probe_root / 'probe_run_manifest.json',
			),
		),
		(
			'summarize_f3_lithology_label_budget_robustness.py',
			('--suite-root', str(label_summary_root), '--dry-run'),
			'execution: dry-run; label-budget robustness summary skipped',
			(label_summary_root / 'reports',),
		),
		(
			'generate_f3_lithology_split_inventories.py',
			('--config', str(split_inventory_config), '--dry-run'),
			'execution: dry-run; split inventories skipped',
			(split_inventory_root,),
		),
		(
			'build_f3_lithology_split_sweep_datasets.py',
			('--config', str(split_dataset_config), '--dry-run'),
			'execution: dry-run; split-sweep token datasets skipped',
			(split_dataset_root,),
		),
		(
			'run_f3_lithology_split_sweep_probes.py',
			('--config', str(split_probe_config), '--dry-run'),
			'execution: dry-run; probe training skipped',
			(
				split_probe_root / 'probes',
				split_probe_root / 'split_probe_run_manifest.json',
			),
		),
		(
			'summarize_f3_lithology_split_robustness.py',
			('--suite-root', str(split_summary_root), '--dry-run'),
			'execution: dry-run; split robustness summary skipped',
			(split_summary_root / 'reports',),
		),
	]


def _write_invalid_label_budget_build_config(tmp_path: Path) -> Path:
	baseline, candidate = _write_label_budget_sources(tmp_path)
	config_path = tmp_path / 'invalid_label_budget_build.yaml'
	_write_label_budget_build_config(
		config_path,
		baseline=baseline,
		candidate=candidate,
		output_root=Path('relative/out'),
	)
	return config_path


def _write_invalid_label_budget_probe_config(tmp_path: Path) -> Path:
	baseline, candidate = _write_label_budget_sources(tmp_path)
	output_root = tmp_path / 'label_budget_probe_invalid_source'
	build_f3_lithology_label_budget_datasets(
		_label_budget_config(baseline, candidate, output_root),
	)
	config_path = tmp_path / 'invalid_label_budget_probe.yaml'
	_write_label_budget_probe_config(
		config_path,
		output_root=Path('relative/out'),
		class_info=_write_class_info(tmp_path),
		manifest=output_root / 'suite_manifest.json',
	)
	return config_path


def _write_invalid_split_inventory_config(tmp_path: Path) -> Path:
	paths = _write_split_fixture(tmp_path)
	config_path = tmp_path / 'invalid_split_inventory.yaml'
	_write_split_inventory_config(
		config_path,
		tmp_path=tmp_path,
		paths=paths,
		output_root=Path('relative/out'),
	)
	return config_path


def _write_invalid_split_dataset_config(tmp_path: Path) -> Path:
	paths = _write_split_fixture(tmp_path)
	inventory_result = build_f3_lithology_split_inventories(
		_split_inventory_config(tmp_path, paths, output_name='split_dataset_invalid'),
	)
	config_path = tmp_path / 'invalid_split_dataset.yaml'
	_write_split_dataset_config(
		config_path,
		tmp_path=tmp_path,
		paths=paths,
		inventory_manifest=inventory_result.manifest_json,
		output_root=Path('relative/out'),
	)
	return config_path


def _write_invalid_split_probe_config(tmp_path: Path) -> Path:
	paths = _write_split_fixture(tmp_path)
	inventory_result = build_f3_lithology_split_inventories(
		_split_inventory_config(tmp_path, paths, output_name='split_probe_invalid'),
	)
	output_root = tmp_path / 'split_probe_invalid_source'
	build_f3_lithology_split_sweep_datasets(
		_split_dataset_config(
			tmp_path,
			paths,
			inventory_result.manifest_json,
			output_root=output_root,
		),
	)
	config_path = tmp_path / 'invalid_split_probe.yaml'
	_write_split_probe_config(
		config_path,
		output_root=Path('relative/out'),
		class_info=_write_class_info(tmp_path),
		dataset_manifest=output_root / 'split_dataset_manifest.json',
	)
	return config_path


def _write_label_budget_build_config(
	path: Path,
	*,
	baseline: Path,
	candidate: Path,
	output_root: Path,
) -> None:
	path.write_text(
		f"""
paths:
  artifact_root: {path.parent}
suite:
  name: label_budget_e2e
  output_root: {output_root}
models:
  baseline:
    model_tag: baseline_model
    token_dataset_root: {baseline}
  candidate:
    model_tag: candidate_model
    token_dataset_root: {candidate}
label_budget:
  mode: per_class_cap
  per_class_caps: [2]
  subsample_seeds: [0]
  require_all_classes: true
validation:
  reuse_full_validation: true
outputs:
  overwrite: false
""",
		encoding='utf-8',
	)


def _write_label_budget_probe_config(
	path: Path,
	*,
	output_root: Path,
	class_info: Path,
	manifest: Path | None = None,
) -> None:
	path.write_text(
		f"""
suite:
  manifest: {manifest or output_root / 'suite_manifest.json'}
  output_root: {output_root}
probe:
  spec: linear_balanced_v1
  type: logistic_regression
  feature_scaling: standard
  class_weight: balanced
  max_iter: 300
labels:
  class_info: {class_info}
evaluation:
  metrics: [accuracy, balanced_accuracy, macro_f1, weighted_f1, mean_iou]
  figure:
    dpi: 100
outputs:
  overwrite: false
""",
		encoding='utf-8',
	)


def _write_split_inventory_config(
	path: Path,
	*,
	tmp_path: Path,
	paths: dict[str, Path],
	output_root: Path,
) -> None:
	path.write_text(
		f"""
paths:
  artifact_root: {tmp_path}
inputs:
  base_png_label_inventory: {paths['inventory']}
  source_label_volume: {paths['label_volume']}
  segy_geometry_json: {paths['geometry']}
  class_info: {paths['class_info']}
  reference_embedding_metadata: {paths['embedding_metadata']}
split_sweep:
  name: split_index_e2e
  output_root: {output_root}
  split_ids: [split_000, split_001]
  random_seeds: [0, 1]
  validation_slice_count: 2
  require_validation_all_classes: true
  min_validation_tokens_per_class:
    default: 1
  include_base_split_as_split_000: true
tokenization:
  min_labeled_fraction: 1.0
  min_majority_fraction: 0.5
  ignore_z_border_samples: 0
  patch_size: [1, 1, 1]
outputs:
  overwrite: false
""",
		encoding='utf-8',
	)


def _write_split_dataset_config(
	path: Path,
	*,
	tmp_path: Path,
	paths: dict[str, Path],
	inventory_manifest: Path,
	output_root: Path,
) -> None:
	path.write_text(
		f"""
suite:
  split_inventory_manifest: {inventory_manifest}
  output_root: {output_root}
models:
  baseline:
    model_tag: baseline_model
    embeddings_dir: {paths['baseline_embeddings']}
    checkpoint: {paths['baseline_checkpoint']}
  candidate:
    model_tag: candidate_model
    embeddings_dir: {paths['candidate_embeddings']}
    checkpoint: {paths['candidate_checkpoint']}
common:
  f3_root: {tmp_path / 'F3'}
  artifact_root: {tmp_path / 'artifacts'}
  dataset:
    name: f3_facies_benchmark
    version: facies_benchmark_v1
  labels:
    source_label_segy: {tmp_path / 'F3' / 'f3_labels.sgy'}
    source_label_volume: {paths['label_volume']}
    class_info: {paths['class_info']}
    segy_geometry_json: {paths['geometry']}
  registry:
    seismic_volume: {paths['seismic_volume']}
    label_volume: {paths['label_volume']}
    metadata_json: {paths['volume_metadata']}
  tokenization:
    min_labeled_fraction: 1.0
    min_majority_fraction: 0.5
    ignore_z_border_samples: 0
outputs:
  overwrite: false
""",
		encoding='utf-8',
	)


def _write_split_probe_config(
	path: Path,
	*,
	output_root: Path,
	class_info: Path,
	dataset_manifest: Path | None = None,
) -> None:
	path.write_text(
		f"""
suite:
  dataset_manifest: {dataset_manifest or output_root / 'split_dataset_manifest.json'}
  output_root: {output_root}
probe:
  spec: linear_balanced_v1
  type: logistic_regression
  feature_scaling: standard
  class_weight: balanced
  max_iter: 300
labels:
  class_info: {class_info}
evaluation:
  metrics: [accuracy, balanced_accuracy, macro_f1, weighted_f1, mean_iou]
  figure:
    dpi: 100
outputs:
  overwrite: false
""",
		encoding='utf-8',
	)


def _label_budget_config(
	baseline: Path,
	candidate: Path,
	output_root: Path,
	*,
	overwrite: bool = False,
):
	return f3_lithology_label_budget_config_from_mapping(
		{
			'paths': {'artifact_root': str(output_root.parent)},
			'suite': {
				'name': 'label_budget_e2e',
				'output_root': str(output_root),
			},
			'models': {
				'baseline': {
					'model_tag': 'baseline_model',
					'token_dataset_root': str(baseline),
				},
				'candidate': {
					'model_tag': 'candidate_model',
					'token_dataset_root': str(candidate),
				},
			},
			'label_budget': {
				'mode': 'per_class_cap',
				'per_class_caps': [2],
				'subsample_seeds': [0, 1],
				'require_all_classes': True,
			},
			'validation': {'reuse_full_validation': True},
			'outputs': {'overwrite': overwrite},
		},
	)


def _label_budget_probe_mapping(
	output_root: Path,
	class_info: Path,
) -> dict[str, object]:
	return {
		'suite': {
			'manifest': str(output_root / 'suite_manifest.json'),
			'output_root': str(output_root),
		},
		'probe': {
			'spec': 'linear_balanced_v1',
			'type': 'logistic_regression',
			'feature_scaling': 'standard',
			'class_weight': 'balanced',
			'max_iter': 300,
		},
		'labels': {'class_info': str(class_info)},
		'evaluation': {
			'metrics': [
				'accuracy',
				'balanced_accuracy',
				'macro_f1',
				'weighted_f1',
				'mean_iou',
			],
			'figure': {'dpi': 100},
		},
		'outputs': {'overwrite': False},
	}


def _write_label_budget_sources(tmp_path: Path) -> tuple[Path, Path]:
	baseline = tmp_path / 'baseline_tokens'
	candidate = tmp_path / 'candidate_tokens'
	if baseline.exists():
		shutil.rmtree(baseline)
	if candidate.exists():
		shutil.rmtree(candidate)
	for root, quality in ((baseline, 'weak'), (candidate, 'strong')):
		root.mkdir(parents=True)
		save_token_dataset_npz(
			_label_budget_dataset(split='train', count=30, quality=quality),
			root / 'train_tokens.npz',
		)
		save_token_dataset_npz(
			_label_budget_dataset(
				split='validation',
				count=18,
				token_xyz_start=10_000,
				quality=quality,
			),
			root / 'validation_tokens.npz',
		)
		(root / 'token_dataset_metadata.json').write_text(
			json.dumps({'source': root.name}) + '\n',
			encoding='utf-8',
		)
	return baseline, candidate


def _label_budget_dataset(
	*,
	split: str,
	count: int,
	quality: str,
	token_xyz_start: int = 0,
) -> F3LithologyTokenDataset:
	labels = np.asarray(([0, 1, 2] * ((count + 2) // 3))[:count], dtype=np.int64)
	token_indices = np.arange(token_xyz_start, token_xyz_start + count, dtype=np.int64)
	if quality == 'strong':
		features = np.column_stack(
			(
				labels.astype(np.float32) * 5.0,
				labels.astype(np.float32) * -4.0,
				(labels == 0).astype(np.float32),
				(labels == 2).astype(np.float32),
			),
		)
	else:
		features = np.column_stack(
			(
				(token_indices % 2).astype(np.float32),
				(token_indices % 3).astype(np.float32),
				np.zeros(count, dtype=np.float32),
				np.ones(count, dtype=np.float32),
			),
		)
	return _token_dataset(
		split=split,
		labels=labels,
		features=features.astype(np.float32),
		token_xyz_start=token_xyz_start,
	)


def _split_inventory_config(
	tmp_path: Path,
	paths: dict[str, Path],
	*,
	output_name: str = 'split_out',
):
	return f3_lithology_split_inventory_config_from_mapping(
		{
			'paths': {'artifact_root': str(tmp_path)},
			'inputs': {
				'base_png_label_inventory': str(paths['inventory']),
				'source_label_volume': str(paths['label_volume']),
				'segy_geometry_json': str(paths['geometry']),
				'class_info': str(paths['class_info']),
				'reference_embedding_metadata': str(paths['embedding_metadata']),
			},
			'split_sweep': {
				'name': 'split_index_e2e',
				'output_root': str(tmp_path / output_name),
				'split_ids': ['split_000', 'split_001'],
				'random_seeds': [0, 1],
				'validation_slice_count': 2,
				'require_validation_all_classes': True,
				'min_validation_tokens_per_class': {'default': 1},
				'include_base_split_as_split_000': True,
			},
			'tokenization': {
				'min_labeled_fraction': 1.0,
				'min_majority_fraction': 0.5,
				'ignore_z_border_samples': 0,
				'patch_size': [1, 1, 1],
			},
			'outputs': {'overwrite': False},
		},
	)


def _split_dataset_config(
	tmp_path: Path,
	paths: dict[str, Path],
	inventory_manifest: Path,
	*,
	output_root: Path | None = None,
):
	return f3_lithology_split_sweep_dataset_config_from_mapping(
		{
			'suite': {
				'split_inventory_manifest': str(inventory_manifest),
				'output_root': str(output_root or tmp_path / 'split_out'),
			},
			'models': {
				'baseline': {
					'model_tag': 'baseline_model',
					'embeddings_dir': str(paths['baseline_embeddings']),
					'checkpoint': str(paths['baseline_checkpoint']),
				},
				'candidate': {
					'model_tag': 'candidate_model',
					'embeddings_dir': str(paths['candidate_embeddings']),
					'checkpoint': str(paths['candidate_checkpoint']),
				},
			},
			'common': {
				'f3_root': str(tmp_path / 'F3'),
				'artifact_root': str(tmp_path / 'artifacts'),
				'dataset': {
					'name': 'f3_facies_benchmark',
					'version': 'facies_benchmark_v1',
				},
				'labels': {
					'source_label_segy': str(tmp_path / 'F3' / 'f3_labels.sgy'),
					'source_label_volume': str(paths['label_volume']),
					'class_info': str(paths['class_info']),
					'segy_geometry_json': str(paths['geometry']),
				},
				'registry': {
					'seismic_volume': str(paths['seismic_volume']),
					'label_volume': str(paths['label_volume']),
					'metadata_json': str(paths['volume_metadata']),
				},
				'tokenization': {
					'min_labeled_fraction': 1.0,
					'min_majority_fraction': 0.5,
					'ignore_z_border_samples': 0,
				},
			},
			'outputs': {'overwrite': False},
		},
	)


def _split_probe_mapping(output_root: Path, class_info: Path) -> dict[str, object]:
	return {
		'suite': {
			'dataset_manifest': str(output_root / 'split_dataset_manifest.json'),
			'output_root': str(output_root),
		},
		'probe': {
			'spec': 'linear_balanced_v1',
			'type': 'logistic_regression',
			'feature_scaling': 'standard',
			'class_weight': 'balanced',
			'max_iter': 300,
		},
		'labels': {'class_info': str(class_info)},
		'evaluation': {
			'metrics': [
				'accuracy',
				'balanced_accuracy',
				'macro_f1',
				'weighted_f1',
				'mean_iou',
			],
			'figure': {'dpi': 100},
		},
		'outputs': {'overwrite': False},
	}


def _write_split_fixture(tmp_path: Path) -> dict[str, Path]:
	artifacts = tmp_path / 'artifacts'
	inventory = tmp_path / 'png_label_inventory.csv'
	_write_inventory(inventory)
	label_volume = artifacts / 'registry' / 'f3_facies_labels.npy'
	seismic_volume = artifacts / 'registry' / 'f3_seismic.npy'
	label_volume.parent.mkdir(parents=True, exist_ok=True)
	labels = _split_label_volume()
	np.save(label_volume, labels)
	np.save(
		seismic_volume,
		np.arange(labels.size, dtype=np.float32).reshape(labels.shape),
	)
	volume_metadata = artifacts / 'registry' / 'f3_metadata.json'
	_write_json(volume_metadata, {'shape': list(labels.shape)})
	class_info = _write_class_info(tmp_path)
	geometry = tmp_path / 'segy_geometry.json'
	_write_json(
		geometry,
		{
			'segy_files': {
				'label': {
					'cube_shape': [6, 2, 1],
					'iline_min': 100,
					'iline_max': 105,
					'xline_min': 200,
					'xline_max': 201,
				},
			},
		},
	)
	embedding_metadata = tmp_path / 'embedding_metadata.json'
	_write_json(
		embedding_metadata,
		{
			'patch_size': [1, 1, 1],
			'token_grid_shape': [6, 2, 1],
			'volume_shape_xyz': [6, 2, 1],
		},
	)
	baseline_embeddings = artifacts / 'embeddings' / 'baseline' / 'overlap_x16'
	candidate_embeddings = artifacts / 'embeddings' / 'candidate' / 'overlap_x16'
	_write_split_embedding_artifacts(baseline_embeddings, labels=labels, quality='weak')
	_write_split_embedding_artifacts(
		candidate_embeddings,
		labels=labels,
		quality='strong',
	)
	baseline_checkpoint = artifacts / 'checkpoints' / 'baseline.pt'
	candidate_checkpoint = artifacts / 'checkpoints' / 'candidate.pt'
	baseline_checkpoint.parent.mkdir(parents=True, exist_ok=True)
	baseline_checkpoint.write_bytes(b'baseline')
	candidate_checkpoint.write_bytes(b'candidate')
	return {
		'inventory': inventory,
		'label_volume': label_volume,
		'seismic_volume': seismic_volume,
		'volume_metadata': volume_metadata,
		'class_info': class_info,
		'geometry': geometry,
		'embedding_metadata': embedding_metadata,
		'baseline_embeddings': baseline_embeddings,
		'candidate_embeddings': candidate_embeddings,
		'baseline_checkpoint': baseline_checkpoint,
		'candidate_checkpoint': candidate_checkpoint,
	}


def _write_inventory(path: Path) -> None:
	rows = [
		{
			'relative_path': f'labels/inline_{100 + index}.png',
			'absolute_path': f'/fixture/inline_{100 + index}.png',
			'split': 'validation' if index in {0, 1} else 'train',
			'slice_type': 'inline',
			'slice_index': str(100 + index),
		}
		for index in range(6)
	]
	with path.open('w', encoding='utf-8', newline='') as handle:
		writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
		writer.writeheader()
		writer.writerows(rows)


def _split_label_volume() -> np.ndarray:
	labels = np.zeros((6, 2, 1), dtype=np.int32)
	for index in range(labels.shape[0]):
		labels[index, :, 0] = np.asarray([index % 3, (index + 1) % 3])
	return labels


def _write_split_embedding_artifacts(
	output_dir: Path,
	*,
	labels: np.ndarray,
	quality: str,
) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	if quality == 'strong':
		embeddings = np.stack(
			(
				labels.astype(np.float32) * 5.0,
				labels.astype(np.float32) * -4.0,
				(labels == 0).astype(np.float32),
			),
			axis=-1,
		)
	else:
		x = np.arange(labels.shape[0], dtype=np.float32)[:, None, None]
		y = np.arange(labels.shape[1], dtype=np.float32)[None, :, None]
		embeddings = np.stack(
			(
				np.broadcast_to(x, labels.shape),
				np.broadcast_to(y, labels.shape),
				np.ones(labels.shape, dtype=np.float32),
			),
			axis=-1,
		)
	np.save(
		output_dir / 'f3_facies_benchmark.embeddings.npy',
		embeddings.astype(np.float16),
	)
	np.save(
		output_dir / 'f3_facies_benchmark.valid_tokens.npy',
		np.ones(labels.shape, dtype=np.bool_),
	)
	_write_json(
		output_dir / 'f3_facies_benchmark.embedding_metadata.json',
		{
			'patch_size': [1, 1, 1],
			'token_grid_shape': list(labels.shape),
			'volume_shape_xyz': list(labels.shape),
		},
	)


def _token_dataset(
	*,
	split: str,
	labels: np.ndarray,
	features: np.ndarray,
	token_xyz_start: int,
) -> F3LithologyTokenDataset:
	count = int(labels.shape[0])
	token_indices = np.arange(token_xyz_start, token_xyz_start + count, dtype=np.int64)
	return F3LithologyTokenDataset(
		features=features,
		labels=labels.astype(np.int64),
		survey_id=np.asarray(['f3'] * count),
		split=np.asarray([split] * count),
		slice_type=np.asarray(['inline'] * count),
		slice_index=np.arange(count, dtype=np.int64),
		token_xyz=np.column_stack(
			(
				token_indices,
				np.zeros(count, dtype=np.int64),
				np.zeros(count, dtype=np.int64),
			),
		),
		voxel_center_xyz=np.zeros((count, 3), dtype=np.float32),
		majority_fraction=np.full(count, 1.0, dtype=np.float32),
		labeled_fraction=np.full(count, 1.0, dtype=np.float32),
		metadata={'fixture': True},
	)


def _write_class_info(tmp_path: Path) -> Path:
	path = tmp_path / 'class_info.json'
	if not path.exists():
		_write_json(
			path,
			{
				'class_count': 3,
				'classes': [
					{'class_id': 0, 'class_name': 'class 0', 'rgb': [230, 159, 0]},
					{'class_id': 1, 'class_name': 'class 1', 'rgb': [86, 180, 233]},
					{'class_id': 2, 'class_name': 'class 2', 'rgb': [0, 158, 115]},
				],
			},
		)
	return path


def _selected_rows_and_hashes(output_root: Path) -> tuple[tuple[object, ...], ...]:
	manifest = _read_json(output_root / 'suite_manifest.json')
	rows = []
	for row in manifest['rows']:
		root = Path(row['token_dataset_root'])
		train = load_token_dataset_npz(root / 'train_tokens.npz')
		validation = load_token_dataset_npz(root / 'validation_tokens.npz')
		rows.append(
			(
				row['model_role'],
				row['budget_id'],
				row['subsample_seed'],
				tuple(int(value) for value in train.token_xyz[:, 0]),
				paired_token_identity_hash(train, validation),
			),
		)
	return tuple(sorted(rows))


def _validation_slices_by_split(
	output_root: Path,
) -> dict[str, tuple[tuple[str, int], ...]]:
	result = {}
	for metadata_path in sorted(
		(output_root / 'split_inventories').glob('split_*/split_metadata.json'),
	):
		metadata = _read_json(metadata_path)
		result[str(metadata['split_id'])] = tuple(
			(str(row['slice_type']), int(row['slice_index']))
			for row in metadata['validation_slices']
		)
	return result


def _hashes_match_by_condition(
	rows: list[dict[str, object]],
	*,
	key_fields: tuple[str, ...],
) -> None:
	hashes: dict[tuple[object, ...], dict[str, str]] = {}
	for row in rows:
		key = tuple(row[field] for field in key_fields)
		hashes.setdefault(key, {})[str(row['model_role'])] = str(
			row['paired_identity_hash'],
		)
	for values in hashes.values():
		assert values['baseline'] == values['candidate']


def _run_cli(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
	env = os.environ.copy()
	env['PYTHONPATH'] = os.pathsep.join(
		(str(REPO_ROOT / 'src'), env.get('PYTHONPATH', '')),
	)
	return subprocess.run(  # noqa: S603
		[sys.executable, str(script), *args],
		cwd=REPO_ROOT,
		env=env,
		text=True,
		capture_output=True,
		check=False,
		timeout=30,
	)


def _read_csv(path: Path) -> list[dict[str, str]]:
	with path.open(encoding='utf-8', newline='') as handle:
		return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, object]:
	return json.loads(path.read_text(encoding='utf-8'))


def _write_json(path: Path, payload: object) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)

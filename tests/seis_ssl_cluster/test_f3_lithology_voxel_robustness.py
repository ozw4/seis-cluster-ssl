from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from seis_ssl_cluster.config.f3_lithology_voxel_decoder import (
	VoxelDecoderSpec,
	VoxelDecoderTileSettings,
	VoxelDecoderTrainSettings,
)
from seis_ssl_cluster.config.f3_lithology_voxel_robustness import (
	BASELINE_MODEL_TAG,
	CANDIDATE_MODEL_TAG,
	F3VoxelDecoderSplitSuiteConfig,
	F3VoxelSplitDatasetSuiteConfig,
	F3VoxelSplitRobustnessSummaryConfig,
	F3VoxelV0SplitSuiteConfig,
	VoxelRobustnessModel,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology import voxel_robustness
from seis_ssl_cluster.f3.lithology.voxel_evaluation import EVALUATION_OUTPUT_FILES
from seis_ssl_cluster.f3.lithology.voxel_robustness import (
	V0_RUN_MANIFEST_TYPE,
	V1_RUN_MANIFEST_TYPE,
	VoxelSplitJob,
	_read_manifest_rows,
	build_f3_lithology_voxel_split_datasets,
	run_f3_lithology_voxel_decoder_split_suite,
	run_f3_lithology_voxel_v0_split_suite,
	summarize_f3_lithology_voxel_split_robustness,
)


class _SyntheticProbe:
	classes_ = np.asarray([3, 5], dtype=np.int64)

	def predict(self, features: np.ndarray) -> np.ndarray:
		return np.where(features[:, 0] < 0, 3, 5)

	def predict_proba(self, features: np.ndarray) -> np.ndarray:
		positive = (features[:, 0] >= 0).astype(np.float64)
		return np.stack((1.0 - positive, positive), axis=1)


class _IdentityScaler:
	def transform(self, features: np.ndarray) -> np.ndarray:
		return features


def test_two_split_voxel_workflow_runs_end_to_end(tmp_path: Path) -> None:
	build_config, v0_config, v1_config = _synthetic_workflow_configs(tmp_path)

	datasets = build_f3_lithology_voxel_split_datasets(build_config)
	v0 = run_f3_lithology_voxel_v0_split_suite(v0_config)
	resumed_v0 = run_f3_lithology_voxel_v0_split_suite(v0_config, only_missing=True)
	probe_path = (
		v0_config.artifact_root / 'probes' / 'split_000' / 'baseline' / 'probe.joblib'
	)
	_probe_bytes = probe_path.read_bytes()
	probe_path.write_bytes(_probe_bytes + b'tampered')
	with pytest.raises(ValueError, match='probe identity does not match config'):
		run_f3_lithology_voxel_v0_split_suite(v0_config, only_missing=True)
	probe_path.write_bytes(_probe_bytes)
	validation_tokens = (
		v0_config.artifact_root
		/ 'tokens'
		/ 'split_000'
		/ 'baseline'
		/ 'validation_tokens.npz'
	)
	_token_bytes = validation_tokens.read_bytes()
	validation_tokens.write_bytes(_token_bytes + b'tampered')
	with pytest.raises(ValueError, match='lithology identity does not match config'):
		run_f3_lithology_voxel_v0_split_suite(v0_config, only_missing=True)
	validation_tokens.write_bytes(_token_bytes)
	v1 = run_f3_lithology_voxel_decoder_split_suite(v1_config, device='cpu')

	assert [row['split_id'] for row in datasets.rows] == [
		'split_000',
		'split_001',
	]
	assert len(v0.rows) == len(resumed_v0.rows) == len(v1.rows) == 4
	assert all(row['status'] == 'complete' for row in (*v0.rows, *v1.rows))
	assert all(
		Path(str(row['evaluation_dir']), 'evaluation_metadata.json').is_file()
		for row in (*v0.rows, *v1.rows)
	)


def test_v0_rejects_cross_model_paired_identity_mismatch(tmp_path: Path) -> None:
	build_config, v0_config, _ = _synthetic_workflow_configs(tmp_path)
	build_f3_lithology_voxel_split_datasets(build_config)
	for path in (v0_config.split_dataset_manifest, v0_config.probe_run_manifest):
		payload = json.loads(path.read_text(encoding='utf-8'))
		candidate = next(
			row
			for row in payload['rows']
			if row['split_id'] == 'split_000' and row['model_role'] == 'candidate'
		)
		candidate['paired_identity_hash'] = 'mismatched-cross-model-pair'
		path.write_text(json.dumps(payload), encoding='utf-8')

	with pytest.raises(ValueError, match='paired identity hash mismatch for split_000'):
		voxel_robustness.voxel_v0_split_jobs(v0_config)


def test_v0_rejects_token_suite_from_different_split_inventory(
	tmp_path: Path,
) -> None:
	build_config, v0_config, _ = _synthetic_workflow_configs(tmp_path)
	build_f3_lithology_voxel_split_datasets(build_config)
	payload = json.loads(v0_config.split_dataset_manifest.read_text(encoding='utf-8'))
	payload['suite']['split_inventory_manifest'] = str(tmp_path / 'other.json')
	v0_config.split_dataset_manifest.write_text(json.dumps(payload), encoding='utf-8')

	with pytest.raises(ValueError, match='different split inventories'):
		voxel_robustness.voxel_v0_split_jobs(v0_config)


def test_synthetic_split_summary_uses_paired_split_deltas(tmp_path: Path) -> None:
	v0_rows = []
	v1_rows = []
	for split_index in range(6):
		split_id = f'split_{split_index:03d}'
		for role, tag, offset in (
			('baseline', BASELINE_MODEL_TAG, 0.0),
			('candidate', CANDIDATE_MODEL_TAG, 0.04),
		):
			v0_dir = tmp_path / 'v0' / split_id / role
			v1_dir = tmp_path / 'v1' / split_id / role
			_write_evaluation(
				v0_dir,
				score=0.50 + offset,
				boundary_mae=2.0 - offset,
				model_tag=tag,
			)
			_write_evaluation(
				v1_dir,
				score=0.60 + offset,
				boundary_mae=1.5 - offset,
				model_tag=tag,
			)
			base = {
				'split_id': split_id,
				'model_role': role,
				'model_tag': tag,
				'voxel_dataset_identity': f'voxel-{split_id}',
				'status': 'complete',
			}
			v0_rows.append({**base, 'evaluation_dir': str(v0_dir)})
			v1_rows.append({**base, 'evaluation_dir': str(v1_dir)})
	v0_manifest = tmp_path / 'v0.json'
	v1_manifest = tmp_path / 'v1.json'
	_write_manifest(v0_manifest, V0_RUN_MANIFEST_TYPE, v0_rows)
	_write_manifest(v1_manifest, V1_RUN_MANIFEST_TYPE, v1_rows)

	result = summarize_f3_lithology_voxel_split_robustness(
		F3VoxelSplitRobustnessSummaryConfig(
			suite_root=tmp_path / 'suite',
			v0_run_manifest=v0_manifest,
			v1_run_manifest=v1_manifest,
			baseline_model_tag=BASELINE_MODEL_TAG,
			candidate_model_tag=CANDIDATE_MODEL_TAG,
		)
	)

	payload = json.loads(result.summary_json.read_text(encoding='utf-8'))
	assert result.status == 'positive'
	assert payload['statistical_unit'] == 'split'
	assert payload['voxel_level_significance_computed'] is False
	assert payload['p_values_computed'] is False
	assert payload['confidence_intervals_computed'] is False
	primary = next(
		row
		for row in payload['aggregates']
		if row['comparison'] == 'v1_candidate_minus_baseline'
		and row['metric'] == 'primary_metrics_simultaneous'
	)
	assert primary['win_count'] == 6
	assert primary['win_rate'] == 1.0
	assert any(
		row['comparison'] == 'candidate_v1_minus_v0' for row in payload['raw_rows']
	)
	v1_payload = json.loads(v1_manifest.read_text(encoding='utf-8'))
	v1_payload['rows'][0]['voxel_dataset_identity'] = 'different-voxel-dataset'
	v1_manifest.write_text(json.dumps(v1_payload), encoding='utf-8')
	with pytest.raises(ValueError, match='voxel dataset identity mismatch'):
		summarize_f3_lithology_voxel_split_robustness(
			F3VoxelSplitRobustnessSummaryConfig(
				suite_root=tmp_path / 'rejected-suite',
				v0_run_manifest=v0_manifest,
				v1_run_manifest=v1_manifest,
				baseline_model_tag=BASELINE_MODEL_TAG,
				candidate_model_tag=CANDIDATE_MODEL_TAG,
			)
		)


def test_partial_split_summary_is_rejected(tmp_path: Path) -> None:
	rows = []
	for role, tag in (
		('baseline', BASELINE_MODEL_TAG),
		('candidate', CANDIDATE_MODEL_TAG),
	):
		evaluation = tmp_path / role
		_write_evaluation(evaluation, score=0.5, boundary_mae=2.0, model_tag=tag)
		rows.append(
			{
				'split_id': 'split_000',
				'model_role': role,
				'model_tag': tag,
				'voxel_dataset_identity': 'voxel-split_000',
				'evaluation_dir': str(evaluation),
				'status': 'complete',
			}
		)
	v0_manifest = tmp_path / 'v0.json'
	v1_manifest = tmp_path / 'v1.json'
	_write_manifest(v0_manifest, V0_RUN_MANIFEST_TYPE, rows)
	_write_manifest(v1_manifest, V1_RUN_MANIFEST_TYPE, rows)

	with pytest.raises(ValueError, match='all six'):
		summarize_f3_lithology_voxel_split_robustness(
			F3VoxelSplitRobustnessSummaryConfig(
				suite_root=tmp_path / 'suite',
				v0_run_manifest=v0_manifest,
				v1_run_manifest=v1_manifest,
				baseline_model_tag=BASELINE_MODEL_TAG,
				candidate_model_tag=CANDIDATE_MODEL_TAG,
			)
		)


def test_null_deltas_remain_in_split_denominators() -> None:
	rows = []
	for split_index in range(6):
		for metric in voxel_robustness.SUMMARY_METRICS:
			delta = (
				(-0.1 if metric == 'boundary_position_mae' else 0.1)
				if split_index == 0
				else None
			)
			rows.append(
				{
					'split_id': f'split_{split_index:03d}',
					'comparison': 'v1_candidate_minus_baseline',
					'metric': metric,
					'delta': delta,
				}
			)

	aggregates = voxel_robustness._aggregate_metric_rows(rows)  # noqa: SLF001
	macro = next(row for row in aggregates if row['metric'] == 'macro_f1')
	primary = next(
		row for row in aggregates if row['metric'] == 'primary_metrics_simultaneous'
	)
	assert macro['split_count'] == 6
	assert macro['win_count'] == 1
	assert macro['win_rate'] == pytest.approx(1 / 6)
	assert primary['split_count'] == 6
	assert primary['win_count'] == 1
	assert primary['win_rate'] == pytest.approx(1 / 6)
	status, _ = voxel_robustness._provisional_status(  # noqa: SLF001
		rows, aggregates
	)
	assert status == 'hold'


def test_all_tied_primary_result_remains_hold() -> None:
	rows = [
		{
			'split_id': f'split_{split_index:03d}',
			'comparison': 'v1_candidate_minus_baseline',
			'metric': metric,
			'delta': 0.0,
		}
		for split_index in range(6)
		for metric in voxel_robustness.SUMMARY_METRICS
	]

	aggregates = voxel_robustness._aggregate_metric_rows(rows)  # noqa: SLF001
	status, _ = voxel_robustness._provisional_status(  # noqa: SLF001
		rows, aggregates
	)

	assert status == 'hold'


def test_duplicate_split_run_is_rejected(tmp_path: Path) -> None:
	path = tmp_path / 'manifest.json'
	row = {
		'split_id': 'split_000',
		'model_role': 'baseline',
		'model_tag': BASELINE_MODEL_TAG,
	}
	_write_manifest(path, V0_RUN_MANIFEST_TYPE, [row, row])
	with pytest.raises(ValueError, match='duplicate split/run'):
		_read_manifest_rows(
			path,
			artifact_type=V0_RUN_MANIFEST_TYPE,
			required=('split_id', 'model_role', 'model_tag'),
		)


def test_v1_paired_identity_failure_is_recorded_for_resume(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	models = (
		SimpleNamespace(role='baseline', model_tag=BASELINE_MODEL_TAG),
		SimpleNamespace(role='candidate', model_tag=CANDIDATE_MODEL_TAG),
	)
	config = SimpleNamespace(
		output_root=tmp_path,
		voxel_dataset_manifest=tmp_path / 'voxel.json',
		dataset={'name': 'f3'},
		models=models,
	)
	jobs = tuple(
		VoxelSplitJob(
			'split_000',
			model.role,
			model.model_tag,
			tmp_path / model.role,
		)
		for model in models
	)
	voxel_row = {'split_id': 'split_000'}
	monkeypatch.setattr(voxel_robustness, 'voxel_decoder_split_jobs', lambda _: jobs)
	monkeypatch.setattr(voxel_robustness, '_voxel_rows', lambda _: (voxel_row,))

	def skip_job(  # noqa: PLR0913
		config, job, *, model, voxel_row, device, resume, only_missing
	):
		del config, job, model, voxel_row, device, resume, only_missing

	monkeypatch.setattr(voxel_robustness, '_run_v1_job', skip_job)

	def manifest_row(  # noqa: PLR0913
		job, voxel_row, *, model, dataset_name, status, failure=None
	):
		del voxel_row, model, dataset_name
		row = {
			'split_id': job.split_id,
			'model_role': job.model_role,
			'model_tag': job.model_tag,
			'status': status,
			'resume_path': str(job.output_root / 'decoder' / 'latest.pt'),
		}
		if failure is not None:
			row['failure'] = failure
		return row

	monkeypatch.setattr(voxel_robustness, '_v1_manifest_row', manifest_row)

	def reject_mismatched_pair(rows, *, split_id):
		if len(rows) == 2:
			raise ValueError(f'paired decoder class_weights mismatch for {split_id}')

	monkeypatch.setattr(
		voxel_robustness,
		'_validate_paired_decoder_identity',
		reject_mismatched_pair,
	)

	with pytest.raises(ValueError, match='class_weights mismatch'):
		run_f3_lithology_voxel_decoder_split_suite(config)

	payload = json.loads(
		(tmp_path / 'v1_split_run_manifest.json').read_text(encoding='utf-8')
	)
	assert [row['status'] for row in payload['rows']] == ['complete', 'failed']
	failed = payload['rows'][1]
	assert failed['resume_path'].endswith('/candidate/decoder/latest.pt')
	assert failed['failure'].startswith('ValueError: paired decoder class_weights')


def _write_evaluation(
	root: Path, *, score: float, boundary_mae: float, model_tag: str
) -> None:
	root.mkdir(parents=True)
	metrics = {
		'macro_f1': score,
		'mean_iou': score,
		'balanced_accuracy': score,
		'per_class_f1': {'3': score, '5': score},
		'per_class_iou': {'3': score, '5': score},
	}
	boundary = {
		'vertical_boundary_f1_at_2': score,
		'vertical_boundary_f1_at_4': score,
		'vertical_boundary_position_mae_at_4': boundary_mae,
		'vertical_boundary_class_3_recall_at_2': score,
		'vertical_boundary_class_3_recall_at_4': score,
		'vertical_boundary_class_5_recall_at_2': score,
		'vertical_boundary_class_5_recall_at_4': score,
	}
	(root / 'metrics.json').write_text(json.dumps(metrics), encoding='utf-8')
	(root / 'boundary_metrics.json').write_text(json.dumps(boundary), encoding='utf-8')
	(root / 'evaluation_metadata.json').write_text(
		json.dumps(
			{
				'artifact_type': 'f3_lithology_voxel_evaluation',
				'model_tag': model_tag,
			}
		),
		encoding='utf-8',
	)
	with (root / 'boundary_region_metrics.csv').open(
		'w', encoding='utf-8', newline=''
	) as file_obj:
		writer = csv.DictWriter(
			file_obj, fieldnames=('region', 'radius', 'macro_f1', 'mean_iou')
		)
		writer.writeheader()
		for radius in (2, 4):
			writer.writerow(
				{
					'region': 'boundary',
					'radius': radius,
					'macro_f1': score,
					'mean_iou': score,
				}
			)
	for name in EVALUATION_OUTPUT_FILES:
		path = root / name
		if not path.exists():
			path.write_text(
				'{}\n' if path.suffix == '.json' else '\n', encoding='utf-8'
			)


def _write_manifest(
	path: Path,
	artifact_type: str,
	rows: object,
	**extra: object,
) -> None:
	path.write_text(
		json.dumps({'artifact_type': artifact_type, **extra, 'rows': rows}),
		encoding='utf-8',
	)


def _synthetic_workflow_configs(  # noqa: PLR0915
	tmp_path: Path,
) -> tuple[
	F3VoxelSplitDatasetSuiteConfig,
	F3VoxelV0SplitSuiteConfig,
	F3VoxelDecoderSplitSuiteConfig,
]:
	artifact_root = tmp_path / 'artifacts'
	f3_root = tmp_path / 'f3'
	suite_root = artifact_root / 'voxel_suite'
	artifact_root.mkdir()
	f3_root.mkdir()
	dataset = {'name': 'synthetic_f3', 'version': 'voxel_robustness_test'}
	labels = np.empty((4, 4, 4), dtype=np.int16)
	for x, y, z in np.ndindex(labels.shape):
		labels[x, y, z] = 3 if (x // 2 + y // 2 + z // 2) % 2 == 0 else 5
	label_volume = artifact_root / 'labels.npy'
	np.save(label_volume, labels, allow_pickle=False)
	source_label_segy = f3_root / 'labels.sgy'
	source_label_segy.write_bytes(b'synthetic label segy')
	class_info = artifact_root / 'class_info.json'
	class_info.write_text(
		json.dumps(
			{
				'3': {'name': 'class three', 'color': [3, 3, 3]},
				'5': {'name': 'class five', 'color': [5, 5, 5]},
			}
		),
		encoding='utf-8',
	)
	geometry = artifact_root / 'geometry.json'
	geometry.write_text(
		json.dumps(
			{
				'segy_files': {
					'label': {
						'cube_shape': [4, 4, 4],
						'iline_min': 100,
						'iline_max': 103,
						'xline_min': 200,
						'xline_max': 203,
					}
				}
			}
		),
		encoding='utf-8',
	)
	inventory_rows = []
	for split_index in range(2):
		split_id = f'split_{split_index:03d}'
		inventory = artifact_root / f'{split_id}_inventory.csv'
		with inventory.open('w', encoding='utf-8', newline='') as file_obj:
			writer = csv.DictWriter(
				file_obj,
				fieldnames=('relative_path', 'split', 'slice_type', 'slice_index'),
			)
			writer.writeheader()
			writer.writerows(
				(
					{
						'relative_path': f'{split_id}_train.png',
						'split': 'train',
						'slice_type': 'inline',
						'slice_index': 100 + split_index,
					},
					{
						'relative_path': f'{split_id}_validation.png',
						'split': 'validation',
						'slice_type': 'crossline',
						'slice_index': 203 - split_index,
					},
				)
			)
		inventory_rows.append(
			{'split_id': split_id, 'png_label_inventory': str(inventory)}
		)
	inventory_manifest = artifact_root / 'split_inventory_manifest.json'
	_write_manifest(
		inventory_manifest,
		voxel_robustness.INVENTORY_ARTIFACT_TYPE,
		inventory_rows,
	)
	valid_tokens = np.ones((2, 2, 2), dtype=np.bool_)
	model_specs = (
		('baseline', BASELINE_MODEL_TAG),
		('candidate', CANDIDATE_MODEL_TAG),
	)
	models = []
	for role, model_tag in model_specs:
		checkpoint = artifact_root / 'pretraining' / model_tag / 'best.pt'
		checkpoint.parent.mkdir(parents=True)
		checkpoint.write_bytes(f'{model_tag} checkpoint'.encode())
		embeddings_dir = (
			artifact_root
			/ 'embeddings'
			/ 'f3'
			/ dataset['version']
			/ model_tag
			/ 'synthetic'
		)
		embeddings_dir.mkdir(parents=True)
		token_classes = np.asarray([[[3, 5], [5, 3]], [[5, 3], [3, 5]]], dtype=np.int16)
		embeddings = np.stack(
			(
				np.where(token_classes == 3, -1.0, 1.0),
				np.ones(token_classes.shape),
			),
			axis=-1,
		).astype(np.float32)
		np.save(
			embeddings_dir / f'{dataset["name"]}.embeddings.npy',
			embeddings,
			allow_pickle=False,
		)
		np.save(
			embeddings_dir / f'{dataset["name"]}.valid_tokens.npy',
			valid_tokens,
			allow_pickle=False,
		)
		metadata = {
			'volume_shape_xyz': [4, 4, 4],
			'patch_size': [2, 2, 2],
			'token_grid_shape': [2, 2, 2],
			'embedding_dim': 2,
			'checkpoint_path': str(checkpoint),
			'checkpoint_sha256': file_sha256(checkpoint),
			'preprocessing': {'kind': 'synthetic'},
			'zero_mask': {'enabled': True},
		}
		(embeddings_dir / f'{dataset["name"]}.embedding_metadata.json').write_text(
			json.dumps(metadata), encoding='utf-8'
		)
		models.append(VoxelRobustnessModel(role, model_tag, embeddings_dir, checkpoint))
	reference_metadata = (
		models[0].embeddings_dir / f'{dataset["name"]}.embedding_metadata.json'
	)
	reference_valid_tokens = (
		models[0].embeddings_dir / f'{dataset["name"]}.valid_tokens.npy'
	)
	build_config = F3VoxelSplitDatasetSuiteConfig(
		split_inventory_manifest=inventory_manifest,
		output_root=suite_root,
		artifact_root=artifact_root,
		f3_root=f3_root,
		dataset=dataset,
		source_label_volume=label_volume,
		source_label_segy=source_label_segy,
		class_info=class_info,
		segy_geometry_json=geometry,
		reference_metadata_json=reference_metadata,
		reference_valid_tokens=reference_valid_tokens,
		ignore_z_border_samples=0,
		overwrite=False,
	)
	dataset_rows = []
	probe_rows = []
	for split_index in range(2):
		split_id = f'split_{split_index:03d}'
		paired_hash = f'synthetic-paired-{split_id}'
		for role, model_tag in model_specs:
			token_root = artifact_root / 'tokens' / split_id / role
			token_root.mkdir(parents=True)
			validation_tokens_path = token_root / 'validation_tokens.npz'
			_write_empty_validation_tokens(validation_tokens_path, dataset['name'])
			probe_dir = artifact_root / 'probes' / split_id / role
			probe_dir.mkdir(parents=True)
			joblib.dump(_SyntheticProbe(), probe_dir / 'probe.joblib')
			joblib.dump(_IdentityScaler(), probe_dir / 'scaler.joblib')
			dataset_rows.append(
				{
					'split_id': split_id,
					'model_role': role,
					'model_tag': model_tag,
					'token_dataset_root': str(token_root),
					'validation_tokens': str(validation_tokens_path),
					'paired_identity_hash': paired_hash,
				}
			)
			probe_rows.append(
				{
					'split_id': split_id,
					'model_role': role,
					'model_tag': model_tag,
					'token_dataset_root': str(token_root),
					'probe_output_dir': str(probe_dir),
					'paired_identity_hash': paired_hash,
				}
			)
	dataset_manifest = artifact_root / 'split_dataset_manifest.json'
	probe_manifest = artifact_root / 'split_probe_run_manifest.json'
	_write_manifest(
		dataset_manifest,
		'f3_lithology_split_sweep_token_dataset_manifest',
		dataset_rows,
		suite={'split_inventory_manifest': str(inventory_manifest)},
	)
	_write_manifest(probe_manifest, 'f3_lithology_split_probe_run_manifest', probe_rows)
	evaluation = {
		'monitored_class_ids': [3, 5],
		'boundary_tolerances': [2, 4],
		'boundary_region_radii': [2, 4],
		'chunk_size_x': 2,
	}
	v0_config = F3VoxelV0SplitSuiteConfig(
		voxel_dataset_manifest=suite_root / 'voxel_split_dataset_manifest.json',
		split_dataset_manifest=dataset_manifest,
		probe_run_manifest=probe_manifest,
		output_root=suite_root,
		artifact_root=artifact_root,
		f3_root=f3_root,
		dataset=dataset,
		models=tuple(models),
		source_label_volume=label_volume,
		source_label_segy=source_label_segy,
		class_info=class_info,
		segy_geometry_json=geometry,
		batch_size=8,
		tokenization={
			'min_labeled_fraction': 1.0,
			'min_majority_fraction': 1.0,
			'ignore_z_border_samples': 0,
		},
		evaluation=evaluation,
		overwrite=False,
	)
	v1_models = tuple(
		VoxelRobustnessModel(model.role, model.model_tag, model.embeddings_dir)
		for model in models
	)
	v1_config = F3VoxelDecoderSplitSuiteConfig(
		voxel_dataset_manifest=suite_root / 'voxel_split_dataset_manifest.json',
		output_root=suite_root,
		artifact_root=artifact_root,
		f3_root=f3_root,
		dataset=dataset,
		models=v1_models,
		source_label_volume=label_volume,
		source_label_segy=source_label_segy,
		class_info=class_info,
		segy_geometry_json=geometry,
		decoder=VoxelDecoderSpec(
			spec='frozen_embedding_decoder_v1',
			embedding_dim=2,
			class_count=2,
			hidden_channels=(2,),
			upsample_factors=((2, 2, 2),),
		),
		tiles=VoxelDecoderTileSettings(
			core_size_tokens=(2, 2, 2), context_halo_tokens=(0, 0, 0)
		),
		train=VoxelDecoderTrainSettings(
			epochs=1,
			batch_size=1,
			learning_rate=0.01,
			weight_decay=0.0,
			class_weight='balanced',
			seed=7,
			num_workers=0,
			amp=False,
			gradient_clip_norm=1.0,
		),
		evaluation=evaluation,
		write_probabilities=False,
		overwrite=False,
	)
	return build_config, v0_config, v1_config


def _write_empty_validation_tokens(path: Path, survey_id: str) -> None:
	np.savez_compressed(
		path,
		features=np.empty((0, 2), dtype=np.float32),
		labels=np.empty((0,), dtype=np.int64),
		survey_id=np.empty((0,), dtype=f'<U{len(survey_id)}'),
		split=np.empty((0,), dtype='<U10'),
		slice_type=np.empty((0,), dtype='<U9'),
		slice_index=np.empty((0,), dtype=np.int64),
		token_xyz=np.empty((0, 3), dtype=np.int64),
		voxel_center_xyz=np.empty((0, 3), dtype=np.float32),
		majority_fraction=np.empty((0,), dtype=np.float32),
		labeled_fraction=np.empty((0,), dtype=np.float32),
	)

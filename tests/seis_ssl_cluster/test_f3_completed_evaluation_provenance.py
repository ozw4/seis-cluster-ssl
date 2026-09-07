from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.completed_evaluation_provenance import (
	read_f3_completed_evaluation_provenance,
)

MODEL_ID = 'candidate'
LAYOUT_ID = 'layout_000'
DATA_SIZE = 'small'
DATASET = {'name': 'f3_facies_benchmark', 'version': 'facies_benchmark_v2'}
PREDICTION_FILES = {
	'predictions': ('voxel_predictions', 'f3_voxel_predictions.npy'),
	'confidence': ('voxel_confidence', 'f3_voxel_confidence.npy'),
	'valid_mask': ('voxel_valid_mask', 'f3_valid_voxel_mask.npy'),
}


def _write_json(path: Path, payload: object) -> None:
	path.write_text(json.dumps(payload, sort_keys=True), encoding='utf-8')


def _read_json(path: Path) -> dict[str, object]:
	payload = json.loads(path.read_text(encoding='utf-8'))
	assert isinstance(payload, dict)
	return payload


def _identity(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}


def _write_job(root: Path) -> Path:
	job_dir = root / f'model={MODEL_ID}' / f'layout={LAYOUT_ID}' / f'size={DATA_SIZE}'
	evaluation_dir = job_dir / 'evaluation'
	prediction_dir = job_dir / 'prediction'
	decoder_dir = job_dir / 'decoder'
	for directory in (evaluation_dir, prediction_dir, decoder_dir):
		directory.mkdir(parents=True, exist_ok=True)
	source_dir = root / 'canonical_sources'
	source_dir.mkdir()
	ground_truth_paths = {
		'label_volume': source_dir / 'f3_facies_labels.npy',
		'class_info': source_dir / 'class_info.json',
		'source_label_segy': source_dir / 'f3_labels.sgy',
		'png_label_inventory': source_dir / 'section_inventory_v2.csv',
		'segy_geometry_json': source_dir / 'segy_geometry.json',
	}
	for index, path in enumerate(ground_truth_paths.values(), start=1):
		path.write_bytes(bytes([index]) * (index + 3))
	voxel_dataset_dir = (
		root
		/ 'conditions/datasets'
		/ f'layout={LAYOUT_ID}'
		/ f'size={DATA_SIZE}'
		/ 'voxel_supervision'
	)
	voxel_dataset_dir.mkdir(parents=True)
	voxel_metadata_path = voxel_dataset_dir / 'voxel_dataset_metadata.json'
	_write_json(
		voxel_metadata_path,
		{
			'artifact_type': 'f3_lithology_voxel_supervision',
			'schema_version': 1,
			'dataset': DATASET,
			'section_layout': {
				'layout_id': LAYOUT_ID,
				'data_size': DATA_SIZE,
			},
			'labels': {
				'class_info': str(ground_truth_paths['class_info']),
				'source_label_segy': str(ground_truth_paths['source_label_segy']),
			},
			'label_volume': _identity(ground_truth_paths['label_volume']),
			'inventory': _identity(ground_truth_paths['png_label_inventory']),
			'source_identities': {
				'class_info': _identity(ground_truth_paths['class_info']),
				'source_label_segy': _identity(ground_truth_paths['source_label_segy']),
				'segy_geometry_json': _identity(
					ground_truth_paths['segy_geometry_json']
				),
			},
		},
	)

	decoder_architecture = {
		'spec': 'frozen_embedding_decoder_nearest_voxel_ln_v1',
		'embedding_dim': 384,
		'class_count': 6,
		'hidden_channels': [128, 64, 32],
		'upsample_factors': [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
		'upsample_mode': 'nearest',
		'normalization': 'voxelwise_layer_norm',
	}
	resolved_config = {
		'dataset': DATASET,
		'model': {'tag': MODEL_ID, 'freeze_encoder': True},
		'outputs': {'output_dir': str(decoder_dir)},
		'decoder': decoder_architecture,
		'voxel_dataset': {'input_dir': str(voxel_dataset_dir)},
	}
	decoder_artifact_identities = {
		'name': 'f3_voxel_decoder_sources',
		'voxel_dataset_metadata': _identity(voxel_metadata_path),
		'label_volume': _identity(ground_truth_paths['label_volume']),
	}
	decoder_config_path = decoder_dir / 'resolved_config.json'
	_write_json(decoder_config_path, resolved_config)
	decoder_checkpoint_path = decoder_dir / 'best.pt'
	torch.save(
		{
			'schema_version': 5,
			'epoch': 1,
			'global_step': 1,
			'model_state_dict': {},
			'optimizer_state_dict': {},
			'amp_scaler_state_dict': None,
			'runtime_identity': {},
			'best_selection_state': None,
			'training_history': [],
			'current_metrics': {},
			'resolved_config': resolved_config,
			'decoder_architecture': decoder_architecture,
			'class_weights': [],
			'artifact_identities': decoder_artifact_identities,
			'tile_manifest_hashes': {},
			'rng_states': {},
			'checkpoint_kind': 'completed',
			'best_checkpoint_sha256': None,
		},
		decoder_checkpoint_path,
	)

	prediction_outputs = {}
	for index, (output_key, (_, name)) in enumerate(PREDICTION_FILES.items(), start=1):
		path = prediction_dir / name
		path.write_bytes(bytes([index]) * (index + 2))
		prediction_outputs[output_key] = str(path)
	prediction_metadata_path = prediction_dir / 'prediction_metadata.json'
	_write_json(
		prediction_metadata_path,
		{
			'artifact_type': 'f3_lithology_voxel_predictions',
			'schema_version': 1,
			'prediction_kind': 'frozen_embedding_decoder',
			'model_tag': MODEL_ID,
			'decoder_architecture': decoder_architecture,
			'inputs': {
				'decoder_checkpoint': str(decoder_checkpoint_path),
				'class_info': str(ground_truth_paths['class_info']),
			},
			'outputs': prediction_outputs,
			'source_identity': {
				'decoder_checkpoint': _identity(decoder_checkpoint_path),
				'resolved_decoder_config': _identity(decoder_config_path),
				'class_info': _identity(ground_truth_paths['class_info']),
				'artifact_identities': decoder_artifact_identities,
			},
		},
	)

	metrics_path = evaluation_dir / 'metrics.json'
	_write_json(
		metrics_path,
		{
			'aggregation_unit': 'unique_validation_voxel',
			'macro_f1': 0.625,
			'evaluation_voxel_count': 496,
		},
	)
	evaluation_inputs = {'prediction_metadata': _identity(prediction_metadata_path)}
	for input_key, name in PREDICTION_FILES.values():
		evaluation_inputs[input_key] = _identity(prediction_dir / name)
	evaluation_inputs['voxel_dataset_metadata'] = _identity(voxel_metadata_path)
	for key, path in ground_truth_paths.items():
		evaluation_inputs[key] = _identity(path)
	evaluation_metadata_path = evaluation_dir / 'evaluation_metadata.json'
	_write_json(
		evaluation_metadata_path,
		{
			'artifact_type': 'f3_lithology_voxel_evaluation',
			'schema_version': 2,
			'dataset': DATASET,
			'prediction_kind': 'frozen_embedding_decoder',
			'model_tag': MODEL_ID,
			'decoder_architecture': decoder_architecture,
			'aggregation': {'primary_unit': 'unique_validation_voxel'},
			'inputs': evaluation_inputs,
			'outputs': {'metrics.json': _identity(metrics_path)},
			'metadata_path': str(evaluation_metadata_path),
			'summary': {'unique_validation_voxel_count': 496},
		},
	)
	return job_dir


@pytest.fixture
def completed_job(tmp_path: Path) -> Path:
	return _write_job(tmp_path)


def _read(completed_job: Path):
	return read_f3_completed_evaluation_provenance(
		completed_job,
		model_id=MODEL_ID,
		layout_id=LAYOUT_ID,
		data_size=DATA_SIZE,
	)


def _refresh_prediction_metadata_identity(job_dir: Path) -> None:
	evaluation_path = job_dir / 'evaluation/evaluation_metadata.json'
	metadata = _read_json(evaluation_path)
	inputs = metadata['inputs']
	assert isinstance(inputs, dict)
	inputs['prediction_metadata'] = _identity(
		job_dir / 'prediction/prediction_metadata.json'
	)
	_write_json(evaluation_path, metadata)


def _refresh_decoder_checkpoint_identity(job_dir: Path) -> None:
	prediction_path = job_dir / 'prediction/prediction_metadata.json'
	metadata = _read_json(prediction_path)
	source = metadata['source_identity']
	assert isinstance(source, dict)
	source['decoder_checkpoint'] = _identity(job_dir / 'decoder/best.pt')
	_write_json(prediction_path, metadata)
	_refresh_prediction_metadata_identity(job_dir)


def test_reads_complete_evaluation_lineage(completed_job: Path) -> None:
	evidence = _read(completed_job)

	assert evidence['model_id'] == MODEL_ID
	assert evidence['layout_id'] == LAYOUT_ID
	assert evidence['data_size'] == DATA_SIZE
	assert evidence['macro_f1'] == 0.625
	assert evidence['evaluation_voxel_count'] == 496
	for key in (
		'metrics',
		'evaluation_metadata',
		'prediction_metadata',
		'voxel_predictions',
		'voxel_confidence',
		'voxel_valid_mask',
		'decoder_checkpoint',
		'decoder_resolved_config',
		'voxel_dataset_metadata',
		'label_volume',
		'class_info',
		'source_label_segy',
		'png_label_inventory',
		'segy_geometry_json',
	):
		path = Path(evidence[f'{key}_path'])
		assert path.is_absolute()
		assert evidence[f'{key}_sha256'] == file_sha256(path)


def test_rejects_metrics_mutated_after_evaluation(completed_job: Path) -> None:
	metrics_path = completed_job / 'evaluation/metrics.json'
	metrics = _read_json(metrics_path)
	metrics['macro_f1'] = 0.75
	_write_json(metrics_path, metrics)

	with pytest.raises(ValueError, match='changed after its provenance was recorded'):
		_read(completed_job)


@pytest.mark.parametrize(
	('container_key', 'identity_key', 'foreign_name', 'message'),
	[
		(
			'outputs',
			'metrics.json',
			'metrics.json',
			r'outputs\.metrics\.json\.path must equal the exact path',
		),
		(
			'inputs',
			'prediction_metadata',
			'prediction_metadata.json',
			r'inputs\.prediction_metadata\.path must equal the exact path',
		),
	],
)
def test_rejects_foreign_evaluation_paths(
	completed_job: Path,
	container_key: str,
	identity_key: str,
	foreign_name: str,
	message: str,
) -> None:
	path = completed_job / 'evaluation/evaluation_metadata.json'
	payload = _read_json(path)
	container = payload[container_key]
	assert isinstance(container, dict)
	entry = container[identity_key]
	assert isinstance(entry, dict)
	entry['path'] = str(completed_job.parent / 'foreign' / foreign_name)
	_write_json(path, payload)

	with pytest.raises(ValueError, match=message):
		_read(completed_job)


@pytest.mark.parametrize(
	('source_key', 'foreign_name'),
	[
		('decoder_checkpoint', 'other.pt'),
		('resolved_decoder_config', 'other.json'),
	],
)
def test_rejects_foreign_prediction_source_paths(
	completed_job: Path,
	source_key: str,
	foreign_name: str,
) -> None:
	prediction_path = completed_job / 'prediction/prediction_metadata.json'
	payload = _read_json(prediction_path)
	source = payload['source_identity']
	assert isinstance(source, dict)
	entry = source[source_key]
	assert isinstance(entry, dict)
	entry['path'] = str(completed_job / 'decoder' / foreign_name)
	_write_json(prediction_path, payload)
	_refresh_prediction_metadata_identity(completed_job)

	with pytest.raises(ValueError, match='must equal the exact path'):
		_read(completed_job)


@pytest.mark.parametrize('bad_sha', ['A' * 64, 'a' * 63, 'g' * 64])
def test_rejects_non_lowercase_sha(completed_job: Path, bad_sha: str) -> None:
	path = completed_job / 'evaluation/evaluation_metadata.json'
	payload = _read_json(path)
	outputs = payload['outputs']
	assert isinstance(outputs, dict)
	metrics = outputs['metrics.json']
	assert isinstance(metrics, dict)
	metrics['sha256'] = bad_sha
	_write_json(path, payload)

	with pytest.raises(ValueError, match='must be a lowercase SHA-256'):
		_read(completed_job)


def test_rejects_foreign_lowercase_sha(completed_job: Path) -> None:
	path = completed_job / 'evaluation/evaluation_metadata.json'
	payload = _read_json(path)
	outputs = payload['outputs']
	assert isinstance(outputs, dict)
	metrics = outputs['metrics.json']
	assert isinstance(metrics, dict)
	metrics['sha256'] = '0' * 64
	_write_json(path, payload)

	with pytest.raises(ValueError, match='changed after its provenance was recorded'):
		_read(completed_job)


@pytest.mark.parametrize(
	('key', 'bad_value', 'exception', 'message'),
	[
		('macro_f1', True, TypeError, 'must be numeric'),
		('macro_f1', float('nan'), ValueError, 'must be finite'),
		('macro_f1', -0.1, ValueError, r'must be within \[0, 1\]'),
		('macro_f1', 1.1, ValueError, r'must be within \[0, 1\]'),
		('evaluation_voxel_count', True, ValueError, 'positive integer'),
		('evaluation_voxel_count', 0, ValueError, 'positive integer'),
	],
)
def test_rejects_malformed_metric_evidence(
	completed_job: Path,
	key: str,
	bad_value: object,
	exception: type[Exception],
	message: str,
) -> None:
	path = completed_job / 'evaluation/metrics.json'
	payload = _read_json(path)
	payload[key] = bad_value
	_write_json(path, payload)

	with pytest.raises(exception, match=message):
		_read(completed_job)


def test_rejects_prediction_array_mutated_after_evaluation(
	completed_job: Path,
) -> None:
	path = completed_job / 'prediction/f3_voxel_predictions.npy'
	path.write_bytes(b'mutated predictions')

	with pytest.raises(ValueError, match='changed after its provenance was recorded'):
		_read(completed_job)


def test_rejects_ground_truth_mutated_after_evaluation(completed_job: Path) -> None:
	path = completed_job.parents[2] / 'canonical_sources/f3_facies_labels.npy'
	path.write_bytes(b'mutated labels')

	with pytest.raises(ValueError, match='changed after its provenance was recorded'):
		_read(completed_job)


def test_rejects_foreign_ground_truth_path(completed_job: Path) -> None:
	path = completed_job / 'evaluation/evaluation_metadata.json'
	payload = _read_json(path)
	inputs = payload['inputs']
	assert isinstance(inputs, dict)
	class_info = inputs['class_info']
	assert isinstance(class_info, dict)
	class_info['path'] = str(completed_job.parent / 'foreign/class_info.json')
	_write_json(path, payload)

	with pytest.raises(ValueError, match='must equal the exact path'):
		_read(completed_job)


def test_rejects_uppercase_ground_truth_sha(completed_job: Path) -> None:
	path = completed_job / 'evaluation/evaluation_metadata.json'
	payload = _read_json(path)
	inputs = payload['inputs']
	assert isinstance(inputs, dict)
	label_segy = inputs['source_label_segy']
	assert isinstance(label_segy, dict)
	sha256 = label_segy['sha256']
	assert isinstance(sha256, str)
	label_segy['sha256'] = sha256.upper()
	_write_json(path, payload)

	with pytest.raises(ValueError, match='must be a lowercase SHA-256'):
		_read(completed_job)


def test_rejects_noncanonical_evaluation_dataset(completed_job: Path) -> None:
	path = completed_job / 'evaluation/evaluation_metadata.json'
	payload = _read_json(path)
	payload['dataset'] = {'name': 'foreign', 'version': 'facies_benchmark_v2'}
	_write_json(path, payload)

	with pytest.raises(ValueError, match='evaluation dataset must equal'):
		_read(completed_job)


def test_rejects_current_decoder_config_mutation(completed_job: Path) -> None:
	path = completed_job / 'decoder/resolved_config.json'
	payload = _read_json(path)
	model = payload['model']
	assert isinstance(model, dict)
	model['tag'] = 'foreign'
	_write_json(path, payload)

	with pytest.raises(ValueError, match='changed after its provenance was recorded'):
		_read(completed_job)


def test_rejects_checkpoint_resolved_config_mismatch(completed_job: Path) -> None:
	path = completed_job / 'decoder/best.pt'
	payload = torch.load(path, map_location='cpu', weights_only=False)
	resolved = payload['resolved_config']
	assert isinstance(resolved, dict)
	model = resolved['model']
	assert isinstance(model, dict)
	model['freeze_encoder'] = False
	torch.save(payload, path)
	_refresh_decoder_checkpoint_identity(completed_job)

	with pytest.raises(ValueError, match='resolved config does not match checkpoint'):
		_read(completed_job)


def test_rejects_evaluation_decoder_architecture_mismatch(
	completed_job: Path,
) -> None:
	path = completed_job / 'evaluation/evaluation_metadata.json'
	payload = _read_json(path)
	architecture = payload['decoder_architecture']
	assert isinstance(architecture, dict)
	architecture['class_count'] = 5
	_write_json(path, payload)

	with pytest.raises(ValueError, match='decoder architecture differs'):
		_read(completed_job)


def test_reads_existing_control_artifact() -> None:
	repository = Path(__file__).resolve().parents[2]
	job_dir = (
		repository
		/ 'artifacts/seis_ssl_cluster/f3_lithology_benchmark'
		/ 'local_bt_noise_rotation_search_v1/runs'
		/ 'model=local_bt_nr_rot90_asym_g060_3ep'
		/ 'layout=layout_000/size=small'
	)
	if not job_dir.is_dir():
		pytest.skip('local F3 control artifact is unavailable')

	evidence = read_f3_completed_evaluation_provenance(
		job_dir,
		model_id='local_bt_nr_rot90_asym_g060_3ep',
		layout_id='layout_000',
		data_size='small',
	)

	assert evidence['macro_f1'] == pytest.approx(0.5135372372225744)
	assert evidence['evaluation_voxel_count'] == 470136
	assert (
		evidence['metrics_sha256']
		== '1f9adfd776a5b5630311e3fa8b7bf7a238dff52f709984d720f160376468420f'
	)
	assert (
		evidence['prediction_metadata_sha256']
		== '16dcc5370d9368677020677d406f49c0b4426dee0654eee6f24ca1c544540ba5'
	)

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	FIXED_DECODER_CONTRACT,
)
from seis_ssl_cluster.embedding.writer import file_sha256, output_paths
from seis_ssl_cluster.f3.lithology.completed_decoder_contract import (
	read_f3_completed_decoder_contract,
)
from seis_ssl_cluster.f3.lithology.five_way_runner import FIVE_WAY_TILE_SETTINGS
from seis_ssl_cluster.f3.lithology.voxel_dataset import GRID_NAME, METADATA_NAME
from seis_ssl_cluster.f3.lithology.voxel_tiles import (
	VoxelTileManifest,
	write_voxel_tile_manifest,
)
from seis_ssl_cluster.training.voxel_decoder.checkpoint import (
	load_voxel_decoder_checkpoint,
	make_best_selection_state,
	save_voxel_decoder_checkpoint,
)

if TYPE_CHECKING:
	from collections.abc import Callable

_MODEL_ID = 'candidate'
_LAYOUT_ID = 'layout_000'
_DATA_SIZE = 'small'


def _plain(value: object) -> object:
	if isinstance(value, dict):
		return {key: _plain(item) for key, item in value.items()}
	if isinstance(value, tuple | list):
		return [_plain(item) for item in value]
	return value


def _resolved_config(
	root: Path,
	*,
	model_id: str,
	layout_id: str,
	data_size: str,
) -> dict[str, object]:
	job_dir = root / f'model={model_id}' / f'layout={layout_id}' / f'size={data_size}'
	artifact_root = root / 'artifacts'
	embeddings_dir = (
		artifact_root
		/ 'embeddings/f3/facies_benchmark_v2/suite'
		/ model_id
		/ 'overlap_x64'
	)
	voxel_dataset = (
		artifact_root
		/ 'lithology/f3/facies_benchmark_v2/conditions/datasets'
		/ f'layout={layout_id}'
		/ f'size={data_size}'
		/ 'voxel_supervision'
	)
	return {
		'paths': {
			'artifact_root': str(artifact_root),
			'f3_root': str(root / 'f3'),
		},
		'dataset': {
			'name': 'f3_facies_benchmark',
			'version': 'facies_benchmark_v2',
		},
		'model': {'tag': model_id, 'freeze_encoder': True},
		'embeddings': {
			'spec': 'overlap_x64',
			'input_dir': str(embeddings_dir),
		},
		'voxel_dataset': {'input_dir': str(voxel_dataset)},
		'decoder': {
			key: _plain(FIXED_DECODER_CONTRACT[key])
			for key in (
				'spec',
				'embedding_dim',
				'class_count',
				'hidden_channels',
				'upsample_factors',
				'upsample_mode',
				'normalization',
			)
		},
		'tiles': _plain(dict(FIVE_WAY_TILE_SETTINGS)),
		'train': {
			**{
				key: _plain(FIXED_DECODER_CONTRACT[key])
				for key in (
					'epochs',
					'batch_size',
					'learning_rate',
					'weight_decay',
					'class_weight',
					'seed',
					'amp',
					'gradient_clip_norm',
					'sampling_mode',
					'steps_per_epoch',
				)
			},
			'num_workers': 0,
		},
		'outputs': {'output_dir': str(job_dir / 'decoder')},
	}


def _write_json(path: Path, payload: object) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
	)


def _identity(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}


def _history(epoch_count: int) -> list[dict[str, int]]:
	return [
		{'epoch': epoch, 'global_step': (epoch + 1) * 440}
		for epoch in range(epoch_count)
	]


def _manifest(split: str) -> VoxelTileManifest:
	return VoxelTileManifest(
		split=split,
		volume_shape_xyz=(8, 8, 8),
		token_grid_shape_xyz=(1, 1, 1),
		patch_size_xyz=(8, 8, 8),
		core_size_tokens=(8, 8, 8),
		context_halo_tokens=(1, 1, 1),
		class_ids=(0, 1, 2, 3, 4, 5),
		tiles=(),
	)


def _write_job(
	root: Path,
	*,
	model_id: str = _MODEL_ID,
	layout_id: str = _LAYOUT_ID,
	data_size: str = _DATA_SIZE,
) -> Path:
	job_dir = root / f'model={model_id}' / f'layout={layout_id}' / f'size={data_size}'
	decoder_dir = job_dir / 'decoder'
	decoder_dir.mkdir(parents=True)
	resolved = _resolved_config(
		root,
		model_id=model_id,
		layout_id=layout_id,
		data_size=data_size,
	)

	dataset = resolved['dataset']
	embeddings = resolved['embeddings']
	voxel_dataset = resolved['voxel_dataset']
	assert isinstance(dataset, dict)
	assert isinstance(embeddings, dict)
	assert isinstance(voxel_dataset, dict)
	embedding_files = output_paths(
		Path(str(embeddings['input_dir'])), str(dataset['name'])
	)
	for path, contents in (
		(embedding_files.embeddings, b'embeddings'),
		(embedding_files.valid_tokens, b'valid-tokens'),
		(embedding_files.metadata, b'{}\n'),
	):
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_bytes(contents)
	voxel_dir = Path(str(voxel_dataset['input_dir']))
	voxel_metadata = voxel_dir / METADATA_NAME
	voxel_split_grid = voxel_dir / GRID_NAME
	label_volume = root / 'sources/f3_facies_labels.npy'
	for path, contents in (
		(voxel_metadata, b'{}\n'),
		(voxel_split_grid, b'split-grid'),
		(label_volume, b'labels'),
	):
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_bytes(contents)
	artifact_identities = {
		'name': 'f3_voxel_decoder_sources',
		'embeddings': _identity(embedding_files.embeddings),
		'embedding_metadata': _identity(embedding_files.metadata),
		'valid_tokens': _identity(embedding_files.valid_tokens),
		'voxel_dataset_metadata': _identity(voxel_metadata),
		'voxel_split_grid': _identity(voxel_split_grid),
		'label_volume': _identity(label_volume),
	}

	train_manifest = _manifest('train')
	validation_manifest = _manifest('validation')
	train_manifest_path = decoder_dir / 'train_tile_manifest.json'
	validation_manifest_path = decoder_dir / 'validation_tile_manifest.json'
	write_voxel_tile_manifest(train_manifest_path, train_manifest)
	write_voxel_tile_manifest(validation_manifest_path, validation_manifest)
	manifest_hashes = {
		'train': train_manifest.identity_sha256,
		'validation': validation_manifest.identity_sha256,
	}
	resolved_path = decoder_dir / 'resolved_config.json'
	_write_json(resolved_path, resolved)
	_write_json(
		decoder_dir / 'run_metadata.json',
		{
			'created_at_utc': '2026-09-07T00:00:00+00:00',
			'git_commit': None,
			'package_version': '0.1.0',
			'source_embedding_metadata': str(embedding_files.metadata),
			'source_valid_tokens': str(embedding_files.valid_tokens),
			'voxel_dataset_metadata': str(voxel_metadata),
			'initial_model_state_sha256': 'a' * 64,
			'sampling_mode': FIXED_DECODER_CONTRACT['sampling_mode'],
			'steps_per_epoch': FIXED_DECODER_CONTRACT['steps_per_epoch'],
			'train_seed': FIXED_DECODER_CONTRACT['seed'],
			'train_tile_manifest_sha256': manifest_hashes['train'],
			'validation_tile_manifest_sha256': manifest_hashes['validation'],
		},
	)

	model = torch.nn.Linear(1, 1)
	optimizer = torch.optim.AdamW(model.parameters())
	validation_metrics = {
		'macro_f1': 0.5,
		'mean_iou': 0.4,
		'weighted_cross_entropy': 1.0,
	}
	best_epoch = 4
	best_state = make_best_selection_state(
		epoch=best_epoch, validation_metrics=validation_metrics
	)
	best_path = save_voxel_decoder_checkpoint(
		decoder_dir / 'best.pt',
		model=model,
		optimizer=optimizer,
		epoch=best_epoch,
		global_step=(best_epoch + 1) * 440,
		resolved_config=resolved,
		class_weights=[1.0] * 6,
		artifact_identities=artifact_identities,
		tile_manifest_hashes=manifest_hashes,
		best_selection_state=best_state,
		training_history=_history(best_epoch + 1),
		current_metrics={'train': {}, 'validation': validation_metrics},
		checkpoint_kind='epoch',
		best_checkpoint_sha256=None,
	)
	best_sha256 = file_sha256(best_path)
	save_voxel_decoder_checkpoint(
		decoder_dir / 'latest.pt',
		model=model,
		optimizer=optimizer,
		epoch=49,
		global_step=22_000,
		resolved_config=resolved,
		class_weights=[1.0] * 6,
		artifact_identities=artifact_identities,
		tile_manifest_hashes=manifest_hashes,
		best_selection_state=best_state,
		training_history=_history(50),
		current_metrics={'train': {}, 'validation': {'macro_f1': 0.4}},
		checkpoint_kind='completed',
		best_checkpoint_sha256=best_sha256,
	)

	class_info = root / 'sources/class_info.json'
	_write_json(class_info, {'classes': []})
	_write_json(
		job_dir / 'prediction/prediction_metadata.json',
		{
			'prediction_kind': 'frozen_embedding_decoder',
			'model_tag': model_id,
			'decoder_architecture': resolved['decoder'],
			'training_sampling': {
				'sampling_mode': FIXED_DECODER_CONTRACT['sampling_mode'],
				'steps_per_epoch': FIXED_DECODER_CONTRACT['steps_per_epoch'],
				'train_seed': FIXED_DECODER_CONTRACT['seed'],
				'train_tile_manifest_sha256': manifest_hashes['train'],
				'validation_tile_manifest_sha256': manifest_hashes['validation'],
			},
			'inputs': {
				'embeddings': str(embedding_files.embeddings),
				'embedding_metadata': str(embedding_files.metadata),
				'valid_tokens': str(embedding_files.valid_tokens),
				'class_info': str(class_info),
				'decoder_checkpoint': str(best_path),
			},
			'source_identity': {
				'decoder_checkpoint': _identity(best_path),
				'resolved_decoder_config': _identity(resolved_path),
				'class_info': _identity(class_info),
				'artifact_identities': artifact_identities,
				'tile_manifests': {
					'train': _identity(train_manifest_path),
					'validation': _identity(validation_manifest_path),
				},
			},
		},
	)
	return job_dir


def _read_json(path: Path) -> dict[str, object]:
	payload = json.loads(path.read_text(encoding='utf-8'))
	assert isinstance(payload, dict)
	return payload


def _mutate_checkpoint(
	path: Path, mutation: Callable[[dict[str, object]], None]
) -> None:
	payload = load_voxel_decoder_checkpoint(path, map_location='cpu')
	mutation(payload)
	torch.save(payload, path)


def _read_contract(job_dir: Path) -> dict[str, object]:
	return read_f3_completed_decoder_contract(
		job_dir,
		model_id=_MODEL_ID,
		layout_id=_LAYOUT_ID,
		data_size=_DATA_SIZE,
	)


def test_reads_completed_contract_and_returns_full_provenance(
	tmp_path: Path,
) -> None:
	job_dir = _write_job(tmp_path)
	evidence = _read_contract(job_dir)

	assert evidence['decoder_initial_state_sha256'] == 'a' * 64
	assert evidence['decoder_completed_epoch'] == 49
	assert evidence['decoder_completed_global_step'] == 22_000
	assert evidence['decoder_best_epoch'] == 4
	assert evidence['decoder_best_global_step'] == 2_200
	assert evidence['decoder_latest_checkpoint_sha256'] == file_sha256(
		job_dir / 'decoder/latest.pt'
	)
	assert evidence['decoder_best_checkpoint_sha256'] == file_sha256(
		job_dir / 'decoder/best.pt'
	)
	assert evidence['prediction_metadata_sha256'] == file_sha256(
		job_dir / 'prediction/prediction_metadata.json'
	)


def test_contract_identity_is_model_independent(tmp_path: Path) -> None:
	first = _write_job(tmp_path / 'first', model_id='candidate')
	second = _write_job(tmp_path / 'second', model_id='control')
	first_evidence = read_f3_completed_decoder_contract(
		first,
		model_id='candidate',
		layout_id=_LAYOUT_ID,
		data_size=_DATA_SIZE,
	)
	second_evidence = read_f3_completed_decoder_contract(
		second,
		model_id='control',
		layout_id=_LAYOUT_ID,
		data_size=_DATA_SIZE,
	)
	assert (
		first_evidence['decoder_contract_identity_sha256']
		== second_evidence['decoder_contract_identity_sha256']
	)


def test_reads_real_completed_control_cell_when_available() -> None:
	workspace = Path(__file__).resolve().parents[2]
	model_id = 'local_bt_nr_rot90_asym_g060_3ep'
	job_dir = (
		workspace
		/ 'artifacts/seis_ssl_cluster/f3_lithology_benchmark'
		/ 'local_bt_noise_rotation_search_v1/runs'
		/ f'model={model_id}/layout=layout_001/size=small'
	)
	if not (job_dir / 'decoder/latest.pt').is_file():
		pytest.skip('real F3 local-BT control artifact is not present')
	evidence = read_f3_completed_decoder_contract(
		job_dir,
		model_id=model_id,
		layout_id='layout_001',
		data_size='small',
	)
	assert evidence['decoder_completed_epoch'] == 49
	assert evidence['decoder_completed_global_step'] == 22_000
	assert evidence['decoder_best_checkpoint_sha256'] == file_sha256(
		job_dir / 'decoder/best.pt'
	)


@pytest.mark.parametrize(
	('section', 'key', 'bad_value', 'message'),
	[
		('train', 'num_workers', 1, r'train\.num_workers'),
		('train', 'seed', True, r'train\.seed'),
		('decoder', 'embedding_dim', True, r'decoder\.embedding_dim'),
		('train', 'unexpected_key', 0, r'unexpected_key'),
	],
)
def test_rejects_fixed_contract_drift_and_bool_integer_aliases(
	tmp_path: Path,
	section: str,
	key: str,
	bad_value: object,
	message: str,
) -> None:
	job_dir = _write_job(tmp_path)
	path = job_dir / 'decoder/resolved_config.json'
	payload = _read_json(path)
	section_payload = payload[section]
	assert isinstance(section_payload, dict)
	section_payload[key] = bad_value
	_write_json(path, payload)

	with pytest.raises((TypeError, ValueError), match=message):
		_read_contract(job_dir)


@pytest.mark.parametrize(
	('mutation', 'message'),
	[
		(
			lambda payload: payload.__setitem__('checkpoint_kind', 'epoch'),
			'checkpoint_kind',
		),
		(lambda payload: payload.__setitem__('epoch', 48), r'epoch must equal 49'),
		(
			lambda payload: payload.__setitem__('global_step', 21_999),
			r'global_step must equal 22000',
		),
		(
			lambda payload: payload.__setitem__(
				'training_history', payload['training_history'][:-1]
			),
			r'must contain 50 completed epochs',
		),
	],
)
def test_rejects_partial_or_incomplete_latest_checkpoint(
	tmp_path: Path,
	mutation: Callable[[dict[str, object]], None],
	message: str,
) -> None:
	job_dir = _write_job(tmp_path)
	_mutate_checkpoint(job_dir / 'decoder/latest.pt', mutation)

	with pytest.raises((TypeError, ValueError), match=message):
		_read_contract(job_dir)


def test_rejects_checkpoint_resolved_config_and_manifest_drift(
	tmp_path: Path,
) -> None:
	job_dir = _write_job(tmp_path / 'config')

	def drift_config(payload: dict[str, object]) -> None:
		resolved = payload['resolved_config']
		assert isinstance(resolved, dict)
		train = resolved['train']
		assert isinstance(train, dict)
		train['num_workers'] = 1

	_mutate_checkpoint(job_dir / 'decoder/latest.pt', drift_config)
	with pytest.raises(ValueError, match=r'resolved_config\.train\.num_workers'):
		_read_contract(job_dir)

	job_dir = _write_job(tmp_path / 'manifest')

	def drift_manifest(payload: dict[str, object]) -> None:
		hashes = payload['tile_manifest_hashes']
		assert isinstance(hashes, dict)
		hashes['train'] = '0' * 64

	_mutate_checkpoint(job_dir / 'decoder/latest.pt', drift_manifest)
	with pytest.raises(ValueError, match=r'tile_manifest_hashes\.train'):
		_read_contract(job_dir)


def test_rejects_best_checkpoint_linkage_drift(tmp_path: Path) -> None:
	job_dir = _write_job(tmp_path)
	_mutate_checkpoint(
		job_dir / 'decoder/latest.pt',
		lambda payload: payload.__setitem__('best_checkpoint_sha256', '0' * 64),
	)

	with pytest.raises(ValueError, match=r'best\.pt SHA-256 linkage'):
		_read_contract(job_dir)


@pytest.mark.parametrize(
	('key', 'bad_value'),
	[
		('train_seed', True),
		('sampling_mode', 'all_tiles_once'),
		('steps_per_epoch', 441),
		('train_tile_manifest_sha256', '0' * 64),
		('validation_tile_manifest_sha256', '0' * 64),
	],
)
def test_rejects_run_metadata_contract_drift(
	tmp_path: Path, key: str, bad_value: object
) -> None:
	job_dir = _write_job(tmp_path)
	path = job_dir / 'decoder/run_metadata.json'
	payload = _read_json(path)
	payload[key] = bad_value
	_write_json(path, payload)

	with pytest.raises((TypeError, ValueError), match=key):
		_read_contract(job_dir)


@pytest.mark.parametrize(
	('mutation', 'message'),
	[
		(
			lambda payload: payload['source_identity'][
				'decoder_checkpoint'
			].__setitem__('sha256', '0' * 64),
			'decoder_checkpoint.sha256',
		),
		(
			lambda payload: payload['source_identity'][
				'resolved_decoder_config'
			].__setitem__('sha256', '0' * 64),
			'resolved_decoder_config.sha256',
		),
		(
			lambda payload: payload['training_sampling'].__setitem__(
				'train_seed',
				True,  # noqa: FBT003 - deliberate bool/int drift
			),
			'training_sampling.train_seed',
		),
		(
			lambda payload: payload['source_identity']['artifact_identities'][
				'embeddings'
			].__setitem__('sha256', '0' * 64),
			'artifact_identities.embeddings.sha256',
		),
		(
			lambda payload: payload['training_sampling'].__setitem__(
				'unexpected_key', 1
			),
			'unexpected_key',
		),
	],
)
def test_rejects_prediction_provenance_drift(
	tmp_path: Path,
	mutation: Callable[[dict[str, object]], None],
	message: str,
) -> None:
	job_dir = _write_job(tmp_path)
	path = job_dir / 'prediction/prediction_metadata.json'
	payload = _read_json(path)
	mutation(payload)
	_write_json(path, payload)

	with pytest.raises((TypeError, ValueError), match=message):
		_read_contract(job_dir)

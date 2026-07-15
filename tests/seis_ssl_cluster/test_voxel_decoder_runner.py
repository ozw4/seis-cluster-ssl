from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

import seis_ssl_cluster.training.voxel_decoder.runner as voxel_decoder_runner
from seis_ssl_cluster.config.f3_lithology_voxel_decoder import (
	UNIFORM_TILES_WITH_REPLACEMENT,
	f3_lithology_voxel_decoder_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_inference import (
	f3_lithology_voxel_inference_config_from_mapping,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.voxel_decoder_inference import (
	predict_f3_lithology_voxels,
)
from seis_ssl_cluster.models.voxel_decoder.spec import (
	VOXEL_DECODER_NORMALIZATION,
	VOXEL_DECODER_SPEC,
	VOXEL_DECODER_UPSAMPLE_MODE,
)
from seis_ssl_cluster.training.voxel_decoder.checkpoint import (
	load_voxel_decoder_checkpoint,
)
from seis_ssl_cluster.training.voxel_decoder.runner import (
	inspect_f3_lithology_voxel_decoder,
	run_f3_lithology_voxel_decoder,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = REPO_ROOT / 'proc' / 'seis_ssl_cluster' / 'train_f3_lithology_voxel_decoder.py'


def _job(tmp_path, name: str, *, epochs: int = 2):
	root = tmp_path / 'artifacts'
	embedding_dir = root / 'embeddings' / 'f3' / 'v1' / 'encoder-v1' / 'tiny-spec'
	voxel_dir = root / 'voxel'
	embedding_dir.mkdir(parents=True)
	voxel_dir.mkdir(parents=True)
	embeddings = np.array(
		[[[[1.0, -1.0]]], [[[0.5, -0.5]]], [[[-1.0, 1.0]]], [[[1.0, 1.0]]]],
		dtype=np.float32,
	)
	valid = np.ones((4, 1, 1), dtype=np.bool_)
	labels = np.array([0, 1, 0, 1], dtype=np.int16).reshape(4, 1, 1)
	split = np.array([1, 1, 1, 2], dtype=np.uint8).reshape(4, 1, 1)
	embedding_path = embedding_dir / 'tiny.embeddings.npy'
	valid_path = embedding_dir / 'tiny.valid_tokens.npy'
	metadata_path = embedding_dir / 'tiny.embedding_metadata.json'
	label_path = root / 'labels.npy'
	np.save(embedding_path, embeddings, allow_pickle=False)
	np.save(valid_path, valid, allow_pickle=False)
	np.save(label_path, labels, allow_pickle=False)
	np.save(voxel_dir / 'supervision_split_grid.npy', split, allow_pickle=False)
	metadata = {
		'volume_shape_xyz': [4, 1, 1],
		'patch_size': [1, 1, 1],
		'token_grid_shape': [4, 1, 1],
		'embedding_dim': 2,
		'checkpoint_path': str(
			root / 'pretraining' / 'f3' / 'v1' / 'encoder-v1' / 'best.pt'
		),
		'preprocessing': {'kind': 'tiny'},
		'zero_mask': {'enabled': True},
	}
	metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
	classes = [
		{
			'class_id': 0,
			'class_name': 'zero',
			'rgb': [0, 0, 0],
			'hex_color': '#000000',
		},
		{
			'class_id': 1,
			'class_name': 'one',
			'rgb': [1, 1, 1],
			'hex_color': '#010101',
		},
	]
	class_info_path = root / 'class_info.json'
	class_info_path.write_text(
		json.dumps(
			{
				'0': {'name': 'zero', 'color': [0, 0, 0]},
				'1': {'name': 'one', 'color': [1, 1, 1]},
			}
		),
		encoding='utf-8',
	)
	voxel_metadata = {
		'dataset': {'name': 'tiny', 'version': 'v1'},
		'classes': classes,
		'reference_embedding': {
			'path': str(metadata_path),
			'sha256': file_sha256(metadata_path),
			'metadata': metadata,
		},
		'reference_valid_tokens': {
			'path': str(valid_path),
			'sha256': file_sha256(valid_path),
		},
		'label_volume': {'path': str(label_path), 'sha256': file_sha256(label_path)},
	}
	(voxel_dir / 'voxel_dataset_metadata.json').write_text(
		json.dumps(voxel_metadata), encoding='utf-8'
	)
	raw = {
		'paths': {'artifact_root': str(root), 'f3_root': str(tmp_path / 'f3')},
		'dataset': {'name': 'tiny', 'version': 'v1'},
		'model': {'tag': 'encoder-v1', 'freeze_encoder': True},
		'embeddings': {'input_dir': str(embedding_dir)},
		'voxel_dataset': {'input_dir': str(voxel_dir)},
		'decoder': {
			'spec': VOXEL_DECODER_SPEC,
			'embedding_dim': 2,
			'class_count': 2,
			'hidden_channels': [2],
			'upsample_factors': [[1, 1, 1]],
			'upsample_mode': VOXEL_DECODER_UPSAMPLE_MODE,
			'normalization': VOXEL_DECODER_NORMALIZATION,
		},
		'tiles': {'core_size_tokens': [1, 1, 1], 'context_halo_tokens': [1, 0, 0]},
		'train': {
			'epochs': epochs,
			'batch_size': 1,
			'learning_rate': 0.01,
			'weight_decay': 0.0,
			'class_weight': 'balanced',
			'seed': 7,
			'num_workers': 0,
			'amp': False,
			'gradient_clip_norm': 1.0,
		},
		'outputs': {'output_dir': str(root / name)},
	}
	return raw, embedding_path


def test_cpu_step_resume_matches_uninterrupted_run(tmp_path) -> None:
	raw, embedding_path = _job(tmp_path, 'full')
	before = file_sha256(embedding_path)
	full = run_f3_lithology_voxel_decoder(
		f3_lithology_voxel_decoder_config_from_mapping(raw), device='cpu'
	)
	resume_raw = deepcopy(raw)
	resume_raw['outputs']['output_dir'] = str(tmp_path / 'artifacts' / 'resumed')
	resume_config = f3_lithology_voxel_decoder_config_from_mapping(resume_raw)
	partial = run_f3_lithology_voxel_decoder(resume_config, device='cpu', max_steps=2)
	assert not partial.completed
	resumed = run_f3_lithology_voxel_decoder(
		resume_config, device='cpu', resume=partial.latest_checkpoint
	)
	assert resumed.completed
	full_payload = load_voxel_decoder_checkpoint(full.latest_checkpoint)
	resumed_payload = load_voxel_decoder_checkpoint(resumed.latest_checkpoint)
	for key, tensor in full_payload['model_state_dict'].items():
		assert torch.equal(tensor, resumed_payload['model_state_dict'][key])
	assert file_sha256(embedding_path) == before
	assert resumed.history_csv.read_text(encoding='utf-8').startswith(
		'epoch,global_step,'
	)


def test_replacement_sampling_has_fixed_steps_and_run_identity(tmp_path) -> None:
	raw, _ = _job(tmp_path, 'replacement-fixed-steps', epochs=2)
	raw['train']['sampling_mode'] = UNIFORM_TILES_WITH_REPLACEMENT
	raw['train']['steps_per_epoch'] = 4
	config = f3_lithology_voxel_decoder_config_from_mapping(raw)

	result = run_f3_lithology_voxel_decoder(config, device='cpu')

	assert result.completed
	assert result.global_step == 8
	with result.history_csv.open(encoding='utf-8', newline='') as file_obj:
		history = list(csv.DictReader(file_obj))
	assert [int(row['global_step']) for row in history] == [4, 8]
	latest = load_voxel_decoder_checkpoint(result.latest_checkpoint)
	best = load_voxel_decoder_checkpoint(result.best_checkpoint)
	for payload in (latest, best):
		assert payload['resolved_config']['train']['sampling_mode'] == (
			UNIFORM_TILES_WITH_REPLACEMENT
		)
		assert payload['resolved_config']['train']['steps_per_epoch'] == 4
	metadata = json.loads(
		(config.output_dir / 'run_metadata.json').read_text(encoding='utf-8')
	)
	train_manifest = json.loads(
		(config.output_dir / 'train_tile_manifest.json').read_text(encoding='utf-8')
	)
	validation_manifest = json.loads(
		(config.output_dir / 'validation_tile_manifest.json').read_text(
			encoding='utf-8'
		)
	)
	assert len(metadata['initial_model_state_sha256']) == 64
	assert metadata['sampling_mode'] == UNIFORM_TILES_WITH_REPLACEMENT
	assert metadata['steps_per_epoch'] == 4
	assert metadata['train_seed'] == 7
	assert metadata['train_tile_manifest_sha256'] == train_manifest['identity_sha256']
	assert (
		metadata['validation_tile_manifest_sha256']
		== validation_manifest['identity_sha256']
	)


def test_replacement_mid_epoch_resume_matches_uninterrupted_run(tmp_path) -> None:
	raw, _ = _job(tmp_path, 'replacement-full', epochs=2)
	raw['train']['sampling_mode'] = UNIFORM_TILES_WITH_REPLACEMENT
	raw['train']['steps_per_epoch'] = 4
	full = run_f3_lithology_voxel_decoder(
		f3_lithology_voxel_decoder_config_from_mapping(raw), device='cpu'
	)
	resume_raw = deepcopy(raw)
	resume_raw['outputs']['output_dir'] = str(
		tmp_path / 'artifacts' / 'replacement-resumed'
	)
	resume_config = f3_lithology_voxel_decoder_config_from_mapping(resume_raw)
	partial = run_f3_lithology_voxel_decoder(resume_config, device='cpu', max_steps=3)
	assert not partial.completed

	resumed = run_f3_lithology_voxel_decoder(
		resume_config, device='cpu', resume=partial.latest_checkpoint
	)

	assert resumed.completed
	assert full.global_step == resumed.global_step == 8
	full_payload = load_voxel_decoder_checkpoint(full.latest_checkpoint)
	resumed_payload = load_voxel_decoder_checkpoint(resumed.latest_checkpoint)
	for key, tensor in full_payload['model_state_dict'].items():
		assert torch.equal(tensor, resumed_payload['model_state_dict'][key])
	assert full_payload['training_history'] == resumed_payload['training_history']
	full_metadata = json.loads(
		(Path(raw['outputs']['output_dir']) / 'run_metadata.json').read_text(
			encoding='utf-8'
		)
	)
	resumed_metadata = json.loads(
		(resume_config.output_dir / 'run_metadata.json').read_text(encoding='utf-8')
	)
	assert (
		full_metadata['initial_model_state_sha256']
		== resumed_metadata['initial_model_state_sha256']
	)


def test_run_passes_decoder_architecture_identity_to_model(
	tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
	raw, _ = _job(tmp_path, 'architecture-identity')
	config = f3_lithology_voxel_decoder_config_from_mapping(raw)
	actual_decoder = voxel_decoder_runner.VoxelDecoder3D
	construction: dict[str, object] = {}

	def capture_decoder(**kwargs):
		construction.update(kwargs)
		return actual_decoder(**kwargs)

	monkeypatch.setattr(voxel_decoder_runner, 'VoxelDecoder3D', capture_decoder)
	run_f3_lithology_voxel_decoder(config, device='cpu', max_steps=1)

	assert construction['spec'] == config.decoder.spec
	assert construction['upsample_mode'] == config.decoder.upsample_mode
	assert construction['normalization'] == config.decoder.normalization


def test_dry_run_inspection_does_not_create_output(tmp_path) -> None:
	raw, _ = _job(tmp_path, 'dry-run')
	config = f3_lithology_voxel_decoder_config_from_mapping(raw)
	plan = inspect_f3_lithology_voxel_decoder(config)
	assert plan.token_grid_shape_xyz == (4, 1, 1)
	assert not config.output_dir.exists()


def test_training_inspection_rejects_insufficient_context_halo(tmp_path) -> None:
	raw, _ = _job(tmp_path, 'insufficient-halo')
	raw['tiles']['context_halo_tokens'] = [0, 0, 0]
	config = f3_lithology_voxel_decoder_config_from_mapping(raw)

	with pytest.raises(ValueError, match='decoder receptive field'):
		inspect_f3_lithology_voxel_decoder(config)


def test_training_inspection_rejects_incompatible_decoder_geometry(tmp_path) -> None:
	raw, _ = _job(tmp_path, 'incompatible-decoder')
	raw['decoder']['upsample_factors'] = [[2, 1, 1]]
	config = f3_lithology_voxel_decoder_config_from_mapping(raw)

	with pytest.raises(ValueError, match='products must equal patch_size_xyz'):
		inspect_f3_lithology_voxel_decoder(config)


def test_completed_training_checkpoint_runs_chunked_inference(tmp_path) -> None:
	raw, _ = _job(tmp_path, 'train-to-inference', epochs=1)
	training = run_f3_lithology_voxel_decoder(
		f3_lithology_voxel_decoder_config_from_mapping(raw), device='cpu'
	)
	artifact_root = Path(raw['paths']['artifact_root'])
	inference_raw = {
		'paths': raw['paths'],
		'dataset': raw['dataset'],
		'model': raw['model'],
		'labels': {'class_info': str(artifact_root / 'class_info.json')},
		'embeddings': raw['embeddings'],
		'decoder': {'checkpoint': str(training.best_checkpoint)},
		'tiles': raw['tiles'],
		'inference': {'write_probabilities': False, 'overwrite': False},
		'outputs': {'output_dir': str(artifact_root / 'predictions')},
	}

	result = predict_f3_lithology_voxels(
		f3_lithology_voxel_inference_config_from_mapping(inference_raw),
		device='cpu',
	)

	assert result.valid_voxel_count == 4
	assert result.tile_count == 4


def test_dry_run_inspection_does_not_hash_array_contents(
	tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
	raw, _ = _job(tmp_path, 'dry-run-no-array-hash')
	config = f3_lithology_voxel_decoder_config_from_mapping(raw)

	def reject_array_hash(path: str | Path) -> str:
		if Path(path).suffix == '.npy':
			raise AssertionError('dry-run hashed an array artifact')
		return file_sha256(path)

	monkeypatch.setattr(voxel_decoder_runner, 'file_sha256', reject_array_hash)
	plan = inspect_f3_lithology_voxel_decoder(config)

	assert plan.volume_shape_xyz == (4, 1, 1)


def test_cli_dry_run_does_not_write_output(tmp_path) -> None:
	raw, _ = _job(tmp_path, 'cli-dry-run')
	config_path = tmp_path / 'decoder.yaml'
	config_path.write_text(json.dumps(raw), encoding='utf-8')
	env = os.environ.copy()
	env['PYTHONPATH'] = os.pathsep.join(
		(str(REPO_ROOT / 'src'), env.get('PYTHONPATH', ''))
	)
	completed = subprocess.run(  # noqa: S603
		[sys.executable, str(CLI), '--config', str(config_path), '--dry-run'],
		cwd=REPO_ROOT,
		env=env,
		text=True,
		capture_output=True,
		check=True,
		timeout=30,
	)
	assert 'execution: dry-run' in completed.stdout
	assert f'decoder.spec: {VOXEL_DECODER_SPEC}' in completed.stdout
	assert f'decoder.upsample_mode: {VOXEL_DECODER_UPSAMPLE_MODE}' in completed.stdout
	assert f'decoder.normalization: {VOXEL_DECODER_NORMALIZATION}' in completed.stdout
	assert not Path(raw['outputs']['output_dir']).exists()


def test_nonempty_output_collision_is_rejected(tmp_path) -> None:
	raw, _ = _job(tmp_path, 'collision')
	config = f3_lithology_voxel_decoder_config_from_mapping(raw)
	config.output_dir.mkdir()
	(config.output_dir / 'sentinel.txt').write_text('keep', encoding='utf-8')
	with pytest.raises(FileExistsError, match='non-empty'):
		run_f3_lithology_voxel_decoder(config, device='cpu')
	assert (config.output_dir / 'sentinel.txt').read_text(encoding='utf-8') == 'keep'


def test_resume_rejects_decoder_identity_mismatch(tmp_path) -> None:
	raw, _ = _job(tmp_path, 'identity-mismatch')
	config = f3_lithology_voxel_decoder_config_from_mapping(raw)
	partial = run_f3_lithology_voxel_decoder(config, device='cpu', max_steps=1)
	mismatched_raw = deepcopy(raw)
	mismatched_raw['decoder']['hidden_channels'] = [3]
	mismatched = f3_lithology_voxel_decoder_config_from_mapping(mismatched_raw)
	with pytest.raises(ValueError, match='resume identity mismatch: decoder'):
		run_f3_lithology_voxel_decoder(
			mismatched, device='cpu', resume=partial.latest_checkpoint
		)


def test_resume_rejects_runtime_device_and_scaler_mismatch(tmp_path) -> None:
	raw, _ = _job(tmp_path, 'runtime-mismatch')
	config = f3_lithology_voxel_decoder_config_from_mapping(raw)
	partial = run_f3_lithology_voxel_decoder(config, device='cpu', max_steps=1)
	payload = torch.load(
		partial.latest_checkpoint,
		map_location='cpu',
		weights_only=False,
	)
	payload['runtime_identity'] = {'device': 'cuda:0', 'amp_scaler': True}
	torch.save(payload, partial.latest_checkpoint)

	with pytest.raises(ValueError, match='resume runtime mismatch'):
		run_f3_lithology_voxel_decoder(
			config,
			device='cpu',
			resume=partial.latest_checkpoint,
		)


def test_training_enables_deterministic_torch_execution(tmp_path) -> None:
	raw, _ = _job(tmp_path, 'deterministic-execution')
	result = run_f3_lithology_voxel_decoder(
		f3_lithology_voxel_decoder_config_from_mapping(raw),
		device='cpu',
		max_steps=1,
	)
	payload = load_voxel_decoder_checkpoint(result.latest_checkpoint)

	assert torch.are_deterministic_algorithms_enabled()
	assert torch.backends.cudnn.deterministic
	assert not torch.backends.cudnn.benchmark
	assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
	assert payload['runtime_identity'] == {
		'device': 'cpu',
		'amp_scaler': False,
	}


@pytest.mark.parametrize(('setting', 'value'), [('batch_size', 2), ('seed', 8)])
def test_resume_rejects_training_identity_mismatch(
	tmp_path, setting: str, value: object
) -> None:
	raw, _ = _job(tmp_path, f'train-identity-{setting}')
	config = f3_lithology_voxel_decoder_config_from_mapping(raw)
	partial = run_f3_lithology_voxel_decoder(config, device='cpu', max_steps=1)
	mismatched_raw = deepcopy(raw)
	mismatched_raw['train'][setting] = value
	mismatched = f3_lithology_voxel_decoder_config_from_mapping(mismatched_raw)
	with pytest.raises(ValueError, match='resume identity mismatch: train'):
		run_f3_lithology_voxel_decoder(
			mismatched, device='cpu', resume=partial.latest_checkpoint
		)


def test_inspection_rejects_model_tag_mismatch(tmp_path) -> None:
	raw, _ = _job(tmp_path, 'model-tag-mismatch')
	raw['model']['tag'] = 'other-encoder'
	config = f3_lithology_voxel_decoder_config_from_mapping(raw)
	with pytest.raises(ValueError, match=r'model\.tag does not match'):
		inspect_f3_lithology_voxel_decoder(config)


def test_inspection_rejects_voxel_dataset_identity_mismatch(tmp_path) -> None:
	raw, _ = _job(tmp_path, 'dataset-mismatch')
	metadata_path = (
		Path(raw['voxel_dataset']['input_dir']) / 'voxel_dataset_metadata.json'
	)
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	metadata['dataset']['version'] = 'other-version'
	metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
	config = f3_lithology_voxel_decoder_config_from_mapping(raw)

	with pytest.raises(ValueError, match='dataset does not match'):
		inspect_f3_lithology_voxel_decoder(config)


def test_inspection_rejects_embedding_checkpoint_tag_mismatch(tmp_path) -> None:
	raw, _ = _job(tmp_path, 'checkpoint-tag-mismatch')
	embedding_dir = Path(raw['embeddings']['input_dir'])
	embedding_metadata_path = embedding_dir / 'tiny.embedding_metadata.json'
	embedding_metadata = json.loads(embedding_metadata_path.read_text(encoding='utf-8'))
	embedding_metadata['checkpoint_path'] = str(
		tmp_path / 'pretraining' / 'other-encoder' / 'best.pt'
	)
	embedding_metadata_path.write_text(json.dumps(embedding_metadata), encoding='utf-8')
	voxel_metadata_path = (
		Path(raw['voxel_dataset']['input_dir']) / 'voxel_dataset_metadata.json'
	)
	voxel_metadata = json.loads(voxel_metadata_path.read_text(encoding='utf-8'))
	voxel_metadata['reference_embedding'].update(
		{
			'sha256': file_sha256(embedding_metadata_path),
			'metadata': embedding_metadata,
		}
	)
	voxel_metadata_path.write_text(json.dumps(voxel_metadata), encoding='utf-8')
	config = f3_lithology_voxel_decoder_config_from_mapping(raw)
	with pytest.raises(ValueError, match='checkpoint_path'):
		inspect_f3_lithology_voxel_decoder(config)


def test_inspection_allows_model_independent_reference_artifact_paths(
	tmp_path,
) -> None:
	raw, _ = _job(tmp_path, 'model-independent-reference')
	embedding_metadata_path = (
		Path(raw['embeddings']['input_dir']) / 'tiny.embedding_metadata.json'
	)
	embedding_metadata = json.loads(embedding_metadata_path.read_text(encoding='utf-8'))
	embedding_metadata['candidate_encoder_identity'] = 'model-b'
	embedding_metadata_path.write_text(json.dumps(embedding_metadata), encoding='utf-8')
	metadata_path = (
		Path(raw['voxel_dataset']['input_dir']) / 'voxel_dataset_metadata.json'
	)
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	metadata['reference_embedding']['path'] = str(tmp_path / 'model-a.json')
	metadata['reference_embedding']['sha256'] = '0' * 64
	metadata['reference_valid_tokens']['path'] = str(tmp_path / 'other.npy')
	metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
	config = f3_lithology_voxel_decoder_config_from_mapping(raw)

	plan = inspect_f3_lithology_voxel_decoder(config)

	assert plan.embedding_metadata == embedding_metadata_path


def test_run_rejects_voxel_source_hash_mismatch(tmp_path) -> None:
	raw, _ = _job(tmp_path, 'source-hash-mismatch')
	metadata_path = (
		Path(raw['voxel_dataset']['input_dir']) / 'voxel_dataset_metadata.json'
	)
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	metadata['reference_valid_tokens']['sha256'] = '0' * 64
	metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
	config = f3_lithology_voxel_decoder_config_from_mapping(raw)
	with pytest.raises(ValueError, match='valid-token hash'):
		run_f3_lithology_voxel_decoder(config, device='cpu')


@pytest.mark.parametrize(
	'source', ['embeddings', 'valid_tokens', 'split_grid', 'labels']
)
def test_dry_run_inspection_rejects_missing_source_arrays(
	tmp_path, source: str
) -> None:
	raw, embedding_path = _job(tmp_path, f'missing-{source}')
	voxel_dir = Path(raw['voxel_dataset']['input_dir'])
	metadata = json.loads(
		(voxel_dir / 'voxel_dataset_metadata.json').read_text(encoding='utf-8')
	)
	paths = {
		'embeddings': embedding_path,
		'valid_tokens': Path(raw['embeddings']['input_dir']) / 'tiny.valid_tokens.npy',
		'split_grid': voxel_dir / 'supervision_split_grid.npy',
		'labels': Path(metadata['label_volume']['path']),
	}
	paths[source].unlink()
	config = f3_lithology_voxel_decoder_config_from_mapping(raw)

	with pytest.raises(FileNotFoundError, match='missing voxel decoder input'):
		inspect_f3_lithology_voxel_decoder(config)


def test_dry_run_inspection_rejects_malformed_source_array(tmp_path) -> None:
	raw, _ = _job(tmp_path, 'malformed-split')
	voxel_dir = Path(raw['voxel_dataset']['input_dir'])
	np.save(
		voxel_dir / 'supervision_split_grid.npy',
		np.ones((3, 1, 1), dtype=np.uint8),
		allow_pickle=False,
	)
	config = f3_lithology_voxel_decoder_config_from_mapping(raw)

	with pytest.raises(ValueError, match='shape does not match label_volume'):
		inspect_f3_lithology_voxel_decoder(config)


def test_run_rejects_label_volume_hash_mismatch(tmp_path) -> None:
	raw, _ = _job(tmp_path, 'label-hash-mismatch')
	metadata_path = (
		Path(raw['voxel_dataset']['input_dir']) / 'voxel_dataset_metadata.json'
	)
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	label_path = Path(metadata['label_volume']['path'])
	labels = np.load(label_path, allow_pickle=False)
	labels[0, 0, 0] = 1
	np.save(label_path, labels, allow_pickle=False)
	config = f3_lithology_voxel_decoder_config_from_mapping(raw)
	with pytest.raises(ValueError, match='label_volume hash'):
		run_f3_lithology_voxel_decoder(config, device='cpu')


def test_latest_and_best_checkpoints_follow_selection_rule(
	tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
	raw, _ = _job(tmp_path, 'selection')
	actual_validate = voxel_decoder_runner.validate_voxel_decoder_one_epoch
	validation_calls = 0

	def controlled_validate(**kwargs):
		nonlocal validation_calls
		metrics = dict(actual_validate(**kwargs))
		metrics['macro_f1'] = 1.0 if validation_calls == 0 else 0.0
		validation_calls += 1
		return metrics

	monkeypatch.setattr(
		voxel_decoder_runner,
		'validate_voxel_decoder_one_epoch',
		controlled_validate,
	)
	result = run_f3_lithology_voxel_decoder(
		f3_lithology_voxel_decoder_config_from_mapping(raw), device='cpu'
	)
	latest = load_voxel_decoder_checkpoint(result.latest_checkpoint)
	best = load_voxel_decoder_checkpoint(result.best_checkpoint)
	resolved = json.loads(
		(Path(raw['outputs']['output_dir']) / 'resolved_config.json').read_text(
			encoding='utf-8'
		)
	)
	assert latest['decoder_architecture'] == resolved['decoder']
	assert best['decoder_architecture'] == resolved['decoder']
	assert latest['epoch'] == 1
	assert latest['checkpoint_kind'] == 'completed'
	assert latest['best_selection_state']['epoch'] == 0
	assert best['epoch'] == 0
	assert best['checkpoint_kind'] == 'epoch'
	assert best['current_metrics']['validation']['macro_f1'] == 1.0


def test_run_json_artifacts_are_strict_standard_json(tmp_path) -> None:
	raw, _ = _job(tmp_path, 'standard-json')
	config = f3_lithology_voxel_decoder_config_from_mapping(raw)
	run_f3_lithology_voxel_decoder(config, device='cpu', max_steps=1)

	def reject_constant(value: str) -> None:
		raise AssertionError(f'non-standard JSON constant: {value}')

	for name in (
		'resolved_config.json',
		'run_metadata.json',
		'train_tile_manifest.json',
		'validation_tile_manifest.json',
	):
		text = (config.output_dir / name).read_text(encoding='utf-8')
		assert text.endswith('\n')
		assert isinstance(json.loads(text, parse_constant=reject_constant), dict)


def test_completed_checkpoint_cannot_be_resumed(tmp_path) -> None:
	raw, _ = _job(tmp_path, 'completed', epochs=1)
	config = f3_lithology_voxel_decoder_config_from_mapping(raw)
	result = run_f3_lithology_voxel_decoder(config, device='cpu')
	with pytest.raises(ValueError, match='completed'):
		run_f3_lithology_voxel_decoder(
			config, device='cpu', resume=result.latest_checkpoint
		)


def test_resume_rejects_best_checkpoint_path(tmp_path) -> None:
	raw, _ = _job(tmp_path, 'resume-best')
	config = f3_lithology_voxel_decoder_config_from_mapping(raw)
	partial = run_f3_lithology_voxel_decoder(config, device='cpu', max_steps=1)
	best_path = config.output_dir / 'best.pt'
	shutil.copy2(partial.latest_checkpoint, best_path)

	with pytest.raises(ValueError, match=r'must be latest\.pt'):
		run_f3_lithology_voxel_decoder(config, device='cpu', resume=best_path)


def test_resume_rejects_missing_resolved_config_snapshot(tmp_path) -> None:
	raw, _ = _job(tmp_path, 'resume-missing-config')
	config = f3_lithology_voxel_decoder_config_from_mapping(raw)
	partial = run_f3_lithology_voxel_decoder(config, device='cpu', max_steps=1)
	(config.output_dir / 'resolved_config.json').unlink()

	with pytest.raises(FileNotFoundError, match=r'resolved_config\.json'):
		run_f3_lithology_voxel_decoder(
			config, device='cpu', resume=partial.latest_checkpoint
		)


def test_resume_rejects_architecture_change_before_model_restore(
	tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
	raw, _ = _job(tmp_path, 'resume-architecture')
	config = f3_lithology_voxel_decoder_config_from_mapping(raw)
	partial = run_f3_lithology_voxel_decoder(config, device='cpu', max_steps=1)
	changed_raw = deepcopy(raw)
	changed_raw['decoder']['hidden_channels'] = [3]
	changed_config = f3_lithology_voxel_decoder_config_from_mapping(changed_raw)

	def reject_restore(*_args, **_kwargs):
		raise AssertionError('model restore ran before identity validation')

	monkeypatch.setattr(
		voxel_decoder_runner,
		'restore_voxel_decoder_checkpoint',
		reject_restore,
	)
	with pytest.raises(ValueError, match='decoder architecture'):
		run_f3_lithology_voxel_decoder(
			changed_config, device='cpu', resume=partial.latest_checkpoint
		)


def test_resume_rejects_modified_tile_manifest_snapshot(tmp_path) -> None:
	raw, _ = _job(tmp_path, 'resume-modified-manifest')
	config = f3_lithology_voxel_decoder_config_from_mapping(raw)
	partial = run_f3_lithology_voxel_decoder(config, device='cpu', max_steps=1)
	path = config.output_dir / 'train_tile_manifest.json'
	payload = json.loads(path.read_text(encoding='utf-8'))
	payload['tile_count'] += 1
	path.write_text(json.dumps(payload), encoding='utf-8')

	with pytest.raises(ValueError, match='tile_count'):
		run_f3_lithology_voxel_decoder(
			config, device='cpu', resume=partial.latest_checkpoint
		)


def test_resume_rejects_modified_best_checkpoint_snapshot(tmp_path) -> None:
	raw, _ = _job(tmp_path, 'resume-modified-best')
	config = f3_lithology_voxel_decoder_config_from_mapping(raw)
	partial = run_f3_lithology_voxel_decoder(config, device='cpu', max_steps=4)
	best_path = config.output_dir / 'best.pt'
	best_path.write_bytes(best_path.read_bytes() + b'tampered')

	with pytest.raises(ValueError, match=r'best\.pt'):
		run_f3_lithology_voxel_decoder(
			config, device='cpu', resume=partial.latest_checkpoint
		)

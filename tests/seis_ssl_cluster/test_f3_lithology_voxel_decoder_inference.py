from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from seis_ssl_cluster.config.f3_lithology_voxel_inference import (
	f3_lithology_voxel_inference_config_from_mapping,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.voxel_decoder_inference import (
	VoxelDecoderInferencePlan,
	_load_decoder,
	_write_inference_tiles,
	inspect_f3_lithology_voxel_inference,
	predict_f3_lithology_voxels,
)
from seis_ssl_cluster.f3.lithology.voxel_prediction_artifact import (
	F3VoxelPredictionArrays,
	validate_f3_voxel_prediction_artifact,
)
from seis_ssl_cluster.f3.lithology.voxel_tiles import (
	build_voxel_tile_manifests,
	write_voxel_tile_manifest,
)
from seis_ssl_cluster.models.voxel_decoder import (
	VOXEL_DECODER_NORMALIZATION,
	VOXEL_DECODER_SPEC,
	VOXEL_DECODER_UPSAMPLE_MODE,
	VoxelDecoder3D,
	validate_context_halo_tokens,
)
from seis_ssl_cluster.training.voxel_decoder.checkpoint import (
	save_voxel_decoder_checkpoint,
)
from tests.helpers import run_python_proc


class _LocalVoxelModel(torch.nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.convolution = torch.nn.Conv3d(1, 2, kernel_size=3, padding=1)
		with torch.no_grad():
			weights = torch.arange(54, dtype=torch.float32).reshape(2, 1, 3, 3, 3)
			self.convolution.weight.copy_((weights - 26.5) / 54.0)
			self.convolution.bias.copy_(torch.tensor([-0.25, 0.5]))

	def forward(
		self,
		embeddings: torch.Tensor,
		token_valid_mask: torch.Tensor,
	) -> torch.Tensor:
		return self.convolution(
			embeddings.masked_fill(~token_valid_mask.unsqueeze(1), 0.0)
		)


def test_tiled_core_outputs_match_whole_grid_forward() -> None:
	torch.manual_seed(231)
	embeddings = torch.randn(1, 1, 5, 3, 2)
	valid_tokens = torch.ones((1, 5, 3, 2), dtype=torch.bool)
	model = _LocalVoxelModel().eval()
	with torch.inference_mode():
		whole_probabilities = torch.softmax(model(embeddings, valid_tokens), dim=1)[0]
		whole_confidence, whole_indices = whole_probabilities.max(dim=0)

	shape = (5, 3, 2)
	arrays = F3VoxelPredictionArrays(
		predictions=np.full(shape, -1, dtype=np.int16),
		confidence=np.full(shape, np.nan, dtype=np.float16),
		valid_mask=np.zeros(shape, dtype=np.bool_),
		probabilities=np.full((*shape, 2), np.nan, dtype=np.float16),
	)
	plan = VoxelDecoderInferencePlan(
		embeddings=Path('embeddings.npy'),
		valid_tokens=Path('valid_tokens.npy'),
		embedding_metadata=Path('embedding_metadata.json'),
		checkpoint=Path('best.pt'),
		resolved_decoder_config=Path('resolved_config.json'),
		train_tile_manifest=Path('train_tile_manifest.json'),
		validation_tile_manifest=Path('validation_tile_manifest.json'),
		volume_shape_xyz=shape,
		token_grid_shape_xyz=shape,
		patch_size_xyz=(1, 1, 1),
		embedding_dim=1,
		class_ids=(3, 7),
		classes=(),
		decoder_spec={},
		checkpoint_payload={},
		artifact_identities={},
	)

	coverage = _write_inference_tiles(
		model,
		embeddings=embeddings[0].movedim(0, -1).numpy(),
		valid_tokens=valid_tokens[0].numpy(),
		arrays=arrays,
		plan=plan,
		core_size_tokens=(2, 2, 1),
		context_halo_tokens=(1, 1, 1),
		device=torch.device('cpu'),
	)

	expected_predictions = np.asarray((3, 7), dtype=np.int16)[whole_indices.numpy()]
	np.testing.assert_array_equal(arrays.predictions, expected_predictions)
	np.testing.assert_array_equal(arrays.valid_mask, np.ones(shape, dtype=np.bool_))
	np.testing.assert_allclose(
		arrays.confidence,
		whole_confidence.numpy().astype(np.float16),
		rtol=0,
		atol=0,
	)
	np.testing.assert_allclose(
		arrays.probabilities,
		whole_probabilities.movedim(0, -1).numpy().astype(np.float16),
		rtol=0,
		atol=0,
	)
	assert coverage['exact_once'] is True
	assert coverage['written_voxel_count'] == np.prod(shape)


def test_production_decoder_tiled_outputs_match_whole_grid_forward() -> None:
	torch.manual_seed(232)
	token_shape = (5, 3, 2)
	patch_size = (8, 8, 8)
	volume_shape = tuple(
		tokens * patch for tokens, patch in zip(token_shape, patch_size, strict=True)
	)
	embedding_dim = 4
	class_count = 3
	embeddings = torch.randn(1, embedding_dim, *token_shape)
	valid_tokens = torch.ones((1, *token_shape), dtype=torch.bool)
	model = VoxelDecoder3D(
		embedding_dim=embedding_dim,
		class_count=class_count,
		hidden_channels=(8, 4, 4),
		upsample_factors=((2, 2, 2),) * 3,
		patch_size_xyz=patch_size,
	).eval()
	with torch.inference_mode():
		whole_probabilities = torch.softmax(model(embeddings, valid_tokens), dim=1)[0]
		whole_confidence, whole_indices = whole_probabilities.max(dim=0)

	arrays = F3VoxelPredictionArrays(
		predictions=np.full(volume_shape, -1, dtype=np.int16),
		confidence=np.full(volume_shape, np.nan, dtype=np.float16),
		valid_mask=np.zeros(volume_shape, dtype=np.bool_),
		probabilities=np.full((*volume_shape, class_count), np.nan, dtype=np.float16),
	)
	plan = VoxelDecoderInferencePlan(
		embeddings=Path('embeddings.npy'),
		valid_tokens=Path('valid_tokens.npy'),
		embedding_metadata=Path('embedding_metadata.json'),
		checkpoint=Path('best.pt'),
		resolved_decoder_config=Path('resolved_config.json'),
		train_tile_manifest=Path('train_tile_manifest.json'),
		validation_tile_manifest=Path('validation_tile_manifest.json'),
		volume_shape_xyz=volume_shape,
		token_grid_shape_xyz=token_shape,
		patch_size_xyz=patch_size,
		embedding_dim=embedding_dim,
		class_ids=(3, 7, 9),
		classes=(),
		decoder_spec={},
		checkpoint_payload={},
		artifact_identities={},
	)

	_write_inference_tiles(
		model,
		embeddings=embeddings[0].movedim(0, -1).numpy(),
		valid_tokens=valid_tokens[0].numpy(),
		arrays=arrays,
		plan=plan,
		core_size_tokens=(2, 2, 1),
		context_halo_tokens=(1, 1, 1),
		device=torch.device('cpu'),
	)

	expected_predictions = np.asarray((3, 7, 9), dtype=np.int16)[whole_indices.numpy()]
	np.testing.assert_array_equal(arrays.predictions, expected_predictions)
	np.testing.assert_array_equal(
		arrays.confidence, whole_confidence.numpy().astype(np.float16)
	)
	np.testing.assert_array_equal(
		arrays.probabilities,
		whole_probabilities.movedim(0, -1).numpy().astype(np.float16),
	)


def test_inference_rejects_halo_smaller_than_decoder_receptive_field() -> None:
	with pytest.raises(ValueError, match='decoder receptive field'):
		validate_context_halo_tokens(
			context_halo_tokens=(1, 0, 0),
			core_size_tokens=(2, 1, 1),
			token_grid_shape_xyz=(5, 1, 1),
			upsample_factors=((1, 1, 1), (1, 1, 1)),
		)


def test_inference_requires_best_checkpoint_path(tmp_path: Path) -> None:
	raw, _ = _write_job(tmp_path)
	raw['decoder']['checkpoint'] = str(
		Path(raw['decoder']['checkpoint']).with_name('latest.pt')
	)
	config = f3_lithology_voxel_inference_config_from_mapping(raw)

	with pytest.raises(ValueError, match=r'best\.pt'):
		inspect_f3_lithology_voxel_inference(config)


def test_cpu_chunked_inference_crops_volume_and_masks_invalid_tokens(
	tmp_path: Path,
) -> None:
	raw, embedding_path = _write_job(tmp_path)
	config = f3_lithology_voxel_inference_config_from_mapping(raw)
	before = embedding_path.read_bytes()
	plan = inspect_f3_lithology_voxel_inference(config)
	model_architecture = _load_decoder(plan, device=torch.device('cpu')).architecture

	result = predict_f3_lithology_voxels(config, device='cpu')
	artifact = validate_f3_voxel_prediction_artifact(result.output_dir)

	assert result.tile_count == 2
	assert artifact.arrays.predictions.shape == (5, 1, 1)
	assert artifact.arrays.valid_mask[:, 0, 0].tolist() == [
		True,
		True,
		False,
		False,
		True,
	]
	assert np.all(artifact.arrays.predictions[~artifact.arrays.valid_mask] == -1)
	assert np.all(np.isnan(artifact.arrays.confidence[~artifact.arrays.valid_mask]))
	assert artifact.arrays.probabilities is not None
	assert artifact.metadata['prediction_kind'] == 'frozen_embedding_decoder'
	assert artifact.metadata['decoder_architecture'] == model_architecture
	assert artifact.metadata['training_sampling'] == {
		'sampling_mode': 'uniform_tiles_with_replacement',
		'steps_per_epoch': 3,
		'train_seed': 11,
		'train_tile_manifest_sha256': plan.checkpoint_payload['tile_manifest_hashes'][
			'train'
		],
		'validation_tile_manifest_sha256': plan.checkpoint_payload[
			'tile_manifest_hashes'
		]['validation'],
	}
	assert artifact.metadata['coverage']['exact_once'] is True
	assert artifact.metadata['coverage']['written_voxel_count'] == 5
	assert embedding_path.read_bytes() == before
	assert model_architecture == plan.decoder_spec


@pytest.mark.parametrize(
	('field', 'value'),
	[
		('spec', 'frozen_embedding_decoder_v1'),
		('upsample_mode', 'trilinear'),
		('normalization', 'batch_norm'),
	],
)
def test_inspection_rejects_tampered_decoder_implementation_identity(
	tmp_path: Path, field: str, value: str
) -> None:
	raw, _ = _write_job(tmp_path)
	checkpoint_path = Path(raw['decoder']['checkpoint'])
	resolved_path = checkpoint_path.parent / 'resolved_config.json'
	resolved = json.loads(resolved_path.read_text(encoding='utf-8'))
	resolved['decoder'][field] = value
	resolved_path.write_text(json.dumps(resolved), encoding='utf-8')
	config = f3_lithology_voxel_inference_config_from_mapping(raw)

	with pytest.raises(ValueError, match=field):
		inspect_f3_lithology_voxel_inference(config)


def test_inspection_rejects_checkpoint_and_resolved_architecture_mismatch(
	tmp_path: Path,
) -> None:
	raw, _ = _write_job(tmp_path)
	checkpoint_path = Path(raw['decoder']['checkpoint'])
	resolved_path = checkpoint_path.parent / 'resolved_config.json'
	resolved = json.loads(resolved_path.read_text(encoding='utf-8'))
	resolved['decoder']['hidden_channels'] = [5]
	resolved_path.write_text(json.dumps(resolved), encoding='utf-8')
	config = f3_lithology_voxel_inference_config_from_mapping(raw)

	with pytest.raises(ValueError, match='architecture identity mismatch'):
		inspect_f3_lithology_voxel_inference(config)


def test_inspection_rejects_checkpoint_embedding_identity_mismatch(
	tmp_path: Path,
) -> None:
	raw, embedding_path = _write_job(tmp_path)
	embeddings = np.load(embedding_path)
	embeddings[0, 0, 0, 0] += 1.0
	np.save(embedding_path, embeddings)
	config = f3_lithology_voxel_inference_config_from_mapping(raw)

	with pytest.raises(ValueError, match='embeddings hash'):
		inspect_f3_lithology_voxel_inference(config, verify_array_hashes=True)


def test_inspection_rejects_class_palette_mismatch(tmp_path: Path) -> None:
	raw, _ = _write_job(tmp_path)
	class_info_path = Path(raw['labels']['class_info'])
	class_info = json.loads(class_info_path.read_text(encoding='utf-8'))
	class_info['0']['color'] = [9, 9, 9]
	class_info_path.write_text(json.dumps(class_info), encoding='utf-8')
	config = f3_lithology_voxel_inference_config_from_mapping(raw)

	with pytest.raises(ValueError, match='class_info does not match'):
		inspect_f3_lithology_voxel_inference(config)


def test_dry_run_inspection_does_not_hash_array_contents(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	raw, _ = _write_job(tmp_path)
	config = f3_lithology_voxel_inference_config_from_mapping(raw)
	actual_file_sha256 = file_sha256

	def reject_array_hash(path: str | Path) -> str:
		if Path(path).suffix == '.npy':
			raise AssertionError('dry-run hashed an array artifact')
		return actual_file_sha256(path)

	monkeypatch.setattr(
		'seis_ssl_cluster.f3.lithology.voxel_decoder_inference.file_sha256',
		reject_array_hash,
	)
	plan = inspect_f3_lithology_voxel_inference(config)

	assert plan.token_grid_shape_xyz == (3, 1, 1)


def test_inference_inspection_rejects_incompatible_decoder_stages(
	tmp_path: Path,
) -> None:
	raw, _ = _write_job(tmp_path)
	checkpoint_path = Path(raw['decoder']['checkpoint'])
	payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
	payload['resolved_config']['decoder']['hidden_channels'] = [4, 4]
	payload['decoder_architecture']['hidden_channels'] = [4, 4]
	torch.save(payload, checkpoint_path)
	resolved_path = checkpoint_path.parent / 'resolved_config.json'
	resolved = json.loads(resolved_path.read_text(encoding='utf-8'))
	resolved['decoder']['hidden_channels'] = [4, 4]
	resolved_path.write_text(json.dumps(resolved), encoding='utf-8')
	config = f3_lithology_voxel_inference_config_from_mapping(raw)

	with pytest.raises(ValueError, match='must have the same length'):
		inspect_f3_lithology_voxel_inference(config)


def test_cli_dry_run_does_not_write(tmp_path: Path) -> None:
	raw, _ = _write_job(tmp_path)
	config_path = tmp_path / 'inference.json'
	config_path.write_text(json.dumps(raw), encoding='utf-8')
	config = f3_lithology_voxel_inference_config_from_mapping(raw)

	completed = run_python_proc(
		Path('proc/seis_ssl_cluster/predict_f3_lithology_voxels.py'),
		'--config',
		config_path,
		'--dry-run',
	)

	assert completed.returncode == 0, completed.stderr
	assert 'execution: dry-run; voxel decoder inference skipped' in completed.stdout
	assert f'decoder.spec: {VOXEL_DECODER_SPEC}' in completed.stdout
	assert f'decoder.upsample_mode: {VOXEL_DECODER_UPSAMPLE_MODE}' in completed.stdout
	assert f'decoder.normalization: {VOXEL_DECODER_NORMALIZATION}' in completed.stdout
	assert not config.output_dir.exists()


def test_output_collision_is_rejected(tmp_path: Path) -> None:
	raw, _ = _write_job(tmp_path)
	config = f3_lithology_voxel_inference_config_from_mapping(raw)
	config.output_dir.mkdir()
	(config.output_dir / 'partial.txt').write_text('partial', encoding='utf-8')

	with pytest.raises(FileExistsError, match='refusing to overwrite'):
		predict_f3_lithology_voxels(config, device='cpu')


def _write_job(tmp_path: Path) -> tuple[dict[str, object], Path]:
	root = tmp_path / 'artifacts'
	embedding_dir = root / 'embeddings'
	decoder_dir = root / 'decoder'
	voxel_dir = root / 'voxel'
	for path in (embedding_dir, decoder_dir, voxel_dir):
		path.mkdir(parents=True)
	embeddings = np.arange(12, dtype=np.float32).reshape(3, 1, 1, 4)
	valid_tokens = np.asarray([True, False, True]).reshape(3, 1, 1)
	embedding_path = embedding_dir / 'tiny.embeddings.npy'
	valid_path = embedding_dir / 'tiny.valid_tokens.npy'
	embedding_metadata_path = embedding_dir / 'tiny.embedding_metadata.json'
	label_path = root / 'labels.npy'
	split_path = voxel_dir / 'supervision_split_grid.npy'
	voxel_metadata_path = voxel_dir / 'voxel_dataset_metadata.json'
	np.save(embedding_path, embeddings)
	np.save(valid_path, valid_tokens)
	np.save(label_path, np.asarray([0, 1, 0, 1, 0], dtype=np.int16).reshape(5, 1, 1))
	np.save(split_path, np.asarray([1, 1, 1, 2, 2], dtype=np.uint8).reshape(5, 1, 1))
	embedding_metadata = {
		'volume_shape_xyz': [5, 1, 1],
		'patch_size': [2, 1, 1],
		'token_grid_shape': [3, 1, 1],
		'embedding_dim': 4,
		'checkpoint_path': str(root / 'encoder-v1' / 'best.pt'),
		'preprocessing': {'kind': 'tiny'},
		'zero_mask': {'enabled': True},
	}
	embedding_metadata_path.write_text(json.dumps(embedding_metadata), encoding='utf-8')
	voxel_metadata = {
		'dataset': {'name': 'tiny', 'version': 'v1'},
		'classes': [
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
		],
		'reference_embedding': {
			'path': str(embedding_metadata_path),
			'sha256': file_sha256(embedding_metadata_path),
			'metadata': embedding_metadata,
		},
		'reference_valid_tokens': {
			'path': str(valid_path),
			'sha256': file_sha256(valid_path),
		},
		'label_volume': {'path': str(label_path), 'sha256': file_sha256(label_path)},
	}
	voxel_metadata_path.write_text(json.dumps(voxel_metadata), encoding='utf-8')
	class_info = root / 'class_info.json'
	class_info.write_text(
		json.dumps(
			{
				'0': {'name': 'zero', 'color': [0, 0, 0]},
				'1': {'name': 'one', 'color': [1, 1, 1]},
			}
		),
		encoding='utf-8',
	)
	resolved_config = {
		'paths': {'artifact_root': str(root), 'f3_root': str(tmp_path / 'f3')},
		'dataset': {'name': 'tiny', 'version': 'v1'},
		'model': {'tag': 'encoder-v1', 'freeze_encoder': True},
		'embeddings': {'input_dir': str(embedding_dir)},
		'voxel_dataset': {'input_dir': str(voxel_dir)},
		'decoder': {
			'spec': VOXEL_DECODER_SPEC,
			'embedding_dim': 4,
			'class_count': 2,
			'hidden_channels': [4],
			'upsample_factors': [[2, 1, 1]],
			'upsample_mode': VOXEL_DECODER_UPSAMPLE_MODE,
			'normalization': VOXEL_DECODER_NORMALIZATION,
		},
		'tiles': {'core_size_tokens': [2, 1, 1], 'context_halo_tokens': [1, 0, 0]},
		'train': {
			'epochs': 1,
			'sampling_mode': 'uniform_tiles_with_replacement',
			'steps_per_epoch': 3,
			'seed': 11,
		},
		'outputs': {'output_dir': str(decoder_dir)},
	}
	(decoder_dir / 'resolved_config.json').write_text(
		json.dumps(resolved_config), encoding='utf-8'
	)
	labels = np.load(label_path)
	split = np.load(split_path)
	manifests = build_voxel_tile_manifests(
		split,
		labels,
		patch_size_xyz=(2, 1, 1),
		core_size_tokens=(2, 1, 1),
		context_halo_tokens=(1, 0, 0),
		class_ids=(0, 1),
		canonical_valid_tokens=valid_tokens,
	)
	for name, manifest in manifests.items():
		write_voxel_tile_manifest(decoder_dir / f'{name}_tile_manifest.json', manifest)
	identities = {
		'name': 'f3_voxel_decoder_sources',
		'embeddings': _identity(embedding_path),
		'embedding_metadata': _identity(embedding_metadata_path),
		'valid_tokens': _identity(valid_path),
		'voxel_dataset_metadata': _identity(voxel_metadata_path),
		'voxel_split_grid': _identity(split_path),
		'label_volume': _identity(label_path),
	}
	model = VoxelDecoder3D(
		embedding_dim=4,
		class_count=2,
		hidden_channels=(4,),
		upsample_factors=((2, 1, 1),),
		patch_size_xyz=(2, 1, 1),
	)
	optimizer = torch.optim.AdamW(model.parameters())
	save_voxel_decoder_checkpoint(
		decoder_dir / 'best.pt',
		model=model,
		optimizer=optimizer,
		epoch=0,
		global_step=1,
		resolved_config=resolved_config,
		class_weights=(1.0, 1.0),
		artifact_identities=identities,
		tile_manifest_hashes={
			name: item.identity_sha256 for name, item in manifests.items()
		},
		best_selection_state=None,
		training_history=[],
		current_metrics={},
		checkpoint_kind='completed',
	)
	return (
		{
			'paths': resolved_config['paths'],
			'dataset': resolved_config['dataset'],
			'model': resolved_config['model'],
			'labels': {'class_info': str(class_info)},
			'embeddings': {'input_dir': str(embedding_dir)},
			'decoder': {'checkpoint': str(decoder_dir / 'best.pt')},
			'tiles': resolved_config['tiles'],
			'inference': {'write_probabilities': True, 'overwrite': False},
			'outputs': {'output_dir': str(root / 'predictions')},
		},
		embedding_path,
	)


def _identity(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}

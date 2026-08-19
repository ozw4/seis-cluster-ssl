"""Focused routing and provenance contracts for XY consensus hard labels."""

from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from seis_ssl_cluster.config import pretraining
from seis_ssl_cluster.data import target_providers
from seis_ssl_cluster.data.target_providers import (
	MultiHeadStratPseudoTargetProvider,
	TargetProviderContext,
	load_strat_multi_head_xy_neighbor_consensus_target_manifest_adapter,
)
from seis_ssl_cluster.embedding.extractor import _stratigraphy_pretext_metadata
from seis_ssl_cluster.stratigraphy.prototypes import (
	MultiResolutionOrderedPrototypeHeads,
)
from seis_ssl_cluster.training import strat_hmm_checkpoint
from seis_ssl_cluster.training.checkpoint import capture_rng_state, load_checkpoint
from seis_ssl_cluster.training.strat_hmm import epoch, runner
from seis_ssl_cluster.training.strat_hmm.resume import (
	restore_strat_hmm_training_checkpoint,
)
from seis_ssl_cluster.training.strat_hmm_checkpoint import (
	save_strat_hmm_checkpoint,
	validate_stratigraphy_checkpoint_payload,
)
from tests.seis_ssl_cluster.test_config_strat_hmm_pretext import _minimal_config


def test_xy_neighbor_consensus_adapter_reuses_hard_provider_without_posterior(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""The source-hard-only export stays lazy and uses the existing hard fields."""
	payload, paths = _consensus_payload(tmp_path)
	calls: list[object] = []

	def load_manifest(*_args: object, **kwargs: object) -> dict[str, object]:
		calls.append(kwargs.get('validate_array_semantics'))
		return payload

	monkeypatch.setattr(
		target_providers,
		'load_multi_head_xy_neighbor_consensus_target_manifest',
		load_manifest,
	)

	adapted = load_strat_multi_head_xy_neighbor_consensus_target_manifest_adapter(
		tmp_path / 'consensus.json'
	)
	assert calls == [False]
	assert not hasattr(adapted, 'source_posterior_manifest')
	provider = MultiHeadStratPseudoTargetProvider(adapted.target_manifest)
	assert provider._pseudo_target_arrays == {}  # noqa: SLF001
	sample: dict[str, object] = {'coords': {}}
	provider.add_targets(
		sample,
		TargetProviderContext(
			manifest=type('Manifest', (), {'survey_id': 'survey'})(),
			crop_request=type('Request', (), {})(),
			patch_size_xyz=(1, 1, 1),
			token_start_xyz=(0, 0, 0),
			token_size_xyz=(2, 2, 2),
			token_valid_mask=np.ones((2, 2, 2), dtype=bool),
		),
	)
	targets = sample['strat_multi_targets']
	assert isinstance(targets, dict)
	assert targets['k6']['labels'][0, 0, 0] == -1
	assert targets['k6']['valid_mask'][0, 0, 0] is np.False_
	assert all(paths[k]['labels'].is_file() for k in (6, 8, 10))


def test_xy_neighbor_consensus_config_uses_its_strict_manifest_loader(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Config resolution cannot substitute a generic hard or posterior loader."""
	calls: list[tuple[str, object]] = []
	manifest = {'head_ks': [6, 8, 10]}

	def consensus_loader(_path: str, **kwargs: object) -> dict[str, object]:
		calls.append(
			('xy_neighbor_consensus_targets', kwargs.get('validate_array_semantics'))
		)
		return manifest

	module = SimpleNamespace(
		load_multi_head_xy_neighbor_consensus_target_manifest=consensus_loader,
	)
	monkeypatch.setattr(pretraining.importlib, 'import_module', lambda _name: module)

	resolved = pretraining._validate_strat_hmm_multi_head_manifest(  # noqa: SLF001
		{'manifest': 'targets.json'},
		{'ks': [6, 8, 10]},
		target_representation='xy_neighbor_consensus_hard_labels_v1',
	)

	assert resolved is manifest
	assert calls == [('xy_neighbor_consensus_targets', False)]


def test_xy_neighbor_consensus_config_rejects_legacy_posterior_identity(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""This successor cannot silently retain M5-U or M5-LS provenance."""
	manifest_path = tmp_path / 'consensus-targets.json'
	manifest_path.write_text('{"consensus": true}\n', encoding='utf-8')
	target_manifest = _identity_manifest()
	monkeypatch.setattr(
		pretraining.importlib,
		'import_module',
		lambda _name: SimpleNamespace(
			load_multi_head_xy_neighbor_consensus_target_manifest=(
				lambda _path, **_kwargs: target_manifest
			)
		),
	)
	config = _xy_neighbor_consensus_config(manifest_path, target_manifest)

	resolved = pretraining.resolve_strat_hmm_pretext_config(config)
	identity = resolved['identity']['scientific_identity']
	assert identity['source_hard_manifest_sha256'] == 'a' * 64
	assert 'source_posterior_manifest_sha256' not in identity
	assert 'lateral_smoothing' not in identity

	invalid = deepcopy(config)
	invalid_identity = invalid['identity']['scientific_identity']
	invalid_identity['source_posterior_manifest_sha256'] = 'b' * 64
	with pytest.raises(ValueError, match='source_posterior_manifest_sha256'):
		pretraining.resolve_strat_hmm_pretext_config(invalid)

	invalid = deepcopy(config)
	invalid['pseudo_targets']['min_confidence'] = 0.1
	with pytest.raises(ValueError, match='min_confidence'):
		pretraining.resolve_strat_hmm_pretext_config(invalid)


def test_xy_neighbor_consensus_runner_uses_adapter_and_hard_dataset(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	"""Consensus labels use hard collation/loss, never the posterior route."""
	dispatch: list[object] = []

	class StopAfterDataloaderError(Exception):
		"""Stop the runner immediately after its dataset routing decision."""

	def hard_dataset(
		_manifests: object, target_manifest: object, **kwargs: object
	) -> list[object]:
		dispatch.append(('dataset', target_manifest, kwargs['min_confidence']))
		return []

	def hard_dataloader(_dataset: object, **_kwargs: object) -> object:
		dispatch.append('dataloader')
		raise StopAfterDataloaderError

	monkeypatch.setattr(runner, 'read_manifest_json', lambda _path: [])
	monkeypatch.setattr(
		runner,
		'load_strat_multi_head_xy_neighbor_consensus_target_manifest_adapter',
		lambda _path: SimpleNamespace(target_manifest='adapted-consensus-manifest'),
	)
	monkeypatch.setattr(runner, 'NopimsStratMultiHeadTargetDataset', hard_dataset)
	monkeypatch.setattr(
		runner, 'build_strat_multi_head_target_dataloader', hard_dataloader
	)
	monkeypatch.setattr(
		runner,
		'NopimsStratMultiHeadPosteriorDataset',
		lambda *_args, **_kwargs: pytest.fail(
			'XY consensus targets must not use posterior data'
		),
	)
	monkeypatch.setattr(runner, '_strat_hmm_control_identity', lambda _config: None)
	monkeypatch.setattr(runner, '_snapshot_run_inputs', lambda **_kwargs: None)
	monkeypatch.setattr(runner, 'prepare_run_directory', lambda **_kwargs: None)

	with pytest.raises(StopAfterDataloaderError):
		runner.run_strat_hmm_pretext_training(
			{
				'paths': {'output_root': str(tmp_path)},
				'manifests': {'train': str(tmp_path / 'train.json')},
				'head': {'spec': 'multi_resolution_ordered_prototypes_v1'},
				'pseudo_targets': {
					'manifest': str(tmp_path / 'consensus.json'),
					'target_representation': 'xy_neighbor_consensus_hard_labels_v1',
					'min_confidence': 0.0,
				},
				'data': {'local_crop_size': [2, 2, 2]},
				'model': {'patch_size': [1, 1, 1]},
				'zero_mask': {},
				'loss': {},
				'train': {
					'device': 'cpu',
					'seed': 1,
					'samples_per_epoch': 1,
				},
			}
		)

	assert dispatch == [('dataset', 'adapted-consensus-manifest', 0.0), 'dataloader']


def test_xy_neighbor_consensus_epoch_uses_existing_hard_loss(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Only posterior supervision is allowed to select the soft loss path."""
	calls: list[str] = []

	class Student(torch.nn.Module):
		def __init__(self) -> None:
			super().__init__()
			self.scale = torch.nn.Parameter(torch.tensor(1.0))

		def encode_tokens(
			self, x: torch.Tensor, *, valid_mask: torch.Tensor
		) -> dict[str, torch.Tensor]:
			del valid_mask
			return {'tokens': x * self.scale}

	def hard_loss(**kwargs: object) -> dict[str, torch.Tensor]:
		calls.append('hard')
		encoded = kwargs['encoded']
		assert isinstance(encoded, dict)
		return {'loss': encoded['tokens'].sum()}

	monkeypatch.setattr(epoch, 'compute_strat_hmm_multi_head_losses', hard_loss)
	monkeypatch.setattr(
		epoch,
		'compute_strat_hmm_multi_head_posterior_losses',
		lambda **_kwargs: pytest.fail('XY consensus must not use posterior loss'),
	)
	student = Student()
	heads = torch.nn.Linear(1, 1)

	epoch.train_strat_hmm_multi_head_one_epoch(
		student=student,
		heads=heads,  # type: ignore[arg-type]
		dataloader=[
			{
				'x': torch.ones((1,)),
				'local_valid_mask': torch.ones((1,), dtype=torch.bool),
			}
		],
		optimizer=torch.optim.AdamW([*student.parameters(), *heads.parameters()]),
		device=torch.device('cpu'),
		epoch=1,
		loss_config={},
		pseudo_target_config={'min_confidence': 0.0},
		target_representation='xy_neighbor_consensus_hard_labels_v1',
	)

	assert calls == ['hard']


def test_xy_neighbor_consensus_checkpoint_has_schema_five_and_rejects_mixing(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""Schema v5 binds only source-hard consensus provenance on resume."""
	manifest_path = tmp_path / 'consensus-targets.json'
	manifest_path.write_text('{"consensus": true}\n', encoding='utf-8')
	target_manifest = _identity_manifest()
	monkeypatch.setattr(
		strat_hmm_checkpoint,
		'load_multi_head_xy_neighbor_consensus_target_manifest',
		lambda path, *, validate_array_semantics: (
			_assert_consensus_manifest_load(
				path,
				manifest_path,
				validate_array_semantics=validate_array_semantics,
			)
			or target_manifest
		),
	)
	config = _xy_neighbor_consensus_checkpoint_config(
		manifest_path,
		target_manifest,
	)
	student, heads, optimizer = _new_multi_head_components()
	checkpoint_path = _save_consensus_checkpoint(
		tmp_path / 'consensus.pt',
		config=config,
		student=student,
		heads=heads,
		optimizer=optimizer,
	)
	payload = load_checkpoint(checkpoint_path, map_location='cpu')
	identity = payload['stratigraphy_checkpoint']
	assert isinstance(identity, dict)
	assert identity['schema_version'] == 5
	assert identity['target_representation'] == 'xy_neighbor_consensus_hard_labels_v1'
	assert 'source_posterior_manifest_sha256' not in identity
	assert 'lateral_smoothing' not in identity
	validate_stratigraphy_checkpoint_payload(payload, expected_config=config)

	metadata = _stratigraphy_pretext_metadata(payload)
	assert metadata is not None
	assert metadata['xy_neighbor_consensus_target_manifest_path'] == str(manifest_path)
	assert (
		metadata['per_head_xy_neighbor_consensus_target_sha256']
		== identity['per_head_xy_neighbor_consensus_targets']
	)

	for representation in (
		None,
		'ordered_path_state_posterior_v1',
		'lateral_mean_field_hard_labels_v1',
	):
		incompatible = deepcopy(config)
		pseudo_targets = incompatible['pseudo_targets']
		assert isinstance(pseudo_targets, dict)
		if representation is None:
			pseudo_targets.pop('target_representation')
		else:
			pseudo_targets['target_representation'] = representation
		other_student, other_heads, other_optimizer = _new_multi_head_components()
		with pytest.raises(ValueError, match='target_representation'):
			restore_strat_hmm_training_checkpoint(
				payload=payload,
				student=other_student,
				head=other_heads,
				optimizer=other_optimizer,
				scaler=None,
				amp_enabled=False,
				config=incompatible,
			)

	legacy_payload = deepcopy(payload)
	legacy_config = legacy_payload['stratigraphy_config']
	assert isinstance(legacy_config, dict)
	legacy_identity = legacy_config['identity']['scientific_identity']
	legacy_identity['source_posterior_manifest_sha256'] = 'b' * 64
	with pytest.raises(ValueError, match='source_posterior_manifest_sha256'):
		validate_stratigraphy_checkpoint_payload(legacy_payload)

	legacy_payload = deepcopy(payload)
	legacy_checkpoint_identity = legacy_payload['stratigraphy_checkpoint']
	assert isinstance(legacy_checkpoint_identity, dict)
	legacy_checkpoint_identity['source_posterior_manifest'] = {
		'path': 'posterior.json',
		'sha256': 'b' * 64,
	}
	with pytest.raises(ValueError, match='source_posterior_manifest'):
		validate_stratigraphy_checkpoint_payload(legacy_payload)


def _consensus_payload(
	tmp_path: Path,
) -> tuple[dict[str, object], dict[int, dict[str, Path]]]:
	paths: dict[int, dict[str, Path]] = {}
	heads: dict[str, object] = {}
	for k in (6, 8, 10):
		root = tmp_path / f'k{k}'
		root.mkdir()
		arrays = {
			'labels': np.full((2, 2, 2), k - 1, dtype=np.int32),
			'confidence': np.ones((2, 2, 2), dtype=np.float32),
			'valid_tokens': np.ones((2, 2, 2), dtype=bool),
		}
		arrays['labels'][0, 0, 0] = -17
		arrays['confidence'][0, 0, 0] = 0.0
		arrays['valid_tokens'][0, 0, 0] = False
		paths[k] = {}
		for name, value in arrays.items():
			path = root / f'{name}.npy'
			np.save(path, value, allow_pickle=False)
			paths[k][name] = path
		metadata = root / 'metadata.json'
		metadata.write_text(
			json.dumps(
				{
					'artifact_type': 'strat_hmm_pseudo_target',
					'schema_version': 1,
					'k': k,
					'survey_id': 'survey',
				}
			),
			encoding='utf-8',
		)
		paths[k]['metadata'] = metadata
		heads[str(k)] = {
			'surveys': {
				'survey': {
					name: {
						'path': str(path),
						'sha256': sha256(path.read_bytes()).hexdigest(),
					}
					for name, path in paths[k].items()
				}
			}
		}
	return (
		{
			'artifact_type': (
				'strat_hmm_multi_head_xy_neighbor_consensus_target_manifest'
			),
			'schema_version': 1,
			'target_representation': 'xy_neighbor_consensus_hard_labels_v1',
			'target_semantics': 'xy_neighbor_consensus_hard_label_smoothing_v1',
			'head_ks': [6, 8, 10],
			'source_hard_manifest': {'path': 'hard.json', 'sha256': 'a' * 64},
			'smoothing': {
				'neighborhood': 'same_z_xy_four_neighbors',
				'application': 'single_pass_synchronous_source_labels',
			},
			'heads': heads,
		},
		paths,
	)


def _identity_manifest() -> dict[str, object]:
	head_hashes = {
		str(k): {
			'survey': {
				name: f'{k:02d}{index:062d}'
				for index, name in enumerate(
					('labels', 'confidence', 'valid_tokens', 'metadata')
				)
			}
		}
		for k in (6, 8, 10)
	}
	return {
		'head_ks': [6, 8, 10],
		'target_representation': 'xy_neighbor_consensus_hard_labels_v1',
		'target_semantics': 'xy_neighbor_consensus_hard_label_smoothing_v1',
		'source_hard_manifest': {'path': 'hard.json', 'sha256': 'a' * 64},
		'smoothing': {
			'neighborhood': 'same_z_xy_four_neighbors',
			'neighbor_order': ('x_minus', 'x_plus', 'y_minus', 'y_plus'),
			'four_valid_neighbors_minimum_agreement': 3,
			'three_valid_neighbors_minimum_agreement': 3,
			'fewer_than_three_valid_neighbors': 'unchanged',
			'tied_or_nonunique_consensus': 'unchanged',
			'center_matching_consensus': 'unchanged',
			'temporal_guard': 'internal_valid_token_source_label_bounds',
			'application': 'single_pass_synchronous_source_labels',
		},
		'heads': {
			str(k): {
				'surveys': {
					'survey': {
						name: {'sha256': value}
						for name, value in head_hashes[str(k)]['survey'].items()
					}
				}
			}
			for k in (6, 8, 10)
		},
	}


def _xy_neighbor_consensus_config(
	manifest_path: Path,
	target_manifest: dict[str, object],
) -> dict[str, object]:
	config = _minimal_config(manifest_path.parent)
	config['pseudo_targets'] = {
		'manifest': str(manifest_path),
		'min_confidence': 0.0,
		'target_representation': 'xy_neighbor_consensus_hard_labels_v1',
	}
	config['head'] = {
		'spec': 'multi_resolution_ordered_prototypes_v1',
		'ks': [6, 8, 10],
		'projection_dim': 128,
		'temperature': 0.1,
		'normalize': True,
	}
	config['loss'] = {
		'prototype_weight': 1.0,
		'usage_weight': 0.005,
		'entropy_floor': None,
		'consistency_weight': 0.0,
		'consistency_beta': 0.1,
		'distillation_weight': 0.2,
	}
	config['identity'] = {
		'model_tag': 'strat_hmm_pretext_mh_k6810_xycons1_nocons_topblock1_distill_v1',
		'scientific_identity': {
			'experiment_role': 'multi_head_ordered_xy_neighbor_consensus_hard_pretext',
			'variant': 'xycons1_nocons',
			'target_representation': 'xy_neighbor_consensus_hard_labels_v1',
			'target_semantics': 'xy_neighbor_consensus_hard_label_smoothing_v1',
			'xy_neighbor_consensus_target_manifest_sha256': sha256(
				manifest_path.read_bytes()
			).hexdigest(),
			'xy_neighbor_consensus_target_head_hashes': (
				pretraining._multi_head_target_hashes(target_manifest)  # noqa: SLF001
			),
			'source_hard_manifest_sha256': 'a' * 64,
			'xy_neighbor_consensus_smoothing': target_manifest['smoothing'],
			'supervised_loss': 'structured_hmm_hard_categorical_v1',
			'head_spec': 'multi_resolution_ordered_prototypes_v1',
			'head_ks': [6, 8, 10],
			'consistency_policy': 'disabled_for_xy_neighbor_consensus_v1',
			'consistency_weight': 0.0,
		},
	}
	return config


def _xy_neighbor_consensus_checkpoint_config(
	manifest_path: Path,
	target_manifest: dict[str, object],
) -> dict[str, object]:
	teacher_checkpoint = manifest_path.parent / 'teacher.pt'
	student_checkpoint = manifest_path.parent / 'student.pt'
	teacher_checkpoint.write_bytes(b'teacher')
	student_checkpoint.write_bytes(b'student')
	head_hashes = pretraining._multi_head_target_hashes(target_manifest)  # noqa: SLF001
	return {
		'stage': 'train_strat_hmm_pretext',
		'paths': {'output_root': str(manifest_path.parent / 'consensus-run')},
		'pseudo_targets': {
			'manifest': str(manifest_path),
			'target_representation': 'xy_neighbor_consensus_hard_labels_v1',
		},
		'teacher': {'checkpoint': str(teacher_checkpoint)},
		'student': {
			'init_checkpoint': str(student_checkpoint),
			'unfreeze_top_blocks': 1,
		},
		'head': {
			'spec': 'multi_resolution_ordered_prototypes_v1',
			'ks': [6, 8, 10],
			'projection_dim': 2,
			'temperature': 0.1,
			'normalize': True,
		},
		'loss': {
			'prototype_weight': 1.0,
			'usage_weight': 0.005,
			'consistency_weight': 0.0,
			'consistency_beta': 0.1,
			'distillation_weight': 0.2,
		},
		'train': {'lr': 1.0e-3, 'encoder_lr': 1.0e-3},
		'identity': {
			'model_tag': (
				'strat_hmm_pretext_mh_k6810_xycons1_nocons_topblock1_distill_v1'
			),
			'scientific_identity': {
				'experiment_role': (
					'multi_head_ordered_xy_neighbor_consensus_hard_pretext'
				),
				'variant': 'xycons1_nocons',
				'target_representation': 'xy_neighbor_consensus_hard_labels_v1',
				'target_semantics': 'xy_neighbor_consensus_hard_label_smoothing_v1',
				'xy_neighbor_consensus_target_manifest_sha256': sha256(
					manifest_path.read_bytes()
				).hexdigest(),
				'xy_neighbor_consensus_target_head_hashes': head_hashes,
				'source_hard_manifest_sha256': 'a' * 64,
				'xy_neighbor_consensus_smoothing': target_manifest['smoothing'],
				'supervised_loss': 'structured_hmm_hard_categorical_v1',
				'head_spec': 'multi_resolution_ordered_prototypes_v1',
				'head_ks': [6, 8, 10],
				'head_temperature': 0.1,
				'head_normalize': True,
				'consistency_policy': 'disabled_for_xy_neighbor_consensus_v1',
				'prototype_weight': 1.0,
				'usage_weight': 0.005,
				'consistency_weight': 0.0,
				'consistency_beta': 0.1,
				'distillation_weight': 0.2,
			},
		},
	}


def _new_multi_head_components() -> tuple[
	torch.nn.Linear,
	MultiResolutionOrderedPrototypeHeads,
	torch.optim.AdamW,
]:
	student = torch.nn.Linear(2, 3)
	heads = MultiResolutionOrderedPrototypeHeads(
		feature_dim=3,
		ks=(6, 8, 10),
		projection_dim=2,
		temperature=0.1,
		normalize=True,
	)
	optimizer = torch.optim.AdamW(
		[
			{'params': heads.parameters(), 'name': 'head'},
			{'params': student.parameters(), 'name': 'encoder'},
		],
		lr=1.0e-3,
	)
	return student, heads, optimizer


def _save_consensus_checkpoint(
	path: Path,
	*,
	config: dict[str, object],
	student: torch.nn.Module,
	heads: MultiResolutionOrderedPrototypeHeads,
	optimizer: torch.optim.AdamW,
) -> Path:
	teacher_checkpoint = Path(config['teacher']['checkpoint'])
	student_checkpoint = Path(config['student']['init_checkpoint'])
	rng_state = capture_rng_state()
	rng_state['dataloader_generator'] = torch.Generator().manual_seed(274).get_state()
	return save_strat_hmm_checkpoint(
		path,
		student=student,
		head=heads,
		optimizer=optimizer,
		epoch=1,
		mae_config={'stage': 'train_amp_mae'},
		stratigraphy_config=config,
		metrics={'loss': 1.0},
		global_step=2,
		checkpoint_kind='step',
		batch_index=1,
		rng_state=rng_state,
		control_identity={
			'input_identities': {
				'teacher_checkpoint': {
					'sha256': sha256(teacher_checkpoint.read_bytes()).hexdigest()
				},
				'student_init_checkpoint': {
					'sha256': sha256(student_checkpoint.read_bytes()).hexdigest()
				},
			},
			'initial_state_sha256': {
				'student': '0' * 64,
				'head': '1' * 64,
			},
		},
	)


def _assert_consensus_manifest_load(
	path: Path,
	expected_path: Path,
	*,
	validate_array_semantics: bool,
) -> None:
	assert path == expected_path
	assert validate_array_semantics is False

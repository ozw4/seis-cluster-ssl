"""Focused M5-LS adapter and hard-loss dispatch contracts."""

from __future__ import annotations

import json
from hashlib import sha256
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from seis_ssl_cluster.config import pretraining
from seis_ssl_cluster.data import target_providers
from seis_ssl_cluster.data.target_providers import (
	MultiHeadStratPseudoTargetProvider,
	TargetProviderContext,
	load_strat_multi_head_lateral_target_manifest_adapter,
)
from seis_ssl_cluster.training.strat_hmm import epoch, runner

if TYPE_CHECKING:
	from pathlib import Path


def test_lateral_adapter_reuses_hard_provider_fields_without_array_load(
	tmp_path: Path, monkeypatch
) -> None:
	"""M5-LS keeps arrays lazy and emits the historical hard sample contract."""
	payload, paths = _lateral_payload(tmp_path)
	calls: list[object] = []

	def load_manifest(*_args: object, **kwargs: object) -> dict[str, object]:
		calls.append(kwargs.get('validate_array_semantics'))
		return payload

	monkeypatch.setattr(
		target_providers,
		'load_multi_head_lateral_target_manifest',
		load_manifest,
	)

	adapted = load_strat_multi_head_lateral_target_manifest_adapter(
		tmp_path / 'lateral.json'
	)
	assert calls == [False]
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
	for k in (6, 8, 10):
		target = targets[f'k{k}']
		assert set(target) == {
			'labels',
			'confidence',
			'boundary_weight',
			'valid_mask',
		}
		assert np.array_equal(target['labels'], np.full((2, 2, 2), k - 1))
		assert np.array_equal(target['confidence'], np.ones((2, 2, 2)))
		assert np.array_equal(target['boundary_weight'], np.ones((2, 2, 2)))
		assert np.array_equal(target['valid_mask'], np.ones((2, 2, 2), dtype=bool))
	assert all(paths[k]['labels'].is_file() for k in (6, 8, 10))


@pytest.mark.parametrize(
	('representation', 'expected_loader'),
	[
		('hard_viterbi_labels_v1', ('multi_head', False)),
		('ordered_path_state_posterior_v1', ('state_posterior', None)),
		('lateral_mean_field_hard_labels_v1', ('lateral_targets', False)),
	],
)
def test_config_manifest_resolution_dispatches_each_representation_explicitly(
	monkeypatch: pytest.MonkeyPatch,
	representation: str,
	expected_loader: tuple[str, bool | None],
) -> None:
	"""Hard, posterior, and lateral configs retain distinct manifest loaders."""
	calls: list[tuple[str, object]] = []
	manifest = {'head_ks': [6, 8, 10]}

	def hard_loader(_path: str, **kwargs: object) -> dict[str, object]:
		calls.append(('multi_head', kwargs.get('validate_array_semantics')))
		return manifest

	def posterior_loader(_path: str, **kwargs: object) -> dict[str, object]:
		calls.append(('state_posterior', kwargs.get('validate_array_semantics')))
		return manifest

	def lateral_loader(_path: str, **kwargs: object) -> dict[str, object]:
		calls.append(('lateral_targets', kwargs.get('validate_array_semantics')))
		return manifest

	module = SimpleNamespace(
		load_multi_head_target_manifest=hard_loader,
		load_multi_head_state_posterior_manifest=posterior_loader,
		load_multi_head_lateral_target_manifest=lateral_loader,
	)
	monkeypatch.setattr(pretraining.importlib, 'import_module', lambda _name: module)

	resolved = pretraining._validate_strat_hmm_multi_head_manifest(  # noqa: SLF001
		{'manifest': 'targets.json'},
		{'ks': [6, 8, 10]},
		target_representation=representation,
	)

	assert resolved is manifest
	assert calls == [expected_loader]


def test_lateral_runner_uses_adapter_and_existing_hard_dataset(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	"""M5-LS enters the historical hard dataset path, never the posterior path."""
	dispatch: list[object] = []

	class _StopAfterDataloaderError(Exception):
		pass

	def hard_dataset(
		_manifests: object, target_manifest: object, **kwargs: object
	) -> list[object]:
		dispatch.append(('dataset', target_manifest, kwargs['min_confidence']))
		return []

	def hard_dataloader(_dataset: object, **_kwargs: object) -> object:
		dispatch.append('dataloader')
		raise _StopAfterDataloaderError

	monkeypatch.setattr(runner, 'read_manifest_json', lambda _path: [])
	monkeypatch.setattr(
		runner,
		'load_strat_multi_head_lateral_target_manifest_adapter',
		lambda _path: SimpleNamespace(target_manifest='adapted-lateral-manifest'),
	)
	monkeypatch.setattr(runner, 'NopimsStratMultiHeadTargetDataset', hard_dataset)
	monkeypatch.setattr(
		runner, 'build_strat_multi_head_target_dataloader', hard_dataloader
	)
	monkeypatch.setattr(
		runner,
		'NopimsStratMultiHeadPosteriorDataset',
		lambda *_args, **_kwargs: pytest.fail(
			'lateral targets must not use posterior data'
		),
	)
	monkeypatch.setattr(runner, '_strat_hmm_control_identity', lambda _config: None)
	monkeypatch.setattr(runner, '_snapshot_run_inputs', lambda **_kwargs: None)
	monkeypatch.setattr(runner, 'prepare_run_directory', lambda **_kwargs: None)

	with pytest.raises(_StopAfterDataloaderError):
		runner.run_strat_hmm_pretext_training(
			{
				'paths': {'output_root': str(tmp_path)},
				'manifests': {'train': str(tmp_path / 'train.json')},
				'head': {'spec': 'multi_resolution_ordered_prototypes_v1'},
				'pseudo_targets': {
					'manifest': str(tmp_path / 'lateral.json'),
					'target_representation': 'lateral_mean_field_hard_labels_v1',
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

	assert dispatch == [('dataset', 'adapted-lateral-manifest', 0.0), 'dataloader']


@pytest.mark.parametrize(
	('representation', 'expected_loss'),
	[
		('hard_viterbi_labels_v1', 'hard'),
		('lateral_mean_field_hard_labels_v1', 'hard'),
		('ordered_path_state_posterior_v1', 'posterior'),
	],
)
def test_epoch_routes_lateral_targets_to_existing_hard_loss(
	monkeypatch: pytest.MonkeyPatch,
	representation: str,
	expected_loss: str,
) -> None:
	"""Only posterior supervision may select the posterior loss implementation."""
	calls: list[str] = []

	class _Student(torch.nn.Module):
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

	def posterior_loss(**kwargs: object) -> dict[str, torch.Tensor]:
		calls.append('posterior')
		encoded = kwargs['encoded']
		assert isinstance(encoded, dict)
		return {'loss': encoded['tokens'].sum()}

	monkeypatch.setattr(epoch, 'compute_strat_hmm_multi_head_losses', hard_loss)
	monkeypatch.setattr(
		epoch, 'compute_strat_hmm_multi_head_posterior_losses', posterior_loss
	)
	student = _Student()
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
		target_representation=representation,
	)

	assert calls == [expected_loss]


def _lateral_payload(
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
			'artifact_type': 'strat_hmm_multi_head_lateral_target_manifest',
			'schema_version': 1,
			'target_semantics': 'ordered_hmm_edge_aware_lateral_mean_field_hard_v1',
			'head_ks': [6, 8, 10],
			'source_hard_manifest': {'path': 'hard.json', 'sha256': 'hard'},
			'source_posterior_manifest': {'path': 'posterior.json', 'sha256': 'post'},
			'smoothing': {'pairwise_strength_ratio': 0.25},
			'heads': heads,
		},
		paths,
	)

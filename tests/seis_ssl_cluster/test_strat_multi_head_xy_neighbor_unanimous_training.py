"""Focused hard-route and schema-v6 contracts for unanimous XY targets."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from seis_ssl_cluster.config import pretraining
from seis_ssl_cluster.data import target_providers
from seis_ssl_cluster.data.target_providers import (
	load_strat_multi_head_xy_neighbor_unanimous_target_manifest_adapter,
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

_REPRESENTATION = 'xy_neighbor_unanimous_hard_labels_v1'
_SEMANTICS = 'xy_neighbor_unanimous_outlier_correction_v1'
_MODEL_TAG = 'strat_hmm_pretext_mh_k6810_xyunanim1_nocons_topblock1_distill_v1'


def test_unanimous_config_uses_its_strict_manifest_loader(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""The unanimous representation cannot fall through to another loader."""
	calls: list[tuple[str, object]] = []
	manifest = {'head_ks': [6, 8, 10]}

	def loader(_path: str, **kwargs: object) -> dict[str, object]:
		calls.append(('unanimous', kwargs.get('validate_array_semantics')))
		return manifest

	module = SimpleNamespace(
		load_multi_head_xy_neighbor_unanimous_target_manifest=loader,
	)
	monkeypatch.setattr(pretraining.importlib, 'import_module', lambda _name: module)

	resolved = pretraining._validate_strat_hmm_multi_head_manifest(  # noqa: SLF001
		{'manifest': 'targets.json'},
		{'ks': [6, 8, 10]},
		target_representation=_REPRESENTATION,
	)

	assert resolved is manifest
	assert calls == [('unanimous', False)]


def test_unanimous_identity_rejects_fixed_training_contract_drift(
	tmp_path: Path,
) -> None:
	"""The candidate cannot silently alter its fixed no-consistency settings."""
	manifest_path = tmp_path / 'unanimous-targets.json'
	manifest_path.write_text('{}\n', encoding='utf-8')
	target = _target_manifest()
	config = _checkpoint_config(manifest_path, target)
	identity = config['identity']
	assert isinstance(identity, dict)
	scientific = identity['scientific_identity']
	assert isinstance(scientific, dict)
	pretraining._validate_xy_neighbor_unanimous_scientific_identity(  # noqa: SLF001
		scientific,
		model_tag=_MODEL_TAG,
		manifest_sha256=sha256(manifest_path.read_bytes()).hexdigest(),
		manifest=target,
		loss=config['loss'],  # type: ignore[arg-type]
	)

	wrong_loss = deepcopy(config['loss'])
	assert isinstance(wrong_loss, dict)
	wrong_loss['consistency_beta'] = 0.2
	with pytest.raises(ValueError, match=r'loss\.consistency_beta'):
		pretraining._validate_xy_neighbor_unanimous_scientific_identity(  # noqa: SLF001
			scientific,
			model_tag=_MODEL_TAG,
			manifest_sha256=sha256(manifest_path.read_bytes()).hexdigest(),
			manifest=target,
			loss=wrong_loss,
		)

	wrong_scientific = deepcopy(scientific)
	wrong_scientific['student_unfreeze_top_blocks'] = 0
	with pytest.raises(ValueError, match='student_unfreeze_top_blocks'):
		pretraining._validate_xy_neighbor_unanimous_scientific_identity(  # noqa: SLF001
			wrong_scientific,
			model_tag=_MODEL_TAG,
			manifest_sha256=sha256(manifest_path.read_bytes()).hexdigest(),
			manifest=target,
			loss=config['loss'],  # type: ignore[arg-type]
		)


def test_unanimous_runner_and_epoch_use_existing_hard_route(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	"""Unanimous labels use hard collation and never select posterior losses."""
	dispatch: list[object] = []

	class StopAfterDataloaderError(Exception):
		"""Stop the runner after its dataset selection."""

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
		'load_strat_multi_head_xy_neighbor_unanimous_target_manifest_adapter',
		lambda _path: SimpleNamespace(target_manifest='adapted-unanimous-manifest'),
	)
	monkeypatch.setattr(runner, 'NopimsStratMultiHeadTargetDataset', hard_dataset)
	monkeypatch.setattr(
		runner, 'build_strat_multi_head_target_dataloader', hard_dataloader
	)
	monkeypatch.setattr(
		runner,
		'NopimsStratMultiHeadPosteriorDataset',
		lambda *_args, **_kwargs: pytest.fail(
			'XY unanimous targets must not use posterior data'
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
					'manifest': str(tmp_path / 'unanimous.json'),
					'target_representation': _REPRESENTATION,
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

	assert dispatch == [('dataset', 'adapted-unanimous-manifest', 0.0), 'dataloader']

	calls: list[str] = []

	class Student(torch.nn.Module):
		def __init__(self) -> None:
			super().__init__()
			self.scale = torch.nn.Parameter(torch.tensor(1.0))

		def encode_tokens(
			self, value: torch.Tensor, *, valid_mask: torch.Tensor
		) -> dict[str, torch.Tensor]:
			del valid_mask
			return {'tokens': value * self.scale}

	def hard_loss(**kwargs: object) -> dict[str, torch.Tensor]:
		calls.append('hard')
		encoded = kwargs['encoded']
		assert isinstance(encoded, dict)
		return {
			'loss': encoded['tokens'].sum(),
			'loss_consistency': encoded['tokens'].sum() * 0.25,
		}

	monkeypatch.setattr(epoch, 'compute_strat_hmm_multi_head_losses', hard_loss)
	monkeypatch.setattr(
		epoch,
		'compute_strat_hmm_multi_head_posterior_losses',
		lambda **_kwargs: pytest.fail('unanimous targets must not use posterior loss'),
	)
	student = Student()
	heads = torch.nn.Linear(1, 1)
	state = epoch.train_strat_hmm_multi_head_one_epoch(
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
		target_representation=_REPRESENTATION,
	)
	assert calls == ['hard']
	assert state.metrics['loss_consistency_contribution'] == 0.0


def test_unanimous_target_adapter_uses_existing_hard_provider(
	monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
	"""The strict manifest is adapted lazily with source-invalid preservation."""
	calls: list[object] = []
	payload = _target_manifest()
	heads = payload['heads']
	assert isinstance(heads, dict)
	for k in (6, 8, 10):
		head = heads[str(k)]
		assert isinstance(head, dict)
		surveys = head['surveys']
		assert isinstance(surveys, dict)
		survey = surveys['survey']
		assert isinstance(survey, dict)
		for name in ('labels', 'confidence', 'valid_tokens', 'metadata'):
			reference = survey[name]
			assert isinstance(reference, dict)
			reference['path'] = str(tmp_path / f'k{k}_{name}')

	def loader(_path: Path, **kwargs: object) -> dict[str, object]:
		calls.append(kwargs.get('validate_array_semantics'))
		return payload

	monkeypatch.setattr(
		target_providers,
		'load_multi_head_xy_neighbor_unanimous_target_manifest',
		loader,
	)
	monkeypatch.setattr(
		target_providers,
		'_coerce_multi_head_target_manifest',
		lambda _manifest: None,
	)

	adapted = load_strat_multi_head_xy_neighbor_unanimous_target_manifest_adapter(
		tmp_path / 'unanimous.json'
	)

	assert calls == [False]
	assert adapted.target_representation == _REPRESENTATION
	assert adapted.target_semantics == _SEMANTICS
	assert adapted.target_manifest.invalid_label_policy == 'preserve_source'
	assert not hasattr(adapted, 'source_posterior_manifest')


def test_unanimous_checkpoint_uses_schema_six_and_rejects_schema_five(  # noqa: PLR0915
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""Schema 6 binds unanimous provenance and exposes it to extraction."""
	manifest_path = tmp_path / 'unanimous-targets.json'
	manifest_path.write_text('{"unanimous": true}\n', encoding='utf-8')
	target = _target_manifest()
	monkeypatch.setattr(
		strat_hmm_checkpoint,
		'load_multi_head_xy_neighbor_unanimous_target_manifest',
		lambda path, *, validate_array_semantics: (
			_assert_manifest_load(
				path,
				manifest_path,
				validate_array_semantics=validate_array_semantics,
			)
			or target
		),
	)
	config = _checkpoint_config(manifest_path, target)
	student, heads, optimizer = _components()
	rng_state = capture_rng_state()
	rng_state['dataloader_generator'] = torch.Generator().manual_seed(9).get_state()
	checkpoint_path = save_strat_hmm_checkpoint(
		tmp_path / 'unanimous.pt',
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
			'schema_version': 2,
			'model_tag': _MODEL_TAG,
			'scientific_identity': config['identity']['scientific_identity'],  # type: ignore[index]
			'input_identities': {
				'teacher_checkpoint': {
					'path': config['teacher']['checkpoint'],  # type: ignore[index]
					'sha256': sha256(
						Path(config['teacher']['checkpoint']).read_bytes()  # type: ignore[index]
					).hexdigest(),
				},
				'student_init_checkpoint': {
					'path': config['student']['init_checkpoint'],  # type: ignore[index]
					'sha256': sha256(
						Path(config['student']['init_checkpoint']).read_bytes()  # type: ignore[index]
					).hexdigest(),
				},
				'target_manifest': {
					'path': str(manifest_path),
					'sha256': sha256(manifest_path.read_bytes()).hexdigest(),
				},
			},
			'initial_state_sha256': {'student': '0' * 64, 'head': '1' * 64},
		},
	)
	payload = load_checkpoint(checkpoint_path, map_location='cpu')
	identity = payload['stratigraphy_checkpoint']
	assert isinstance(identity, dict)
	assert identity['schema_version'] == 6
	assert identity['target_representation'] == _REPRESENTATION
	assert 'xy_neighbor_consensus_target_manifest' not in identity
	validate_stratigraphy_checkpoint_payload(payload, expected_config=config)

	metadata = _stratigraphy_pretext_metadata(payload)
	assert metadata is not None
	assert metadata['xy_neighbor_unanimous_target_manifest_path'] == str(manifest_path)
	assert (
		metadata['per_head_xy_neighbor_unanimous_target_sha256']
		== identity['per_head_xy_neighbor_unanimous_targets']
	)

	wrong_schema = deepcopy(payload)
	wrong_schema_identity = wrong_schema['stratigraphy_checkpoint']
	assert isinstance(wrong_schema_identity, dict)
	wrong_schema_identity['schema_version'] = 5
	with pytest.raises(ValueError, match='schema_version'):
		validate_stratigraphy_checkpoint_payload(wrong_schema)

	tampered_initial = deepcopy(payload)
	tampered_initial_identity = tampered_initial['stratigraphy_checkpoint']
	assert isinstance(tampered_initial_identity, dict)
	tampered_initial_identity['initial_student_state_sha256'] = 'f' * 64
	with pytest.raises(ValueError, match='control initial student hash'):
		validate_stratigraphy_checkpoint_payload(tampered_initial)

	tampered_semantics = deepcopy(payload)
	tampered_semantics_identity = tampered_semantics['stratigraphy_checkpoint']
	tampered_semantics_config = tampered_semantics['stratigraphy_config']
	tampered_semantics_control = tampered_semantics['control_identity']
	assert isinstance(tampered_semantics_identity, dict)
	assert isinstance(tampered_semantics_config, dict)
	assert isinstance(tampered_semantics_control, dict)
	tampered_scientific = tampered_semantics_config['identity']['scientific_identity']  # type: ignore[index]
	assert isinstance(tampered_scientific, dict)
	tampered_scientific['target_semantics'] = 'forged_semantics'
	tampered_semantics_identity['target_semantics'] = 'forged_semantics'
	tampered_semantics_identity['scientific_identity_sha256'] = (
		strat_hmm_checkpoint.scientific_identity_sha256(tampered_scientific)
	)
	tampered_semantics_control['scientific_identity'] = tampered_scientific
	with pytest.raises(ValueError, match='fixed XY unanimous identity'):
		validate_stratigraphy_checkpoint_payload(tampered_semantics)

	for representation in (
		None,
		'ordered_path_state_posterior_v1',
		'lateral_mean_field_hard_labels_v1',
		'xy_neighbor_consensus_hard_labels_v1',
	):
		incompatible = deepcopy(config)
		pseudo_targets = incompatible['pseudo_targets']
		assert isinstance(pseudo_targets, dict)
		if representation is None:
			pseudo_targets.pop('target_representation')
		else:
			pseudo_targets['target_representation'] = representation
		other_student, other_heads, other_optimizer = _components()
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


def _target_manifest() -> dict[str, object]:
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
		'target_representation': _REPRESENTATION,
		'target_semantics': _SEMANTICS,
		'source_hard_manifest': {'path': 'hard.json', 'sha256': 'a' * 64},
		'smoothing': {
			'neighborhood': 'same_z_xy_four_neighbors',
			'neighbor_order': ['x_minus', 'x_plus', 'y_minus', 'y_plus'],
			'four_valid_neighbors_minimum_agreement': 4,
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


def _checkpoint_config(
	manifest_path: Path, target: dict[str, object]
) -> dict[str, object]:
	teacher = manifest_path.parent / 'teacher.pt'
	student = manifest_path.parent / 'student.pt'
	teacher.write_bytes(b'teacher')
	student.write_bytes(b'student')
	head_hashes = pretraining._multi_head_target_hashes(target)  # noqa: SLF001
	return {
		'stage': 'train_strat_hmm_pretext',
		'paths': {'output_root': str(manifest_path.parent / 'unanimous-run')},
		'pseudo_targets': {
			'manifest': str(manifest_path),
			'target_representation': _REPRESENTATION,
		},
		'teacher': {'checkpoint': str(teacher)},
		'student': {'init_checkpoint': str(student), 'unfreeze_top_blocks': 1},
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
			'model_tag': _MODEL_TAG,
			'scientific_identity': {
				'experiment_role': (
					'multi_head_ordered_xy_neighbor_unanimous_hard_pretext'
				),
				'variant': 'xyunanim1_nocons',
				'head_spec': 'multi_resolution_ordered_prototypes_v1',
				'head_ks': [6, 8, 10],
				'head_temperature': 0.1,
				'head_normalize': True,
				'target_representation': _REPRESENTATION,
				'target_semantics': _SEMANTICS,
				'xy_neighbor_unanimous_target_manifest_sha256': sha256(
					manifest_path.read_bytes()
				).hexdigest(),
				'xy_neighbor_unanimous_target_head_hashes': head_hashes,
				'source_hard_manifest_sha256': 'a' * 64,
				'xy_neighbor_unanimous_smoothing': target['smoothing'],
				'supervised_loss': 'structured_hmm_hard_categorical_v1',
				'consistency_policy': 'disabled_for_xy_neighbor_unanimous_v1',
				'prototype_weight': 1.0,
				'usage_weight': 0.005,
				'consistency_weight': 0.0,
				'consistency_beta': 0.1,
				'distillation_weight': 0.2,
				'student_unfreeze_top_blocks': 1,
			},
		},
	}


def _components() -> tuple[
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


def _assert_manifest_load(
	path: Path, expected_path: Path, *, validate_array_semantics: bool
) -> None:
	assert path == expected_path
	assert validate_array_semantics is False

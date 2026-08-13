
from __future__ import annotations

import json
import shutil
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import torch

import seis_ssl_cluster.parihaka.mae_validation as validation_module
import seis_ssl_cluster.training.mae_checkpoint as mae_checkpoint_module
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.parihaka import (
	ParihakaMaeValidationResult,
	prepare_parihaka_volume,
	validate_parihaka_mae_inputs_from_configs,
	write_parihaka_mae_validation_report,
)
from seis_ssl_cluster.training.checkpoint import capture_rng_state, load_checkpoint
from seis_ssl_cluster.training.mae_checkpoint import _save_mae_checkpoint
from tests.seis_ssl_cluster.test_parihaka_prepare_volume import _fixture_config

if TYPE_CHECKING:
	from collections.abc import Mapping

PRETRAIN_ROOT = Path(
	'experiments/parihaka/facies_benchmark_v1/20_pretrain/'
	'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1',
)
SMOKE_CONFIG = PRETRAIN_ROOT / '01_smoke_2step.yaml'
FULL_CONFIG = PRETRAIN_ROOT / '02_full_100ep.yaml'


def test_valid_synthetic_parihaka_inputs_pass(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	prepare, smoke, full = _inputs_fixture(tmp_path, monkeypatch)

	result = validate_parihaka_mae_inputs_from_configs(
		prepare=prepare,
		smoke_raw=smoke,
		full_raw=full,
	)

	assert result.source_sha256
	assert result.prepared_sha256
	assert result.smoke['train']['device'] == 'cpu'
	assert result.full['train']['device'] == 'cuda'


def test_runtime_validation_does_not_read_nopims_config(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	prepare, smoke, full = _inputs_fixture(tmp_path, monkeypatch)

	def reject_external_config(_path: str | Path) -> dict[str, object]:
		raise AssertionError('runtime validation read an external experiment config')

	monkeypatch.setattr(validation_module, 'load_config', reject_external_config)
	result = validate_parihaka_mae_inputs_from_configs(
		prepare=prepare,
		smoke_raw=smoke,
		full_raw=full,
	)

	assert result.full['train']['epochs'] == 100
	module_source = Path(validation_module.__file__).read_text(encoding='utf-8')
	assert 'experiments/nopims' not in module_source


@pytest.mark.parametrize(
	'target',
	['source', 'prepared', 'metadata', 'manifest', 'path_list', 'stats'],
)
def test_inputs_reject_missing_files(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	target: str,
) -> None:
	prepare, smoke, full = _inputs_fixture(tmp_path, monkeypatch)
	paths = {
		'source': prepare.inputs.amplitude_npz,
		'prepared': prepare.outputs.amplitude_npy,
		'metadata': prepare.outputs.metadata,
		'manifest': prepare.outputs.manifest,
		'path_list': prepare.outputs.path_list,
		'stats': prepare.outputs.normalization_stats,
	}
	paths[target].unlink()

	with pytest.raises((FileNotFoundError, FileExistsError, ValueError)):
		validate_parihaka_mae_inputs_from_configs(
			prepare=prepare,
			smoke_raw=smoke,
			full_raw=full,
		)


@pytest.mark.parametrize(
	'target', ['source', 'prepared', 'manifest', 'path_list', 'stats']
)
def test_inputs_reject_live_hash_drift(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	target: str,
) -> None:
	prepare, smoke, full = _inputs_fixture(tmp_path, monkeypatch)
	paths = {
		'source': prepare.inputs.amplitude_npz,
		'prepared': prepare.outputs.amplitude_npy,
		'manifest': prepare.outputs.manifest,
		'path_list': prepare.outputs.path_list,
		'stats': prepare.outputs.normalization_stats,
	}
	with paths[target].open('ab') as file_obj:
		file_obj.write(b'drift')

	with pytest.raises((ValueError, OSError)):
		validate_parihaka_mae_inputs_from_configs(
			prepare=prepare,
			smoke_raw=smoke,
			full_raw=full,
		)


@pytest.mark.parametrize(
	('field', 'value', 'match'),
	[
		(('conversion', 'axis_mapping'), 'XYZ -> ZXY', 'axis_mapping'),
		(('conversion', 'transpose_axes'), [0, 1, 2], 'transpose_axes'),
		(('conversion', 'verification'), 'sampled', 'verification'),
		(('dataset', 'version'), 'foreign', 'dataset.version'),
		(('dataset', 'survey_id'), 'foreign', 'dataset.survey_id'),
		(('schema_version',), 2, 'identity|schema_version'),
		(('source', 'dtype'), 'float64', 'source.dtype'),
		(
			('outputs', 'amplitude_npy', 'path'),
			'/artifacts/foreign.npy',
			'hash/path drift|amplitude_npy.path',
		),
		(('outputs', 'amplitude_npy', 'order'), 'F', 'order'),
	],
)
def test_inputs_reject_metadata_contract_drift(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	field: tuple[str, ...],
	value: object,
	match: str,
) -> None:
	prepare, smoke, full = _inputs_fixture(tmp_path, monkeypatch)
	metadata = json.loads(prepare.outputs.metadata.read_text(encoding='utf-8'))
	parent = metadata
	for key in field[:-1]:
		parent = parent[key]
	parent[field[-1]] = value
	prepare.outputs.metadata.write_text(json.dumps(metadata), encoding='utf-8')

	with pytest.raises(ValueError, match=match):
		validate_parihaka_mae_inputs_from_configs(
			prepare=prepare,
			smoke_raw=smoke,
			full_raw=full,
		)


@pytest.mark.parametrize('contract', ['shape', 'dtype', 'order'])
def test_inputs_reject_live_prepared_npy_contract_drift(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	contract: str,
) -> None:
	prepare, smoke, full = _inputs_fixture(tmp_path, monkeypatch)
	expected_shape = (
		prepare.source.shape_zxy[1],
		prepare.source.shape_zxy[2],
		prepare.source.shape_zxy[0],
	)
	if contract == 'shape':
		array = np.zeros((2, 2, 2), dtype=np.float32)
	elif contract == 'dtype':
		array = np.zeros(expected_shape, dtype=np.float64)
	else:
		array = np.asfortranarray(np.zeros(expected_shape, dtype=np.float32))
	np.save(prepare.outputs.amplitude_npy, array)
	metadata = json.loads(prepare.outputs.metadata.read_text(encoding='utf-8'))
	record = metadata['outputs']['amplitude_npy']
	record['sha256'] = sha256(prepare.outputs.amplitude_npy.read_bytes()).hexdigest()
	record['size_bytes'] = prepare.outputs.amplitude_npy.stat().st_size
	prepare.outputs.metadata.write_text(json.dumps(metadata), encoding='utf-8')

	match = 'C-contiguous|order' if contract == 'order' else contract
	with pytest.raises(ValueError, match=match):
		validate_parihaka_mae_inputs_from_configs(
			prepare=prepare,
			smoke_raw=smoke,
			full_raw=full,
		)


def test_inputs_reject_normalization_config_mismatch(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	prepare, smoke, full = _inputs_fixture(tmp_path, monkeypatch)
	prepare = prepare.__class__(
		**{
			**prepare.__dict__,
			'normalization': prepare.normalization.__class__(
				clip_low_percentile=1.0,
				clip_high_percentile=99.5,
				eps=1.0e-6,
				max_samples=1_000_000,
				seed=42,
			),
		},
	)

	with pytest.raises(ValueError, match='normalization low percentile'):
		validate_parihaka_mae_inputs_from_configs(
			prepare=prepare,
			smoke_raw=smoke,
			full_raw=full,
		)


def test_inputs_reject_full_scientific_and_non_allowlisted_smoke_drift(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	prepare, smoke, full = _inputs_fixture(tmp_path, monkeypatch)
	full['loss']['gradient_weight'] = 0.25
	with pytest.raises(ValueError, match='gradient_weight'):
		validate_parihaka_mae_inputs_from_configs(
			prepare=prepare,
			smoke_raw=smoke,
			full_raw=full,
		)

	prepare, smoke, full = _inputs_fixture(tmp_path / 'second', monkeypatch)
	smoke['data']['min_valid_fraction'] = 0.2
	with pytest.raises(ValueError, match=r'data\.min_valid_fraction'):
		validate_parihaka_mae_inputs_from_configs(
			prepare=prepare,
			smoke_raw=smoke,
			full_raw=full,
		)


def test_inputs_allow_allowlisted_field_to_match_full(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	prepare, smoke, full = _inputs_fixture(tmp_path, monkeypatch)
	smoke['train']['batch_size'] = full['train']['batch_size']

	result = validate_parihaka_mae_inputs_from_configs(
		prepare=prepare,
		smoke_raw=smoke,
		full_raw=full,
	)

	assert result.smoke['train']['batch_size'] == result.full['train']['batch_size']


@pytest.mark.parametrize(
	'key',
	['device', 'amp', 'epochs', 'samples_per_epoch', 'max_steps'],
)
def test_inputs_require_mandatory_smoke_full_train_differences(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	key: str,
) -> None:
	prepare, smoke, full = _inputs_fixture(tmp_path, monkeypatch)
	smoke['train'][key] = full['train'].get(key)

	with pytest.raises(ValueError, match=key):
		validate_parihaka_mae_inputs_from_configs(
			prepare=prepare,
			smoke_raw=smoke,
			full_raw=full,
		)


@pytest.mark.parametrize(
	('section', 'key'),
	[
		('manifests', 'label_path'),
		('data', 'label_key'),
		('train', 'class_count'),
	],
)
def test_inputs_reject_label_or_class_fields(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	section: str,
	key: str,
) -> None:
	prepare, smoke, full = _inputs_fixture(tmp_path, monkeypatch)
	smoke[section][key] = 'forbidden'

	with pytest.raises(ValueError, match='forbidden label/class field'):
		validate_parihaka_mae_inputs_from_configs(
			prepare=prepare,
			smoke_raw=smoke,
			full_raw=full,
		)


@pytest.mark.parametrize('relation', ['same', 'nested', 'full_as_smoke'])
def test_inputs_reject_overlapping_or_foreign_smoke_root(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	relation: str,
) -> None:
	prepare, smoke, full = _inputs_fixture(tmp_path, monkeypatch)
	full_root = full['paths']['output_root']
	if relation in {'same', 'full_as_smoke'}:
		smoke['paths']['output_root'] = full_root
	else:
		smoke['paths']['output_root'] = f'{full_root}/nested'

	with pytest.raises(ValueError, match=r'smoke paths.output_root|disjoint'):
		validate_parihaka_mae_inputs_from_configs(
			prepare=prepare,
			smoke_raw=smoke,
			full_raw=full,
		)


def test_valid_schema2_two_step_smoke_passes_without_touching_full(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	inputs, base = _smoke_fixture(tmp_path, monkeypatch)
	full_root = Path(cast('Mapping[str, object]', inputs.full['paths'])['output_root'])
	full_root.mkdir(parents=True)
	marker = full_root / 'existing.txt'
	marker.write_text('unchanged\n', encoding='utf-8')
	before = marker.read_bytes()

	result = validation_module._validate_smoke(inputs, base=base)  # noqa: SLF001

	assert result.status == 'pass'
	assert result.checkpoint_schema_version == 2
	assert result.checkpoint_epoch == 1
	assert result.checkpoint_global_step == 2
	assert result.resolved_precision == 'float32'
	assert marker.read_bytes() == before


@pytest.mark.parametrize(
	('field', 'value', 'match'),
	[
		(('training_state', 'stage'), 'full', 'stage'),
		(('training_state', 'schema_version'), 1, 'schema_version'),
		(('training_state', 'checkpoint_kind'), 'step', 'checkpoint_kind|batch_index'),
		(('training_state', 'resolved_precision'), 'float16', 'precision'),
		(('epoch',), 2, 'epoch'),
		(('global_step',), 3, 'global_step'),
		(('config', 'train', 'seed'), 7, 'config'),
	],
)
def test_smoke_rejects_checkpoint_identity_drift(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	field: tuple[str, ...],
	value: object,
	match: str,
) -> None:
	inputs, base = _smoke_fixture(tmp_path, monkeypatch)
	latest = (
		Path(cast('Mapping[str, object]', inputs.smoke['paths'])['output_root'])
		/ 'latest.pt'
	)
	_mutate_checkpoint(latest, field, value)

	with pytest.raises((TypeError, ValueError), match=match):
		validation_module._validate_smoke(inputs, base=base)  # noqa: SLF001


@pytest.mark.parametrize('target', ['metric', 'model', 'optimizer'])
def test_smoke_rejects_nonfinite_checkpoint_state(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	target: str,
) -> None:
	inputs, base = _smoke_fixture(tmp_path, monkeypatch)
	latest = (
		Path(cast('Mapping[str, object]', inputs.smoke['paths'])['output_root'])
		/ 'latest.pt'
	)
	payload = load_checkpoint(latest, map_location='cpu')
	if target == 'metric':
		payload['metrics']['loss'] = float('nan')
	elif target == 'model':
		first = next(iter(payload['model_state_dict'].values()))
		first.reshape(-1)[0] = float('nan')
	else:
		state = next(iter(payload['optimizer_state_dict']['state'].values()))
		state['exp_avg'].reshape(-1)[0] = float('inf')
	torch.save(payload, latest)

	with pytest.raises(ValueError, match=r'finite|nonfinite'):
		validation_module._validate_smoke(inputs, base=base)  # noqa: SLF001


@pytest.mark.parametrize(
	'target',
	['resolved_config.json', 'manifest.json', 'path_list', 'run_metadata.json'],
)
def test_smoke_rejects_snapshot_drift(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	target: str,
) -> None:
	inputs, base = _smoke_fixture(tmp_path, monkeypatch)
	root = Path(cast('Mapping[str, object]', inputs.smoke['paths'])['output_root'])
	path = (
		root / 'inputs' / inputs.prepare.outputs.path_list.name
		if target == 'path_list'
		else root / target
	)
	path.write_text('{}\n', encoding='utf-8')

	with pytest.raises(ValueError, match=r'snapshot|metadata'):
		validation_module._validate_smoke(inputs, base=base)  # noqa: SLF001


def test_valid_schema2_completed_full_passes(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	inputs, base = _full_fixture(tmp_path, monkeypatch)
	monkeypatch.setattr(
		torch.cuda,
		'is_available',
		lambda: pytest.fail('post-hoc full validation queried CUDA availability'),
	)

	result = validation_module._validate_full(inputs, base=base)  # noqa: SLF001

	assert result.status == 'pass'
	assert result.checkpoint_schema_version == 2
	assert result.checkpoint_epoch == 100
	assert result.checkpoint_global_step == 250_000
	assert result.resolved_precision == 'bfloat16'
	assert result.scaler_present is False
	assert result.best_checkpoint_epoch == 40
	assert result.best_checkpoint_global_step == 100_000
	assert result.best_metric_value == 1.0


def test_full_loads_latest_and_best_once_each(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	inputs, base = _full_fixture(tmp_path, monkeypatch)
	loads: list[Path] = []
	original = mae_checkpoint_module.load_checkpoint

	def counted_load(path: str | Path, **kwargs: object) -> object:
		loads.append(Path(path))
		return original(path, **kwargs)

	monkeypatch.setattr(mae_checkpoint_module, 'load_checkpoint', counted_load)
	validation_module._validate_full(inputs, base=base)  # noqa: SLF001

	root = Path(cast('Mapping[str, object]', inputs.full['paths'])['output_root'])
	assert loads.count(root / 'latest.pt') == 1
	assert loads.count(root / 'best.pt') == 1


def test_valid_float16_completed_full_requires_scaler(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	inputs, base = _full_fixture(tmp_path, monkeypatch, resolved='float16')

	result = validation_module._validate_full(inputs, base=base)  # noqa: SLF001

	assert result.resolved_precision == 'float16'
	assert result.scaler_present is True


@pytest.mark.parametrize(
	('field', 'value', 'match'),
	[
		(('training_state', 'stage'), 'foreign', 'stage'),
		(('training_state', 'schema_version'), 1, 'schema_version'),
		(('training_state', 'checkpoint_kind'), 'step', 'checkpoint_kind|batch_index'),
		(('epoch',), 99, 'epoch'),
		(('global_step',), 249_999, 'global_step'),
		(('amp_enabled',), False, 'amp_enabled'),
		(('training_state', 'resolved_precision'), 'float16', 'precision'),
		(('config', 'train', 'seed'), 7, 'config'),
	]
)
def test_full_rejects_checkpoint_contract_drift(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	field: tuple[str, ...],
	value: object,
	match: str,
) -> None:
	inputs, base = _full_fixture(tmp_path, monkeypatch)
	root = Path(cast('Mapping[str, object]', inputs.full['paths'])['output_root'])
	_mutate_checkpoint(root / 'latest.pt', field, value)

	with pytest.raises((TypeError, ValueError), match=match):
		validation_module._validate_full(inputs, base=base)  # noqa: SLF001


@pytest.mark.parametrize('target', ['metric', 'model', 'optimizer', 'scaler'])
def test_full_rejects_nonfinite_checkpoint_state(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	target: str,
) -> None:
	inputs, base = _full_fixture(tmp_path, monkeypatch)
	root = Path(cast('Mapping[str, object]', inputs.full['paths'])['output_root'])
	latest = root / 'latest.pt'
	payload = load_checkpoint(latest, map_location='cpu')
	if target == 'metric':
		payload['metrics']['loss'] = float('nan')
	elif target == 'model':
		first = next(iter(payload['model_state_dict'].values()))
		first.reshape(-1)[0] = float('nan')
	elif target == 'optimizer':
		state = next(iter(payload['optimizer_state_dict']['state'].values()))
		state['exp_avg'].reshape(-1)[0] = float('inf')
	else:
		inputs, base = _full_fixture(
			tmp_path / 'float16', monkeypatch, resolved='float16'
		)
		root = Path(cast('Mapping[str, object]', inputs.full['paths'])['output_root'])
		latest = root / 'latest.pt'
		payload = load_checkpoint(latest, map_location='cpu')
		payload['scaler_state_dict']['scale'] = torch.tensor(float('inf'))
	torch.save(payload, latest)

	with pytest.raises(ValueError, match=r'finite|nonfinite'):
		validation_module._validate_full(inputs, base=base)  # noqa: SLF001


def test_full_rejects_model_geometry_mismatch(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	inputs, base = _full_fixture(tmp_path, monkeypatch)
	root = Path(cast('Mapping[str, object]', inputs.full['paths'])['output_root'])
	latest = root / 'latest.pt'
	payload = load_checkpoint(latest, map_location='cpu')
	payload['model_state_dict'].pop(next(iter(payload['model_state_dict'])))
	torch.save(payload, latest)

	with pytest.raises(ValueError, match=r'geometry|state mismatch'):
		validation_module._validate_full(inputs, base=base)  # noqa: SLF001


def test_full_rejects_float16_checkpoint_without_scaler(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	inputs, base = _full_fixture(tmp_path, monkeypatch, resolved='float16')
	root = Path(cast('Mapping[str, object]', inputs.full['paths'])['output_root'])
	latest = root / 'latest.pt'
	payload = load_checkpoint(latest, map_location='cpu')
	payload['scaler_state_dict'] = None
	torch.save(payload, latest)

	with pytest.raises(ValueError, match='scaler_state_dict'):
		validation_module._validate_full(inputs, base=base)  # noqa: SLF001


@pytest.mark.parametrize(
	('target', 'value', 'match'),
	[
		('amp_requested', False, 'requested AMP'),
		('amp_dtype_requested', 'float16', 'requested AMP dtype'),
		('resolved_dtype', 'float16', 'scaler contract'),
		('amp_enabled', False, 'enabled CUDA AMP'),
		('grad_scaler_enabled', True, 'scaler contract'),
	],
)
def test_full_rejects_precision_contract_mismatch(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	target: str,
	value: object,
	match: str,
) -> None:
	inputs, base = _full_fixture(tmp_path, monkeypatch)
	root = Path(cast('Mapping[str, object]', inputs.full['paths'])['output_root'])
	metadata_path = root / 'run_metadata.json'
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	metadata['precision'][target] = value
	metadata_path.write_text(json.dumps(metadata), encoding='utf-8')

	with pytest.raises((TypeError, ValueError), match=match):
		validation_module._validate_full(inputs, base=base)  # noqa: SLF001


@pytest.mark.parametrize(
	('target', 'match'),
	[
		('resolved_config.json', 'snapshot'),
		('manifest.json', 'snapshot'),
		('path_list', 'snapshot'),
		('run_metadata.json', 'metadata'),
	]
)
def test_full_rejects_snapshot_drift(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	target: str,
	match: str,
) -> None:
	inputs, base = _full_fixture(tmp_path, monkeypatch)
	root = Path(cast('Mapping[str, object]', inputs.full['paths'])['output_root'])
	path = (
		root / 'inputs' / inputs.prepare.outputs.path_list.name
		if target == 'path_list'
		else root / target
	)
	path.write_text('{}\n', encoding='utf-8')

	with pytest.raises((TypeError, ValueError), match=match):
		validation_module._validate_full(inputs, base=base)  # noqa: SLF001


@pytest.mark.parametrize(
	('target', 'value', 'match'),
	[
		('best_metric', 3.0, 'no greater'),
		('best_epoch', 101, r'\[1, 100\]'),
		('best_step', 100_001, 'global_step'),
		('best_config', 13, 'config'),
	]
)
def test_full_rejects_invalid_or_foreign_best(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	target: str,
	value: object,
	match: str,
) -> None:
	inputs, base = _full_fixture(tmp_path, monkeypatch)
	root = Path(cast('Mapping[str, object]', inputs.full['paths'])['output_root'])
	best = root / 'best.pt'
	fields = {
		'best_metric': ('metrics', 'loss'),
		'best_epoch': ('epoch',),
		'best_step': ('global_step',),
		'best_config': ('config', 'train', 'seed'),
	}
	_mutate_checkpoint(best, fields[target], value)

	with pytest.raises(ValueError, match=match):
		validation_module._validate_full(inputs, base=base)  # noqa: SLF001


def test_inputs_do_not_require_checkpoint_and_json_is_only_explicit(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	prepare, smoke, full = _inputs_fixture(tmp_path, monkeypatch)
	inputs = validate_parihaka_mae_inputs_from_configs(
		prepare=prepare,
		smoke_raw=smoke,
		full_raw=full,
	)
	assert not Path(
		cast('Mapping[str, object]', inputs.smoke['paths'])['output_root']
	).exists()
	assert not list(tmp_path.rglob('*validation*.json'))

	report = tmp_path / 'explicit' / 'report.json'
	result = ParihakaMaeValidationResult(
		check='inputs',
		status='pass',
		prepare_config=tmp_path / 'prepare.yaml',
		smoke_config=tmp_path / 'smoke.yaml',
		full_config=tmp_path / 'full.yaml',
		source_npz=prepare.inputs.amplitude_npz,
		source_sha256=inputs.source_sha256,
		prepared_npy=prepare.outputs.amplitude_npy,
		prepared_sha256=inputs.prepared_sha256,
		smoke_output_root=tmp_path / 'smoke',
		full_output_root=tmp_path / 'full',
	)
	write_parihaka_mae_validation_report(result, report)
	assert json.loads(report.read_text(encoding='utf-8'))['status'] == 'pass'


def _inputs_fixture(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, dict[str, object], dict[str, object]]:
	source = np.asfortranarray(
		(np.arange(5 * 3 * 4, dtype=np.float32) - np.float32(12)).reshape(5, 3, 4),
	)
	prepare = _fixture_config(tmp_path, source=source, chunk_size_z=2)
	prepare_parihaka_volume(prepare)
	monkeypatch.setenv(
		'SEIS_SSL_CLUSTER_ARTIFACT_ROOT',
		str(prepare.paths.artifact_root),
	)
	smoke = load_config(SMOKE_CONFIG)
	full = load_config(FULL_CONFIG)
	return prepare, smoke, full


def _smoke_fixture(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, dict[str, object]]:
	prepare, smoke_raw, full_raw = _inputs_fixture(tmp_path, monkeypatch)
	inputs = validate_parihaka_mae_inputs_from_configs(
		prepare=prepare,
		smoke_raw=smoke_raw,
		full_raw=full_raw,
	)
	smoke_root = Path(
		cast('Mapping[str, object]', inputs.smoke['paths'])['output_root']
	)
	smoke_root.mkdir(parents=True)
	model = torch.nn.Linear(2, 2)
	optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
	loss = model(torch.ones(1, 2)).sum()
	loss.backward()
	optimizer.step()
	rng_state = capture_rng_state()
	rng_state['dataloader_generator'] = torch.Generator().get_state()
	for name in ('latest.pt', 'best.pt'):
		_save_mae_checkpoint(
			smoke_root / name,
			model=model,
			optimizer=optimizer,
			epoch=1,
			config=inputs.smoke,
			metrics={'loss': 1.25, 'amp_enabled': 0.0},
			global_step=2,
			amp_enabled=False,
			scaler=None,
			checkpoint_kind='epoch',
			batch_index=None,
			rng_state=rng_state,
		)
	(smoke_root / 'resolved_config.json').write_text(
		json.dumps(inputs.smoke, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)
	shutil.copy2(prepare.outputs.manifest, smoke_root / 'manifest.json')
	inputs_dir = smoke_root / 'inputs'
	inputs_dir.mkdir()
	shutil.copy2(prepare.outputs.path_list, inputs_dir / prepare.outputs.path_list.name)
	(smoke_root / 'run_metadata.json').write_text(
		json.dumps(
			{
				'runtime_check_mode': 'once',
				'precision': {
					'amp_requested': False,
					'amp_dtype_requested': 'auto',
					'resolved_dtype': 'float32',
					'amp_enabled': False,
					'grad_scaler_enabled': False,
				},
			},
		),
		encoding='utf-8',
	)
	monkeypatch.setattr(
		validation_module,
		'_build_mae_model',
		lambda _config: torch.nn.Linear(2, 2),
	)
	base = {
		'check': 'smoke',
		'status': 'pass',
		'prepare_config': tmp_path / 'prepare.yaml',
		'smoke_config': SMOKE_CONFIG,
		'full_config': FULL_CONFIG,
		'source_npz': prepare.inputs.amplitude_npz,
		'source_sha256': inputs.source_sha256,
		'prepared_npy': prepare.outputs.amplitude_npy,
		'prepared_sha256': inputs.prepared_sha256,
		'smoke_output_root': smoke_root,
		'full_output_root': Path(
			cast('Mapping[str, object]', inputs.full['paths'])['output_root'],
		),
	}
	return inputs, base


def _full_fixture(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	*,
	resolved: str = 'bfloat16',
) -> tuple[object, dict[str, object]]:
	prepare, smoke_raw, full_raw = _inputs_fixture(tmp_path, monkeypatch)
	inputs = validate_parihaka_mae_inputs_from_configs(
		prepare=prepare,
		smoke_raw=smoke_raw,
		full_raw=full_raw,
	)
	full_root = Path(cast('Mapping[str, object]', inputs.full['paths'])['output_root'])
	full_root.mkdir(parents=True)
	model = torch.nn.Linear(2, 2)
	optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
	loss = model(torch.ones(1, 2)).sum()
	loss.backward()
	optimizer.step()
	rng_state = capture_rng_state()
	rng_state['dataloader_generator'] = torch.Generator().get_state()
	scaler = torch.amp.GradScaler('cpu') if resolved == 'float16' else None
	for name, epoch, step, metric in (
		('latest.pt', 100, 250_000, 2.0),
		('best.pt', 40, 100_000, 1.0),
	):
		_save_mae_checkpoint(
			full_root / name,
			model=model,
			optimizer=optimizer,
			epoch=epoch,
			config=inputs.full,
			metrics={'loss': metric, 'amp_enabled': 1.0},
			global_step=step,
			amp_enabled=True,
			scaler=scaler,
			checkpoint_kind='epoch',
			batch_index=None,
			rng_state=rng_state,
		)
	(full_root / 'resolved_config.json').write_text(
		json.dumps(inputs.full, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)
	shutil.copy2(prepare.outputs.manifest, full_root / 'manifest.json')
	inputs_dir = full_root / 'inputs'
	inputs_dir.mkdir()
	shutil.copy2(prepare.outputs.path_list, inputs_dir / prepare.outputs.path_list.name)
	(full_root / 'run_metadata.json').write_text(
		json.dumps(
			{
				'runtime_check_mode': 'once',
				'precision': _precision_contract(resolved),
			},
		),
		encoding='utf-8',
	)
	monkeypatch.setattr(
		validation_module,
		'_build_mae_model',
		lambda _config: torch.nn.Linear(2, 2),
	)
	base = {
		'check': 'full',
		'status': 'pass',
		'prepare_config': tmp_path / 'prepare.yaml',
		'smoke_config': SMOKE_CONFIG,
		'full_config': FULL_CONFIG,
		'source_npz': prepare.inputs.amplitude_npz,
		'source_sha256': inputs.source_sha256,
		'prepared_npy': prepare.outputs.amplitude_npy,
		'prepared_sha256': inputs.prepared_sha256,
		'smoke_output_root': Path(
			cast('Mapping[str, object]', inputs.smoke['paths'])['output_root']
		),
		'full_output_root': full_root,
	}
	return inputs, base


def _precision_contract(resolved: str) -> dict[str, object]:
	return {
		'amp_requested': True,
		'amp_dtype_requested': 'auto',
		'resolved_dtype': resolved,
		'amp_enabled': True,
		'grad_scaler_enabled': resolved == 'float16',
	}


def _mutate_checkpoint(path: Path, field: tuple[str, ...], value: object) -> None:
	payload = load_checkpoint(path, map_location='cpu')
	parent = payload
	for key in field[:-1]:
		parent = parent[key]
	parent[field[-1]] = value
	torch.save(payload, path)

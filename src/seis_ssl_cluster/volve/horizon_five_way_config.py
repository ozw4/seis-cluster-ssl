'''Strict configuration for the Volve horizon five-way comparison.'''

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from seis_ssl_cluster.volve.horizon_frozen import (
	FrozenHorizonTrainSettings,
	frozen_horizon_config_from_mapping,
)

if TYPE_CHECKING:
	from seis_ssl_cluster.volve.horizon_tiles import HorizonTileSettings

FIVE_WAY_MODEL_IDS = (
	'mae',
	'mae_hmm_k6',
	'local_barlow_twins',
	'local_barlow_twins_hmm_k6',
	'random',
)
VOLVE_HORIZON_FIVE_WAY_MODEL_IDS = FIVE_WAY_MODEL_IDS
FIVE_WAY_HMM_K = 6
FIVE_WAY_RANDOM_SEED = 42
FIVE_WAY_STAGE1_EPOCHS = 100
FIVE_WAY_STAGE2_EPOCHS = 25
FIVE_WAY_UNFREEZE_TOP_BLOCKS = 1
LOCAL_BARLOW_TWINS_METHOD = 'local_barlow_twins_3d'
LOCAL_BARLOW_TWINS_PAIRS_PER_CROP = 128
MAE_OBJECTIVE = 'amp_mae3d'
RANDOM_OBJECTIVE = 'random_encoder'

EXPECTED_MODEL_IDENTITIES: Mapping[str, Mapping[str, object]] = {
	'mae': {
		'objective': MAE_OBJECTIVE,
		'stratigraphy_pretext': False,
	},
	'mae_hmm_k6': {
		'objective': MAE_OBJECTIVE,
		'base_objective': MAE_OBJECTIVE,
		'hmm_k': FIVE_WAY_HMM_K,
		'stratigraphy_pretext': True,
	},
	'local_barlow_twins': {
		'objective': LOCAL_BARLOW_TWINS_METHOD,
		'local_pairs_per_crop': LOCAL_BARLOW_TWINS_PAIRS_PER_CROP,
		'stratigraphy_pretext': False,
	},
	'local_barlow_twins_hmm_k6': {
		'objective': LOCAL_BARLOW_TWINS_METHOD,
		'base_objective': LOCAL_BARLOW_TWINS_METHOD,
		'hmm_k': FIVE_WAY_HMM_K,
		'local_pairs_per_crop': LOCAL_BARLOW_TWINS_PAIRS_PER_CROP,
		'stratigraphy_pretext': True,
	},
	'random': {
		'objective': RANDOM_OBJECTIVE,
		'random_seed': FIVE_WAY_RANDOM_SEED,
		'stratigraphy_pretext': False,
	},
}

_TOP_LEVEL_KEYS = frozenset(
	{
		'paths',
		'dataset',
		'inputs',
		'models',
		'outputs',
		'decoder',
		'tiles',
		'train',
	}
)


@dataclass(frozen=True)
class VolveHorizonFiveWayModelSource:
	'''One configured frozen encoder source.'''

	model_id: str
	checkpoint: Path
	embeddings_dir: Path
	expected: Mapping[str, object]


@dataclass(frozen=True)
class VolveHorizonFiveWayConfig:
	'''Resolved common settings and five source identities for 75 jobs.'''

	artifact_root: Path
	volve_root: Path
	survey_id: str
	canonical_input_metadata: Path
	models: tuple[VolveHorizonFiveWayModelSource, ...]
	runs_root: Path
	summary_root: Path
	train: FrozenHorizonTrainSettings
	tiles: HorizonTileSettings

	@property
	def model_ids(self) -> tuple[str, ...]:
		'''Return the fixed scientific comparison order.'''
		return tuple(model.model_id for model in self.models)

	def model_by_id(self, model_id: str) -> VolveHorizonFiveWayModelSource:
		'''Return one configured model or reject an unknown identity.'''
		for model in self.models:
			if model.model_id == model_id:
				return model
		raise ValueError(
			f'unknown Volve horizon five-way model: {model_id!r}; '
			f'expected one of {list(FIVE_WAY_MODEL_IDS)!r}'
		)


def volve_horizon_five_way_config_from_mapping(
	config: Mapping[str, object],
) -> VolveHorizonFiveWayConfig:
	'''Resolve the strict five-model config without touching artifacts.'''
	_validate_exact_keys(config, _TOP_LEVEL_KEYS, 'config')
	paths = _required_mapping(config, 'paths', 'config')
	dataset = _required_mapping(config, 'dataset', 'config')
	inputs = _required_mapping(config, 'inputs', 'config')
	outputs = _required_mapping(config, 'outputs', 'config')
	_validate_exact_keys(paths, frozenset({'artifact_root', 'volve_root'}), 'paths')
	_validate_exact_keys(dataset, frozenset({'survey_id'}), 'dataset')
	_validate_exact_keys(
		inputs,
		frozenset({'canonical_input_metadata'}),
		'inputs',
	)
	_validate_exact_keys(
		outputs,
		frozenset({'runs_root', 'summary_root'}),
		'outputs',
	)

	artifact_root = _absolute_path(paths, 'artifact_root', 'paths')
	volve_root = _absolute_path(paths, 'volve_root', 'paths')
	models = _resolve_models(config.get('models'))
	runs_root = _absolute_path(outputs, 'runs_root', 'outputs')
	summary_root = _absolute_path(outputs, 'summary_root', 'outputs')
	if _same_path(runs_root, summary_root):
		raise ValueError('outputs.runs_root and outputs.summary_root must differ')
	for key, output_root in (
		('runs_root', runs_root),
		('summary_root', summary_root),
	):
		if not _is_relative_to(output_root, artifact_root):
			raise ValueError(f'outputs.{key} must be below paths.artifact_root')
		if _is_relative_to(output_root, volve_root):
			raise ValueError(
				f'outputs.{key} must not be below public paths.volve_root'
			)

	# Reuse the established Volve decoder/tile/train validators verbatim while
	# keeping the legacy parser and its public dataclasses unchanged.
	legacy_config = frozen_horizon_config_from_mapping(
		{
			'paths': dict(paths),
			'dataset': dict(dataset),
			'inputs': dict(inputs),
			'embeddings': {
				'pretrained_dir': str(models[0].embeddings_dir),
				'random_dir': str(models[-1].embeddings_dir),
			},
			'outputs': {'runs_root': str(runs_root)},
			'decoder': _required_mapping(config, 'decoder', 'config'),
			'tiles': _required_mapping(config, 'tiles', 'config'),
			'train': _required_mapping(config, 'train', 'config'),
		}
	)
	return VolveHorizonFiveWayConfig(
		artifact_root=artifact_root,
		volve_root=volve_root,
		survey_id=_non_empty_string(dataset.get('survey_id'), 'dataset.survey_id'),
		canonical_input_metadata=_absolute_path(
			inputs,
			'canonical_input_metadata',
			'inputs',
		),
		models=models,
		runs_root=runs_root,
		summary_root=summary_root,
		train=legacy_config.train,
		tiles=legacy_config.tiles,
	)


def _resolve_models(value: object) -> tuple[VolveHorizonFiveWayModelSource, ...]:
	if not isinstance(value, Mapping):
		raise TypeError('models must be a mapping keyed by fixed model ID')
	if set(value) != set(FIVE_WAY_MODEL_IDS):
		missing = sorted(set(FIVE_WAY_MODEL_IDS) - set(value))
		extra = sorted(set(value) - set(FIVE_WAY_MODEL_IDS))
		raise ValueError(
			'models must define exactly the five fixed model IDs; '
			f'missing={missing!r}, extra={extra!r}'
		)
	models = tuple(
		_resolve_model(model_id, value[model_id]) for model_id in FIVE_WAY_MODEL_IDS
	)
	checkpoint_identities = {
		model.checkpoint.resolve(strict=False) for model in models
	}
	embedding_identities = {
		model.embeddings_dir.resolve(strict=False) for model in models
	}
	if len(checkpoint_identities) != len(models):
		raise ValueError('model checkpoint paths must be distinct')
	if len(embedding_identities) != len(models):
		raise ValueError('model embedding directory paths must be distinct')
	return models


def _resolve_model(
	model_id: str,
	value: object,
) -> VolveHorizonFiveWayModelSource:
	if not isinstance(value, Mapping):
		raise TypeError(f'models.{model_id} must be a mapping')
	_validate_exact_keys(
		value,
		frozenset({'checkpoint', 'embeddings_dir'}),
		f'models.{model_id}',
	)
	checkpoint = _absolute_path(value, 'checkpoint', f'models.{model_id}')
	embeddings_dir = _absolute_path(value, 'embeddings_dir', f'models.{model_id}')
	for key, path in (
		('checkpoint', checkpoint),
		('embeddings_dir', embeddings_dir),
	):
		if 'trace_drop' in str(path).lower():
			raise ValueError(
				f'models.{model_id}.{key} must not reference a trace-drop artifact'
			)
	return VolveHorizonFiveWayModelSource(
		model_id=model_id,
		checkpoint=checkpoint,
		embeddings_dir=embeddings_dir,
		expected=dict(EXPECTED_MODEL_IDENTITIES[model_id]),
	)


def _required_mapping(
	value: Mapping[str, object],
	key: str,
	prefix: str,
) -> Mapping[str, object]:
	child = value.get(key)
	if not isinstance(child, Mapping):
		raise TypeError(f'{prefix}.{key} must be a mapping')
	return child


def _validate_exact_keys(
	value: Mapping[str, object],
	expected: frozenset[str],
	prefix: str,
) -> None:
	if set(value) == set(expected):
		return
	missing = sorted(set(expected) - set(value))
	extra = sorted(set(value) - set(expected))
	raise ValueError(
		f'{prefix} keys differ from the fixed contract; '
		f'missing={missing!r}, extra={extra!r}'
	)


def _absolute_path(
	value: Mapping[str, object],
	key: str,
	prefix: str,
) -> Path:
	path = Path(_non_empty_string(value.get(key), f'{prefix}.{key}'))
	if not path.is_absolute():
		raise ValueError(f'{prefix}.{key} must be absolute')
	return path


def _non_empty_string(value: object, label: str) -> str:
	if not isinstance(value, str) or not value:
		raise TypeError(f'{label} must be a non-empty string')
	return value


def _same_path(first: Path, second: Path) -> bool:
	return first.resolve(strict=False) == second.resolve(strict=False)


def _is_relative_to(path: Path, root: Path) -> bool:
	try:
		path.resolve(strict=False).relative_to(root.resolve(strict=False))
	except ValueError:
		return False
	return True


__all__ = [
	'EXPECTED_MODEL_IDENTITIES',
	'FIVE_WAY_HMM_K',
	'FIVE_WAY_MODEL_IDS',
	'FIVE_WAY_RANDOM_SEED',
	'FIVE_WAY_STAGE1_EPOCHS',
	'FIVE_WAY_STAGE2_EPOCHS',
	'FIVE_WAY_UNFREEZE_TOP_BLOCKS',
	'LOCAL_BARLOW_TWINS_METHOD',
	'LOCAL_BARLOW_TWINS_PAIRS_PER_CROP',
	'MAE_OBJECTIVE',
	'RANDOM_OBJECTIVE',
	'VOLVE_HORIZON_FIVE_WAY_MODEL_IDS',
	'VolveHorizonFiveWayConfig',
	'VolveHorizonFiveWayModelSource',
	'volve_horizon_five_way_config_from_mapping',
]

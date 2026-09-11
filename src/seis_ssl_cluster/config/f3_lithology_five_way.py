"""Strict configuration for the F3 lithology five-way frozen-encoder comparison."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from seis_ssl_cluster.config.f3_lithology_common import (
	_optional_str,
	_required_absolute_path,
	_required_mapping,
	_required_str,
	_validate_allowed_keys,
)

if TYPE_CHECKING:
	from pathlib import Path

FIVE_WAY_MODEL_IDS = (
	'mae',
	'mae_hmm_k6',
	'local_barlow_twins',
	'local_barlow_twins_hmm_k6',
	'random',
)
FIVE_WAY_RANDOM_SEED = 42
FIVE_WAY_HMM_K = 6
FIVE_WAY_MIN_CONFIDENCE = 0.0
FIVE_WAY_STAGE1_EPOCHS = 100
FIVE_WAY_STAGE2_EPOCHS = 25
FIVE_WAY_STAGE2_GLOBAL_STEPS = 15_625
FIVE_WAY_UNFREEZE_TOP_BLOCKS = 1
LOCAL_BARLOW_TWINS_METHOD = 'local_barlow_twins_3d'
LOCAL_BARLOW_TWINS_PAIRS_PER_CROP = 128
MAE_OBJECTIVE = 'amp_mae3d'
RANDOM_OBJECTIVE = 'random_encoder'
DEFAULT_FIVE_WAY_SUMMARY_NAME = 'f3_lithology_mae_local_bt_five_way_v1'
FIVE_WAY_LABEL_KEYS = (
	'source_label_volume',
	'source_label_segy',
	'png_label_inventory',
	'segy_geometry_json',
	'class_info',
)

EXPECTED_MODEL_IDENTITIES: Mapping[str, Mapping[str, object]] = {
	'mae': {
		'objective': MAE_OBJECTIVE,
		'stratigraphy_pretext': False,
	},
	'mae_hmm_k6': {
		'objective': MAE_OBJECTIVE,
		'stratigraphy_pretext': True,
		'base_objective': MAE_OBJECTIVE,
		'hmm_k': FIVE_WAY_HMM_K,
	},
	'local_barlow_twins': {
		'objective': LOCAL_BARLOW_TWINS_METHOD,
		'local_pairs_per_crop': LOCAL_BARLOW_TWINS_PAIRS_PER_CROP,
		'stratigraphy_pretext': False,
	},
	'local_barlow_twins_hmm_k6': {
		'objective': LOCAL_BARLOW_TWINS_METHOD,
		'stratigraphy_pretext': True,
		'base_objective': LOCAL_BARLOW_TWINS_METHOD,
		'hmm_k': FIVE_WAY_HMM_K,
	},
	'random': {
		'objective': RANDOM_OBJECTIVE,
		'random_seed': FIVE_WAY_RANDOM_SEED,
		'stratigraphy_pretext': False,
	},
}


@dataclass(frozen=True)
class F3FiveWayModelSource:
	"""One frozen encoder source of the five-way comparison."""

	model_id: str
	checkpoint: Path
	embeddings_dir: Path
	expected: Mapping[str, object]


@dataclass(frozen=True)
class F3FiveWayConfig:
	"""Resolved single source of truth for all 75 downstream jobs."""

	artifact_root: Path
	f3_root: Path
	dataset: Mapping[str, str]
	labels: Mapping[str, Path]
	section_layout_dataset_root: Path
	models: tuple[F3FiveWayModelSource, ...]
	runs_root: Path
	summary_root: Path
	summary_name: str = DEFAULT_FIVE_WAY_SUMMARY_NAME

	@property
	def model_ids(self) -> tuple[str, ...]:
		"""Return the fixed comparison order."""
		return tuple(model.model_id for model in self.models)

	def model_by_id(self, model_id: str) -> F3FiveWayModelSource:
		"""Return one configured source, rejecting unknown model IDs."""
		for model in self.models:
			if model.model_id == model_id:
				return model
		raise ValueError(
			f'unknown five-way model: {model_id!r}; '
			f'expected one of {list(FIVE_WAY_MODEL_IDS)!r}'
		)


def f3_lithology_five_way_config_from_mapping(
	config: Mapping[str, object],
) -> F3FiveWayConfig:
	"""Strictly resolve the canonical five-way comparison config."""
	_validate_allowed_keys(
		config,
		frozenset(
			{'paths', 'dataset', 'labels', 'section_layout', 'models', 'outputs'}
		),
		prefix='config',
	)
	paths = _required_mapping(config, 'paths')
	dataset = _required_mapping(config, 'dataset')
	labels = _required_mapping(config, 'labels')
	section_layout = _required_mapping(config, 'section_layout')
	outputs = _required_mapping(config, 'outputs')
	_validate_allowed_keys(
		paths, frozenset({'artifact_root', 'f3_root'}), prefix='paths'
	)
	_validate_allowed_keys(dataset, frozenset({'name', 'version'}), prefix='dataset')
	_validate_allowed_keys(
		labels, frozenset(FIVE_WAY_LABEL_KEYS), prefix='labels'
	)
	if set(labels) != set(FIVE_WAY_LABEL_KEYS):
		missing = sorted(set(FIVE_WAY_LABEL_KEYS) - set(labels))
		raise ValueError(f'labels must define every source; missing={missing!r}')
	_validate_allowed_keys(
		section_layout, frozenset({'dataset_root'}), prefix='section_layout'
	)
	_validate_allowed_keys(
		outputs,
		frozenset({'runs_root', 'summary_root', 'summary_name'}),
		prefix='outputs',
	)

	artifact_root = _required_absolute_path(paths, 'artifact_root', prefix='paths')
	models = _resolve_models(config.get('models'), artifact_root=artifact_root)
	runs_root = _required_absolute_path(outputs, 'runs_root', prefix='outputs')
	summary_root = _required_absolute_path(outputs, 'summary_root', prefix='outputs')
	if runs_root == summary_root:
		raise ValueError('outputs.runs_root and outputs.summary_root must differ')
	return F3FiveWayConfig(
		artifact_root=artifact_root,
		f3_root=_required_absolute_path(paths, 'f3_root', prefix='paths'),
		dataset={
			'name': _required_str(dataset, 'name', prefix='dataset'),
			'version': _required_str(dataset, 'version', prefix='dataset'),
		},
		labels={
			key: _required_absolute_path(labels, key, prefix='labels')
			for key in FIVE_WAY_LABEL_KEYS
		},
		section_layout_dataset_root=_required_absolute_path(
			section_layout, 'dataset_root', prefix='section_layout'
		),
		models=models,
		runs_root=runs_root,
		summary_root=summary_root,
		summary_name=_optional_str(
			outputs,
			'summary_name',
			default=DEFAULT_FIVE_WAY_SUMMARY_NAME,
			prefix='outputs',
		),
	)


def _resolve_models(
	value: object, *, artifact_root: Path
) -> tuple[F3FiveWayModelSource, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		raise TypeError('models must be a list of model source mappings')
	if len(value) != len(FIVE_WAY_MODEL_IDS):
		raise ValueError(
			f'models must contain exactly {len(FIVE_WAY_MODEL_IDS)} entries'
		)
	models = tuple(
		_resolve_model(item, index=index, artifact_root=artifact_root)
		for index, item in enumerate(value)
	)
	if tuple(model.model_id for model in models) != FIVE_WAY_MODEL_IDS:
		raise ValueError(
			'models must define exactly '
			f'{list(FIVE_WAY_MODEL_IDS)!r} in this order'
		)
	checkpoints = {model.checkpoint for model in models}
	embedding_dirs = {model.embeddings_dir for model in models}
	if len(checkpoints) != len(models) or len(embedding_dirs) != len(models):
		raise ValueError('model checkpoints and embedding dirs must be distinct')
	return models


def _resolve_model(
	value: object, *, index: int, artifact_root: Path
) -> F3FiveWayModelSource:
	if not isinstance(value, Mapping):
		raise TypeError(f'models[{index}] must be a mapping; got {value!r}')
	prefix = f'models[{index}]'
	_validate_allowed_keys(
		value,
		frozenset({'model_id', 'checkpoint', 'embeddings_dir', 'expected'}),
		prefix=prefix,
	)
	model_id = _required_str(value, 'model_id', prefix=prefix)
	if model_id not in EXPECTED_MODEL_IDENTITIES:
		raise ValueError(
			f'{prefix}.model_id must be one of {list(FIVE_WAY_MODEL_IDS)!r}; '
			f'got {model_id!r}'
		)
	expected = value.get('expected')
	if expected != EXPECTED_MODEL_IDENTITIES[model_id]:
		raise ValueError(
			f'{prefix}.expected must restate the fixed {model_id} identity '
			f'{dict(EXPECTED_MODEL_IDENTITIES[model_id])!r}'
		)
	checkpoint = _required_absolute_path(value, 'checkpoint', prefix=prefix)
	embeddings_dir = _required_absolute_path(value, 'embeddings_dir', prefix=prefix)
	for label, path in (
		('checkpoint', checkpoint),
		('embeddings_dir', embeddings_dir),
	):
		try:
			relevant = path.relative_to(artifact_root)
		except ValueError:
			relevant = path
		if 'trace_drop' in str(relevant):
			raise ValueError(
				f'{prefix}.{label} must not reference a trace-drop artifact'
			)
	return F3FiveWayModelSource(
		model_id=model_id,
		checkpoint=checkpoint,
		embeddings_dir=embeddings_dir,
		expected=dict(EXPECTED_MODEL_IDENTITIES[model_id]),
	)


__all__ = [
	'DEFAULT_FIVE_WAY_SUMMARY_NAME',
	'EXPECTED_MODEL_IDENTITIES',
	'FIVE_WAY_HMM_K',
	'FIVE_WAY_LABEL_KEYS',
	'FIVE_WAY_MIN_CONFIDENCE',
	'FIVE_WAY_MODEL_IDS',
	'FIVE_WAY_RANDOM_SEED',
	'FIVE_WAY_STAGE1_EPOCHS',
	'FIVE_WAY_STAGE2_EPOCHS',
	'FIVE_WAY_STAGE2_GLOBAL_STEPS',
	'FIVE_WAY_UNFREEZE_TOP_BLOCKS',
	'LOCAL_BARLOW_TWINS_METHOD',
	'LOCAL_BARLOW_TWINS_PAIRS_PER_CROP',
	'MAE_OBJECTIVE',
	'RANDOM_OBJECTIVE',
	'F3FiveWayConfig',
	'F3FiveWayModelSource',
	'f3_lithology_five_way_config_from_mapping',
]

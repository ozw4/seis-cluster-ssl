"""Strict configuration for the center-trace masked six-split preflight."""
# ruff: noqa: CPY001, C901, E501

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from seis_ssl_cluster.config.f3_lithology_common import (
	_required_absolute_path,
	_required_mapping,
	_required_str,
	_validate_allowed_keys,
)
from seis_ssl_cluster.config.io import load_config

SPLIT_IDS = tuple(f'split_{index:03d}' for index in range(6))
BUDGETS = ('cap25', 'cap50', 'cap100')
CANDIDATE_MODEL_ID = 'mh_ctmask010_nocons'
CANDIDATE_MODEL_TAG = (
	'strat_hmm_pretext_mh_k6810_ctmask010_nocons_topblock1_distill_v1'
)
BASELINE_MODEL_ID = 'mh_nocons'
BASELINE_MODEL_TAG = 'strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1'
PRIMARY_MODEL_ROLES = (CANDIDATE_MODEL_ID, BASELINE_MODEL_ID)
PRIMARY_METRICS = ('macro_f1', 'mean_iou')
MONITORED_CLASS_IDS = (3, 5)
MONITORED_CLASS_METRICS = (
	'f1',
	'iou',
	'boundary_recall_t2',
	'boundary_recall_t4',
)
EXPECTED_OUTPUT_RELATIVE_PATH = Path(
	'lithology/f3/facies_benchmark_v1/'
	'voxel_label_budget_center_trace_masked_k6810_six_split_v1'
)
SOURCE_IDENTITY_KEYS = (
	'seismic_volume',
	'source_label_segy',
	'class_info',
	'segy_geometry_json',
)

_CONFIG_KEYS = frozenset({'paths', 'matrix', 'inputs'})
_PATH_KEYS = frozenset({'artifact_root', 'output_root'})
_MATRIX_KEYS = frozenset(
	{
		'candidate',
		'baseline',
		'split_ids',
		'budgets',
		'label_subset_seed',
		'decoder_seed',
	}
)
_MODEL_KEYS = frozenset({'model_id', 'model_tag'})
_INPUT_KEYS = frozenset(
	{
		'original_split_handoff',
		'candidate_pretraining_handoff',
		'candidate_embeddings_dir',
		'hard_baseline_pretraining_handoff',
		'hard_baseline_embeddings_dir',
		'experiment96_dataset_manifest',
		'experiment96_scientific_run_manifest',
		'split_inventory_manifest',
		'split_token_dataset_manifest',
		'full_voxel_split_dataset_manifest',
		'original_split_dataset_manifest',
		'seismic_volume',
		'source_label_segy',
		'class_info',
		'segy_geometry_json',
		'source_identities',
	}
)


@dataclass(frozen=True)
class F3CenterTraceMaskedSixSplitConfig:
	"""Immutable inputs and the pre-registered six-split matrix."""

	artifact_root: Path
	output_root: Path
	original_split_handoff: Path
	candidate_pretraining_handoff: Path
	candidate_embeddings_dir: Path
	hard_baseline_pretraining_handoff: Path
	hard_baseline_embeddings_dir: Path
	experiment96_dataset_manifest: Path
	experiment96_scientific_run_manifest: Path
	split_inventory_manifest: Path
	split_token_dataset_manifest: Path
	full_voxel_split_dataset_manifest: Path
	original_split_dataset_manifest: Path
	seismic_volume: Path
	source_label_segy: Path
	class_info: Path
	segy_geometry_json: Path
	source_identities: Mapping[str, Mapping[str, str]]
	candidate_model_id: str
	candidate_model_tag: str
	baseline_model_id: str
	baseline_model_tag: str
	split_ids: tuple[str, ...]
	budgets: tuple[str, ...]
	label_subset_seed: int
	decoder_seed: int

	@property
	def primary_model_roles(self) -> tuple[str, str]:
		"""Return candidate then primary-baseline roles."""
		return (self.candidate_model_id, self.baseline_model_id)

	@property
	def primary_model_tags(self) -> Mapping[str, str]:
		"""Return the fixed role-to-tag binding."""
		return {
			self.candidate_model_id: self.candidate_model_tag,
			self.baseline_model_id: self.baseline_model_tag,
		}

	@property
	def primary_matrix_row_count(self) -> int:
		"""Return the six-split by budget by role row count."""
		return len(self.split_ids) * len(self.budgets) * len(PRIMARY_MODEL_ROLES)

	@property
	def future_candidate_jobs(self) -> int:
		"""Return the candidate-owned future job count."""
		return len(self.split_ids) * len(self.budgets)

	@property
	def future_new_baseline_jobs(self) -> int:
		"""Return the cap100 baseline additions."""
		return len(self.split_ids)

	@property
	def historical_baseline_rows(self) -> int:
		"""Return the experiment-96 primary baseline rows."""
		return len(self.split_ids) * 2

	@property
	def future_new_scientific_jobs(self) -> int:
		"""Return the candidate and cap100 baseline additions."""
		return self.future_candidate_jobs + self.future_new_baseline_jobs

	@property
	def audit_output_path(self) -> Path:
		"""Return the only output owned by this preflight."""
		return self.output_root / 'preflight/center_trace_masked_six_split_audit.json'

	@property
	def source_paths(self) -> Mapping[str, Path]:
		"""Return the four immutable source files in contract order."""
		return {
			'seismic_volume': self.seismic_volume,
			'source_label_segy': self.source_label_segy,
			'class_info': self.class_info,
			'segy_geometry_json': self.segy_geometry_json,
		}


def f3_lithology_voxel_label_budget_center_trace_masked_split_config_from_mapping(
	config: Mapping[str, object],
) -> F3CenterTraceMaskedSixSplitConfig:
	"""Resolve the closed six-split contract and reject every unknown key."""
	if not isinstance(config, Mapping):
		raise TypeError('center-trace masked six-split config must be a mapping')
	_validate_allowed_keys(config, _CONFIG_KEYS, prefix='config')
	paths = _required_mapping(config, 'paths')
	matrix = _required_mapping(config, 'matrix')
	inputs = _required_mapping(config, 'inputs')
	_validate_allowed_keys(paths, _PATH_KEYS, prefix='paths')
	_validate_allowed_keys(matrix, _MATRIX_KEYS, prefix='matrix')
	_validate_allowed_keys(inputs, _INPUT_KEYS, prefix='inputs')

	artifact_root = _required_absolute_path(paths, 'artifact_root', prefix='paths')
	output_root = _required_absolute_path(paths, 'output_root', prefix='paths')
	if output_root != (artifact_root / EXPECTED_OUTPUT_RELATIVE_PATH).resolve(
		strict=False
	):
		raise ValueError(
			'paths.output_root must be the candidate-owned six-split output root'
		)

	candidate = _model_identity(matrix, 'candidate')
	baseline = _model_identity(matrix, 'baseline')
	if candidate != (CANDIDATE_MODEL_ID, CANDIDATE_MODEL_TAG):
		raise ValueError('matrix.candidate identity is not the fixed candidate')
	if baseline != (BASELINE_MODEL_ID, BASELINE_MODEL_TAG):
		raise ValueError('matrix.baseline identity is not the fixed baseline')
	if candidate[0] == baseline[0]:
		raise ValueError('candidate and baseline model IDs must be distinct')

	split_ids = _strings(matrix.get('split_ids'), 'matrix.split_ids')
	budgets = _strings(matrix.get('budgets'), 'matrix.budgets')
	label_subset_seed = _integer(
		matrix.get('label_subset_seed'), 'matrix.label_subset_seed'
	)
	decoder_seed = _integer(matrix.get('decoder_seed'), 'matrix.decoder_seed')
	if split_ids != SPLIT_IDS or len(set(split_ids)) != len(SPLIT_IDS):
		raise ValueError('matrix.split_ids must be the canonical six unique split IDs')
	if budgets != BUDGETS or len(set(budgets)) != len(BUDGETS):
		raise ValueError('matrix.budgets must be cap25, cap50, cap100')
	if label_subset_seed != 0 or decoder_seed != 42000:
		raise ValueError(
		'matrix seeds must be label_subset_seed=0 and decoder_seed=42000'
	)

	resolved_inputs = {
		name: _required_absolute_path(inputs, name, prefix='inputs')
		for name in _INPUT_KEYS
		if name != 'source_identities'
	}
	for name in ('candidate_embeddings_dir', 'hard_baseline_embeddings_dir'):
		if not resolved_inputs[name].is_absolute():
			raise ValueError(f'inputs.{name} must be absolute')

	source_identities = _source_identities(inputs, resolved_inputs)
	return F3CenterTraceMaskedSixSplitConfig(
		artifact_root=artifact_root,
		output_root=output_root,
		original_split_handoff=resolved_inputs['original_split_handoff'],
		candidate_pretraining_handoff=resolved_inputs[
			'candidate_pretraining_handoff'
		],
		candidate_embeddings_dir=resolved_inputs['candidate_embeddings_dir'],
		hard_baseline_pretraining_handoff=resolved_inputs[
			'hard_baseline_pretraining_handoff'
		],
		hard_baseline_embeddings_dir=resolved_inputs[
			'hard_baseline_embeddings_dir'
		],
		experiment96_dataset_manifest=resolved_inputs[
			'experiment96_dataset_manifest'
		],
		experiment96_scientific_run_manifest=resolved_inputs[
			'experiment96_scientific_run_manifest'
		],
		split_inventory_manifest=resolved_inputs['split_inventory_manifest'],
		split_token_dataset_manifest=resolved_inputs[
			'split_token_dataset_manifest'
		],
		full_voxel_split_dataset_manifest=resolved_inputs[
			'full_voxel_split_dataset_manifest'
		],
		original_split_dataset_manifest=resolved_inputs[
			'original_split_dataset_manifest'
		],
		seismic_volume=resolved_inputs['seismic_volume'],
		source_label_segy=resolved_inputs['source_label_segy'],
		class_info=resolved_inputs['class_info'],
		segy_geometry_json=resolved_inputs['segy_geometry_json'],
		source_identities=source_identities,
		candidate_model_id=candidate[0],
		candidate_model_tag=candidate[1],
		baseline_model_id=baseline[0],
		baseline_model_tag=baseline[1],
		split_ids=split_ids,
		budgets=budgets,
		label_subset_seed=label_subset_seed,
		decoder_seed=decoder_seed,
	)


def load_f3_lithology_voxel_label_budget_center_trace_masked_split_config(
	path: str | Path,
) -> F3CenterTraceMaskedSixSplitConfig:
	"""Load and resolve the six-split audit YAML."""
	return f3_lithology_voxel_label_budget_center_trace_masked_split_config_from_mapping(
		load_config(path)
	)


def _model_identity(
	matrix: Mapping[str, object], name: str
) -> tuple[str, str]:
	value = _required_mapping(matrix, name)
	_validate_allowed_keys(value, _MODEL_KEYS, prefix=f'matrix.{name}')
	return (
		_required_str(value, 'model_id', prefix=f'matrix.{name}'),
		_required_str(value, 'model_tag', prefix=f'matrix.{name}'),
	)


def _source_identities(
	inputs: Mapping[str, object], paths: Mapping[str, Path]
) -> Mapping[str, Mapping[str, str]]:
	raw = _required_mapping(inputs, 'source_identities')
	_validate_allowed_keys(
		raw, frozenset(SOURCE_IDENTITY_KEYS), prefix='inputs.source_identities'
	)
	result: dict[str, dict[str, str]] = {}
	for name in SOURCE_IDENTITY_KEYS:
		identity = _required_mapping(raw, name)
		_validate_allowed_keys(
			identity,
			frozenset({'path', 'sha256'}),
			prefix=f'inputs.source_identities.{name}',
		)
		path = _required_absolute_path(
			identity, 'path', prefix=f'inputs.source_identities.{name}'
		)
		if path != paths[name]:
			raise ValueError(
				f'inputs.source_identities.{name}.path must match inputs.{name}'
			)
		sha256 = _required_str(
			identity, 'sha256', prefix=f'inputs.source_identities.{name}'
		)
		if len(sha256) != 64 or any(
			character not in '0123456789abcdef' for character in sha256
		):
			raise ValueError(
				f'inputs.source_identities.{name}.sha256 must be a lowercase SHA-256'
			)
		result[name] = {'path': str(path), 'sha256': sha256}
	return result


def _strings(value: object, label: str) -> tuple[str, ...]:
	if not isinstance(value, list) or not all(
		isinstance(item, str) and item for item in value
	):
		raise TypeError(f'{label} must be a list of non-empty strings')
	return tuple(value)


def _integer(value: object, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool):
		raise TypeError(f'{label} must be an integer')
	return value


config_from_mapping = (
	f3_lithology_voxel_label_budget_center_trace_masked_split_config_from_mapping
)


__all__ = [
	'BASELINE_MODEL_ID',
	'BASELINE_MODEL_TAG',
	'BUDGETS',
	'CANDIDATE_MODEL_ID',
	'CANDIDATE_MODEL_TAG',
	'MONITORED_CLASS_IDS',
	'MONITORED_CLASS_METRICS',
	'PRIMARY_METRICS',
	'PRIMARY_MODEL_ROLES',
	'SPLIT_IDS',
	'F3CenterTraceMaskedSixSplitConfig',
	'config_from_mapping',
	'f3_lithology_voxel_label_budget_center_trace_masked_split_config_from_mapping',
	'load_f3_lithology_voxel_label_budget_center_trace_masked_split_config',
]

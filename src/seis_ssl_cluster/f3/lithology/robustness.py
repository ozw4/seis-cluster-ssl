"""Contracts and paired dataset helpers for F3 lithology robustness suites."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from seis_ssl_cluster.f3.lithology.token_dataset import (
	F3LithologyTokenDataset,
	load_f3_lithology_token_dataset,
	save_f3_lithology_token_dataset,
	validate_f3_lithology_token_dataset,
)

if TYPE_CHECKING:
	from numpy.typing import NDArray

F3_ROBUSTNESS_CONTRACT_VERSION = 'f3_lithology_robustness_m1_v1'
F3_M1_MODEL_ROLES = frozenset({'baseline', 'candidate'})
F3_M1_DISALLOWED_SUITE_NAMES = frozenset({'seed_sweep'})
F3_M1_DISALLOWED_CONFIG_KEYS = frozenset({'checkpoint_policy'})
F3_PAIRED_DELTA_FIELDS = (
	'delta_macro_f1',
	'delta_mean_iou',
	'delta_balanced_accuracy',
	'delta_accuracy',
)


@dataclass(frozen=True)
class F3RobustnessModelSpec:
	"""One encoder identity participating in a paired robustness comparison."""

	model_tag: str
	role: str
	embedding_spec: str
	label_set: str
	probe_spec: str

	def __post_init__(self) -> None:
		"""Validate model identity fields."""
		_validate_non_empty_str(self.model_tag, 'model_tag')
		_validate_non_empty_str(self.role, 'role')
		_validate_non_empty_str(self.embedding_spec, 'embedding_spec')
		_validate_non_empty_str(self.label_set, 'label_set')
		_validate_non_empty_str(self.probe_spec, 'probe_spec')
		if self.role not in F3_M1_MODEL_ROLES:
			msg = (
				f'role must be one of {sorted(F3_M1_MODEL_ROLES)!r}; '
				f'got {self.role!r}'
			)
			raise ValueError(msg)


@dataclass(frozen=True)
class F3LabelBudgetSpec:
	"""Named small-label condition for a paired label-budget suite."""

	budget_id: str
	train_fraction: float | None = None
	max_train_tokens: int | None = None
	subsample_seeds: tuple[int, ...] = ()

	def __post_init__(self) -> None:
		"""Validate label-budget fields without generating a subsample."""
		_validate_non_empty_str(self.budget_id, 'budget_id')
		if self.train_fraction is None and self.max_train_tokens is None:
			msg = 'label budget requires train_fraction or max_train_tokens'
			raise ValueError(msg)
		if self.train_fraction is not None:
			_validate_fraction(self.train_fraction, 'train_fraction')
		if self.max_train_tokens is not None:
			_validate_positive_int(self.max_train_tokens, 'max_train_tokens')
		_validate_int_sequence(self.subsample_seeds, 'subsample_seeds')


@dataclass(frozen=True)
class F3RobustnessSuiteManifest:
	"""Resolved, versioned manifest for one F3 M1 robustness suite."""

	suite_name: str
	contract_version: str
	output_root: Path
	models: tuple[F3RobustnessModelSpec, ...]
	label_budgets: tuple[F3LabelBudgetSpec, ...] = ()
	split_ids: tuple[str, ...] = ()
	report_paths: Mapping[str, Path] = field(default_factory=dict)

	def __post_init__(self) -> None:
		"""Validate suite-level invariants shared by B and C suites."""
		validate_f3_robustness_suite_name(self.suite_name)
		_validate_non_empty_str(self.contract_version, 'contract_version')
		validate_f3_robustness_output_root(self.output_root)
		validate_f3_m1_model_pair(self.models)
		_validate_str_sequence(self.split_ids, 'split_ids')
		_validate_report_paths(self.report_paths)


@dataclass(frozen=True)
class F3PairedMetricRow:
	"""One paired metric row comparing strat-HMM against MAE."""

	suite_name: str
	condition_id: str
	baseline_model_tag: str
	candidate_model_tag: str
	metric_name: str
	baseline_value: float
	candidate_value: float
	delta_value: float

	def __post_init__(self) -> None:
		"""Validate paired row identity fields."""
		validate_f3_robustness_suite_name(self.suite_name)
		_validate_non_empty_str(self.condition_id, 'condition_id')
		_validate_non_empty_str(self.baseline_model_tag, 'baseline_model_tag')
		_validate_non_empty_str(self.candidate_model_tag, 'candidate_model_tag')
		_validate_non_empty_str(self.metric_name, 'metric_name')


def load_token_dataset_npz(path: str | Path) -> F3LithologyTokenDataset:
	"""Load an F3 lithology token dataset NPZ using the shared schema loader."""
	return load_f3_lithology_token_dataset(Path(path))


def save_token_dataset_npz(
	dataset: F3LithologyTokenDataset,
	path: str | Path,
) -> None:
	"""Save an F3 lithology token dataset NPZ using the shared schema writer."""
	save_f3_lithology_token_dataset(dataset, Path(path))


def token_identity_frame(dataset: F3LithologyTokenDataset) -> NDArray[np.generic]:
	"""Return canonical row identity fields for paired token dataset checks."""
	validate_f3_lithology_token_dataset(dataset)
	count = dataset.count
	frame = np.empty(
		count,
		dtype=[
			('survey_id', object),
			('split', object),
			('slice_type', object),
			('slice_index', np.int64),
			('token_xyz', np.int64, (3,)),
			('labels', np.int64),
			('majority_fraction', np.float32),
			('labeled_fraction', np.float32),
		],
	)
	frame['survey_id'] = np.asarray(dataset.survey_id, dtype=str)
	frame['split'] = np.asarray(dataset.split, dtype=str)
	frame['slice_type'] = np.asarray(dataset.slice_type, dtype=str)
	frame['slice_index'] = np.asarray(dataset.slice_index, dtype=np.int64)
	frame['token_xyz'] = np.asarray(dataset.token_xyz, dtype=np.int64)
	frame['labels'] = np.asarray(dataset.labels, dtype=np.int64)
	frame['majority_fraction'] = np.asarray(
		dataset.majority_fraction,
		dtype=np.float32,
	)
	frame['labeled_fraction'] = np.asarray(
		dataset.labeled_fraction,
		dtype=np.float32,
	)
	return frame


def assert_same_token_identity(
	reference: F3LithologyTokenDataset,
	candidate: F3LithologyTokenDataset,
	*,
	reference_label: str,
	candidate_label: str,
) -> None:
	"""Raise if two token datasets do not have identical paired row identity."""
	reference_frame = token_identity_frame(reference)
	candidate_frame = token_identity_frame(candidate)
	if reference_frame.shape != candidate_frame.shape:
		msg = (
			'token identity row counts differ between '
			f'{reference_label} and {candidate_label}; '
			f'{reference_label}={reference_frame.shape[0]}, '
			f'{candidate_label}={candidate_frame.shape[0]}'
		)
		raise ValueError(msg)
	if np.array_equal(reference_frame, candidate_frame):
		return
	field_mismatches = [
		name
		for name in reference_frame.dtype.names or ()
		if not np.array_equal(reference_frame[name], candidate_frame[name])
	]
	first_index = _first_identity_mismatch_index(reference_frame, candidate_frame)
	msg = (
		'token identity rows differ between '
		f'{reference_label} and {candidate_label}; '
		f'fields={field_mismatches!r}, first_mismatch_index={first_index}'
	)
	raise ValueError(msg)


def class_stratified_subset_indices(
	labels: NDArray[np.generic],
	*,
	per_class_cap: int | None,
	seed: int,
	class_ids: Sequence[int] | None = None,
	require_all_classes: bool = True,
) -> NDArray[np.int64]:
	"""Select deterministic class-stratified subset row indices."""
	label_array = _label_vector(labels)
	if not isinstance(seed, int) or isinstance(seed, bool):
		msg = f'seed must be an integer; got {seed!r}'
		raise TypeError(msg)
	if per_class_cap is None:
		return np.arange(label_array.shape[0], dtype=np.int64)
	_validate_positive_int(per_class_cap, 'per_class_cap')
	classes = _class_ids(label_array, class_ids)
	rng = np.random.default_rng(seed)
	selected: list[NDArray[np.int64]] = []
	for class_id in classes:
		class_indices = np.flatnonzero(label_array == class_id).astype(np.int64)
		if class_indices.size == 0:
			if require_all_classes:
				msg = f'class_id {int(class_id)} has zero rows'
				raise ValueError(msg)
			continue
		if class_indices.size <= per_class_cap:
			selected.append(class_indices)
			continue
		selected.append(
			np.asarray(
				rng.choice(class_indices, size=per_class_cap, replace=False),
				dtype=np.int64,
			),
		)
	if not selected:
		return np.asarray([], dtype=np.int64)
	return np.sort(np.concatenate(selected).astype(np.int64))


def subset_token_dataset(
	dataset: F3LithologyTokenDataset,
	indices: NDArray[np.generic],
) -> F3LithologyTokenDataset:
	"""Return a token dataset subset preserving all provenance arrays."""
	validate_f3_lithology_token_dataset(dataset)
	index_array = _index_vector(indices, dataset.count)
	subset = F3LithologyTokenDataset(
		features=np.asarray(dataset.features)[index_array],
		labels=np.asarray(dataset.labels)[index_array],
		survey_id=np.asarray(dataset.survey_id)[index_array],
		split=np.asarray(dataset.split)[index_array],
		slice_type=np.asarray(dataset.slice_type)[index_array],
		slice_index=np.asarray(dataset.slice_index)[index_array],
		token_xyz=np.asarray(dataset.token_xyz)[index_array],
		voxel_center_xyz=np.asarray(dataset.voxel_center_xyz)[index_array],
		majority_fraction=np.asarray(dataset.majority_fraction)[index_array],
		labeled_fraction=np.asarray(dataset.labeled_fraction)[index_array],
		metadata=dict(dataset.metadata),
	)
	validate_f3_lithology_token_dataset(subset)
	return subset


def class_count_dict(labels: NDArray[np.generic]) -> dict[str, int]:
	"""Return JSON-safe class count mapping keyed by class id string."""
	label_array = _label_vector(labels)
	class_ids, counts = np.unique(label_array, return_counts=True)
	return {
		str(int(class_id)): int(count)
		for class_id, count in zip(class_ids, counts, strict=True)
	}


def paired_token_identity_hash(
	*datasets: F3LithologyTokenDataset,
) -> str:
	"""Return a SHA256 hash for one or more token identity frames."""
	hasher = hashlib.sha256()
	for dataset in datasets:
		_update_hash_with_identity_frame(hasher, token_identity_frame(dataset))
	return hasher.hexdigest()


def budget_subset_metadata(  # noqa: PLR0913
	*,
	source_train_tokens: str | Path,
	source_validation_tokens: str | Path,
	per_class_cap: int | None,
	subsample_seed: int,
	selected_train_dataset: F3LithologyTokenDataset | None = None,
	validation_dataset: F3LithologyTokenDataset | None = None,
	selected_train_labels: NDArray[np.generic] | None = None,
	validation_labels: NDArray[np.generic] | None = None,
	paired_identity_hash: str | None = None,
) -> dict[str, object]:
	"""Return JSON-safe metadata for a paired label-budget token subset."""
	if selected_train_dataset is not None:
		selected_train_labels = selected_train_dataset.labels
	if validation_dataset is not None:
		validation_labels = validation_dataset.labels
	if selected_train_labels is None:
		msg = 'selected_train_labels or selected_train_dataset is required'
		raise ValueError(msg)
	if validation_labels is None:
		msg = 'validation_labels or validation_dataset is required'
		raise ValueError(msg)
	train_labels = _label_vector(selected_train_labels)
	valid_labels = _label_vector(validation_labels)
	if paired_identity_hash is None:
		if selected_train_dataset is None or validation_dataset is None:
			msg = (
				'paired_identity_hash requires selected_train_dataset and '
				'validation_dataset when not provided explicitly'
			)
			raise ValueError(msg)
		paired_identity_hash = paired_token_identity_hash(
			selected_train_dataset,
			validation_dataset,
		)
	return {
		'source_train_tokens': str(source_train_tokens),
		'source_validation_tokens': str(source_validation_tokens),
		'per_class_cap': None if per_class_cap is None else int(per_class_cap),
		'subsample_seed': int(subsample_seed),
		'selected_train_token_count': int(train_labels.shape[0]),
		'validation_token_count': int(valid_labels.shape[0]),
		'selected_class_counts': class_count_dict(train_labels),
		'validation_class_counts': class_count_dict(valid_labels),
		'paired_identity_hash': str(paired_identity_hash),
	}


def validate_f3_robustness_suite_name(suite_name: str) -> None:
	"""Validate a robustness suite name."""
	_validate_non_empty_str(suite_name, 'suite_name')
	if suite_name in F3_M1_DISALLOWED_SUITE_NAMES:
		msg = (
			f'suite_name {suite_name!r} is out of scope for F3 M1 robustness; '
			'label-budget and split/index suites are the supported contracts'
		)
		raise ValueError(msg)


def validate_f3_robustness_output_root(output_root: Path) -> None:
	"""Validate that a robustness output root is absolute."""
	path = Path(output_root)
	if not path.is_absolute():
		msg = f'output_root must be an absolute path; got {path}'
		raise ValueError(msg)


def validate_f3_m1_model_pair(
	models: Sequence[F3RobustnessModelSpec],
) -> None:
	"""Validate the F3 M1 paired comparison model contract."""
	if isinstance(models, str | bytes):
		msg = f'models must be a sequence of model specs; got {models!r}'
		raise TypeError(msg)
	model_tuple = tuple(models)
	if len(model_tuple) != 2:
		msg = (
			'F3 M1 paired comparisons require exactly one baseline and one '
			f'candidate model; got {len(model_tuple)}'
		)
		raise ValueError(msg)
	roles = [model.role for model in model_tuple]
	if sorted(roles) != ['baseline', 'candidate']:
		msg = (
			'F3 M1 paired comparisons require model roles '
			f"['baseline', 'candidate']; got {roles!r}"
		)
		raise ValueError(msg)


def validate_f3_robustness_config_keys(
	config: Mapping[str, object],
	*,
	prefix: str = 'config',
) -> None:
	"""Reject suite keys that are explicitly out of scope for F3 M1 robustness."""
	if not isinstance(config, Mapping):
		msg = f'{prefix} must be a mapping; got {config!r}'
		raise TypeError(msg)
	for key, value in config.items():
		if key in F3_M1_DISALLOWED_CONFIG_KEYS:
			msg = (
				f'{prefix}.{key} is out of scope for F3 M1 robustness; '
				'checkpoint-selection policy is not part of this contract'
			)
			raise ValueError(msg)
		if key == 'suite_name' and isinstance(value, str):
			validate_f3_robustness_suite_name(value)
		if isinstance(value, Mapping):
			validate_f3_robustness_config_keys(value, prefix=f'{prefix}.{key}')


def _validate_non_empty_str(value: object, label: str) -> str:
	if not isinstance(value, str) or not value:
		msg = f'{label} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return value


def _validate_fraction(value: object, label: str) -> float:
	if not isinstance(value, int | float) or isinstance(value, bool):
		msg = f'{label} must be a number in (0, 1]; got {value!r}'
		raise TypeError(msg)
	fraction = float(value)
	if not 0.0 < fraction <= 1.0:
		msg = f'{label} must be in (0, 1]; got {value!r}'
		raise ValueError(msg)
	return fraction


def _validate_positive_int(value: object, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
		msg = f'{label} must be a positive integer; got {value!r}'
		raise ValueError(msg)
	return value


def _validate_int_sequence(values: object, label: str) -> tuple[int, ...]:
	if not isinstance(values, Sequence) or isinstance(values, str | bytes):
		msg = f'{label} must be a sequence of integers; got {values!r}'
		raise TypeError(msg)
	items = tuple(values)
	if not all(isinstance(item, int) and not isinstance(item, bool) for item in items):
		msg = f'{label} must contain only integers; got {values!r}'
		raise TypeError(msg)
	return items


def _validate_str_sequence(values: object, label: str) -> tuple[str, ...]:
	if not isinstance(values, Sequence) or isinstance(values, str | bytes):
		msg = f'{label} must be a sequence of strings; got {values!r}'
		raise TypeError(msg)
	items = tuple(values)
	for item in items:
		_validate_non_empty_str(item, label)
	return items


def _validate_report_paths(paths: Mapping[str, Path]) -> None:
	if not isinstance(paths, Mapping):
		msg = f'report_paths must be a mapping; got {paths!r}'
		raise TypeError(msg)
	for key, value in paths.items():
		_validate_non_empty_str(key, 'report_paths key')
		if not isinstance(value, Path):
			msg = f'report_paths[{key!r}] must be a Path; got {value!r}'
			raise TypeError(msg)


def _label_vector(labels: NDArray[np.generic]) -> NDArray[np.int64]:
	array = np.asarray(labels)
	if array.ndim != 1:
		msg = f'labels must be a 1D vector; got shape={array.shape!r}'
		raise ValueError(msg)
	if not np.issubdtype(array.dtype, np.integer):
		msg = f'labels must be integer typed; got dtype={array.dtype}'
		raise TypeError(msg)
	return np.asarray(array, dtype=np.int64)


def _index_vector(indices: NDArray[np.generic], count: int) -> NDArray[np.int64]:
	array = np.asarray(indices)
	if array.ndim != 1:
		msg = f'indices must be a 1D vector; got shape={array.shape!r}'
		raise ValueError(msg)
	if not np.issubdtype(array.dtype, np.integer):
		msg = f'indices must be integer typed; got dtype={array.dtype}'
		raise TypeError(msg)
	index_array = np.asarray(array, dtype=np.int64)
	if np.any(index_array < 0) or np.any(index_array >= count):
		msg = f'indices must be in [0, {count}); got {index_array.tolist()!r}'
		raise IndexError(msg)
	return index_array


def _class_ids(
	labels: NDArray[np.int64],
	class_ids: Sequence[int] | None,
) -> NDArray[np.int64]:
	if class_ids is None:
		return np.asarray(np.unique(labels), dtype=np.int64)
	if isinstance(class_ids, str | bytes):
		msg = f'class_ids must be a sequence of integers; got {class_ids!r}'
		raise TypeError(msg)
	classes = np.asarray(list(class_ids))
	if classes.ndim != 1:
		msg = f'class_ids must be a 1D sequence; got shape={classes.shape!r}'
		raise ValueError(msg)
	if not np.issubdtype(classes.dtype, np.integer):
		msg = f'class_ids must contain only integers; got dtype={classes.dtype}'
		raise TypeError(msg)
	return np.asarray(classes, dtype=np.int64)


def _first_identity_mismatch_index(
	reference_frame: NDArray[np.generic],
	candidate_frame: NDArray[np.generic],
) -> int:
	for index, (reference_row, candidate_row) in enumerate(
		zip(reference_frame, candidate_frame, strict=True),
	):
		if reference_row != candidate_row:
			return index
	return -1


def _update_hash_with_identity_frame(
	hasher: hashlib._Hash,
	frame: NDArray[np.generic],
) -> None:
	hasher.update(b'f3_lithology_token_identity_v1')
	hasher.update(json.dumps(frame.dtype.descr, separators=(',', ':')).encode())
	hasher.update(json.dumps(frame.shape, separators=(',', ':')).encode())
	hasher.update(
		json.dumps(
			_identity_records(frame),
			allow_nan=False,
			sort_keys=True,
			separators=(',', ':'),
		).encode(),
	)


def _identity_records(frame: NDArray[np.generic]) -> list[dict[str, object]]:
	names = frame.dtype.names or ()
	records: list[dict[str, object]] = []
	for row in frame:
		record: dict[str, object] = {}
		for name in names:
			record[name] = _json_ready_value(row[name])
		records.append(record)
	return records


def _json_ready_value(value: object) -> object:
	array = np.asarray(value)
	if array.shape != ():
		return [_json_ready_value(item) for item in array.tolist()]
	if isinstance(value, np.generic):
		value = value.item()
	if isinstance(value, bytes):
		return value.decode('utf-8')
	if isinstance(value, int | float | str) or value is None:
		return value
	return str(value)


__all__ = [
	'F3_M1_DISALLOWED_CONFIG_KEYS',
	'F3_M1_DISALLOWED_SUITE_NAMES',
	'F3_M1_MODEL_ROLES',
	'F3_PAIRED_DELTA_FIELDS',
	'F3_ROBUSTNESS_CONTRACT_VERSION',
	'F3LabelBudgetSpec',
	'F3PairedMetricRow',
	'F3RobustnessModelSpec',
	'F3RobustnessSuiteManifest',
	'assert_same_token_identity',
	'budget_subset_metadata',
	'class_count_dict',
	'class_stratified_subset_indices',
	'load_token_dataset_npz',
	'paired_token_identity_hash',
	'save_token_dataset_npz',
	'subset_token_dataset',
	'token_identity_frame',
	'validate_f3_m1_model_pair',
	'validate_f3_robustness_config_keys',
	'validate_f3_robustness_output_root',
	'validate_f3_robustness_suite_name',
]

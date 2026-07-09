"""Contracts and paired dataset helpers for F3 lithology robustness suites."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from seis_ssl_cluster.embedding.sliding_window import token_grid_shape_xyz
from seis_ssl_cluster.f3.lithology.token_dataset import (
	F3LithologyTokenDataset,
	load_f3_lithology_token_dataset,
	save_f3_lithology_token_dataset,
	validate_f3_lithology_token_dataset,
)
from seis_ssl_cluster.f3.lithology.tokens import (
	F3LithologyTokenPolicy,
	read_f3_lithology_class_info,
	tokenize_f3_lithology_slice,
)
from seis_ssl_cluster.f3.splits import (
	F3SliceSplitRecord,
	load_f3_slice_split_records,
	read_f3_line_geometry,
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
F3_LABEL_BUDGET_ARTIFACT_TYPE = 'f3_lithology_label_budget_token_dataset'
F3_LABEL_BUDGET_SUITE_MANIFEST_ARTIFACT_TYPE = (
	'f3_lithology_label_budget_suite_manifest'
)
F3_LABEL_BUDGET_REQUIRED_SOURCE_FILES = (
	'train_tokens.npz',
	'validation_tokens.npz',
	'token_dataset_metadata.json',
)
F3_LABEL_BUDGET_CLASS_COUNTS_FIELDNAMES = (
	'split',
	'class_id',
	'class_name',
	'count',
	'fraction',
)
F3_SPLIT_INVENTORY_MANIFEST_ARTIFACT_TYPE = (
	'f3_lithology_split_inventory_manifest'
)
F3_SPLIT_INVENTORY_MAX_ATTEMPTS = 1000


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
class F3LabelBudgetModelConfig:
	"""One source token dataset participating in a paired label-budget suite."""

	role: str
	model_tag: str
	token_dataset_root: Path

	def __post_init__(self) -> None:
		"""Validate model source fields."""
		_validate_non_empty_str(self.role, 'role')
		_validate_non_empty_str(self.model_tag, 'model_tag')
		if self.role not in F3_M1_MODEL_ROLES:
			msg = (
				f'role must be one of {sorted(F3_M1_MODEL_ROLES)!r}; '
				f'got {self.role!r}'
			)
			raise ValueError(msg)
		if not Path(self.token_dataset_root).is_absolute():
			msg = (
				'token_dataset_root must be an absolute path; '
				f'got {self.token_dataset_root}'
			)
			raise ValueError(msg)

	@property
	def train_tokens(self) -> Path:
		"""Return the source train token dataset path."""
		return self.token_dataset_root / 'train_tokens.npz'

	@property
	def validation_tokens(self) -> Path:
		"""Return the source validation token dataset path."""
		return self.token_dataset_root / 'validation_tokens.npz'

	@property
	def metadata_json(self) -> Path:
		"""Return the source token dataset metadata path."""
		return self.token_dataset_root / 'token_dataset_metadata.json'


@dataclass(frozen=True)
class F3LabelBudgetConfig:
	"""Resolved config for paired F3 lithology label-budget datasets."""

	artifact_root: Path
	suite_name: str
	output_root: Path
	models: tuple[F3LabelBudgetModelConfig, ...]
	per_class_caps: tuple[int | None, ...]
	subsample_seeds: tuple[int, ...]
	require_all_classes: bool
	reuse_full_validation: bool
	overwrite: bool = False

	def __post_init__(self) -> None:
		"""Validate suite-level label-budget settings."""
		if not Path(self.artifact_root).is_absolute():
			msg = f'artifact_root must be an absolute path; got {self.artifact_root}'
			raise ValueError(msg)
		validate_f3_robustness_suite_name(self.suite_name)
		validate_f3_robustness_output_root(self.output_root)
		_validate_label_budget_model_pair(self.models)
		_validate_per_class_caps(self.per_class_caps)
		_validate_int_sequence(self.subsample_seeds, 'subsample_seeds')
		if not self.per_class_caps:
			msg = 'per_class_caps must contain at least one budget'
			raise ValueError(msg)
		if not self.subsample_seeds:
			msg = 'subsample_seeds must contain at least one seed'
			raise ValueError(msg)
		if not isinstance(self.require_all_classes, bool):
			msg = (
				'require_all_classes must be boolean; '
				f'got {self.require_all_classes!r}'
			)
			raise TypeError(msg)
		if not isinstance(self.reuse_full_validation, bool):
			msg = (
				'reuse_full_validation must be boolean; '
				f'got {self.reuse_full_validation!r}'
			)
			raise TypeError(msg)
		if not self.reuse_full_validation:
			msg = 'validation.reuse_full_validation must be true'
			raise ValueError(msg)
		if not isinstance(self.overwrite, bool):
			msg = f'overwrite must be boolean; got {self.overwrite!r}'
			raise TypeError(msg)

	@property
	def baseline(self) -> F3LabelBudgetModelConfig:
		"""Return the baseline model config."""
		return _model_by_role(self.models, 'baseline')

	@property
	def candidate(self) -> F3LabelBudgetModelConfig:
		"""Return the candidate model config."""
		return _model_by_role(self.models, 'candidate')

	@property
	def expected_dataset_count(self) -> int:
		"""Return the number of model/budget/seed datasets to write."""
		return len(self.models) * len(self.per_class_caps) * len(self.subsample_seeds)


@dataclass(frozen=True)
class F3SplitInventoryInputs:
	"""Input artifacts for an F3 lithology split-index suite."""

	base_png_label_inventory: Path
	source_label_volume: Path
	segy_geometry_json: Path
	class_info: Path
	reference_embedding_metadata: Path

	def __post_init__(self) -> None:
		"""Validate split-index source paths."""
		for label, path in (
			('base_png_label_inventory', self.base_png_label_inventory),
			('source_label_volume', self.source_label_volume),
			('segy_geometry_json', self.segy_geometry_json),
			('class_info', self.class_info),
			('reference_embedding_metadata', self.reference_embedding_metadata),
		):
			if not Path(path).is_absolute():
				msg = f'{label} must be an absolute path; got {path}'
				raise ValueError(msg)


@dataclass(frozen=True)
class F3SplitInventoryConfig:
	"""Resolved config for alternative F3 split/index inventories."""

	artifact_root: Path
	inputs: F3SplitInventoryInputs
	suite_name: str
	output_root: Path
	split_ids: tuple[str, ...]
	random_seeds: tuple[int, ...]
	validation_slice_count: int
	require_validation_all_classes: bool
	min_validation_tokens_per_class: Mapping[str, int]
	include_base_split_as_split_000: bool
	tokenization_policy: F3LithologyTokenPolicy
	patch_size_xyz: tuple[int, int, int]
	overwrite: bool = False

	def __post_init__(self) -> None:
		"""Validate split-index suite settings."""
		if not Path(self.artifact_root).is_absolute():
			msg = f'artifact_root must be an absolute path; got {self.artifact_root}'
			raise ValueError(msg)
		validate_f3_robustness_suite_name(self.suite_name)
		validate_f3_robustness_output_root(self.output_root)
		_validate_str_sequence(self.split_ids, 'split_ids')
		if not self.split_ids:
			msg = 'split_ids must contain at least one split id'
			raise ValueError(msg)
		_reject_duplicates(self.split_ids, 'split_ids')
		_validate_int_sequence(self.random_seeds, 'random_seeds')
		randomized_count = len(self.split_ids)
		if self.include_base_split_as_split_000:
			if self.split_ids[0] != 'split_000':
				msg = (
					'include_base_split_as_split_000 requires first split_id '
					'to be "split_000"'
				)
				raise ValueError(msg)
			randomized_count -= 1
		allowed_seed_counts = {randomized_count, len(self.split_ids)}
		if len(self.random_seeds) not in allowed_seed_counts:
			msg = (
				'random_seeds count must match split_ids count or randomized '
				f'split count; got {len(self.random_seeds)} seeds for '
				f'{len(self.split_ids)} split(s), {randomized_count} randomized'
			)
			raise ValueError(msg)
		_validate_positive_int(
			self.validation_slice_count,
			'validation_slice_count',
		)
		if not isinstance(self.require_validation_all_classes, bool):
			msg = (
				'require_validation_all_classes must be boolean; '
				f'got {self.require_validation_all_classes!r}'
			)
			raise TypeError(msg)
		_validate_min_class_counts(self.min_validation_tokens_per_class)
		if not isinstance(self.include_base_split_as_split_000, bool):
			msg = (
				'include_base_split_as_split_000 must be boolean; '
				f'got {self.include_base_split_as_split_000!r}'
			)
			raise TypeError(msg)
		_validate_positive_xyz(self.patch_size_xyz, 'patch_size')
		if not isinstance(self.overwrite, bool):
			msg = f'overwrite must be boolean; got {self.overwrite!r}'
			raise TypeError(msg)

	@property
	def expected_split_count(self) -> int:
		"""Return number of split inventories to write."""
		return len(self.split_ids)


@dataclass(frozen=True)
class F3LabelBudgetBuildResult:
	"""Output locations from a paired label-budget dataset build."""

	suite_manifest_json: Path
	dataset_roots: tuple[Path, ...]
	rows: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class F3SplitInventoryBuildResult:
	"""Output locations from an F3 split-index inventory build."""

	manifest_json: Path
	inventory_paths: tuple[Path, ...]
	metadata_paths: tuple[Path, ...]


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


def build_f3_lithology_label_budget_datasets(
	config: F3LabelBudgetConfig,
) -> F3LabelBudgetBuildResult:
	"""Write paired MAE/strat-HMM token datasets for label-budget conditions."""
	sources = _load_label_budget_sources(config)
	_validate_source_identity_matches(sources)
	_plan = _planned_label_budget_output_paths(config)
	if not config.overwrite:
		_refuse_existing_outputs(_plan)
	rows: list[dict[str, object]] = []
	dataset_roots: list[Path] = []
	baseline_train = sources['baseline']['train']
	if not isinstance(baseline_train, F3LithologyTokenDataset):
		raise TypeError('baseline train dataset missing')
	for per_class_cap in config.per_class_caps:
		budget_id = label_budget_id(per_class_cap)
		for subsample_seed in config.subsample_seeds:
			indices = class_stratified_subset_indices(
				baseline_train.labels,
				per_class_cap=per_class_cap,
				seed=subsample_seed,
				require_all_classes=config.require_all_classes,
			)
			condition_rows, condition_roots = _write_label_budget_condition(
				config,
				sources,
				budget_id=budget_id,
				per_class_cap=per_class_cap,
				subsample_seed=subsample_seed,
				indices=indices,
			)
			rows.extend(condition_rows)
			dataset_roots.extend(condition_roots)
	manifest_path = config.output_root / 'suite_manifest.json'
	_write_json(
		manifest_path,
		{
			'artifact_type': F3_LABEL_BUDGET_SUITE_MANIFEST_ARTIFACT_TYPE,
			'contract_version': F3_ROBUSTNESS_CONTRACT_VERSION,
			'suite': {
				'name': config.suite_name,
				'output_root': str(config.output_root),
			},
			'rows': rows,
		},
	)
	return F3LabelBudgetBuildResult(
		suite_manifest_json=manifest_path,
		dataset_roots=tuple(dataset_roots),
		rows=tuple(rows),
	)


def label_budget_id(per_class_cap: int | None) -> str:
	"""Return the stable output id for a per-class-cap budget."""
	return 'full' if per_class_cap is None else f'cap{per_class_cap}'


def label_budget_dry_run_summary(config: F3LabelBudgetConfig) -> dict[str, object]:
	"""Return source counts and config fields used by the dry-run CLI summary."""
	sources = _load_label_budget_sources(config)
	_validate_source_identity_matches(sources)
	return {
		'suite name': config.suite_name,
		'output root': config.output_root,
		'baseline model tag': config.baseline.model_tag,
		'candidate model tag': config.candidate.model_tag,
		'budgets': [label_budget_id(cap) for cap in config.per_class_caps],
		'subsample seeds': list(config.subsample_seeds),
		'source train counts': {
			role: int(_source_dataset(payload, 'train').count)
			for role, payload in sources.items()
		},
		'source validation counts': {
			role: int(_source_dataset(payload, 'validation').count)
			for role, payload in sources.items()
		},
		'expected dataset count': config.expected_dataset_count,
	}


def build_f3_lithology_split_inventories(
	config: F3SplitInventoryConfig,
) -> F3SplitInventoryBuildResult:
	"""Write alternative PNG-label inventories for F3 split-index conditions."""
	source = _load_split_inventory_sources(config)
	plan = _planned_split_inventory_output_paths(config)
	if not config.overwrite:
		_refuse_existing_split_inventory_outputs(plan)
	rows: list[dict[str, object]] = []
	inventory_paths: list[Path] = []
	metadata_paths: list[Path] = []
	for split_index, split_id in enumerate(config.split_ids):
		if split_index == 0 and config.include_base_split_as_split_000:
			selection = _base_validation_keys(source['records'])
			random_seed = (
				config.random_seeds[0]
				if len(config.random_seeds) == len(config.split_ids)
				else None
			)
		else:
			seed_index = (
				split_index
				if len(config.random_seeds) == len(config.split_ids)
				else split_index - int(config.include_base_split_as_split_000)
			)
			random_seed = config.random_seeds[seed_index]
			selection = _select_random_validation_keys(
				config,
				source['records'],
				source['class_counts_by_key'],
				source['validation_constraints'],
				random_seed=random_seed,
				split_id=split_id,
			)
		inventory_path, metadata_path, manifest_row = _write_split_inventory(
			config,
			split_id=split_id,
			random_seed=random_seed,
			validation_keys=selection,
			source=source,
		)
		rows.append(manifest_row)
		inventory_paths.append(inventory_path)
		metadata_paths.append(metadata_path)
	manifest_path = config.output_root / 'split_inventory_manifest.json'
	_write_json(
		manifest_path,
		{
			'artifact_type': F3_SPLIT_INVENTORY_MANIFEST_ARTIFACT_TYPE,
			'contract_version': F3_ROBUSTNESS_CONTRACT_VERSION,
			'suite': {
				'name': config.suite_name,
				'output_root': str(config.output_root),
			},
			'source_inventory': str(config.inputs.base_png_label_inventory),
			'rows': rows,
		},
	)
	return F3SplitInventoryBuildResult(
		manifest_json=manifest_path,
		inventory_paths=tuple(inventory_paths),
		metadata_paths=tuple(metadata_paths),
	)


def split_inventory_dry_run_summary(
	config: F3SplitInventoryConfig,
) -> dict[str, object]:
	"""Return source counts and planned output paths for split-index dry runs."""
	records = load_f3_slice_split_records(config.inputs.base_png_label_inventory)
	return {
		'base inventory path': config.inputs.base_png_label_inventory,
		'number of source slices': len(records),
		'split IDs': list(config.split_ids),
		'validation slice count': config.validation_slice_count,
		'class support constraints': dict(config.min_validation_tokens_per_class),
		'expected output paths': [
			str(path) for path in _planned_split_inventory_output_paths(config)
		],
	}


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
	classes = _class_ids(label_array, class_ids)
	if require_all_classes:
		for class_id in classes:
			if not np.any(label_array == class_id):
				msg = f'class_id {int(class_id)} has zero rows'
				raise ValueError(msg)
	if per_class_cap is None:
		return np.arange(label_array.shape[0], dtype=np.int64)
	_validate_positive_int(per_class_cap, 'per_class_cap')
	rng = np.random.default_rng(seed)
	selected: list[NDArray[np.int64]] = []
	for class_id in classes:
		class_indices = np.flatnonzero(label_array == class_id).astype(np.int64)
		if class_indices.size == 0:
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


def _load_label_budget_sources(
	config: F3LabelBudgetConfig,
) -> dict[str, dict[str, object]]:
	sources: dict[str, dict[str, object]] = {}
	for model in config.models:
		_require_source_token_dataset_files(model)
		sources[model.role] = {
			'model': model,
			'train': load_token_dataset_npz(model.train_tokens),
			'validation': load_token_dataset_npz(model.validation_tokens),
			'metadata': _read_json(model.metadata_json),
		}
	return sources


def _load_split_inventory_sources(
	config: F3SplitInventoryConfig,
) -> dict[str, object]:
	rows, fieldnames = _read_inventory_csv_rows(config.inputs.base_png_label_inventory)
	records = load_f3_slice_split_records(config.inputs.base_png_label_inventory)
	if len(rows) != len(records):
		msg = (
			'base inventory CSV row count does not match parsed split records; '
			f'rows={len(rows)}, records={len(records)}'
		)
		raise ValueError(msg)
	if config.validation_slice_count >= len(records):
		msg = (
			'validation_slice_count must be smaller than source slice count; '
			f'got {config.validation_slice_count} for {len(records)} slices'
		)
		raise ValueError(msg)
	classes = read_f3_lithology_class_info(config.inputs.class_info)
	label_volume = np.load(config.inputs.source_label_volume)
	geometry = read_f3_line_geometry(config.inputs.segy_geometry_json)
	valid_tokens = np.ones(
		token_grid_shape_xyz(geometry.shape_xyz, config.patch_size_xyz),
		dtype=np.bool_,
	)
	counts = _estimated_slice_class_counts(
		records,
		label_volume=label_volume,
		valid_tokens=valid_tokens,
		config=config,
	)
	constraints = _validation_class_constraints(
		config,
		class_ids=tuple(class_info.class_id for class_info in classes),
	)
	return {
		'rows': rows,
		'fieldnames': fieldnames,
		'records': records,
		'class_counts_by_key': counts,
		'validation_constraints': constraints,
	}


def _read_inventory_csv_rows(
	path: Path,
) -> tuple[list[dict[str, str]], tuple[str, ...]]:
	with path.open(encoding='utf-8', newline='') as file_obj:
		reader = csv.DictReader(file_obj)
		fieldnames = tuple(reader.fieldnames or ())
		rows = [dict(row) for row in reader]
	if 'split' not in fieldnames:
		msg = f'base inventory CSV missing required split column: {path}'
		raise ValueError(msg)
	return rows, fieldnames


def _estimated_slice_class_counts(
	records: Sequence[F3SliceSplitRecord],
	*,
	label_volume: NDArray[np.generic],
	valid_tokens: NDArray[np.bool_],
	config: F3SplitInventoryConfig,
) -> dict[tuple[str, int], dict[str, int]]:
	geometry = read_f3_line_geometry(config.inputs.segy_geometry_json)
	classes = read_f3_lithology_class_info(config.inputs.class_info)
	counts: dict[tuple[str, int], dict[str, int]] = {}
	for record in records:
		tokenization = tokenize_f3_lithology_slice(
			record,
			label_volume=label_volume,
			valid_tokens=valid_tokens,
			geometry=geometry,
			patch_size_xyz=config.patch_size_xyz,
			policy=config.tokenization_policy,
			classes=classes,
		)
		labels = tokenization.tokenization.majority_class_ids[
			tokenization.usable_mask
		].astype(np.int64, copy=False)
		counts[_record_key(record)] = class_count_dict(labels)
	return counts


def _validation_class_constraints(
	config: F3SplitInventoryConfig,
	*,
	class_ids: Sequence[int],
) -> dict[str, int]:
	min_counts = dict(config.min_validation_tokens_per_class)
	default = min_counts.get('default')
	constraints: dict[str, int] = {}
	if config.require_validation_all_classes:
		required = 1 if default is None else default
		constraints.update({str(int(class_id)): required for class_id in class_ids})
	for key, value in min_counts.items():
		if key == 'default':
			continue
		constraints[str(int(key))] = value
	return constraints


def _base_validation_keys(
	records: Sequence[F3SliceSplitRecord],
) -> frozenset[tuple[str, int]]:
	return frozenset(
		_record_key(record) for record in records if record.split == 'validation'
	)


def _select_random_validation_keys(  # noqa: PLR0913
	config: F3SplitInventoryConfig,
	records: Sequence[F3SliceSplitRecord],
	class_counts_by_key: Mapping[tuple[str, int], Mapping[str, int]],
	validation_constraints: Mapping[str, int],
	*,
	random_seed: int,
	split_id: str,
) -> frozenset[tuple[str, int]]:
	rng = np.random.default_rng(random_seed)
	record_count = len(records)
	for _attempt in range(1, F3_SPLIT_INVENTORY_MAX_ATTEMPTS + 1):
		indices = rng.choice(
			record_count,
			size=config.validation_slice_count,
			replace=False,
		)
		keys = frozenset(_record_key(records[int(index)]) for index in indices)
		counts = _sum_class_counts(keys, class_counts_by_key)
		if _satisfies_class_constraints(counts, validation_constraints):
			return keys
	msg = (
		'could not generate F3 split inventory satisfying validation class '
		f'support constraints for {split_id!r} after '
		f'{F3_SPLIT_INVENTORY_MAX_ATTEMPTS} attempts; '
		f'constraints={dict(validation_constraints)!r}'
	)
	raise ValueError(msg)


def _write_split_inventory(
	config: F3SplitInventoryConfig,
	*,
	split_id: str,
	random_seed: int | None,
	validation_keys: frozenset[tuple[str, int]],
	source: Mapping[str, object],
) -> tuple[Path, Path, dict[str, object]]:
	rows = cast('Sequence[Mapping[str, str]]', source['rows'])
	fieldnames = cast('Sequence[str]', source['fieldnames'])
	records = cast('Sequence[F3SliceSplitRecord]', source['records'])
	class_counts_by_key = cast(
		'Mapping[tuple[str, int], Mapping[str, int]]',
		source['class_counts_by_key'],
	)
	output_rows = _rows_with_split_assignment(rows, records, validation_keys)
	output_dir = config.output_root / 'split_inventories' / split_id
	inventory_path = output_dir / 'png_label_inventory.csv'
	metadata_path = output_dir / 'split_metadata.json'
	_write_inventory_csv(inventory_path, fieldnames, output_rows)
	parsed = load_f3_slice_split_records(inventory_path)
	validation_records = [
		record for record in parsed if _record_key(record) in validation_keys
	]
	train_records = [
		record for record in parsed if _record_key(record) not in validation_keys
	]
	validation_counts = _sum_class_counts(validation_keys, class_counts_by_key)
	train_counts = _sum_class_counts(
		frozenset(_record_key(record) for record in train_records),
		class_counts_by_key,
	)
	metadata = {
		'split_id': split_id,
		'random_seed': random_seed,
		'validation_slices': [_slice_metadata(record) for record in validation_records],
		'train_slices': [_slice_metadata(record) for record in train_records],
		'validation_class_counts_estimated': validation_counts,
		'train_class_counts_estimated': train_counts,
		'tokenization': {
			**config.tokenization_policy.to_dict(),
			'patch_size_xyz': list(config.patch_size_xyz),
		},
		'source_inventory': str(config.inputs.base_png_label_inventory),
	}
	_write_json(metadata_path, metadata)
	return inventory_path, metadata_path, {
		'split_id': split_id,
		'random_seed': random_seed,
		'png_label_inventory': str(inventory_path),
		'split_metadata': str(metadata_path),
		'validation_slice_count': len(validation_records),
	}


def _rows_with_split_assignment(
	rows: Sequence[Mapping[str, str]],
	records: Sequence[F3SliceSplitRecord],
	validation_keys: frozenset[tuple[str, int]],
) -> list[dict[str, str]]:
	output_rows: list[dict[str, str]] = []
	record_keys = {_record_key(record) for record in records}
	for row in rows:
		key = _row_key(row)
		if key not in record_keys:
			msg = f'inventory row does not match parsed split records: {row!r}'
			raise ValueError(msg)
		output = dict(row)
		output['split'] = 'validation' if key in validation_keys else 'train'
		output_rows.append(output)
	return output_rows


def _write_inventory_csv(
	path: Path,
	fieldnames: Sequence[str],
	rows: Sequence[Mapping[str, str]],
) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('w', encoding='utf-8', newline='') as file_obj:
		writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def _sum_class_counts(
	keys: frozenset[tuple[str, int]],
	class_counts_by_key: Mapping[tuple[str, int], Mapping[str, int]],
) -> dict[str, int]:
	counts: dict[str, int] = {}
	for key in keys:
		for class_id, count in class_counts_by_key[key].items():
			counts[class_id] = counts.get(class_id, 0) + int(count)
	return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def _satisfies_class_constraints(
	counts: Mapping[str, int],
	constraints: Mapping[str, int],
) -> bool:
	return all(
		int(counts.get(class_id, 0)) >= minimum
		for class_id, minimum in constraints.items()
	)


def _slice_metadata(record: F3SliceSplitRecord) -> dict[str, object]:
	return {
		'slice_type': record.slice_type,
		'slice_index': record.slice_index,
		'relative_path': record.relative_path,
	}


def _record_key(record: F3SliceSplitRecord) -> tuple[str, int]:
	return (record.slice_type, record.slice_index)


def _row_key(row: Mapping[str, str]) -> tuple[str, int]:
	return (str(row['slice_type']).lower(), int(row['slice_index']))


def _planned_split_inventory_output_paths(
	config: F3SplitInventoryConfig,
) -> tuple[Path, ...]:
	paths = [config.output_root / 'split_inventory_manifest.json']
	for split_id in config.split_ids:
		root = config.output_root / 'split_inventories' / split_id
		paths.extend([root / 'png_label_inventory.csv', root / 'split_metadata.json'])
	return tuple(paths)


def _refuse_existing_split_inventory_outputs(paths: Sequence[Path]) -> None:
	existing = [path for path in paths if path.exists()]
	if existing:
		msg = (
			'refusing to overwrite existing split inventory output(s); '
			f'first existing path: {existing[0]}'
		)
		raise FileExistsError(msg)


def _require_source_token_dataset_files(model: F3LabelBudgetModelConfig) -> None:
	for filename in F3_LABEL_BUDGET_REQUIRED_SOURCE_FILES:
		path = model.token_dataset_root / filename
		if not path.is_file():
			msg = (
				f'source token dataset for {model.role} model '
				f'{model.model_tag!r} is missing required file: {path}'
			)
			raise FileNotFoundError(msg)


def _validate_source_identity_matches(
	sources: Mapping[str, Mapping[str, object]],
) -> None:
	try:
		assert_same_token_identity(
			_source_dataset(sources['baseline'], 'train'),
			_source_dataset(sources['candidate'], 'train'),
			reference_label='baseline train',
			candidate_label='candidate train',
		)
		assert_same_token_identity(
			_source_dataset(sources['baseline'], 'validation'),
			_source_dataset(sources['candidate'], 'validation'),
			reference_label='baseline validation',
			candidate_label='candidate validation',
		)
	except ValueError as exc:
		msg = (
			f'{exc}; rebuild the candidate token dataset using the baseline as '
			'reference_token_dataset'
		)
		raise ValueError(msg) from exc


def _planned_label_budget_output_paths(config: F3LabelBudgetConfig) -> tuple[Path, ...]:
	paths = [config.output_root / 'suite_manifest.json']
	for model in config.models:
		for per_class_cap in config.per_class_caps:
			for subsample_seed in config.subsample_seeds:
				root = _budget_token_dataset_root(
					config,
					model,
					budget_id=label_budget_id(per_class_cap),
					subsample_seed=subsample_seed,
				)
				paths.extend(
					[
						root / 'train_tokens.npz',
						root / 'validation_tokens.npz',
						root / 'all_labeled_tokens.npz',
						root / 'token_dataset_metadata.json',
						root / 'class_counts.csv',
						root / 'token_dataset_summary.md',
					],
				)
	return tuple(paths)


def _refuse_existing_outputs(paths: Sequence[Path]) -> None:
	existing = [path for path in paths if path.exists()]
	if existing:
		msg = (
			'refusing to overwrite existing label-budget output(s); '
			f'first existing path: {existing[0]}'
		)
		raise FileExistsError(msg)


def _write_label_budget_condition(  # noqa: PLR0913
	config: F3LabelBudgetConfig,
	sources: Mapping[str, Mapping[str, object]],
	*,
	budget_id: str,
	per_class_cap: int | None,
	subsample_seed: int,
	indices: NDArray[np.generic],
) -> tuple[list[dict[str, object]], list[Path]]:
	rows: list[dict[str, object]] = []
	roots: list[Path] = []
	condition_hash = paired_token_identity_hash(
		subset_token_dataset(_source_dataset(sources['baseline'], 'train'), indices),
		_source_dataset(sources['baseline'], 'validation'),
	)
	for model in config.models:
		source = sources[model.role]
		train = subset_token_dataset(_source_dataset(source, 'train'), indices)
		validation = _source_dataset(source, 'validation')
		model_hash = paired_token_identity_hash(train, validation)
		if model_hash != condition_hash:
			msg = (
				'paired identity hash mismatch after subsetting; '
				f'baseline={condition_hash}, {model.role}={model_hash}'
			)
			raise ValueError(msg)
		root = _budget_token_dataset_root(
			config,
			model,
			budget_id=budget_id,
			subsample_seed=subsample_seed,
		)
		metadata = _label_budget_metadata_payload(
			config,
			model,
			source,
			budget_id=budget_id,
			per_class_cap=per_class_cap,
			subsample_seed=subsample_seed,
			selected_train_dataset=train,
			validation_dataset=validation,
			paired_identity_hash=condition_hash,
			token_dataset_root=root,
		)
		_write_label_budget_token_dataset(root, train, validation, metadata)
		rows.append(
			_suite_manifest_row(
				model,
				root,
				budget_id=budget_id,
				per_class_cap=per_class_cap,
				subsample_seed=subsample_seed,
				metadata=metadata,
			),
		)
		roots.append(root)
	return rows, roots


def _label_budget_metadata_payload(  # noqa: PLR0913
	config: F3LabelBudgetConfig,
	model: F3LabelBudgetModelConfig,
	source: Mapping[str, object],
	*,
	budget_id: str,
	per_class_cap: int | None,
	subsample_seed: int,
	selected_train_dataset: F3LithologyTokenDataset,
	validation_dataset: F3LithologyTokenDataset,
	paired_identity_hash: str,
	token_dataset_root: Path,
) -> dict[str, object]:
	subset_metadata = budget_subset_metadata(
		source_train_tokens=model.train_tokens,
		source_validation_tokens=model.validation_tokens,
		per_class_cap=per_class_cap,
		subsample_seed=subsample_seed,
		selected_train_dataset=selected_train_dataset,
		validation_dataset=validation_dataset,
		paired_identity_hash=paired_identity_hash,
	)
	return {
		'artifact_type': F3_LABEL_BUDGET_ARTIFACT_TYPE,
		'contract_version': F3_ROBUSTNESS_CONTRACT_VERSION,
		'suite': {
			'name': config.suite_name,
			'output_root': str(config.output_root),
		},
		'model': {
			'role': model.role,
			'model_tag': model.model_tag,
		},
		'label_budget': {
			'budget_id': budget_id,
			'per_class_cap': None if per_class_cap is None else int(per_class_cap),
			'subsample_seed': int(subsample_seed),
			'require_all_classes': config.require_all_classes,
		},
		'validation': {
			'reuse_full_validation': config.reuse_full_validation,
		},
		'source_token_dataset': {
			'root': str(model.token_dataset_root),
			'train_tokens': str(model.train_tokens),
			'validation_tokens': str(model.validation_tokens),
			'metadata_json': str(model.metadata_json),
			'metadata': source.get('metadata', {}),
		},
		'outputs': {
			'token_dataset_root': str(token_dataset_root),
			'train_tokens': str(token_dataset_root / 'train_tokens.npz'),
			'validation_tokens': str(token_dataset_root / 'validation_tokens.npz'),
			'all_labeled_tokens': str(token_dataset_root / 'all_labeled_tokens.npz'),
			'metadata_json': str(token_dataset_root / 'token_dataset_metadata.json'),
			'class_counts_csv': str(token_dataset_root / 'class_counts.csv'),
			'summary_markdown': str(token_dataset_root / 'token_dataset_summary.md'),
		},
		'summary': {
			'train_tokens': subset_metadata['selected_train_token_count'],
			'validation_tokens': subset_metadata['validation_token_count'],
			'all_labeled_tokens': (
				int(subset_metadata['selected_train_token_count'])
				+ int(subset_metadata['validation_token_count'])
			),
		},
		**subset_metadata,
	}


def _write_label_budget_token_dataset(
	root: Path,
	train: F3LithologyTokenDataset,
	validation: F3LithologyTokenDataset,
	metadata: Mapping[str, object],
) -> None:
	root.mkdir(parents=True, exist_ok=True)
	save_token_dataset_npz(
		_dataset_with_metadata(
			train,
			metadata,
			split_name='train',
		),
		root / 'train_tokens.npz',
	)
	save_token_dataset_npz(
		_dataset_with_metadata(
			validation,
			metadata,
			split_name='validation',
		),
		root / 'validation_tokens.npz',
	)
	save_token_dataset_npz(
		_concat_token_datasets(
			train,
			validation,
			metadata=_merged_dataset_metadata(
				train,
				metadata,
				split_name='all_labeled',
			),
		),
		root / 'all_labeled_tokens.npz',
	)
	_write_json(root / 'token_dataset_metadata.json', metadata)
	_write_label_budget_class_counts_csv(root / 'class_counts.csv', train, validation)
	_write_text(
		root / 'token_dataset_summary.md',
		_render_label_budget_summary_markdown(metadata),
	)


def _dataset_with_metadata(
	dataset: F3LithologyTokenDataset,
	metadata: Mapping[str, object],
	*,
	split_name: str,
) -> F3LithologyTokenDataset:
	return F3LithologyTokenDataset(
		features=np.asarray(dataset.features),
		labels=np.asarray(dataset.labels),
		survey_id=np.asarray(dataset.survey_id),
		split=np.asarray(dataset.split),
		slice_type=np.asarray(dataset.slice_type),
		slice_index=np.asarray(dataset.slice_index),
		token_xyz=np.asarray(dataset.token_xyz),
		voxel_center_xyz=np.asarray(dataset.voxel_center_xyz),
		majority_fraction=np.asarray(dataset.majority_fraction),
		labeled_fraction=np.asarray(dataset.labeled_fraction),
		metadata=_merged_dataset_metadata(dataset, metadata, split_name=split_name),
	)


def _merged_dataset_metadata(
	dataset: F3LithologyTokenDataset,
	metadata: Mapping[str, object],
	*,
	split_name: str,
) -> dict[str, object]:
	return {**dict(dataset.metadata), **metadata, 'split_name': split_name}


def _concat_token_datasets(
	train: F3LithologyTokenDataset,
	validation: F3LithologyTokenDataset,
	*,
	metadata: Mapping[str, object],
) -> F3LithologyTokenDataset:
	return F3LithologyTokenDataset(
		features=np.concatenate([train.features, validation.features], axis=0),
		labels=np.concatenate([train.labels, validation.labels], axis=0),
		survey_id=np.concatenate([train.survey_id, validation.survey_id], axis=0),
		split=np.concatenate([train.split, validation.split], axis=0),
		slice_type=np.concatenate([train.slice_type, validation.slice_type], axis=0),
		slice_index=np.concatenate([train.slice_index, validation.slice_index], axis=0),
		token_xyz=np.concatenate([train.token_xyz, validation.token_xyz], axis=0),
		voxel_center_xyz=np.concatenate(
			[train.voxel_center_xyz, validation.voxel_center_xyz],
			axis=0,
		),
		majority_fraction=np.concatenate(
			[train.majority_fraction, validation.majority_fraction],
			axis=0,
		),
		labeled_fraction=np.concatenate(
			[train.labeled_fraction, validation.labeled_fraction],
			axis=0,
		),
		metadata=dict(metadata),
	)


def _write_label_budget_class_counts_csv(
	path: Path,
	train: F3LithologyTokenDataset,
	validation: F3LithologyTokenDataset,
) -> None:
	rows: list[dict[str, object]] = []
	rows.extend(_label_budget_class_count_rows('train', train.labels))
	rows.extend(_label_budget_class_count_rows('validation', validation.labels))
	rows.extend(
		_label_budget_class_count_rows(
			'all_labeled',
			np.concatenate([train.labels, validation.labels], axis=0),
		),
	)
	with path.open('w', encoding='utf-8', newline='') as file_obj:
		writer = csv.DictWriter(
			file_obj,
			fieldnames=F3_LABEL_BUDGET_CLASS_COUNTS_FIELDNAMES,
		)
		writer.writeheader()
		writer.writerows(rows)


def _label_budget_class_count_rows(
	split: str,
	labels: NDArray[np.generic],
) -> list[dict[str, object]]:
	counts = class_count_dict(labels)
	total = int(np.asarray(labels).shape[0])
	rows: list[dict[str, object]] = []
	for class_id in sorted((int(value) for value in counts), key=int):
		count = int(counts[str(class_id)])
		rows.append(
			{
				'split': split,
				'class_id': class_id,
				'class_name': f'class_{class_id}',
				'count': count,
				'fraction': 0.0 if total == 0 else count / total,
			},
		)
	return rows


def _render_label_budget_summary_markdown(metadata: Mapping[str, object]) -> str:
	return '\n'.join(
		[
			'# F3 lithology label-budget token dataset',
			'',
			f'- suite: {metadata["suite"]["name"]}',
			f'- model role: {metadata["model"]["role"]}',
			f'- model tag: {metadata["model"]["model_tag"]}',
			f'- budget: {metadata["label_budget"]["budget_id"]}',
			f'- subsample seed: {metadata["label_budget"]["subsample_seed"]}',
			f'- train tokens: {metadata["selected_train_token_count"]}',
			f'- validation tokens: {metadata["validation_token_count"]}',
			f'- paired identity hash: {metadata["paired_identity_hash"]}',
			'',
		],
	)


def _suite_manifest_row(  # noqa: PLR0913
	model: F3LabelBudgetModelConfig,
	root: Path,
	*,
	budget_id: str,
	per_class_cap: int | None,
	subsample_seed: int,
	metadata: Mapping[str, object],
) -> dict[str, object]:
	return {
		'model_role': model.role,
		'model_tag': model.model_tag,
		'budget_id': budget_id,
		'per_class_cap': None if per_class_cap is None else int(per_class_cap),
		'subsample_seed': int(subsample_seed),
		'token_dataset_root': str(root),
		'train_tokens': str(root / 'train_tokens.npz'),
		'validation_tokens': str(root / 'validation_tokens.npz'),
		'metadata_json': str(root / 'token_dataset_metadata.json'),
		'selected_train_token_count': metadata['selected_train_token_count'],
		'validation_token_count': metadata['validation_token_count'],
		'selected_class_counts': metadata['selected_class_counts'],
		'validation_class_counts': metadata['validation_class_counts'],
		'paired_identity_hash': metadata['paired_identity_hash'],
	}


def _budget_token_dataset_root(
	config: F3LabelBudgetConfig,
	model: F3LabelBudgetModelConfig,
	*,
	budget_id: str,
	subsample_seed: int,
) -> Path:
	return (
		config.output_root
		/ 'datasets'
		/ f'model={model.model_tag}'
		/ f'budget={budget_id}'
		/ f'subsample_seed={subsample_seed}'
		/ 'token_dataset'
	)


def _source_dataset(
	source: Mapping[str, object],
	key: str,
) -> F3LithologyTokenDataset:
	value = source.get(key)
	if not isinstance(value, F3LithologyTokenDataset):
		msg = f'source {key} dataset is missing'
		raise TypeError(msg)
	return value


def _model_by_role(
	models: Sequence[F3LabelBudgetModelConfig],
	role: str,
) -> F3LabelBudgetModelConfig:
	for model in models:
		if model.role == role:
			return model
	msg = f'missing {role} model'
	raise ValueError(msg)


def _validate_label_budget_model_pair(
	models: Sequence[F3LabelBudgetModelConfig],
) -> None:
	if isinstance(models, str | bytes):
		msg = f'models must be a sequence of model configs; got {models!r}'
		raise TypeError(msg)
	model_tuple = tuple(models)
	if len(model_tuple) != 2:
		msg = (
			'label-budget datasets require exactly baseline and candidate models; '
			f'got {len(model_tuple)}'
		)
		raise ValueError(msg)
	roles = [model.role for model in model_tuple]
	if sorted(roles) != ['baseline', 'candidate']:
		msg = (
			'label-budget datasets require model roles '
			f"['baseline', 'candidate']; got {roles!r}"
		)
		raise ValueError(msg)


def _validate_per_class_caps(values: object) -> tuple[int | None, ...]:
	if not isinstance(values, Sequence) or isinstance(values, str | bytes):
		msg = f'per_class_caps must be a sequence; got {values!r}'
		raise TypeError(msg)
	caps: list[int | None] = []
	for value in values:
		if value is None:
			caps.append(None)
			continue
		caps.append(_validate_positive_int(value, 'per_class_cap'))
	_reject_duplicates(caps, 'per_class_caps')
	return tuple(caps)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)


def _read_json(path: Path) -> Mapping[str, object]:
	with path.open(encoding='utf-8') as file_obj:
		payload = json.load(file_obj)
	if not isinstance(payload, Mapping):
		msg = f'JSON file must contain an object: {path}'
		raise TypeError(msg)
	return dict(payload)


def _write_text(path: Path, text: str) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(text, encoding='utf-8')


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


def _validate_positive_xyz(values: object, label: str) -> tuple[int, int, int]:
	if (
		not isinstance(values, Sequence)
		or isinstance(values, str | bytes)
		or len(values) != 3
	):
		msg = f'{label} must contain three positive integers; got {values!r}'
		raise TypeError(msg)
	items = tuple(values)
	if not all(
		isinstance(item, int) and not isinstance(item, bool) and item > 0
		for item in items
	):
		msg = f'{label} must contain three positive integers; got {values!r}'
		raise ValueError(msg)
	return (int(items[0]), int(items[1]), int(items[2]))


def _validate_min_class_counts(counts: Mapping[str, int]) -> None:
	if not isinstance(counts, Mapping):
		msg = f'min_validation_tokens_per_class must be a mapping; got {counts!r}'
		raise TypeError(msg)
	for key, value in counts.items():
		if not isinstance(key, str) or not key:
			msg = (
				'min_validation_tokens_per_class keys must be non-empty strings; '
				f'got {key!r}'
			)
			raise TypeError(msg)
		if key != 'default':
			int(key)
		if not isinstance(value, int) or isinstance(value, bool) or value < 0:
			msg = (
				'min_validation_tokens_per_class values must be nonnegative '
				f'integers; got {value!r} for {key!r}'
			)
			raise ValueError(msg)


def _validate_int_sequence(values: object, label: str) -> tuple[int, ...]:
	if not isinstance(values, Sequence) or isinstance(values, str | bytes):
		msg = f'{label} must be a sequence of integers; got {values!r}'
		raise TypeError(msg)
	items = tuple(values)
	if not all(isinstance(item, int) and not isinstance(item, bool) for item in items):
		msg = f'{label} must contain only integers; got {values!r}'
		raise TypeError(msg)
	_reject_duplicates(items, label)
	return items


def _reject_duplicates(values: Sequence[object], label: str) -> None:
	seen: set[object] = set()
	duplicates: list[object] = []
	for value in values:
		if value in seen and value not in duplicates:
			duplicates.append(value)
		seen.add(value)
	if duplicates:
		msg = f'{label} must not contain duplicates; got {duplicates!r}'
		raise ValueError(msg)


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
	'F3LabelBudgetBuildResult',
	'F3LabelBudgetConfig',
	'F3LabelBudgetModelConfig',
	'F3LabelBudgetSpec',
	'F3PairedMetricRow',
	'F3RobustnessModelSpec',
	'F3RobustnessSuiteManifest',
	'F3SplitInventoryBuildResult',
	'F3SplitInventoryConfig',
	'F3SplitInventoryInputs',
	'assert_same_token_identity',
	'budget_subset_metadata',
	'build_f3_lithology_label_budget_datasets',
	'build_f3_lithology_split_inventories',
	'class_count_dict',
	'class_stratified_subset_indices',
	'label_budget_dry_run_summary',
	'label_budget_id',
	'load_token_dataset_npz',
	'paired_token_identity_hash',
	'save_token_dataset_npz',
	'split_inventory_dry_run_summary',
	'subset_token_dataset',
	'token_identity_frame',
	'validate_f3_m1_model_pair',
	'validate_f3_robustness_config_keys',
	'validate_f3_robustness_output_root',
	'validate_f3_robustness_suite_name',
]

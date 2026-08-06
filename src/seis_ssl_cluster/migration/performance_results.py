"""Pure contracts shared by performance-migration validation stages.

The helpers in this module intentionally know nothing about a particular model,
embedding writer, or clustering implementation.  They make it possible for the
stage runners to keep historical artifacts read-only while applying one strict
definition of completion, parity, publication, and migration status.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

import numpy as np

from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.results import (
	PublishedItem,
	PublishItem,
	PublishManifest,
	publish_selected_results,
)

if TYPE_CHECKING:
	from collections.abc import Iterator

	from numpy.typing import ArrayLike, NDArray


COMPLETION_MANIFEST_NAME = 'migration_completion.json'
COMPLETION_ARTIFACT_TYPE = 'performance_migration_completion'
COMPLETION_SCHEMA_VERSION = 1
MIGRATION_STATUSES = frozenset(
	{
		'PASS_REUSE_EXISTING',
		'PASS_WITH_NUMERIC_DRIFT',
		'REEXTRACT_REQUIRED',
		'REBUILD_M1_REQUIRED',
		'BLOCKED_NUMERIC_CONTRACT',
	},
)
MIGRATION_ALLOWED_PUBLISH_SUFFIXES = frozenset({'.md', '.json', '.csv', '.png'})
RAW_ARTIFACT_SUFFIXES = frozenset(
	{
		'.pt',
		'.pth',
		'.ckpt',
		'.npy',
		'.npz',
		'.joblib',
		'.pkl',
		'.pickle',
		'.mmap',
		'.memmap',
		'.h5',
		'.hdf5',
		'.zarr',
		'.sgy',
		'.segy',
		'.parquet',
	}
)
_MISSING = object()
_STATUS_ORDER = (
	'BLOCKED_NUMERIC_CONTRACT',
	'REBUILD_M1_REQUIRED',
	'REEXTRACT_REQUIRED',
	'PASS_WITH_NUMERIC_DRIFT',
	'PASS_REUSE_EXISTING',
)


@dataclass(frozen=True)
class MigrationDecision:
	"""One deterministic decision and the evidence that selected it."""

	status: str
	reasons: tuple[str, ...]
	required_rerun_scope: str
	multi_head_baseline_policy: str

	def __post_init__(self) -> None:
		"""Reject statuses outside the preregistered migration taxonomy."""
		if self.status not in MIGRATION_STATUSES:
			raise ValueError(f'unsupported migration status: {self.status}')


@dataclass(frozen=True)
class MetadataDifference:
	"""A single leaf-level metadata mismatch and its contract category."""

	path: str
	historical: object
	current: object
	classification: Literal['scientific', 'performance', 'path_only', 'environment']


@dataclass(frozen=True)
class MetadataDiff:
	"""Partitioned metadata differences; unknown fields are scientific by default."""

	scientific: tuple[MetadataDifference, ...]
	performance: tuple[MetadataDifference, ...]
	path_only: tuple[MetadataDifference, ...]
	environment: tuple[MetadataDifference, ...]

	@property
	def identical(self) -> bool:
		"""Return whether both metadata documents are structurally identical."""
		return not any(
			(self.scientific, self.performance, self.path_only, self.environment)
		)

	@property
	def has_scientific_drift(self) -> bool:
		"""Return whether any scientific identity field differs."""
		return bool(self.scientific)


@dataclass(frozen=True)
class NumericArrayComparison:
	"""Exact identity and finite numeric error diagnostics for two arrays."""

	historical_shape: tuple[int, ...]
	current_shape: tuple[int, ...]
	historical_dtype: str
	current_dtype: str
	shape_equal: bool
	dtype_equal: bool
	array_equal: bool
	exact_equal: bool
	valid_element_count: int
	invalid_element_count: int
	finite_pair_count: int
	different_element_count: int
	different_element_fraction: float | None
	nan_count_historical: int
	nan_count_current: int
	inf_count_historical: int
	inf_count_current: int
	nonfinite_mismatch_count: int
	max_absolute_error: float | None
	mean_absolute_error: float | None
	median_absolute_error: float | None
	p95_absolute_error: float | None
	p99_absolute_error: float | None
	p999_absolute_error: float | None
	max_stable_relative_error: float | None
	mean_stable_relative_error: float | None


@dataclass(frozen=True)
class CosineSimilaritySummary:
	"""Summary of rowwise cosine similarities over an optional validity mask."""

	row_count: int
	finite_row_count: int
	zero_norm_row_count: int
	minimum: float | None
	p1: float | None
	p5: float | None
	median: float | None
	mean: float | None
	p95: float | None
	maximum: float | None


@dataclass(frozen=True)
class ArtifactReuse:
	"""Result of inspecting a pre-existing migration-owned artifact path."""

	action: Literal['NEW', 'REUSED', 'QUARANTINED']
	path: Path
	quarantine_path: Path | None
	reason: str | None
	manifest: Mapping[str, object] | None


def decide_migration_status(  # noqa: PLR0911, PLR0913
	*,
	blocking_numeric_contract: bool = False,
	valid_token_masks_exact: bool,
	probe_predictions_exact: bool,
	probe_confusion_matrix_exact: bool,
	primary_metrics_exact: bool,
	hmm_labels_exact: bool,
	pseudo_target_labels_exact: bool,
	pseudo_target_valid_tokens_exact: bool,
	confidence_threshold_crossing: bool,
	numeric_drift: bool,
) -> MigrationDecision:
	"""Apply the preregistered migration-status priority without heuristics.

	The function is deliberately conservative.  A mismatched token-valid mask is
	a numeric-contract blocker, while a decoded HMM or pseudo-target label drift
	selects ``REBUILD_M1_REQUIRED``.  This preserves the user's stated priority:
	blocked > rebuild > reextract > numeric drift > reuse.
	"""
	contract_failures = _contract_failure_reasons(
		blocking_numeric_contract=blocking_numeric_contract,
		valid_token_masks_exact=valid_token_masks_exact,
		probe_predictions_exact=probe_predictions_exact,
		probe_confusion_matrix_exact=probe_confusion_matrix_exact,
		primary_metrics_exact=primary_metrics_exact,
		hmm_labels_exact=hmm_labels_exact,
		pseudo_target_labels_exact=pseudo_target_labels_exact,
		pseudo_target_valid_tokens_exact=pseudo_target_valid_tokens_exact,
		confidence_threshold_crossing=confidence_threshold_crossing,
	)
	if blocking_numeric_contract or not valid_token_masks_exact:
		return _decision(
			'BLOCKED_NUMERIC_CONTRACT',
			contract_failures
			or ('checkpoint, shape, finite, or valid-token contract failed',),
		)
	if not hmm_labels_exact or not pseudo_target_labels_exact:
		return _decision(
			'REBUILD_M1_REQUIRED',
			_contract_subset(
				contract_failures,
				{'hmm_labels_exact', 'pseudo_target_labels_exact'},
			),
		)
	if confidence_threshold_crossing:
		return _decision(
			'REBUILD_M1_REQUIRED',
			('pseudo-target confidence crosses the configured threshold',),
		)
	if not pseudo_target_valid_tokens_exact:
		return _decision(
			'BLOCKED_NUMERIC_CONTRACT',
			('pseudo-target valid-token mask differs',),
		)
	if not probe_predictions_exact:
		return _decision(
			'REEXTRACT_REQUIRED',
			('existing linear-probe predicted class differs',),
		)
	if not probe_confusion_matrix_exact or not primary_metrics_exact:
		return _decision(
			'BLOCKED_NUMERIC_CONTRACT',
			(
				'probe parity evidence is internally inconsistent: predictions are '
				'exact but confusion matrix or primary metrics differ',
			),
		)
	if numeric_drift:
		return _decision(
			'PASS_WITH_NUMERIC_DRIFT',
			('numeric embedding or center drift has no downstream label effect',),
		)
	return _decision(
		'PASS_REUSE_EXISTING', ('all required parity contracts are exact',)
	)


def migration_decision_to_dict(decision: MigrationDecision) -> dict[str, object]:
	"""Serialize a deterministic migration decision for a summary manifest."""
	return {
		'status': decision.status,
		'reasons': list(decision.reasons),
		'required_rerun_scope': decision.required_rerun_scope,
		'multi_head_baseline_policy': decision.multi_head_baseline_policy,
		'priority_order': list(_STATUS_ORDER),
	}


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> Path:
	"""Serialize one JSON mapping through a sibling temporary file and replace."""
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.NamedTemporaryFile(
		mode='w',
		encoding='utf-8',
		dir=path.parent,
		prefix=f'.{path.name}.',
		suffix='.tmp',
		delete=False,
	) as file_obj:
		temporary = Path(file_obj.name)
		try:
			json.dump(payload, file_obj, indent=2, sort_keys=True, allow_nan=False)
			file_obj.write('\n')
			file_obj.flush()
			os.fsync(file_obj.fileno())
		except BaseException:
			temporary.unlink(missing_ok=True)
			raise
	temporary.replace(path)
	return path


def write_text_atomic(path: Path, text: str) -> Path:
	"""Write text atomically, preserving a failed write's prior final file."""
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.NamedTemporaryFile(
		mode='w',
		encoding='utf-8',
		dir=path.parent,
		prefix=f'.{path.name}.',
		suffix='.tmp',
		delete=False,
	) as file_obj:
		temporary = Path(file_obj.name)
		try:
			file_obj.write(text)
			file_obj.flush()
			os.fsync(file_obj.fileno())
		except BaseException:
			temporary.unlink(missing_ok=True)
			raise
	temporary.replace(path)
	return path


@contextmanager
def staged_artifact_directory(final_path: Path) -> Iterator[Path]:
	"""Create a same-filesystem staging directory without silently deleting it.

	Call :func:`commit_staged_artifact_directory` after writing and validating the
	completion manifest.  On error the staging directory intentionally remains in
	place, so callers can quarantine it with an explicit reason instead of losing
	partial evidence.
	"""
	final = Path(final_path)
	if final.exists():
		raise FileExistsError(f'final artifact path already exists: {final}')
	final.parent.mkdir(parents=True, exist_ok=True)
	staging = final.parent / f'.{final.name}.staging_{uuid4().hex}'
	staging.mkdir()
	yield staging


def commit_staged_artifact_directory(staging_path: Path, final_path: Path) -> Path:
	"""Atomically rename a same-parent staging directory into a new final path."""
	staging = Path(staging_path)
	final = Path(final_path)
	if not staging.is_dir():
		raise FileNotFoundError(f'staging artifact directory is missing: {staging}')
	if staging.parent.resolve() != final.parent.resolve():
		raise ValueError('staging and final artifact directories must share a parent')
	if final.exists():
		raise FileExistsError(f'final artifact path already exists: {final}')
	staging.replace(final)
	return final


def quarantine_artifact(
	path: Path,
	*,
	reason: str,
	timestamp: datetime | None = None,
) -> Path:
	"""Move an invalid migration-owned file or directory to a timestamped path."""
	source = Path(path)
	if not source.exists():
		raise FileNotFoundError(f'cannot quarantine a missing path: {source}')
	stamp = (timestamp or datetime.now(timezone.utc)).strftime('%Y%m%dT%H%M%SZ')
	safe_reason = _safe_reason(reason)
	candidate = source.with_name(f'{source.name}.quarantine_{stamp}_{safe_reason}')
	counter = 1
	while candidate.exists():
		candidate = source.with_name(
			f'{source.name}.quarantine_{stamp}_{safe_reason}_{counter}'
		)
		counter += 1
	source.replace(candidate)
	return candidate


def artifact_identity(path: Path) -> dict[str, object]:
	"""Return a stable identity record for one regular file or directory tree."""
	value = Path(path)
	if value.is_symlink():
		raise ValueError(f'artifact identity does not accept symlinks: {value}')
	if value.is_file():
		return {
			'path': str(value),
			'file_type': 'file',
			'byte_size': value.stat().st_size,
			'sha256': file_sha256(value),
		}
	if not value.is_dir():
		raise FileNotFoundError(f'artifact path is missing or unsupported: {value}')
	digest = hashlib.sha256()
	byte_size = 0
	file_count = 0
	for child in sorted(
		value.rglob('*'), key=lambda item: item.relative_to(value).as_posix()
	):
		if child.is_symlink():
			raise ValueError(f'artifact identity does not accept symlinks: {child}')
		if child.is_dir():
			continue
		if not child.is_file():
			raise ValueError(
				f'artifact identity encountered unsupported entry: {child}'
			)
		relative = child.relative_to(value).as_posix().encode('utf-8')
		sha256 = file_sha256(child)
		size = child.stat().st_size
		digest.update(relative)
		digest.update(b'\0')
		digest.update(str(size).encode('ascii'))
		digest.update(b'\0')
		digest.update(sha256.encode('ascii'))
		digest.update(b'\n')
		byte_size += size
		file_count += 1
	return {
		'path': str(value),
		'file_type': 'directory',
		'byte_size': byte_size,
		'file_count': file_count,
		'sha256': digest.hexdigest(),
	}


def write_completion_manifest(  # noqa: PLR0913
	artifact_root: Path,
	*,
	artifact_type: str,
	schema_version: int,
	required_files: Sequence[Path | str],
	current_git_sha: str,
	historical_baseline_sha: str,
	source_identities: Mapping[str, object] | None = None,
	extra: Mapping[str, object] | None = None,
) -> Path:
	"""Write a hash-bound complete-artifact manifest only after all files exist."""
	root = Path(artifact_root)
	if not root.is_dir():
		raise FileNotFoundError(f'artifact root must already be a directory: {root}')
	if not artifact_type:
		raise ValueError('artifact_type must be non-empty')
	if isinstance(schema_version, bool) or not isinstance(schema_version, int):
		raise TypeError('schema_version must be an integer')
	_validate_git_sha(current_git_sha, 'current_git_sha')
	_validate_git_sha(historical_baseline_sha, 'historical_baseline_sha')
	required = _required_file_records(root, required_files)
	payload: dict[str, object] = {
		'artifact_type': artifact_type,
		'completion_artifact_type': COMPLETION_ARTIFACT_TYPE,
		'completion_schema_version': COMPLETION_SCHEMA_VERSION,
		'schema_version': schema_version,
		'status': 'COMPLETE',
		'current_git_sha': current_git_sha,
		'historical_baseline_sha': historical_baseline_sha,
		'required_files': required,
		'source_identities': dict(source_identities or {}),
		'extra': dict(extra or {}),
	}
	return write_json_atomic(root / COMPLETION_MANIFEST_NAME, payload)


def validate_completion_manifest(  # noqa: C901, PLR0913
	artifact_root: Path,
	*,
	expected_artifact_type: str | None = None,
	expected_schema_version: int | None = None,
	expected_current_git_sha: str | None = None,
	expected_historical_baseline_sha: str | None = None,
	required_files: Sequence[Path | str] | None = None,
) -> Mapping[str, object]:
	"""Validate an artifact's completion state, paths, hashes, and identities."""
	root = Path(artifact_root)
	manifest_path = root / COMPLETION_MANIFEST_NAME
	if not manifest_path.is_file():
		raise FileNotFoundError(f'completion manifest is missing: {manifest_path}')
	payload = _read_json_mapping(manifest_path)
	if payload.get('completion_artifact_type') != COMPLETION_ARTIFACT_TYPE:
		raise ValueError('completion manifest artifact type is invalid')
	if payload.get('completion_schema_version') != COMPLETION_SCHEMA_VERSION:
		raise ValueError('completion manifest schema version is invalid')
	if payload.get('status') != 'COMPLETE':
		raise ValueError('artifact is not complete')
	artifact_type = payload.get('artifact_type')
	if not isinstance(artifact_type, str) or not artifact_type:
		raise ValueError('completion manifest artifact_type is invalid')
	if expected_artifact_type is not None and artifact_type != expected_artifact_type:
		raise ValueError(
			'completion manifest artifact_type mismatch: '
			f'expected {expected_artifact_type}, got {artifact_type}'
		)
	schema_version = payload.get('schema_version')
	if isinstance(schema_version, bool) or not isinstance(schema_version, int):
		raise TypeError('completion manifest schema_version is invalid')
	if (
		expected_schema_version is not None
		and schema_version != expected_schema_version
	):
		raise ValueError(
			'completion manifest schema_version mismatch: '
			f'expected {expected_schema_version}, got {schema_version}'
		)
	_validate_manifest_git_sha(
		payload,
		key='current_git_sha',
		expected=expected_current_git_sha,
	)
	_validate_manifest_git_sha(
		payload,
		key='historical_baseline_sha',
		expected=expected_historical_baseline_sha,
	)
	recorded_files = _validate_required_file_records(
		root, payload.get('required_files')
	)
	if required_files is not None:
		expected_paths = {
			_relative_file_path(root, item, label='required_files item')
			for item in required_files
		}
		if set(recorded_files) != expected_paths:
			raise ValueError(
				'completion manifest required file set mismatch: '
				f'expected {sorted(expected_paths)}, got {sorted(recorded_files)}'
			)
	return payload


def reuse_or_quarantine_artifact(  # noqa: PLR0913
	artifact_root: Path,
	*,
	expected_artifact_type: str,
	expected_schema_version: int,
	expected_current_git_sha: str,
	expected_historical_baseline_sha: str,
	required_files: Sequence[Path | str],
	reason_prefix: str = 'invalid_or_partial',
) -> ArtifactReuse:
	"""Return a verified reusable artifact or quarantine an invalid existing path."""
	root = Path(artifact_root)
	if not root.exists():
		return ArtifactReuse('NEW', root, None, None, None)
	try:
		manifest = validate_completion_manifest(
			root,
			expected_artifact_type=expected_artifact_type,
			expected_schema_version=expected_schema_version,
			expected_current_git_sha=expected_current_git_sha,
			expected_historical_baseline_sha=expected_historical_baseline_sha,
			required_files=required_files,
		)
	except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError) as error:
		quarantined = quarantine_artifact(root, reason=reason_prefix)
		return ArtifactReuse('QUARANTINED', root, quarantined, str(error), None)
	return ArtifactReuse('REUSED', root, None, None, manifest)


def classify_metadata_diff(
	historical: Mapping[str, object],
	current: Mapping[str, object],
	*,
	field_classifications: Mapping[
		str, Literal['scientific', 'performance', 'path_only', 'environment']
	]
	| None = None,
) -> MetadataDiff:
	"""Classify metadata differences without treating path-only changes as drift.

	Any field not recognized as runtime, path-only, or environment information is
	classified as scientific.  This fail-closed default is intentional: a newly
	introduced preprocessing or model field cannot silently become harmless.
	"""
	historical_flat = _flatten_metadata(historical)
	current_flat = _flatten_metadata(current)
	overrides = dict(field_classifications or {})
	by_category: dict[
		Literal['scientific', 'performance', 'path_only', 'environment'],
		list[MetadataDifference],
	] = {
		'scientific': [],
		'performance': [],
		'path_only': [],
		'environment': [],
	}
	for path in sorted(set(historical_flat) | set(current_flat)):
		old = historical_flat.get(path, _MISSING)
		new = current_flat.get(path, _MISSING)
		if _metadata_values_equal(old, new):
			continue
		classification = _metadata_classification(path, overrides)
		by_category[classification].append(
			MetadataDifference(
				path=path,
				historical=None if old is _MISSING else old,
				current=None if new is _MISSING else new,
				classification=classification,
			)
		)
	return MetadataDiff(
		scientific=tuple(by_category['scientific']),
		performance=tuple(by_category['performance']),
		path_only=tuple(by_category['path_only']),
		environment=tuple(by_category['environment']),
	)


def metadata_diff_to_dict(diff: MetadataDiff) -> dict[str, object]:
	"""Convert a metadata diff into a JSON-safe diagnostic payload."""
	return {
		'identical': diff.identical,
		'has_scientific_drift': diff.has_scientific_drift,
		'scientific': [_metadata_difference_to_dict(item) for item in diff.scientific],
		'performance': [
			_metadata_difference_to_dict(item) for item in diff.performance
		],
		'path_only': [_metadata_difference_to_dict(item) for item in diff.path_only],
		'environment': [
			_metadata_difference_to_dict(item) for item in diff.environment
		],
	}


def compare_numeric_arrays(
	historical: ArrayLike,
	current: ArrayLike,
	*,
	valid_mask: ArrayLike | None = None,
	stable_relative_epsilon: float = 1e-12,
) -> NumericArrayComparison:
	"""Compare numeric arrays while excluding invalid-mask elements from aggregates.

	A mask may have the same shape as the arrays or a matching leading shape (for
	example an ``[x, y, z]`` token mask for ``[x, y, z, channel]`` embeddings).
	The exact check records ``np.array_equal`` first; finite error diagnostics are
	then calculated in float64 only for valid elements finite in both arrays.
	"""
	if not math.isfinite(stable_relative_epsilon) or stable_relative_epsilon <= 0:
		raise ValueError('stable_relative_epsilon must be a finite positive number')
	old = np.asarray(historical)
	new = np.asarray(current)
	_validate_numeric_array(old, 'historical')
	_validate_numeric_array(new, 'current')
	shape_equal = old.shape == new.shape
	dtype_equal = old.dtype == new.dtype
	array_equal = bool(shape_equal and np.array_equal(old, new))
	if not shape_equal:
		return NumericArrayComparison(
			historical_shape=tuple(old.shape),
			current_shape=tuple(new.shape),
			historical_dtype=str(old.dtype),
			current_dtype=str(new.dtype),
			shape_equal=False,
			dtype_equal=dtype_equal,
			array_equal=False,
			exact_equal=False,
			valid_element_count=0,
			invalid_element_count=0,
			finite_pair_count=0,
			different_element_count=0,
			different_element_fraction=None,
			nan_count_historical=0,
			nan_count_current=0,
			inf_count_historical=0,
			inf_count_current=0,
			nonfinite_mismatch_count=0,
			max_absolute_error=None,
			mean_absolute_error=None,
			median_absolute_error=None,
			p95_absolute_error=None,
			p99_absolute_error=None,
			p999_absolute_error=None,
			max_stable_relative_error=None,
			mean_stable_relative_error=None,
		)
	mask = _broadcast_valid_mask(valid_mask, old.shape)
	old_values = old[mask]
	new_values = new[mask]
	valid_count = int(old_values.size)
	invalid_count = int(old.size - valid_count)
	old_nan = _nan_count(old_values)
	new_nan = _nan_count(new_values)
	old_inf = _inf_count(old_values)
	new_inf = _inf_count(new_values)
	equal_with_nan = _equal_with_nan(old_values, new_values)
	different_count = int(np.count_nonzero(~equal_with_nan))
	finite_pair = np.isfinite(old_values) & np.isfinite(new_values)
	finite_count = int(np.count_nonzero(finite_pair))
	nonfinite_mismatch = int(np.count_nonzero((~finite_pair) & (~equal_with_nan)))
	if finite_count:
		old_float = old_values[finite_pair].astype(np.float64, copy=False)
		new_float = new_values[finite_pair].astype(np.float64, copy=False)
		absolute = np.abs(new_float - old_float)
		relative = absolute / np.maximum(np.abs(old_float), stable_relative_epsilon)
		metrics = _numeric_error_metrics(absolute, relative)
	else:
		metrics = _empty_numeric_error_metrics()
	return NumericArrayComparison(
		historical_shape=tuple(old.shape),
		current_shape=tuple(new.shape),
		historical_dtype=str(old.dtype),
		current_dtype=str(new.dtype),
		shape_equal=True,
		dtype_equal=dtype_equal,
		array_equal=array_equal,
		exact_equal=bool(dtype_equal and array_equal),
		valid_element_count=valid_count,
		invalid_element_count=invalid_count,
		finite_pair_count=finite_count,
		different_element_count=different_count,
		different_element_fraction=(
			None if valid_count == 0 else different_count / valid_count
		),
		nan_count_historical=old_nan,
		nan_count_current=new_nan,
		inf_count_historical=old_inf,
		inf_count_current=new_inf,
		nonfinite_mismatch_count=nonfinite_mismatch,
		max_absolute_error=metrics['max_absolute_error'],
		mean_absolute_error=metrics['mean_absolute_error'],
		median_absolute_error=metrics['median_absolute_error'],
		p95_absolute_error=metrics['p95_absolute_error'],
		p99_absolute_error=metrics['p99_absolute_error'],
		p999_absolute_error=metrics['p999_absolute_error'],
		max_stable_relative_error=metrics['max_stable_relative_error'],
		mean_stable_relative_error=metrics['mean_stable_relative_error'],
	)


def numeric_array_comparison_to_dict(
	comparison: NumericArrayComparison,
) -> dict[str, object]:
	"""Serialize a numeric comparison with JSON list shapes."""
	return {
		'historical_shape': list(comparison.historical_shape),
		'current_shape': list(comparison.current_shape),
		'historical_dtype': comparison.historical_dtype,
		'current_dtype': comparison.current_dtype,
		'shape_equal': comparison.shape_equal,
		'dtype_equal': comparison.dtype_equal,
		'np_array_equal': comparison.array_equal,
		'exact_equal': comparison.exact_equal,
		'valid_element_count': comparison.valid_element_count,
		'invalid_element_count': comparison.invalid_element_count,
		'finite_pair_count': comparison.finite_pair_count,
		'different_element_count': comparison.different_element_count,
		'different_element_fraction': comparison.different_element_fraction,
		'nan_count_historical': comparison.nan_count_historical,
		'nan_count_current': comparison.nan_count_current,
		'inf_count_historical': comparison.inf_count_historical,
		'inf_count_current': comparison.inf_count_current,
		'nonfinite_mismatch_count': comparison.nonfinite_mismatch_count,
		'max_absolute_error': comparison.max_absolute_error,
		'mean_absolute_error': comparison.mean_absolute_error,
		'median_absolute_error': comparison.median_absolute_error,
		'p95_absolute_error': comparison.p95_absolute_error,
		'p99_absolute_error': comparison.p99_absolute_error,
		'p999_absolute_error': comparison.p999_absolute_error,
		'max_stable_relative_error': comparison.max_stable_relative_error,
		'mean_stable_relative_error': comparison.mean_stable_relative_error,
	}


def summarize_rowwise_cosine_similarity(
	historical: ArrayLike,
	current: ArrayLike,
	*,
	valid_mask: ArrayLike | None = None,
) -> CosineSimilaritySummary:
	"""Return the requested per-token cosine summary for final-axis features."""
	old = np.asarray(historical)
	new = np.asarray(current)
	_validate_numeric_array(old, 'historical')
	_validate_numeric_array(new, 'current')
	if old.shape != new.shape:
		raise ValueError('rowwise cosine requires matching array shapes')
	if old.ndim < 1 or old.shape[-1] == 0:
		raise ValueError('rowwise cosine requires a non-empty final feature axis')
	row_shape = old.shape[:-1]
	mask = _broadcast_valid_mask(valid_mask, row_shape)
	old_rows = old.reshape((-1, old.shape[-1]))[mask.reshape(-1)]
	new_rows = new.reshape((-1, new.shape[-1]))[mask.reshape(-1)]
	row_count = int(old_rows.shape[0])
	if row_count == 0:
		return _empty_cosine_summary()
	old_float = old_rows.astype(np.float64, copy=False)
	new_float = new_rows.astype(np.float64, copy=False)
	finite_rows = np.isfinite(old_float).all(axis=1) & np.isfinite(new_float).all(
		axis=1
	)
	old_norm = np.linalg.norm(old_float[finite_rows], axis=1)
	new_norm = np.linalg.norm(new_float[finite_rows], axis=1)
	nonzero = (old_norm > 0.0) & (new_norm > 0.0)
	zero_count = int(finite_rows.sum() - nonzero.sum())
	if not np.any(nonzero):
		return CosineSimilaritySummary(
			row_count=row_count,
			finite_row_count=int(finite_rows.sum()),
			zero_norm_row_count=zero_count,
			minimum=None,
			p1=None,
			p5=None,
			median=None,
			mean=None,
			p95=None,
			maximum=None,
		)
	old_nonzero = old_float[finite_rows][nonzero]
	new_nonzero = new_float[finite_rows][nonzero]
	cosine = np.einsum('ij,ij->i', old_nonzero, new_nonzero) / (
		old_norm[nonzero] * new_norm[nonzero]
	)
	return CosineSimilaritySummary(
		row_count=row_count,
		finite_row_count=int(finite_rows.sum()),
		zero_norm_row_count=zero_count,
		minimum=float(np.min(cosine)),
		p1=float(np.percentile(cosine, 1.0)),
		p5=float(np.percentile(cosine, 5.0)),
		median=float(np.median(cosine)),
		mean=float(np.mean(cosine)),
		p95=float(np.percentile(cosine, 95.0)),
		maximum=float(np.max(cosine)),
	)


def cosine_similarity_summary_to_dict(
	summary: CosineSimilaritySummary,
) -> dict[str, object]:
	"""Serialize a rowwise cosine summary."""
	return {
		'row_count': summary.row_count,
		'finite_row_count': summary.finite_row_count,
		'zero_norm_row_count': summary.zero_norm_row_count,
		'minimum': summary.minimum,
		'p1': summary.p1,
		'p5': summary.p5,
		'median': summary.median,
		'mean': summary.mean,
		'p95': summary.p95,
		'maximum': summary.maximum,
	}


def publish_lightweight_migration_results(
	*,
	items: Sequence[PublishItem],
	output_dir: Path,
	source_artifact_root: Path,
	max_file_size_bytes: int = 10 * 1024 * 1024,
	allow_reuse: bool = False,
) -> PublishManifest:
	"""Atomically publish only lightweight migration reports and validate them.

	Raw checkpoints, arrays, feature caches, and clustering products are rejected
	by suffix before any copy.  A fresh directory is staged and renamed only after
	the manifest has been enriched with source and published hashes and fully
	validated.  Existing result directories are never overwritten.
	"""
	final = Path(output_dir)
	source_root = Path(source_artifact_root)
	_validate_publish_sources(items, source_root, max_file_size_bytes)
	if final.exists():
		if not allow_reuse:
			raise FileExistsError(f'publish output already exists: {final}')
		validate_migration_publish_manifest(
			final / 'publish_manifest.json',
			source_artifact_root=source_root,
			max_file_size_bytes=max_file_size_bytes,
		)
		return _publish_manifest_from_payload(final / 'publish_manifest.json')
	with staged_artifact_directory(final) as staging:
		try:
			manifest = publish_selected_results(
				items=items,
				output_dir=staging,
				allowed_suffixes=MIGRATION_ALLOWED_PUBLISH_SUFFIXES,
				max_file_size_bytes=max_file_size_bytes,
				overwrite=False,
			)
			_manifest_enrich_source_identities(manifest.manifest_path)
			validate_migration_publish_manifest(
				manifest.manifest_path,
				source_artifact_root=source_root,
				max_file_size_bytes=max_file_size_bytes,
			)
			committed = commit_staged_artifact_directory(staging, final)
		except BaseException:
			if staging.exists():
				quarantine_artifact(staging, reason='publish_failure')
			raise
	final_manifest = committed / 'publish_manifest.json'
	payload = _read_json_mapping(final_manifest)
	payload['output_dir'] = str(final)
	write_json_atomic(final_manifest, payload)
	validate_migration_publish_manifest(
		final_manifest,
		source_artifact_root=source_root,
		max_file_size_bytes=max_file_size_bytes,
	)
	return replace(manifest, output_dir=final, manifest_path=final_manifest)


def validate_migration_publish_manifest(  # noqa: C901
	manifest_path: Path,
	*,
	source_artifact_root: Path,
	max_file_size_bytes: int = 10 * 1024 * 1024,
) -> tuple[Path, ...]:
	"""Validate hashes, size, source identity, and no-unlisted-file publication."""
	manifest = Path(manifest_path)
	if not manifest.is_file():
		raise FileNotFoundError(f'publish manifest is missing: {manifest}')
	if manifest.name != 'publish_manifest.json':
		raise ValueError(
			'migration publish manifest must be named publish_manifest.json'
		)
	if max_file_size_bytes <= 0:
		raise ValueError('max_file_size_bytes must be positive')
	root = manifest.parent.resolve()
	source_root = Path(source_artifact_root).resolve()
	payload = _read_json_mapping(manifest)
	items = payload.get('items')
	if not isinstance(items, list):
		raise TypeError('publish manifest items must be a list')
	listed_paths: set[Path] = set()
	for index, item in enumerate(items):
		if not isinstance(item, Mapping):
			raise TypeError(f'publish manifest item {index} must be an object')
		target = _manifest_target(root, item, index)
		if target in listed_paths:
			raise ValueError(f'publish manifest lists duplicate target: {target}')
		listed_paths.add(target)
		_validate_lightweight_path(target, label=f'publish target {target}')
		if target.is_symlink() or not target.is_file():
			raise ValueError(f'publish target must be a regular file: {target}')
		if target.stat().st_size > max_file_size_bytes:
			raise ValueError(f'publish target exceeds max_file_size_bytes: {target}')
		_validate_manifest_hashes(item, target, index)
		source = _manifest_source(source_root, item, index)
		_validate_lightweight_path(source, label=f'publish source {source}')
		if source.is_symlink() or not source.is_file():
			raise ValueError(f'publish source must be a regular file: {source}')
		if source.stat().st_size > max_file_size_bytes:
			raise ValueError(f'publish source exceeds max_file_size_bytes: {source}')
		_validate_source_hashes(item, source, target, index)
	actual_paths = {
		item.resolve()
		for item in root.rglob('*')
		if item.is_file() and not item.is_symlink()
	}
	expected_paths = listed_paths | {manifest.resolve()}
	if actual_paths != expected_paths:
		extra = sorted(
			str(item.relative_to(root)) for item in actual_paths - expected_paths
		)
		missing = sorted(
			str(item.relative_to(root)) for item in expected_paths - actual_paths
		)
		raise ValueError(
			'publish directory files do not exactly match the manifest: '
			f'extra={extra}, missing={missing}'
		)
	return tuple(sorted(listed_paths))


def _decision(status: str, reasons: Sequence[str]) -> MigrationDecision:
	policies = {
		'PASS_REUSE_EXISTING': (
			'No M3-V or M3-V-LB rerun is required; historical MAE/M1/M2-A '
			'artifacts remain the baseline.',
			'Historical M1 may be reused for multi-head K=6/8/10; retain the '
			'multi-head no-consistency guardrail.',
		),
		'PASS_WITH_NUMERIC_DRIFT': (
			'Existing scientific results remain historical baselines; no blanket rerun '
			'is required.',
			'Create a current-code single-head K=6 control under the multi-head '
			'conditions before comparing multi-head variants.',
		),
		'REEXTRACT_REQUIRED': (
			'Regenerate versioned current-code MAE/M1/M2-A embeddings and required '
			'downstream baselines; do not overwrite historical artifacts.',
			(
				'Regenerate current-code embeddings and downstream baseline '
				'artifacts before evaluating multi-head models.'
			),
		),
		'REBUILD_M1_REQUIRED': (
			'Rebuild a current-code K=6 pseudo-target and single-head M1 baseline '
			'before future multi-head work; retain old M1 as historical.',
			(
				'Train a current-code single-head M1 from the current K=6 target '
				'and use it as the multi-head baseline.'
			),
		),
		'BLOCKED_NUMERIC_CONTRACT': (
			(
				'Do not proceed to downstream or multi-head work until the '
				'numeric contract failure is corrected and migration validation '
				'is rerun.'
			),
			'Do not proceed to multi-head K=6/8/10.',
		),
	}
	if status not in policies:
		raise ValueError(f'unsupported migration status: {status}')
	required_scope, policy = policies[status]
	return MigrationDecision(
		status=status,
		reasons=tuple(reasons),
		required_rerun_scope=required_scope,
		multi_head_baseline_policy=policy,
	)


def _contract_failure_reasons(**checks: bool) -> tuple[str, ...]:
	return tuple(key for key, value in checks.items() if not value)


def _contract_subset(reasons: Sequence[str], allowed: set[str]) -> tuple[str, ...]:
	selected = tuple(reason for reason in reasons if reason in allowed)
	return selected or ('current K=6 HMM or pseudo-target decoded labels differ',)


def _safe_reason(reason: str) -> str:
	if not isinstance(reason, str) or not reason.strip():
		raise ValueError('quarantine reason must be a non-empty string')
	cleaned = re.sub(r'[^A-Za-z0-9]+', '_', reason).strip('_')[:80]
	return cleaned or 'invalid'


def _required_file_records(
	root: Path,
	required_files: Sequence[Path | str],
) -> list[dict[str, object]]:
	if not required_files:
		raise ValueError('completion manifest requires at least one required file')
	records: list[dict[str, object]] = []
	seen: set[str] = set()
	for item in required_files:
		relative = _relative_file_path(root, item, label='required_files item')
		if relative == COMPLETION_MANIFEST_NAME:
			raise ValueError(
				'completion manifest cannot list itself as a required file'
			)
		if relative in seen:
			raise ValueError(f'duplicate required file: {relative}')
		seen.add(relative)
		path = root / relative
		if path.is_symlink() or not path.is_file():
			raise FileNotFoundError(f'required artifact file is missing: {path}')
		records.append(
			{
				'relative_path': relative,
				'byte_size': path.stat().st_size,
				'sha256': file_sha256(path),
			}
		)
	return sorted(records, key=lambda item: str(item['relative_path']))


def _validate_required_file_records(  # noqa: C901
	root: Path,
	value: object,
) -> dict[str, Mapping[str, object]]:
	if not isinstance(value, list) or not value:
		raise TypeError('completion manifest required_files must be a non-empty list')
	recorded: dict[str, Mapping[str, object]] = {}
	for index, item in enumerate(value):
		if not isinstance(item, Mapping):
			raise TypeError(f'completion manifest required_files[{index}] is invalid')
		relative_value = item.get('relative_path')
		if not isinstance(relative_value, str):
			raise TypeError(
				f'completion manifest required_files[{index}].relative_path is invalid'
			)
		relative = _relative_file_path(root, relative_value, label='required file')
		if relative in recorded:
			raise TypeError(
				f'completion manifest has duplicate required file: {relative}'
			)
		path = root / relative
		if path.is_symlink() or not path.is_file():
			raise FileNotFoundError(
				f'completed artifact required file is missing: {path}'
			)
		byte_size = item.get('byte_size')
		if isinstance(byte_size, bool) or not isinstance(byte_size, int):
			raise TypeError(
				f'completion manifest required_files[{index}].byte_size is invalid'
			)
		if path.stat().st_size != byte_size:
			raise ValueError(f'completed artifact file size mismatch: {path}')
		sha256 = item.get('sha256')
		if not _is_sha256(sha256):
			raise ValueError(
				f'completion manifest required_files[{index}].sha256 is invalid'
			)
		if file_sha256(path) != sha256:
			raise ValueError(f'completed artifact file SHA-256 mismatch: {path}')
		recorded[relative] = item
	return recorded


def _relative_file_path(root: Path, value: Path | str, *, label: str) -> str:
	path = Path(value)
	if path.is_absolute():
		resolved = path.resolve(strict=False)
	else:
		resolved = (root / path).resolve(strict=False)
	try:
		relative = resolved.relative_to(root.resolve()).as_posix()
	except ValueError as error:
		raise ValueError(
			f'{label} must be inside the completion manifest root: {value}'
		) from error
	if relative in {'', '.'} or relative.startswith('../'):
		raise ValueError(f'{label} must be a relative file path: {value}')
	return relative


def _validate_git_sha(value: str, label: str) -> None:
	if not _is_git_sha(value):
		raise ValueError(f'{label} must be a lowercase 40-character git SHA')


def _validate_manifest_git_sha(
	payload: Mapping[str, object],
	*,
	key: str,
	expected: str | None,
) -> None:
	value = payload.get(key)
	if not isinstance(value, str) or not _is_git_sha(value):
		raise ValueError(f'completion manifest {key} is invalid')
	if expected is not None:
		_validate_git_sha(expected, f'expected_{key}')
		if value != expected:
			raise ValueError(
				f'completion manifest {key} mismatch: expected {expected}, got {value}'
			)


def _is_git_sha(value: object) -> bool:
	return isinstance(value, str) and bool(re.fullmatch(r'[0-9a-f]{40}', value))


def _is_sha256(value: object) -> bool:
	return isinstance(value, str) and bool(re.fullmatch(r'[0-9a-f]{64}', value))


def _read_json_mapping(path: Path) -> dict[str, object]:
	payload = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(payload, dict):
		raise TypeError(f'JSON document must be an object: {path}')
	return payload


def _flatten_metadata(value: object, prefix: str = '') -> dict[str, object]:
	if isinstance(value, Mapping):
		if not value:
			return {prefix: {}}
		result: dict[str, object] = {}
		for key in sorted(value, key=str):
			key_text = str(key)
			path = key_text if not prefix else f'{prefix}.{key_text}'
			result.update(_flatten_metadata(value[key], path))
		return result
	if isinstance(value, list | tuple):
		return {prefix: list(value)}
	if isinstance(value, Path):
		return {prefix: str(value)}
	if isinstance(value, np.generic):
		return {prefix: value.item()}
	return {prefix: value}


def _metadata_values_equal(left: object, right: object) -> bool:
	if left is _MISSING or right is _MISSING:
		return left is right
	if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
		return bool(np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True))
	try:
		result = left == right
	except (TypeError, ValueError):
		return False
	if isinstance(result, np.ndarray):
		return bool(np.all(result))
	return bool(result)


def _metadata_classification(
	path: str,
	overrides: Mapping[
		str, Literal['scientific', 'performance', 'path_only', 'environment']
	],
) -> Literal['scientific', 'performance', 'path_only', 'environment']:
	for prefix in sorted(overrides, key=len, reverse=True):
		if path == prefix or path.startswith(f'{prefix}.'):
			return overrides[prefix]
	segments = tuple(part.lower() for part in path.split('.') if part)
	if _matches_environment_field(segments):
		return 'environment'
	if _matches_path_field(segments):
		return 'path_only'
	if _matches_performance_field(segments):
		return 'performance'
	return 'scientific'


def _matches_environment_field(segments: Sequence[str]) -> bool:
	environment = {
		'environment',
		'hostname',
		'platform',
		'python_version',
		'numpy_version',
		'torch_version',
		'cuda_version',
		'gpu_name',
		'cuda_available',
		'bf16_support',
		'fp16_support',
	}
	return any(segment in environment for segment in segments)


def _matches_path_field(segments: Sequence[str]) -> bool:
	path_names = {
		'path',
		'paths',
		'directory',
		'dir',
		'root',
		'output',
		'output_dir',
		'artifact_root',
		'cache_dir',
	}
	return any(
		segment in path_names or segment.endswith(('_path', '_dir', '_root'))
		for segment in segments
	)


def _matches_performance_field(segments: Sequence[str]) -> bool:
	performance = {
		'batch_size',
		'prefetch_queue_depth',
		'amp',
		'amp_dtype',
		'stage_timing',
		'average_chunk_size_x',
		'chunk_size',
		'chunk_size_x',
		'chunk_size_tokens',
		'num_workers',
		'device',
		'device_transfer',
		'pin_memory',
		'non_blocking',
		'cache',
		'preprocessing_cache',
		'prepared_feature_cache',
		'reuse',
		'cleanup',
		'force_rebuild',
		'persist',
	}
	return any(segment in performance for segment in segments)


def _metadata_difference_to_dict(item: MetadataDifference) -> dict[str, object]:
	return {
		'path': item.path,
		'historical': _json_safe(item.historical),
		'current': _json_safe(item.current),
		'classification': item.classification,
	}


def _json_safe(value: object) -> object:
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, Mapping):
		return {str(key): _json_safe(item) for key, item in value.items()}
	if isinstance(value, list | tuple):
		return [_json_safe(item) for item in value]
	return value


def _validate_numeric_array(array: NDArray[np.generic], label: str) -> None:
	if not np.issubdtype(array.dtype, np.number):
		raise TypeError(f'{label} must have a numeric dtype; got {array.dtype}')


def _broadcast_valid_mask(
	valid_mask: ArrayLike | None,
	shape: tuple[int, ...],
) -> NDArray[np.bool_]:
	if valid_mask is None:
		return np.ones(shape, dtype=np.bool_)
	mask = np.asarray(valid_mask)
	if mask.dtype != np.bool_:
		raise TypeError('valid_mask must have boolean dtype')
	if mask.ndim > len(shape) or tuple(mask.shape) != shape[: mask.ndim]:
		raise ValueError(
			'valid_mask must match the array shape or a leading array shape: '
			f'mask={mask.shape}, array={shape}'
		)
	reshaped = mask.reshape(mask.shape + (1,) * (len(shape) - mask.ndim))
	return np.broadcast_to(reshaped, shape)


def _nan_count(values: NDArray[np.generic]) -> int:
	if not np.issubdtype(values.dtype, np.inexact):
		return 0
	return int(np.count_nonzero(np.isnan(values)))


def _inf_count(values: NDArray[np.generic]) -> int:
	if not np.issubdtype(values.dtype, np.inexact):
		return 0
	return int(np.count_nonzero(np.isinf(values)))


def _equal_with_nan(
	left: NDArray[np.generic],
	right: NDArray[np.generic],
) -> NDArray[np.bool_]:
	equal = np.equal(left, right)
	if np.issubdtype(left.dtype, np.inexact) and np.issubdtype(right.dtype, np.inexact):
		equal = equal | (np.isnan(left) & np.isnan(right))
	return np.asarray(equal, dtype=np.bool_)


def _numeric_error_metrics(
	absolute: NDArray[np.float64],
	relative: NDArray[np.float64],
) -> dict[str, float]:
	return {
		'max_absolute_error': float(np.max(absolute)),
		'mean_absolute_error': float(np.mean(absolute)),
		'median_absolute_error': float(np.median(absolute)),
		'p95_absolute_error': float(np.percentile(absolute, 95.0)),
		'p99_absolute_error': float(np.percentile(absolute, 99.0)),
		'p999_absolute_error': float(np.percentile(absolute, 99.9)),
		'max_stable_relative_error': float(np.max(relative)),
		'mean_stable_relative_error': float(np.mean(relative)),
	}


def _empty_numeric_error_metrics() -> dict[str, None]:
	return {
		'max_absolute_error': None,
		'mean_absolute_error': None,
		'median_absolute_error': None,
		'p95_absolute_error': None,
		'p99_absolute_error': None,
		'p999_absolute_error': None,
		'max_stable_relative_error': None,
		'mean_stable_relative_error': None,
	}


def _empty_cosine_summary() -> CosineSimilaritySummary:
	return CosineSimilaritySummary(
		row_count=0,
		finite_row_count=0,
		zero_norm_row_count=0,
		minimum=None,
		p1=None,
		p5=None,
		median=None,
		mean=None,
		p95=None,
		maximum=None,
	)


def _validate_publish_sources(
	items: Sequence[PublishItem],
	source_root: Path,
	max_file_size_bytes: int,
) -> None:
	if not source_root.is_dir():
		raise FileNotFoundError(f'source_artifact_root is missing: {source_root}')
	if max_file_size_bytes <= 0:
		raise ValueError('max_file_size_bytes must be positive')
	if not items:
		raise ValueError('at least one lightweight publish item is required')
	for item in items:
		if item.content_text is not None or item.text_replacements:
			raise ValueError('migration publish accepts immutable source files only')
		source = Path(item.source)
		if source.is_symlink() or not source.is_file():
			raise FileNotFoundError(f'publish source must be a regular file: {source}')
		try:
			source.resolve().relative_to(source_root.resolve())
		except ValueError as error:
			raise ValueError(
				f'publish source must be under source_artifact_root: {source}'
			) from error
		_validate_lightweight_path(source, label=f'publish source {source}')
		_validate_lightweight_path(
			Path(item.relative_target),
			label=f'publish target {item.relative_target}',
		)
		if source.stat().st_size > max_file_size_bytes:
			raise ValueError(f'publish source exceeds max_file_size_bytes: {source}')


def _validate_lightweight_path(path: Path, *, label: str) -> None:
	suffix = path.suffix.lower()
	if suffix in RAW_ARTIFACT_SUFFIXES:
		raise ValueError(f'{label} has forbidden raw-artifact suffix: {suffix}')
	if suffix not in MIGRATION_ALLOWED_PUBLISH_SUFFIXES:
		raise ValueError(
			f'{label} suffix is not allowed for lightweight publication: {suffix}'
		)


def _manifest_enrich_source_identities(manifest_path: Path) -> None:
	payload = _read_json_mapping(manifest_path)
	items = payload.get('items')
	if not isinstance(items, list):
		raise TypeError('publish manifest items must be a list')
	for index, item in enumerate(items):
		if not isinstance(item, dict):
			raise TypeError(f'publish manifest item {index} must be an object')
		source_value = item.get('source')
		if not isinstance(source_value, str):
			raise TypeError(f'publish manifest item {index}.source is invalid')
		source = Path(source_value)
		if not source.is_file():
			raise FileNotFoundError(f'publish source disappeared: {source}')
		item['source_sha256'] = file_sha256(source)
		item['source_size_bytes'] = source.stat().st_size
		item['published_sha256'] = item.get('sha256')
		item['published_size_bytes'] = item.get('size_bytes')
	write_json_atomic(manifest_path, payload)


def _manifest_target(root: Path, item: Mapping[str, object], index: int) -> Path:
	target_value = item.get('target')
	if not isinstance(target_value, str) or not target_value:
		raise ValueError(f'publish manifest item {index}.target is invalid')
	target = Path(target_value)
	if target.is_absolute():
		raise ValueError(f'publish manifest item {index}.target must be relative')
	resolved = (root / target).resolve(strict=False)
	try:
		resolved.relative_to(root)
	except ValueError as error:
		raise ValueError(
			f'publish manifest item {index}.target escapes output root'
		) from error
	return resolved


def _manifest_source(root: Path, item: Mapping[str, object], index: int) -> Path:
	source_value = item.get('source')
	if not isinstance(source_value, str) or not source_value:
		raise ValueError(f'publish manifest item {index}.source is invalid')
	source = Path(source_value).resolve(strict=False)
	try:
		source.relative_to(root)
	except ValueError as error:
		raise ValueError(
			f'publish manifest item {index}.source escapes source root'
		) from error
	return source


def _validate_manifest_hashes(
	item: Mapping[str, object],
	target: Path,
	index: int,
) -> None:
	for key, actual in (
		('size_bytes', target.stat().st_size),
		('published_size_bytes', target.stat().st_size),
	):
		value = item.get(key)
		if isinstance(value, bool) or not isinstance(value, int) or value != actual:
			raise ValueError(f'publish manifest item {index}.{key} mismatch')
	for key in ('sha256', 'published_sha256'):
		value = item.get(key)
		if not _is_sha256(value) or value != file_sha256(target):
			raise ValueError(f'publish manifest item {index}.{key} mismatch')


def _validate_source_hashes(
	item: Mapping[str, object],
	source: Path,
	target: Path,
	index: int,
) -> None:
	source_size = item.get('source_size_bytes')
	if (
		isinstance(source_size, bool)
		or not isinstance(source_size, int)
		or source_size != source.stat().st_size
	):
		raise ValueError(f'publish manifest item {index}.source_size_bytes mismatch')
	source_sha = item.get('source_sha256')
	if not _is_sha256(source_sha) or source_sha != file_sha256(source):
		raise ValueError(f'publish manifest item {index}.source_sha256 mismatch')
	if source_sha != file_sha256(target):
		raise ValueError(f'publish source and target SHA-256 differ: {target}')


def _publish_manifest_from_payload(manifest_path: Path) -> PublishManifest:
	payload = _read_json_mapping(manifest_path)
	items_value = payload.get('items')
	if not isinstance(items_value, list):
		raise TypeError('publish manifest items must be a list')

	published: list[PublishedItem] = []
	for index, item in enumerate(items_value):
		if not isinstance(item, Mapping):
			raise TypeError(f'publish manifest item {index} must be an object')
		source = item.get('source')
		target = item.get('target')
		size = item.get('size_bytes')
		sha256 = item.get('sha256')
		if (
			not isinstance(source, str)
			or not isinstance(target, str)
			or isinstance(size, bool)
			or not isinstance(size, int)
			or not _is_sha256(sha256)
		):
			raise TypeError(f'publish manifest item {index} is invalid')
		published.append(
			PublishedItem(
				source=Path(source),
				target=manifest_path.parent / target,
				size_bytes=size,
				sha256=sha256,
			)
		)
	created_at = payload.get('created_at_utc')
	if not isinstance(created_at, str):
		raise TypeError('publish manifest created_at_utc is invalid')
	output = payload.get('output_dir')
	if not isinstance(output, str):
		raise TypeError('publish manifest output_dir is invalid')
	return PublishManifest(
		created_at_utc=created_at,
		source_artifact_root=(
			None
			if payload.get('source_artifact_root') is None
			else Path(str(payload['source_artifact_root']))
		),
		output_dir=Path(output),
		items=published,
		skipped_optional_items=[],
		warnings=[],
		manifest_path=manifest_path,
	)


__all__ = [
	'COMPLETION_ARTIFACT_TYPE',
	'COMPLETION_MANIFEST_NAME',
	'COMPLETION_SCHEMA_VERSION',
	'MIGRATION_ALLOWED_PUBLISH_SUFFIXES',
	'MIGRATION_STATUSES',
	'RAW_ARTIFACT_SUFFIXES',
	'ArtifactReuse',
	'CosineSimilaritySummary',
	'MetadataDiff',
	'MetadataDifference',
	'MigrationDecision',
	'NumericArrayComparison',
	'artifact_identity',
	'classify_metadata_diff',
	'commit_staged_artifact_directory',
	'compare_numeric_arrays',
	'cosine_similarity_summary_to_dict',
	'decide_migration_status',
	'metadata_diff_to_dict',
	'migration_decision_to_dict',
	'numeric_array_comparison_to_dict',
	'publish_lightweight_migration_results',
	'quarantine_artifact',
	'reuse_or_quarantine_artifact',
	'staged_artifact_directory',
	'summarize_rowwise_cosine_similarity',
	'validate_completion_manifest',
	'validate_migration_publish_manifest',
	'write_completion_manifest',
	'write_json_atomic',
	'write_text_atomic',
]

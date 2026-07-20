"""Strict manifest contract for ordered multi-head HMM pseudo-targets."""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from seis_ssl_cluster.clustering.features import (
	discover_embedding_inputs,
	embedding_input_metadata,
	file_sha256,
)
from seis_ssl_cluster.clustering.ordered_diagnostics import ordered_label_diagnostics
from seis_ssl_cluster.stratigraphy.targets import (
	ARTIFACT_TYPE as PSEUDO_TARGET_ARTIFACT_TYPE,
)
from seis_ssl_cluster.stratigraphy.targets import (
	discover_pseudo_target_inputs,
	load_pseudo_target_arrays,
	load_pseudo_target_metadata,
	validate_pseudo_target_arrays,
)

ARTIFACT_TYPE = 'strat_hmm_multi_head_target_manifest'
SCHEMA_VERSION = 1
_HEAD_KS = (6, 8, 10)


def build_multi_head_target_manifest(  # noqa: PLR0913
	*,
	manifest_path: str | Path,
	source_embedding_dir: str | Path,
	head_roots: Mapping[int | str, str | Path],
	replay_k6_root: str | Path,
	migration_decision: str | Path,
	control_summary: str | Path,
	ordering_orientation: str = 'increasing_downward',
) -> dict[str, object]:
	"""Validate references and atomically write a complete multi-head manifest.

	The manifest intentionally contains paths and hashes only; it never embeds target
	arrays.  K=6 may point to an immutable historical root while exact replay
	evidence is recorded separately before publication.  Publication also requires
	the positive migration and current-control preflight gates.
	"""
	validate_multi_head_target_publication_preflight(
		migration_decision=migration_decision,
		control_summary=control_summary,
	)
	if ordering_orientation != 'increasing_downward':
		raise ValueError('ordering_orientation must be increasing_downward')
	roots = _normalized_head_roots(head_roots)
	ks = _validate_head_ks(roots)
	embeddings = tuple(discover_embedding_inputs(source_embedding_dir))
	if not embeddings:
		raise ValueError('source_embedding_dir contains no embedding artifacts')
	embedding_by_survey = {item.survey_id: item for item in embeddings}
	head_payloads: dict[str, object] = {}
	common: dict[str, object] | None = None
	for k in ks:
		head = _head_reference(Path(roots[k]), k=k)
		if set(head['surveys']) != set(embedding_by_survey):
			raise ValueError(f'head k={k} survey set does not match source embeddings')
		_validate_head_embedding_alignment(head, embedding_by_survey, k=k)
		current_common = _common_contract(head, k=k)
		if common is None:
			common = current_common
		elif common != current_common:
			raise ValueError(f'head k={k} token grids or valid-token masks differ')
		head_payloads[str(k)] = head
	if common is None:
		raise AssertionError('at least one head is required')
	payload: dict[str, object] = {
		'artifact_type': ARTIFACT_TYPE,
		'schema_version': SCHEMA_VERSION,
		'head_ks': list(ks),
		'ordering_orientation': ordering_orientation,
		'source_embedding': _embedding_identity(
			Path(source_embedding_dir),
			embedding_by_survey,
		),
		'common': common,
		'heads': head_payloads,
		'cross_head_diagnostics': multi_head_cross_head_diagnostics(roots),
	}
	payload['k6_replay_parity'] = compare_k6_replay(
		historical_root=Path(roots[6]),
		replay_root=replay_k6_root,
	)
	payload['k6_replay_parity']['replay_root'] = str(replay_k6_root)
	if not payload['k6_replay_parity']['exact']:
		raise ValueError('K=6 replay parity is not exact; refusing complete manifest')
	validate_multi_head_target_manifest(payload, verify_hashes=True)
	_write_json_atomic(Path(manifest_path), payload)
	return payload


def validate_multi_head_target_publication_preflight(
	*,
	migration_decision: str | Path,
	control_summary: str | Path,
) -> None:
	"""Reject manifest publication unless the required K=6 gates are positive."""
	migration_status = _json_object(Path(migration_decision)).get('status')
	if migration_status != 'PASS_WITH_NUMERIC_DRIFT':
		raise ValueError(
			'migration preflight requires PASS_WITH_NUMERIC_DRIFT; '
			f'got {migration_status!r}'
		)
	control_payload = _json_object(Path(control_summary))
	readiness = control_payload.get('readiness', control_payload)
	if not isinstance(readiness, dict):
		raise TypeError('control summary readiness must be an object')
	control_status = readiness.get('status')
	if control_status != 'CONTROL_READY_POSITIVE':
		raise ValueError(
			'control preflight requires CONTROL_READY_POSITIVE; '
			f'got {control_status!r}'
		)


def _json_object(path: Path) -> dict[str, object]:
	try:
		payload = json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		raise ValueError(f'expected JSON object: {path}') from exc
	if not isinstance(payload, dict):
		raise TypeError(f'expected JSON object: {path}')
	return payload


def load_multi_head_target_manifest(
	path: str | Path,
	*,
	validate_array_semantics: bool = True,
) -> dict[str, object]:
	"""Load a manifest with strict reference validation.

	Set ``validate_array_semantics`` to false for configuration-only consumers.
	That mode verifies the schema, metadata identities, and every referenced file
	digest without materializing pseudo-target arrays.  Full target-array semantic
	validation remains the default for artifact validation and publication.
	"""
	try:
		payload = json.loads(Path(path).read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		raise ValueError(f'multi-head manifest must be valid JSON: {path}') from exc
	if not isinstance(payload, dict):
		raise TypeError('multi-head manifest must be a JSON object')
	validate_multi_head_target_manifest(
		payload,
		verify_hashes=True,
		validate_array_semantics=validate_array_semantics,
	)
	return payload


def validate_multi_head_target_manifest(  # noqa: C901, PLR0912
	payload: Mapping[str, object],
	*,
	verify_hashes: bool = False,
	validate_array_semantics: bool = True,
) -> None:
	"""Strictly validate schema-v1 references and shared target semantics."""
	_required_keys(
		payload,
		{
			'artifact_type',
			'schema_version',
			'head_ks',
			'ordering_orientation',
			'source_embedding',
			'common',
			'heads',
			'cross_head_diagnostics',
			'k6_replay_parity',
		},
		'manifest',
	)
	if payload['artifact_type'] != ARTIFACT_TYPE or payload['schema_version'] != 1:
		raise ValueError('unsupported multi-head target manifest schema')
	if payload['ordering_orientation'] != 'increasing_downward':
		raise ValueError('manifest ordering_orientation must be increasing_downward')
	if not isinstance(payload['head_ks'], list):
		raise TypeError('manifest head_ks must be a list')
	ks = tuple(payload['head_ks'])
	if any(isinstance(k, bool) or not isinstance(k, int) for k in ks):
		raise TypeError('manifest head_ks must contain integers')
	if tuple(sorted(ks)) != ks or len(set(ks)) != len(ks) or any(k < 2 for k in ks):
		raise ValueError('manifest head_ks must be unique, ascending integers >= 2')
	if ks != _HEAD_KS:
		raise ValueError(f'manifest head_ks must be {list(_HEAD_KS)!r}')
	head_values = _mapping(payload['heads'], 'manifest heads')
	if set(head_values) != {str(k) for k in ks}:
		raise ValueError('manifest heads must contain exactly one entry per head K')
	common = _mapping(payload['common'], 'manifest common')
	_required_keys(
		common,
		{'survey_ids', 'token_grid_shapes', 'valid_tokens_sha256'},
		'manifest common',
	)
	_validate_cross_head_diagnostics(payload['cross_head_diagnostics'])
	source_embedding = _mapping(payload['source_embedding'], 'source_embedding')
	_validate_embedding_identity(source_embedding, verify_hashes=verify_hashes)
	survey_ids = common['survey_ids']
	if (
		not isinstance(survey_ids, list)
		or not survey_ids
		or survey_ids != sorted(survey_ids)
	):
		raise ValueError('manifest common survey_ids must be a non-empty sorted list')
	common_token_grid_shapes = _mapping(
		common['token_grid_shapes'], 'manifest common token_grid_shapes'
	)
	common_valid_tokens_sha256 = _mapping(
		common['valid_tokens_sha256'], 'manifest common valid_tokens_sha256'
	)
	if set(common_token_grid_shapes) != set(survey_ids):
		raise ValueError('manifest common token_grid_shapes survey set mismatch')
	if set(common_valid_tokens_sha256) != set(survey_ids):
		raise ValueError('manifest common valid_tokens_sha256 survey set mismatch')
	for k in ks:
		head = _mapping(head_values[str(k)], f'head k={k}')
		_required_keys(
			head, {'pseudo_target_root', 'surveys', 'diagnostics'}, f'head k={k}'
		)
		surveys = _mapping(head['surveys'], f'head k={k} surveys')
		if set(surveys) != set(survey_ids):
			raise ValueError(f'head k={k} survey set mismatch')
		_validate_head_diagnostics(head['diagnostics'], surveys=surveys, k=k)
		for survey_id in survey_ids:
			entry = _mapping(surveys[survey_id], f'head k={k} survey {survey_id}')
			_required_keys(
				entry,
				{
					'labels',
					'confidence',
					'valid_tokens',
					'metadata',
					'token_grid_shape',
					'diagnostics',
				},
				'target reference',
			)
			if 'boundary_weight' in entry:
				raise ValueError(
					'multi-head schema-v1 target references forbid boundary weights'
				)
			_validate_target_reference_schema(entry)
			_validate_common_target_contract(
				common_token_grid_shapes,
				common_valid_tokens_sha256,
				entry,
				survey_id=str(survey_id),
			)
			if verify_hashes:
				_validate_reference_hashes(
					entry,
					k=k,
					survey_id=survey_id,
					validate_array_semantics=validate_array_semantics,
				)
	if verify_hashes:
		_validate_manifest_embedding_alignment(
			head_values,
			survey_ids,
			source_embedding,
			ks,
			validate_array_semantics=validate_array_semantics,
		)
	_validate_k6_replay_parity(payload['k6_replay_parity'], survey_ids)


def compare_k6_replay(
	*, historical_root: str | Path, replay_root: str | Path
) -> dict[str, object]:
	"""Require exact K=6 decoded/pseudo-target semantics for replay evidence."""
	historical = _head_reference(Path(historical_root), k=6)
	replay = _head_reference(Path(replay_root), k=6)
	if set(historical['surveys']) != set(replay['surveys']):
		raise ValueError('K=6 replay survey set mismatch')
	checks: dict[str, bool] = {}
	for survey_id in historical['surveys']:
		left = _load_reference_arrays(
			_mapping(historical['surveys'][survey_id], 'historical')
		)
		right = _load_reference_arrays(_mapping(replay['surveys'][survey_id], 'replay'))
		for name in ('labels', 'confidence', 'valid_tokens'):
			checks[f'{survey_id}.{name}'] = bool(
				np.array_equal(left[name], right[name])
			)
		checks[f'{survey_id}.state_occupancy'] = bool(
			np.array_equal(
				np.bincount(left['labels'][left['valid_tokens']], minlength=6),
				np.bincount(right['labels'][right['valid_tokens']], minlength=6),
			)
		)
		checks[f'{survey_id}.transition_counts'] = bool(
			np.array_equal(
				_transition_counts(left['labels']), _transition_counts(right['labels'])
			)
		)
		checks[f'{survey_id}.ordered_violations'] = bool(
			_ordered_violation_count(left['labels'], k=6)
			== _ordered_violation_count(right['labels'], k=6)
		)
	return {'exact': all(checks.values()), 'checks': checks}


def multi_head_cross_head_diagnostics(
	head_roots: Mapping[int | str, str | Path],
) -> dict[str, object]:
	"""Compare normalized ordered labels across heads on their common valid mask."""
	roots = _normalized_head_roots(head_roots)
	ks = _validate_head_ks(roots)
	heads = {k: _head_reference(Path(roots[k]), k=k) for k in ks}
	result: dict[str, object] = {}
	for index, left_k in enumerate(ks):
		for right_k in ks[index + 1 :]:
			rows: list[tuple[np.ndarray, np.ndarray]] = []
			for survey_id in heads[left_k]['surveys']:
				left = _load_reference_arrays(
					_mapping(heads[left_k]['surveys'][survey_id], 'left')
				)
				right = _load_reference_arrays(
					_mapping(heads[right_k]['surveys'][survey_id], 'right')
				)
				if not np.array_equal(left['valid_tokens'], right['valid_tokens']):
					raise ValueError('cross-head valid-token masks differ')
				valid = left['valid_tokens']
				rows.append(
					(
						left['labels'][valid] / (left_k - 1),
						right['labels'][valid] / (right_k - 1),
					)
				)
			one = np.concatenate([item[0] for item in rows])
			two = np.concatenate([item[1] for item in rows])
			result[f'k{left_k}_k{right_k}'] = {
				'mae': float(np.mean(np.abs(one - two))),
				'correlation': float(np.corrcoef(one, two)[0, 1])
				if one.size > 1
				else 1.0,
				'rank_order_disagreement': float(
					np.mean(np.sign(np.diff(one)) != np.sign(np.diff(two)))
					if one.size > 1
					else 0.0
				),
			}
	return result


def _validate_cross_head_diagnostics(value: object) -> None:
	"""Validate the persisted pairwise normalized-coordinate sanity evidence."""
	pairs = _mapping(value, 'cross_head_diagnostics')
	expected_pairs = {'k6_k8', 'k6_k10', 'k8_k10'}
	_required_keys(pairs, expected_pairs, 'cross_head_diagnostics')
	for pair_name in sorted(expected_pairs):
		metrics = _mapping(pairs[pair_name], f'cross-head diagnostics {pair_name}')
		_required_keys(
			metrics,
			{'mae', 'correlation', 'rank_order_disagreement'},
			f'cross-head diagnostics {pair_name}',
		)
		for metric_name, metric_value in metrics.items():
			if (
				isinstance(metric_value, bool)
				or not isinstance(metric_value, (int, float))
				or not math.isfinite(metric_value)
			):
				raise ValueError(
					f'cross-head diagnostics {pair_name} {metric_name} must be finite'
				)


def _validate_head_diagnostics(
	value: object,
	*,
	surveys: Mapping[str, object],
	k: int,
) -> None:
	"""Validate the strict per-head diagnostics payload."""
	diagnostics = _mapping(value, f'head k={k} diagnostics')
	_required_keys(diagnostics, {'per_survey'}, f'head k={k} diagnostics')
	per_survey = _mapping(diagnostics['per_survey'], f'head k={k} per_survey')
	if set(per_survey) != set(surveys):
		raise ValueError(f'head k={k} diagnostics survey set mismatch')
	for survey_id, entry in surveys.items():
		survey_diagnostics = _mapping(
			per_survey[survey_id], f'head k={k} diagnostics {survey_id}'
		)
		_validate_survey_diagnostics(survey_diagnostics, k=k, survey_id=survey_id)
		target_reference = _mapping(entry, f'head k={k} survey {survey_id}')
		if survey_diagnostics != target_reference['diagnostics']:
			raise ValueError(
				f'head k={k} diagnostics do not match target reference for {survey_id}'
			)


def _validate_survey_diagnostics(
	value: Mapping[str, object], *, k: int, survey_id: str
) -> None:
	"""Validate generated diagnostics without accepting extension fields."""
	keys = {
		'valid_token_count',
		'invalid_token_count',
		'state_occupancy_count',
		'state_occupancy_ratio',
		'empty_state_count',
		'effective_k',
		'min_occupancy',
		'max_occupancy',
		'confidence_quantiles',
		'trace_unique_state_count',
		'trace_transition_count',
		'initial_state_distribution',
		'terminal_state_distribution',
		'ordered_path_violation_count',
	}
	name = f'head k={k} diagnostics {survey_id}'
	_required_keys(value, keys, name)
	_validate_occupancy_diagnostics(value, k=k, name=name)
	_validate_confidence_quantiles(value['confidence_quantiles'], name=name)
	for quantile_name in ('trace_unique_state_count', 'trace_transition_count'):
		_validate_trace_quantiles(value[quantile_name], f'{name} {quantile_name}')
	for endpoint_name in (
		'initial_state_distribution',
		'terminal_state_distribution',
	):
		_nonnegative_int_list(value[endpoint_name], k=k, name=f'{name} {endpoint_name}')
	if _nonnegative_int(
		value['ordered_path_violation_count'], f'{name} ordered path violations'
	):
		raise ValueError(f'{name} has ordered path violations')


def _validate_occupancy_diagnostics(
	value: Mapping[str, object], *, k: int, name: str
) -> None:
	"""Validate occupancy-derived diagnostics and their internal consistency."""
	valid_count = _nonnegative_int(value['valid_token_count'], f'{name} valid count')
	_nonnegative_int(value['invalid_token_count'], f'{name} invalid count')
	if valid_count == 0:
		raise ValueError(f'{name} must contain valid tokens')
	counts = _nonnegative_int_list(
		value['state_occupancy_count'], k=k, name=f'{name} occupancy count'
	)
	if sum(counts) != valid_count:
		raise ValueError(f'{name} occupancy count does not match valid token count')
	ratios = _finite_number_list(
		value['state_occupancy_ratio'], k=k, name=f'{name} occupancy ratio'
	)
	if any(ratio < 0.0 for ratio in ratios) or not all(
		math.isclose(ratio, count / valid_count, rel_tol=1e-9, abs_tol=1e-12)
		for count, ratio in zip(counts, ratios, strict=True)
	):
		raise ValueError(f'{name} occupancy ratio does not match occupancy count')
	empty_count = _nonnegative_int(value['empty_state_count'], f'{name} empty state')
	if empty_count != sum(count == 0 for count in counts):
		raise ValueError(f'{name} empty state count does not match occupancy count')
	if empty_count:
		raise ValueError(f'{name} has an empty state')
	if _nonnegative_int(value['min_occupancy'], f'{name} min occupancy') != min(counts):
		raise ValueError(f'{name} min occupancy does not match occupancy count')
	if _nonnegative_int(value['max_occupancy'], f'{name} max occupancy') != max(counts):
		raise ValueError(f'{name} max occupancy does not match occupancy count')
	effective_k = _finite_number(value['effective_k'], f'{name} effective K')
	if not 1.0 <= effective_k <= float(k):
		raise ValueError(f'{name} effective K must be in [1, {k}]')


def _validate_confidence_quantiles(value: object, *, name: str) -> None:
	"""Validate persisted bootstrap-confidence summary quantiles."""
	confidence = _finite_number_list(
		value, k=5, name=f'{name} confidence quantiles'
	)
	if (
		any(quantile < 0.0 or quantile > 1.0 for quantile in confidence)
		or confidence != sorted(confidence)
	):
		raise ValueError(
			f'{name} confidence quantiles must be ordered values in [0, 1]'
		)


def _validate_trace_quantiles(value: object, name: str) -> None:
	quantiles = _mapping(value, name)
	_required_keys(quantiles, {'mean', 'median', 'p05', 'p95', 'max'}, name)
	values = {
		key: _finite_number(quantiles[key], f'{name} {key}')
		for key in ('mean', 'median', 'p05', 'p95', 'max')
	}
	if any(item < 0.0 for item in values.values()) or not (
		values['p05'] <= values['median'] <= values['p95'] <= values['max']
	):
		raise ValueError(f'{name} must contain ordered non-negative quantiles')


def _validate_k6_replay_parity(value: object, survey_ids: Sequence[object]) -> None:
	"""Validate required exact K=6 replay evidence for complete manifests."""
	parity = _mapping(value, 'k6_replay_parity')
	_required_keys(parity, {'exact', 'checks', 'replay_root'}, 'k6_replay_parity')
	if not isinstance(parity['replay_root'], str) or not parity['replay_root']:
		raise TypeError('k6_replay_parity replay_root must be a non-empty string')
	if not isinstance(parity['exact'], bool):
		raise TypeError('k6_replay_parity exact must be a boolean')
	checks = _mapping(parity['checks'], 'k6_replay_parity checks')
	expected = {
		f'{survey_id}.{metric}'
		for survey_id in survey_ids
		for metric in (
			'labels',
			'confidence',
			'valid_tokens',
			'state_occupancy',
			'transition_counts',
			'ordered_violations',
		)
	}
	_required_keys(checks, expected, 'k6_replay_parity checks')
	if not all(isinstance(result, bool) for result in checks.values()):
		raise TypeError('k6_replay_parity checks must contain booleans')
	if not parity['exact'] or not all(checks.values()):
		raise ValueError('K=6 replay parity is not exact')


def _head_reference(root: Path, *, k: int) -> dict[str, object]:
	surveys: dict[str, object] = {}
	for item in discover_pseudo_target_inputs(root, k=k):
		metadata = load_pseudo_target_metadata(item)
		if metadata.get('schema_version') != 1:
			raise ValueError(f'head k={k} must use schema-v1 pseudo-targets')
		if item.boundary_weight_path is not None:
			raise ValueError(f'head k={k} must not contain a boundary-weight artifact')
		arrays = load_pseudo_target_arrays(item)
		if int(np.count_nonzero(arrays.valid_tokens)) == 0:
			raise ValueError(f'head k={k} has no valid tokens for {item.survey_id}')
		occupancy = np.bincount(arrays.labels[arrays.valid_tokens], minlength=k)
		if np.any(occupancy == 0):
			raise ValueError(f'head k={k} has an empty state for {item.survey_id}')
		diagnostics = _head_diagnostics(
			arrays.labels, arrays.confidence, arrays.valid_tokens, k
		)
		if diagnostics['ordered_path_violation_count']:
			raise ValueError(
				f'head k={k} has ordered path violations for {item.survey_id}'
			)
		surveys[item.survey_id] = {
			'labels': _file_reference(item.labels_path),
			'confidence': _file_reference(item.confidence_path),
			'valid_tokens': _file_reference(item.valid_tokens_path),
			'metadata': _file_reference(item.metadata_path),
			'token_grid_shape': list(arrays.labels.shape),
			'diagnostics': diagnostics,
		}
	return {
		'pseudo_target_root': str(root),
		'surveys': surveys,
		'diagnostics': {
			'per_survey': {key: value['diagnostics'] for key, value in surveys.items()}
		},
	}


def _common_contract(head: Mapping[str, object], *, k: int) -> dict[str, object]:
	surveys = _mapping(head['surveys'], f'head k={k} surveys')
	return {
		'survey_ids': sorted(surveys),
		'token_grid_shapes': {
			key: _mapping(value, 'survey')['token_grid_shape']
			for key, value in sorted(surveys.items())
		},
		'valid_tokens_sha256': {
			key: _mapping(_mapping(value, 'survey')['valid_tokens'], 'valid ref')[
				'sha256'
			]
			for key, value in sorted(surveys.items())
		},
	}


def _embedding_identity(root: Path, entries: Mapping[str, Any]) -> dict[str, object]:
	return {
		'input_dir': str(root),
		'surveys': {
			key: {
				'metadata': embedding_input_metadata(item),
				'embedding_path': str(item.embeddings_path),
				'embedding_sha256': file_sha256(item.embeddings_path),
				'metadata_path': str(item.metadata_path),
				'metadata_sha256': file_sha256(item.metadata_path),
				'valid_tokens_path': str(item.valid_tokens_path),
				'valid_tokens_sha256': file_sha256(item.valid_tokens_path),
			}
			for key, item in sorted(entries.items())
		},
	}


def _validate_head_embedding_alignment(
	head: Mapping[str, object],
	embedding_by_survey: Mapping[str, Any],
	*,
	k: int,
) -> None:
	"""Require every target grid and mask to align exactly with its embedding."""
	surveys = _mapping(head['surveys'], f'head k={k} surveys')
	for survey_id, embedding in embedding_by_survey.items():
		entry = _mapping(surveys[survey_id], 'survey')
		target = _load_reference_arrays(entry)
		embedding_valid = np.load(embedding.valid_tokens_path)
		if target['labels'].shape != embedding_valid.shape:
			raise ValueError(
				f'head k={k} {survey_id} token grid does not match source embedding'
			)
		if not np.array_equal(target['valid_tokens'], embedding_valid):
			raise ValueError(
				f'head k={k} {survey_id} valid-token mask does not match '
				'source embedding'
			)


def _validate_embedding_identity(
	value: Mapping[str, object], *, verify_hashes: bool
) -> None:
	_required_keys(value, {'input_dir', 'surveys'}, 'source_embedding')
	for entry in _mapping(value['surveys'], 'source_embedding surveys').values():
		item = _mapping(entry, 'source embedding survey')
		_required_keys(
			item,
			{
				'metadata',
				'embedding_path',
				'embedding_sha256',
				'metadata_path',
				'metadata_sha256',
				'valid_tokens_path',
				'valid_tokens_sha256',
			},
			'source embedding survey',
		)
		if not verify_hashes:
			continue
		for path_key, hash_key in (
			('embedding_path', 'embedding_sha256'),
			('metadata_path', 'metadata_sha256'),
			('valid_tokens_path', 'valid_tokens_sha256'),
		):
			if file_sha256(Path(str(item[path_key]))) != item[hash_key]:
				raise ValueError(f'source embedding {path_key} hash mismatch')


def _validate_manifest_embedding_alignment(
	head_values: Mapping[str, object],
	survey_ids: list[object],
	source_embedding: Mapping[str, object],
	ks: Sequence[int],
	*,
	validate_array_semantics: bool,
) -> None:
	"""Recheck target-to-embedding alignment when loading a manifest.

	The reference-only path proves mask identity from the already verified file
	digests.  It deliberately avoids reading target or embedding arrays; the
	publication path performs the additional shape and bitwise checks.
	"""
	embeddings = _mapping(source_embedding['surveys'], 'source_embedding surveys')
	if set(embeddings) != set(survey_ids):
		raise ValueError('source embedding survey set does not match manifest common')
	for k in ks:
		surveys = _mapping(
			_mapping(head_values[str(k)], f'head k={k}')['surveys'],
			f'head k={k} surveys',
		)
		for survey_id in survey_ids:
			entry = _mapping(surveys[str(survey_id)], 'target reference')
			embedding = _mapping(embeddings[str(survey_id)], 'source embedding survey')
			valid_reference = _mapping(
				entry['valid_tokens'],
				'target valid_tokens reference',
			)
			if valid_reference['sha256'] != embedding['valid_tokens_sha256']:
				raise ValueError(
					f'head k={k} {survey_id} valid-token mask does not match '
					'source embedding'
				)
			if not validate_array_semantics:
				continue
			target = _load_reference_arrays(entry)
			embedding_valid = np.load(Path(str(embedding['valid_tokens_path'])))
			if target['labels'].shape != embedding_valid.shape:
				raise ValueError(
					f'head k={k} {survey_id} token grid does not match source embedding'
				)
			if not np.array_equal(target['valid_tokens'], embedding_valid):
				raise ValueError(
					f'head k={k} {survey_id} valid-token mask does not match '
					'source embedding'
				)


def _head_diagnostics(
	labels: np.ndarray, confidence: np.ndarray, valid: np.ndarray, k: int
) -> dict[str, object]:
	values = labels[valid]
	counts = np.bincount(values, minlength=k)
	transitions = _transition_counts(labels)
	trace_unique: list[int] = []
	for raw_trace in labels.reshape(-1, labels.shape[-1]):
		valid_trace = raw_trace[raw_trace >= 0]
		if valid_trace.size:
			trace_unique.append(int(np.unique(valid_trace).size))
	return {
		'valid_token_count': int(valid.sum()),
		'invalid_token_count': int((~valid).sum()),
		'state_occupancy_count': counts.tolist(),
		'state_occupancy_ratio': (counts / counts.sum()).tolist(),
		'empty_state_count': int((counts == 0).sum()),
		'effective_k': float(
			math.exp(
				-np.sum(
					(counts / counts.sum())
					* np.log(np.maximum(counts / counts.sum(), 1e-30))
				)
			)
		),
		'min_occupancy': int(counts.min()),
		'max_occupancy': int(counts.max()),
		'confidence_quantiles': np.quantile(
			confidence[valid], [0, 0.05, 0.5, 0.95, 1]
		).tolist(),
		'trace_unique_state_count': _quantiles(trace_unique),
		'trace_transition_count': _quantiles(transitions.tolist()),
		'initial_state_distribution': _endpoint_distribution(labels, first=True, k=k),
		'terminal_state_distribution': _endpoint_distribution(labels, first=False, k=k),
		'ordered_path_violation_count': _ordered_violation_count(labels, k=k),
	}


def _transition_counts(labels: np.ndarray) -> np.ndarray:
	return np.asarray(
		[
			np.count_nonzero(np.diff(trace[trace >= 0]) != 0)
			for trace in labels.reshape(-1, labels.shape[-1])
		],
		dtype=np.int64,
	)


def _ordered_violation_count(labels: np.ndarray, *, k: int) -> int:
	return int(ordered_label_diagnostics(labels, k=k)['reverse_transition_count'])


def _endpoint_distribution(labels: np.ndarray, *, first: bool, k: int) -> list[int]:
	values = [
		trace[trace >= 0][0 if first else -1]
		for trace in labels.reshape(-1, labels.shape[-1])
		if np.any(trace >= 0)
	]
	return np.bincount(values, minlength=k).tolist()


def _quantiles(values: Sequence[int]) -> dict[str, float]:
	array = np.asarray(values, dtype=np.float64)
	return {
		'mean': float(np.mean(array)),
		'median': float(np.quantile(array, 0.5)),
		'p05': float(np.quantile(array, 0.05)),
		'p95': float(np.quantile(array, 0.95)),
		'max': float(np.max(array)),
	}


def _file_reference(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': file_sha256(path)}


def _validate_reference_hashes(
	entry: Mapping[str, object],
	*,
	k: int,
	survey_id: str,
	validate_array_semantics: bool,
) -> None:
	refs = {
		name: _mapping(entry[name], f'{name} reference')
		for name in ('labels', 'confidence', 'valid_tokens', 'metadata')
	}
	shape = entry['token_grid_shape']
	if not isinstance(shape, list):
		raise TypeError(f'head k={k} {survey_id} token_grid_shape must be a list')
	validate_multi_head_target_reference(
		k=k,
		survey_id=survey_id,
		labels_path=Path(str(refs['labels']['path'])),
		confidence_path=Path(str(refs['confidence']['path'])),
		valid_tokens_path=Path(str(refs['valid_tokens']['path'])),
		metadata_path=Path(str(refs['metadata']['path'])),
		hashes={name: refs[name]['sha256'] for name in refs},
		expected_token_grid_shape=shape,
		validate_array_semantics=validate_array_semantics,
	)


def validate_multi_head_target_reference(  # noqa: PLR0913
	*,
	k: int,
	survey_id: str,
	labels_path: str | Path,
	confidence_path: str | Path,
	valid_tokens_path: str | Path,
	metadata_path: str | Path,
	hashes: Mapping[str, object],
	expected_token_grid_shape: Sequence[object] | None = None,
	validate_array_semantics: bool = True,
) -> None:
	"""Validate one referenced schema-v1 target without duplicating its contract.

	Reference-only validation verifies immutable file identities and metadata but
	does not call :func:`numpy.load`.  Full validation retains the array-level
	range, validity, occupancy, and ordering checks used at publication time.
	"""
	_required_keys(
		hashes,
		{'labels', 'confidence', 'valid_tokens', 'metadata'},
		'multi-head target hashes',
	)
	paths = {
		'labels': Path(labels_path),
		'confidence': Path(confidence_path),
		'valid_tokens': Path(valid_tokens_path),
		'metadata': Path(metadata_path),
	}
	for name, path in paths.items():
		digest = hashes[name]
		if not isinstance(digest, str) or file_sha256(path) != digest:
			raise ValueError(f'head k={k} {survey_id} {name} hash mismatch')
	_validate_referenced_target_metadata(
		metadata_path=paths['metadata'],
		labels_path=paths['labels'],
		k=k,
		survey_id=survey_id,
	)
	if not validate_array_semantics:
		return
	arrays = {
		name: np.load(path, mmap_mode='r', allow_pickle=False)
		for name, path in paths.items()
		if name != 'metadata'
	}
	if expected_token_grid_shape is not None and arrays['labels'].shape != tuple(
		expected_token_grid_shape
	):
		raise ValueError(f'head k={k} {survey_id} token grid mismatch')
	_validate_referenced_target_semantics(
		arrays,
		k=k,
		survey_id=survey_id,
	)


def _validate_referenced_target_metadata(
	*,
	metadata_path: Path,
	labels_path: Path,
	k: int,
	survey_id: str,
) -> None:
	"""Recheck each referenced schema-v1 target's blocking semantics."""
	try:
		metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		raise ValueError(
			f'head k={k} {survey_id} metadata must be valid JSON'
		) from exc
	if not isinstance(metadata, Mapping):
		raise TypeError(f'head k={k} {survey_id} metadata must be an object')
	if (
		metadata.get('artifact_type') != PSEUDO_TARGET_ARTIFACT_TYPE
		or metadata.get('schema_version') != 1
		or metadata.get('k') != k
		or metadata.get('survey_id') != survey_id
	):
		raise ValueError(
			f'head k={k} {survey_id} must use matching schema-v1 pseudo-target metadata'
		)
	boundary_weight_path = labels_path.with_name(
		f'{survey_id}.hmm_boundary_weight_token.npy'
	)
	if boundary_weight_path.exists():
		raise ValueError(
			f'head k={k} {survey_id} must not contain a boundary-weight artifact'
		)


def _validate_referenced_target_semantics(
	arrays: Mapping[str, np.ndarray],
	*,
	k: int,
	survey_id: str,
) -> None:
	"""Recheck array-level blocking semantics after reference validation."""
	validate_pseudo_target_arrays(
		arrays['labels'],
		arrays['confidence'],
		arrays['valid_tokens'],
		k=k,
		survey_id=survey_id,
	)
	occupancy = np.bincount(arrays['labels'][arrays['valid_tokens']], minlength=k)
	if np.any(occupancy == 0):
		raise ValueError(f'head k={k} has an empty state for {survey_id}')
	if _ordered_violation_count(arrays['labels'], k=k):
		raise ValueError(f'head k={k} has ordered path violations for {survey_id}')


def _validate_target_reference_schema(entry: Mapping[str, object]) -> None:
	for name in ('labels', 'confidence', 'valid_tokens', 'metadata'):
		_required_keys(
			_mapping(entry[name], f'{name} reference'),
			{'path', 'sha256'},
			f'{name} reference',
		)


def _validate_common_target_contract(
	common_token_grid_shapes: Mapping[str, object],
	common_valid_tokens_sha256: Mapping[str, object],
	entry: Mapping[str, object],
	*,
	survey_id: str,
) -> None:
	if common_token_grid_shapes[survey_id] != entry['token_grid_shape']:
		raise ValueError(
		f'manifest common token grid does not match target reference for {survey_id}'
	)
	valid_tokens = _mapping(entry['valid_tokens'], 'valid_tokens reference')
	if common_valid_tokens_sha256[survey_id] != valid_tokens['sha256']:
		raise ValueError(
		f'manifest common valid-token hash does not match target reference for '
		f'{survey_id}'
	)


def _load_reference_arrays(entry: Mapping[str, object]) -> dict[str, np.ndarray]:
	return {
		name: np.load(Path(str(_mapping(entry[name], name)['path'])))
		for name in ('labels', 'confidence', 'valid_tokens')
	}


def _validate_head_ks(head_roots: Mapping[int | str, str | Path]) -> tuple[int, ...]:
	values = _normalized_head_roots(head_roots)
	if tuple(sorted(values)) != _HEAD_KS:
		raise ValueError(f'head roots must contain exactly {_HEAD_KS!r}')
	return _HEAD_KS


def _normalized_head_roots(
	head_roots: Mapping[int | str, str | Path],
) -> dict[int, str | Path]:
	try:
		values = {int(key): value for key, value in head_roots.items()}
	except (TypeError, ValueError) as exc:
		raise TypeError('head roots must use integer K keys') from exc
	if len(values) != len(head_roots):
		raise ValueError('head roots contain duplicate K values')
	return values


def _nonnegative_int(value: object, name: str) -> int:
	if isinstance(value, bool) or not isinstance(value, int) or value < 0:
		raise ValueError(f'{name} must be a non-negative integer')
	return value


def _nonnegative_int_list(value: object, *, k: int, name: str) -> list[int]:
	if not isinstance(value, list) or len(value) != k:
		raise ValueError(f'{name} must be a list of {k} non-negative integers')
	return [_nonnegative_int(item, name) for item in value]


def _finite_number(value: object, name: str) -> float:
	if (
		isinstance(value, bool)
		or not isinstance(value, (int, float))
		or not math.isfinite(value)
	):
		raise ValueError(f'{name} must be finite')
	return float(value)


def _finite_number_list(value: object, *, k: int, name: str) -> list[float]:
	if not isinstance(value, list) or len(value) != k:
		raise ValueError(f'{name} must be a list of {k} finite values')
	return [_finite_number(item, name) for item in value]


def _mapping(value: object, name: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{name} must be a mapping')
	return value


def _required_keys(
	value: Mapping[str, object],
	keys: set[str],
	name: str,
	*,
	optional_keys: set[str] | None = None,
) -> None:
	unknown = set(value) - keys - (optional_keys or set())
	if unknown:
		raise ValueError(f'{name} has unknown fields: {sorted(unknown)!r}')
	missing = keys - set(value)
	if missing:
		raise ValueError(f'{name} is missing fields: {sorted(missing)!r}')


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.NamedTemporaryFile(
		'w', dir=path.parent, delete=False, encoding='utf-8'
	) as handle:
		temporary = Path(handle.name)
		json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
		handle.write('\n')
	try:
		temporary.replace(path)
	finally:
		temporary.unlink(missing_ok=True)


__all__ = [
	'ARTIFACT_TYPE',
	'SCHEMA_VERSION',
	'build_multi_head_target_manifest',
	'compare_k6_replay',
	'load_multi_head_target_manifest',
	'multi_head_cross_head_diagnostics',
	'validate_multi_head_target_manifest',
	'validate_multi_head_target_publication_preflight',
	'validate_multi_head_target_reference',
]

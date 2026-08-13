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
SCHEMA_VERSION = 2
_LEGACY_SCHEMA_VERSION = 1


def build_multi_head_target_manifest(  # noqa: C901, PLR0912, PLR0913
	*,
	manifest_path: str | Path,
	source_embedding_dir: str | Path,
	head_roots: Mapping[int | str, str | Path],
	replay_k6_root: str | Path | None = None,
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
	source_target_alignment: dict[str, object] | None = None
	for k in ks:
		head = _head_reference(Path(roots[k]), k=k)
		if set(head['surveys']) != set(embedding_by_survey):
			raise ValueError(f'head k={k} survey set does not match source embeddings')
		current_alignment = _validate_head_embedding_alignment(
			head,
			embedding_by_survey,
			k=k,
		)
		current_common = _common_contract(head, k=k)
		if common is None:
			common = current_common
		elif common != current_common:
			raise ValueError(f'head k={k} token grids or valid-token masks differ')
		if source_target_alignment is None:
			source_target_alignment = current_alignment
		elif source_target_alignment != current_alignment:
			raise ValueError(f'head k={k} source-to-target alignment differs')
		head_payloads[str(k)] = head
	if common is None or source_target_alignment is None:
		raise AssertionError('at least one head is required')
	common['source_target_alignment'] = source_target_alignment
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
	if 6 in ks:
		if replay_k6_root is None:
			raise ValueError('K=6 manifests require replay_k6_root')
		if _same_resolved_path(Path(roots[6]), Path(replay_k6_root)):
			raise ValueError(
			'K=6 replay root must differ from the immutable historical '
			'training-target root'
		)
		payload['k6_replay_parity'] = compare_k6_replay(
			historical_root=Path(roots[6]),
			replay_root=replay_k6_root,
		)
		payload['k6_replay_parity']['replay_root'] = str(replay_k6_root)
		if not payload['k6_replay_parity']['exact']:
			raise ValueError(
				'K=6 replay parity is not exact; refusing complete manifest'
			)
	elif replay_k6_root is not None:
		raise ValueError('replay_k6_root requires a K=6 head')
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
	"""Load a v1 or v2 manifest with strict reference validation.

	Set ``validate_array_semantics`` to false for configuration-only consumers.
	That mode verifies the schema, metadata identities, and every referenced file
	digest without materializing pseudo-target arrays.  Full target-array semantic
	validation remains the default for artifact validation and publication.  Legacy
	v1 manifests remain loadable only under their original exact source-mask
	contract; newly published manifests always use v2 subset evidence.
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


def validate_multi_head_target_manifest(  # noqa: C901, PLR0912, PLR0915
	payload: Mapping[str, object],
	*,
	verify_hashes: bool = False,
	validate_array_semantics: bool = True,
) -> None:
	"""Strictly validate v1/v2 references and shared target semantics."""
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
		},
		'manifest',
		optional_keys={'k6_replay_parity'},
	)
	if payload['artifact_type'] != ARTIFACT_TYPE or payload['schema_version'] not in {
		_LEGACY_SCHEMA_VERSION,
		SCHEMA_VERSION,
	}:
		raise ValueError('unsupported multi-head target manifest schema')
	legacy_v1 = payload['schema_version'] == _LEGACY_SCHEMA_VERSION
	if payload['ordering_orientation'] != 'increasing_downward':
		raise ValueError('manifest ordering_orientation must be increasing_downward')
	if not isinstance(payload['head_ks'], list):
		raise TypeError('manifest head_ks must be a list')
	ks = tuple(payload['head_ks'])
	if any(isinstance(k, bool) or not isinstance(k, int) for k in ks):
		raise TypeError('manifest head_ks must contain integers')
	if (
		len(ks) < 2
		or tuple(sorted(ks)) != ks
		or len(set(ks)) != len(ks)
		or any(k < 2 for k in ks)
	):
		raise ValueError(
			'manifest head_ks must contain at least two unique, ascending integers >= 2'
		)
	head_values = _mapping(payload['heads'], 'manifest heads')
	if set(head_values) != {str(k) for k in ks}:
		raise ValueError('manifest heads must contain exactly one entry per head K')
	common = _mapping(payload['common'], 'manifest common')
	common_keys = {
		'survey_ids',
		'token_grid_shapes',
		'valid_tokens_sha256',
	}
	if not legacy_v1:
		common_keys.add('source_target_alignment')
	_required_keys(common, common_keys, 'manifest common')
	_validate_cross_head_diagnostics(payload['cross_head_diagnostics'], ks=ks)
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
	common_source_target_alignment: Mapping[str, object] | None = None
	if not legacy_v1:
		common_source_target_alignment = _mapping(
			common['source_target_alignment'],
			'manifest common source_target_alignment',
		)
	if set(common_token_grid_shapes) != set(survey_ids):
		raise ValueError('manifest common token_grid_shapes survey set mismatch')
	if set(common_valid_tokens_sha256) != set(survey_ids):
		raise ValueError('manifest common valid_tokens_sha256 survey set mismatch')
	if common_source_target_alignment is not None:
		_validate_source_target_alignment_contract(
			common_source_target_alignment,
			survey_ids,
		)
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
				optional_keys={'boundary_weight'},
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
	if legacy_v1:
		_validate_legacy_manifest_embedding_alignment(
			head_values,
			survey_ids,
			source_embedding,
			ks,
			validate_array_semantics=validate_array_semantics,
		)
	else:
		if common_source_target_alignment is None:
			raise AssertionError('schema-v2 alignment evidence is required')
		_validate_manifest_embedding_alignment(
			head_values,
			survey_ids,
			source_embedding,
			ks,
			common_source_target_alignment,
			validate_array_semantics=validate_array_semantics,
		)
	if 6 in ks:
		if 'k6_replay_parity' not in payload:
			raise ValueError('manifest is missing K=6 replay parity evidence')
		historical_head = _mapping(head_values['6'], 'head k=6')
		_validate_k6_replay_parity(
			payload['k6_replay_parity'],
			survey_ids,
			historical_root=Path(str(historical_head['pseudo_target_root'])),
			historical_targets=_mapping(historical_head['surveys'], 'head k=6 surveys'),
			verify_hashes=verify_hashes,
		)
	elif 'k6_replay_parity' in payload:
		raise ValueError('K=6 replay parity evidence requires a K=6 head')


def compare_k6_replay(
	*, historical_root: str | Path, replay_root: str | Path
) -> dict[str, object]:
	"""Require exact K=6 decoded/pseudo-target semantics for replay evidence."""
	if _same_resolved_path(Path(historical_root), Path(replay_root)):
		raise ValueError(
			'K=6 replay root must differ from the immutable historical '
			'training-target root'
		)
	historical = _head_reference(Path(historical_root), k=6)
	replay = _head_reference(Path(replay_root), k=6)
	if set(historical['surveys']) != set(replay['surveys']):
		raise ValueError('K=6 replay survey set mismatch')
	historical_decoded = _decoded_label_references(historical)
	replay_decoded = _decoded_label_references(replay)
	checks: dict[str, bool] = {}
	for survey_id in historical['surveys']:
		left = _load_reference_arrays(
			_mapping(historical['surveys'][survey_id], 'historical')
		)
		right = _load_reference_arrays(_mapping(replay['surveys'][survey_id], 'replay'))
		left_decoded = np.load(
			Path(str(historical_decoded[survey_id]['path'])), mmap_mode='r'
		)
		right_decoded = np.load(
			Path(str(replay_decoded[survey_id]['path'])), mmap_mode='r'
		)
		if _same_resolved_path(
			Path(str(historical_decoded[survey_id]['path'])),
			Path(str(replay_decoded[survey_id]['path'])),
		):
			raise ValueError(
				'K=6 replay decoded-label artifact must differ from the '
				'immutable historical artifact'
			)
		for name in ('labels', 'confidence', 'valid_tokens'):
			checks[f'{survey_id}.pseudo_target_{name}'] = bool(
				np.array_equal(left[name], right[name])
			)
		if ('boundary_weight' in left) != ('boundary_weight' in right):
			raise ValueError('K=6 replay boundary-weight policy differs')
		if 'boundary_weight' in left:
			checks[f'{survey_id}.pseudo_target_boundary_weight'] = bool(
				np.array_equal(left['boundary_weight'], right['boundary_weight'])
			)
		checks[f'{survey_id}.decoded_valid_token_mask'] = bool(
			np.array_equal(left_decoded >= 0, right_decoded >= 0)
		)
		checks[f'{survey_id}.decoded_invalid_positions'] = bool(
			np.array_equal(left_decoded < 0, right_decoded < 0)
		)
		checks[f'{survey_id}.decoded_labels'] = bool(
			np.array_equal(left_decoded, right_decoded)
		)
		checks[f'{survey_id}.decoded_state_occupancy'] = bool(
			np.array_equal(
				np.bincount(left_decoded[left_decoded >= 0], minlength=6),
				np.bincount(right_decoded[right_decoded >= 0], minlength=6),
			)
		)
		checks[f'{survey_id}.decoded_transition_counts'] = bool(
			np.array_equal(
				_transition_counts(left_decoded),
				_transition_counts(right_decoded),
			)
		)
		checks[f'{survey_id}.decoded_ordered_violations'] = bool(
			_ordered_violation_count(left_decoded, k=6)
			== _ordered_violation_count(right_decoded, k=6)
		)
	return {
		'exact': all(checks.values()),
		'checks': checks,
		'replay_artifacts': _replay_artifact_references(replay),
		'historical_decoded_labels': historical_decoded,
		'replay_decoded_labels': replay_decoded,
	}


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


def _validate_cross_head_diagnostics(value: object, *, ks: Sequence[int]) -> None:
	"""Validate the persisted pairwise normalized-coordinate sanity evidence."""
	pairs = _mapping(value, 'cross_head_diagnostics')
	expected_pairs = {
		f'k{left_k}_k{right_k}'
		for index, left_k in enumerate(ks)
		for right_k in ks[index + 1 :]
	}
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


def _validate_k6_replay_parity(  # noqa: C901, PLR0912, PLR0915
	value: object,
	survey_ids: Sequence[object],
	*,
	historical_root: Path,
	historical_targets: Mapping[str, object],
	verify_hashes: bool,
) -> None:
	"""Validate required exact K=6 replay evidence for complete manifests."""
	parity = _mapping(value, 'k6_replay_parity')
	_required_keys(
		parity,
		{
			'exact',
			'checks',
			'replay_root',
			'replay_artifacts',
			'historical_decoded_labels',
			'replay_decoded_labels',
		},
		'k6_replay_parity',
	)
	if not isinstance(parity['replay_root'], str) or not parity['replay_root']:
		raise TypeError('k6_replay_parity replay_root must be a non-empty string')
	if _same_resolved_path(historical_root, Path(parity['replay_root'])):
		raise ValueError(
			'K=6 replay root must differ from the immutable historical '
			'training-target root'
		)
	if not isinstance(parity['exact'], bool):
		raise TypeError('k6_replay_parity exact must be a boolean')
	checks = _mapping(parity['checks'], 'k6_replay_parity checks')
	historical_boundary = {
		'boundary_weight' in _mapping(
			historical_targets[str(survey_id)],
			f'K=6 historical target {survey_id}',
		)
		for survey_id in survey_ids
	}
	if len(historical_boundary) != 1:
		raise ValueError('K=6 historical boundary-weight policy is inconsistent')
	has_boundary = historical_boundary.pop()
	expected = {
		f'{survey_id}.{metric}'
		for survey_id in survey_ids
		for metric in (
			'pseudo_target_labels',
			'pseudo_target_confidence',
			'pseudo_target_valid_tokens',
			'decoded_valid_token_mask',
			'decoded_invalid_positions',
			'decoded_labels',
			'decoded_state_occupancy',
			'decoded_transition_counts',
			'decoded_ordered_violations',
		)
	}
	if has_boundary:
		expected.update(
		f'{survey_id}.pseudo_target_boundary_weight' for survey_id in survey_ids
	)
	_required_keys(checks, expected, 'k6_replay_parity checks')
	if not all(isinstance(result, bool) for result in checks.values()):
		raise TypeError('k6_replay_parity checks must contain booleans')
	if not parity['exact'] or not all(checks.values()):
		raise ValueError('K=6 replay parity is not exact')
	replay_artifacts = _mapping(
		parity['replay_artifacts'],
		'k6_replay_parity replay_artifacts',
	)
	if set(replay_artifacts) != {str(survey_id) for survey_id in survey_ids}:
		raise ValueError('K=6 replay artifact survey set mismatch')
	for survey_id in survey_ids:
		artifacts = _mapping(
			replay_artifacts[str(survey_id)],
			f'K=6 replay artifacts {survey_id}',
		)
		_required_keys(
			artifacts,
			{'labels', 'confidence', 'valid_tokens', 'metadata'},
			f'K=6 replay artifacts {survey_id}',
			optional_keys={'boundary_weight'},
		)
		if has_boundary and 'boundary_weight' not in artifacts:
			raise ValueError(
				f'K=6 replay boundary-weight artifact is missing for {survey_id}'
			)
		if not has_boundary and 'boundary_weight' in artifacts:
			raise ValueError(
				f'K=6 replay has an unexpected boundary-weight artifact for {survey_id}'
			)
		_validate_target_reference_schema(artifacts)
		if verify_hashes:
			for name in ('labels', 'confidence', 'valid_tokens', 'metadata'):
				reference = _mapping(artifacts[name], f'K=6 replay {name}')
				if file_sha256(Path(str(reference['path']))) != reference['sha256']:
					raise ValueError(
						f'K=6 replay artifact {name} hash mismatch for {survey_id}'
					)
	_validate_decoded_label_references(
		parity['historical_decoded_labels'],
		survey_ids,
		name='K=6 historical decoded labels',
		verify_hashes=verify_hashes,
	)
	_validate_decoded_label_references(
		parity['replay_decoded_labels'],
		survey_ids,
		name='K=6 replay decoded labels',
		verify_hashes=verify_hashes,
	)
	historical_decoded = _mapping(
		parity['historical_decoded_labels'],
		'K=6 historical decoded labels',
	)
	replay_decoded = _mapping(
		parity['replay_decoded_labels'],
		'K=6 replay decoded labels',
	)
	for survey_id in survey_ids:
		historical_target = _mapping(
			historical_targets[str(survey_id)],
			f'K=6 historical target {survey_id}',
		)
		artifacts = _mapping(
			replay_artifacts[str(survey_id)],
			f'K=6 replay artifacts {survey_id}',
		)
		expected_historical = _decoded_label_reference_from_target_metadata(
			historical_target,
			name=f'K=6 historical target {survey_id}',
		)
		expected_replay = _decoded_label_reference_from_target_metadata(
			artifacts,
			name=f'K=6 replay target {survey_id}',
		)
		if not _same_file_reference(
			historical_decoded[str(survey_id)], expected_historical
		):
			raise ValueError(
				f'K=6 historical decoded-label reference mismatch for {survey_id}'
			)
		if not _same_file_reference(replay_decoded[str(survey_id)], expected_replay):
			raise ValueError(
				f'K=6 replay decoded-label reference mismatch for {survey_id}'
			)
		historical_reference = _mapping(
			historical_decoded[str(survey_id)],
			'reference',
		)
		replay_reference = _mapping(replay_decoded[str(survey_id)], 'reference')
		if _same_resolved_path(
			Path(str(historical_reference['path'])),
			Path(str(replay_reference['path'])),
		):
			raise ValueError(
				'K=6 replay decoded-label artifact must differ from the '
				'immutable historical artifact'
			)


def _head_reference(root: Path, *, k: int) -> dict[str, object]:
	surveys: dict[str, object] = {}
	for item in discover_pseudo_target_inputs(root, k=k):
		metadata = load_pseudo_target_metadata(item)
		if metadata.get('schema_version') not in {1, 2}:
			raise ValueError(f'head k={k} has unsupported pseudo-target schema')
		if (
			metadata.get('schema_version') == 1
			and item.boundary_weight_path is not None
		):
			raise ValueError(
				f'head k={k} schema-v1 target has a boundary-weight artifact'
			)
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
		entry: dict[str, object] = {
			'labels': _file_reference(item.labels_path),
			'confidence': _file_reference(item.confidence_path),
			'valid_tokens': _file_reference(item.valid_tokens_path),
			'metadata': _file_reference(item.metadata_path),
			'token_grid_shape': list(arrays.labels.shape),
			'diagnostics': diagnostics,
		}
		if item.boundary_weight_path is not None:
			entry['boundary_weight'] = _file_reference(item.boundary_weight_path)
		surveys[item.survey_id] = entry
	return {
		'pseudo_target_root': str(root),
		'surveys': surveys,
		'diagnostics': {
			'per_survey': {key: value['diagnostics'] for key, value in surveys.items()}
		},
	}


def _replay_artifact_references(
	head: Mapping[str, object],
) -> dict[str, dict[str, dict[str, str]]]:
	"""Extract replay artifact identities without embedding their arrays."""
	surveys = _mapping(head['surveys'], 'K=6 replay surveys')
	references: dict[str, dict[str, dict[str, str]]] = {}
	for survey_id, value in surveys.items():
		entry = _mapping(value, f'K=6 replay survey {survey_id}')
		references[survey_id] = {
			name: dict(_mapping(entry[name], f'K=6 replay {name}'))
			for name in (
				'labels',
				'confidence',
				'valid_tokens',
				'boundary_weight',
				'metadata',
			)
			if name in entry
		}
	return references


def _decoded_label_references(
	head: Mapping[str, object],
) -> dict[str, dict[str, str]]:
	"""Resolve each exported target's immutable clustering-label input."""
	surveys = _mapping(head['surveys'], 'K=6 surveys')
	return {
		survey_id: _decoded_label_reference_from_target_metadata(
			_mapping(value, f'K=6 target {survey_id}'),
			name=f'K=6 target {survey_id}',
		)
		for survey_id, value in surveys.items()
	}


def _decoded_label_reference_from_target_metadata(
	entry: Mapping[str, object],
	*,
	name: str,
) -> dict[str, str]:
	metadata = _mapping(entry['metadata'], f'{name} metadata reference')
	metadata_path = Path(str(metadata['path']))
	source = _mapping(_json_object(metadata_path).get('source'), f'{name} source')
	label_path = source.get('source_label_path')
	if not isinstance(label_path, str) or not label_path:
		raise ValueError(f'{name} must record source_label_path')
	path = Path(label_path)
	if not path.is_file():
		raise FileNotFoundError(f'{name} decoded-label artifact is missing: {path}')
	return _file_reference(path)


def _validate_decoded_label_references(
	value: object,
	survey_ids: Sequence[object],
	*,
	name: str,
	verify_hashes: bool,
) -> None:
	references = _mapping(value, name)
	if set(references) != {str(survey_id) for survey_id in survey_ids}:
		raise ValueError(f'{name} survey set mismatch')
	for survey_id in survey_ids:
		reference = _mapping(references[str(survey_id)], f'{name} {survey_id}')
		_required_keys(reference, {'path', 'sha256'}, f'{name} {survey_id}')
		if (
			verify_hashes
			and file_sha256(Path(str(reference['path']))) != reference['sha256']
		):
			raise ValueError(f'{name} hash mismatch for {survey_id}')


def _same_file_reference(
	left: Mapping[str, object], right: Mapping[str, object]
) -> bool:
	return (
		_same_resolved_path(Path(str(left['path'])), Path(str(right['path'])))
		and left['sha256'] == right['sha256']
	)


def _same_resolved_path(left: Path, right: Path) -> bool:
	return left.resolve() == right.resolve()


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
) -> dict[str, object]:
	"""Require every target grid and mask to be contained by its embedding."""
	surveys = _mapping(head['surveys'], f'head k={k} surveys')
	alignment: dict[str, object] = {}
	for survey_id, embedding in embedding_by_survey.items():
		entry = _mapping(surveys[survey_id], 'survey')
		target = _load_reference_arrays(entry)
		embedding_valid = np.load(embedding.valid_tokens_path, allow_pickle=False)
		if target['labels'].shape != embedding_valid.shape:
			raise ValueError(
				f'head k={k} {survey_id} token grid does not match source embedding'
			)
		if np.any(target['valid_tokens'] & ~embedding_valid):
			raise ValueError(
				f'head k={k} {survey_id} valid-token mask is not a subset '
				'of source embedding'
			)
		alignment[survey_id] = _source_target_alignment_evidence(
			embedding_valid,
			target['valid_tokens'],
		)
	return alignment


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


def _validate_manifest_embedding_alignment(  # noqa: C901, PLR0913
	head_values: Mapping[str, object],
	survey_ids: list[object],
	source_embedding: Mapping[str, object],
	ks: Sequence[int],
	common_source_target_alignment: Mapping[str, object],
	*,
	validate_array_semantics: bool,
) -> None:
	"""Recheck target-to-embedding alignment when loading a manifest.

	The reference-only path verifies the independently persisted source and target
	identities plus their recorded subset contract without materializing arrays.
	Full validation rechecks grid shape, subset semantics, and exact target-mask
	parity across heads.
	"""
	embeddings = _mapping(source_embedding['surveys'], 'source_embedding surveys')
	if set(embeddings) != set(survey_ids):
		raise ValueError('source embedding survey set does not match manifest common')
	for survey_id in survey_ids:
		if not validate_array_semantics:
			continue
		embedding = _mapping(embeddings[str(survey_id)], 'source embedding survey')
		embedding_valid = np.load(
			Path(str(embedding['valid_tokens_path'])),
			mmap_mode='r',
			allow_pickle=False,
		)
		target_valid: np.ndarray | None = None
		for k in ks:
			surveys = _mapping(
				_mapping(head_values[str(k)], f'head k={k}')['surveys'],
				f'head k={k} surveys',
			)
			entry = _mapping(surveys[str(survey_id)], 'target reference')
			target = _load_reference_arrays(entry)
			if target['labels'].shape != embedding_valid.shape:
				raise ValueError(
					f'head k={k} {survey_id} token grid does not match source embedding'
				)
			if np.any(target['valid_tokens'] & ~embedding_valid):
				raise ValueError(
					f'head k={k} {survey_id} valid-token mask is not a subset '
					'of source embedding'
				)
			if target_valid is None:
				target_valid = target['valid_tokens']
			elif not np.array_equal(target['valid_tokens'], target_valid):
				raise ValueError(
					f'head k={k} {survey_id} valid-token mask differs across heads'
				)
		if target_valid is None:
			raise AssertionError('at least one head is required')
		if _source_target_alignment_evidence(
			embedding_valid,
			target_valid,
		) != common_source_target_alignment[str(survey_id)]:
			raise ValueError(
				f'{survey_id} persisted source-to-target alignment '
				'does not match arrays'
			)


def _validate_legacy_manifest_embedding_alignment(
	head_values: Mapping[str, object],
	survey_ids: list[object],
	source_embedding: Mapping[str, object],
	ks: Sequence[int],
	*,
	validate_array_semantics: bool,
) -> None:
	"""Validate the original v1 exact source-to-target mask contract.

	V1 does not carry subset evidence.  Its persisted source and target mask hashes
	must therefore remain equal, and full validation repeats that bitwise check.
	This compatibility path is intentionally read-only: new manifests are v2.
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
			embedding = _mapping(
				embeddings[str(survey_id)], 'source embedding survey'
			)
			valid_reference = _mapping(
				entry['valid_tokens'],
				'target valid_tokens reference',
			)
			if valid_reference['sha256'] != embedding['valid_tokens_sha256']:
				raise ValueError(
					f'legacy v1 head k={k} {survey_id} valid-token mask does not '
					'match source embedding'
				)
			if not validate_array_semantics:
				continue
			target = _load_reference_arrays(entry)
			embedding_valid = np.load(
				Path(str(embedding['valid_tokens_path'])),
				mmap_mode='r',
				allow_pickle=False,
			)
			if target['labels'].shape != embedding_valid.shape:
				raise ValueError(
					f'head k={k} {survey_id} token grid does not match source embedding'
				)
			if not np.array_equal(target['valid_tokens'], embedding_valid):
				raise ValueError(
					f'legacy v1 head k={k} {survey_id} valid-token mask does not '
					'match source embedding'
				)


def _source_target_alignment_evidence(
	source_valid: np.ndarray,
	target_valid: np.ndarray,
) -> dict[str, object]:
	"""Return the persisted evidence for a target mask contained by its source."""
	source_valid_count = int(np.count_nonzero(source_valid))
	target_valid_count = int(np.count_nonzero(target_valid))
	return {
		'source_valid_count': source_valid_count,
		'target_valid_count': target_valid_count,
		'excluded_from_source_count': source_valid_count - target_valid_count,
		'target_is_subset_of_source': True,
	}


def _validate_source_target_alignment_contract(
	alignment: Mapping[str, object],
	survey_ids: Sequence[object],
) -> None:
	"""Validate subset evidence used by the reference-only manifest path."""
	if set(alignment) != {str(survey_id) for survey_id in survey_ids}:
		raise ValueError('manifest source-to-target alignment survey set mismatch')
	for survey_id in survey_ids:
		entry = _mapping(
			alignment[str(survey_id)],
			f'manifest source-to-target alignment {survey_id}',
		)
		_required_keys(
			entry,
			{
				'source_valid_count',
				'target_valid_count',
				'excluded_from_source_count',
				'target_is_subset_of_source',
			},
			f'manifest source-to-target alignment {survey_id}',
		)
		source_count = _nonnegative_int(
			entry['source_valid_count'],
			f'manifest source-to-target alignment {survey_id} source count',
		)
		target_count = _nonnegative_int(
			entry['target_valid_count'],
			f'manifest source-to-target alignment {survey_id} target count',
		)
		excluded_count = _nonnegative_int(
			entry['excluded_from_source_count'],
			f'manifest source-to-target alignment {survey_id} excluded count',
		)
		if entry['target_is_subset_of_source'] is not True:
			raise ValueError(
				f'manifest source-to-target alignment {survey_id} must record a subset'
			)
		if target_count > source_count or excluded_count != source_count - target_count:
			raise ValueError(
				f'manifest source-to-target alignment {survey_id} '
				'counts are inconsistent'
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
	if 'boundary_weight' in entry:
		refs['boundary_weight'] = _mapping(
			entry['boundary_weight'], 'boundary_weight reference'
		)
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
		boundary_weight_path=(
			None
			if 'boundary_weight' not in refs
			else Path(str(refs['boundary_weight']['path']))
		),
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
	boundary_weight_path: str | Path | None = None,
	expected_token_grid_shape: Sequence[object] | None = None,
	validate_array_semantics: bool = True,
) -> None:
	"""Validate one referenced schema-v1 target without duplicating its contract.

	Reference-only validation verifies immutable file identities and metadata but
	does not call :func:`numpy.load`.  Full validation retains the array-level
	range, validity, occupancy, and ordering checks used at publication time.
	"""
	required_hashes = {'labels', 'confidence', 'valid_tokens', 'metadata'}
	if boundary_weight_path is not None:
		required_hashes.add('boundary_weight')
	_required_keys(hashes, required_hashes, 'multi-head target hashes')
	paths = {
		'labels': Path(labels_path),
		'confidence': Path(confidence_path),
		'valid_tokens': Path(valid_tokens_path),
		'metadata': Path(metadata_path),
	}
	if boundary_weight_path is not None:
		paths['boundary_weight'] = Path(boundary_weight_path)
	for name, path in paths.items():
		digest = hashes[name]
		if not isinstance(digest, str) or file_sha256(path) != digest:
			raise ValueError(f'head k={k} {survey_id} {name} hash mismatch')
	_validate_referenced_target_metadata(
		metadata_path=paths['metadata'],
		boundary_weight_path=paths.get('boundary_weight'),
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
	boundary_weight_path: Path | None,
	k: int,
	survey_id: str,
) -> None:
	"""Recheck each referenced target's blocking semantics."""
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
		or metadata.get('schema_version') not in {1, 2}
		or metadata.get('k') != k
		or metadata.get('survey_id') != survey_id
	):
		raise ValueError(
			f'head k={k} {survey_id} must use matching pseudo-target metadata'
		)
	if metadata.get('schema_version') == 1 and boundary_weight_path is not None:
		raise ValueError(
			f'head k={k} {survey_id} schema-v1 target has a boundary-weight artifact'
		)
	if metadata.get('schema_version') == 2 and boundary_weight_path is None:
		raise ValueError(
			f'head k={k} {survey_id} schema-v2 target is missing a '
			'boundary-weight artifact'
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
		boundary_weight=arrays.get('boundary_weight'),
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
	if 'boundary_weight' in entry:
		_required_keys(
			_mapping(entry['boundary_weight'], 'boundary_weight reference'),
			{'path', 'sha256'},
			'boundary_weight reference',
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
	arrays = {
		name: np.load(Path(str(_mapping(entry[name], name)['path'])))
		for name in ('labels', 'confidence', 'valid_tokens')
	}
	if 'boundary_weight' in entry:
		arrays['boundary_weight'] = np.load(
			Path(str(_mapping(entry['boundary_weight'], 'boundary_weight')['path']))
		)
	return arrays


def _validate_head_ks(head_roots: Mapping[int | str, str | Path]) -> tuple[int, ...]:
	values = _normalized_head_roots(head_roots)
	ks = tuple(sorted(values))
	if len(ks) < 2 or any(k < 2 for k in ks):
		raise ValueError('head roots must contain at least two K values >= 2')
	return ks


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

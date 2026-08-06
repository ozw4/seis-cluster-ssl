"""Publish lightweight, target-only F3 M5-LS review artifacts.

This module deliberately consumes the immutable target-calibration evidence and
an optional already-validated smoke evidence payload.  It never opens target
arrays, posterior arrays, facies/lithology labels, decoder outputs, or
downstream evaluation metrics.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from seis_ssl_cluster.f3.lateral_smoothing_target_calibration import (
	load_f3_m5_lateral_target_calibration_handoff,
)
from seis_ssl_cluster.results import (
	PublishItem,
	PublishManifest,
	publish_manifest_to_dict,
	publish_selected_results,
)

_ARTIFACT_ROOT_PLACEHOLDER = '${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}'
_CALIBRATION_REPORT_TYPE = 'f3_m5_lateral_target_calibration_report'
_CANONICAL_CANDIDATES = (
	('beta010', 0.10),
	('beta025', 0.25),
	('beta050', 0.50),
)
_CANONICAL_HEADS = (6, 8, 10)
_CONFIG_KEYS = frozenset(
	{
		'artifact_root',
		'workspace_root',
		'calibration_handoff',
		'calibration_report',
		'output_dir',
		'smoke_evidence',
	}
)
_REQUIRED_CONFIG_KEYS = _CONFIG_KEYS - {'smoke_evidence'}
_REPORT_KEYS = (
	'artifact_type',
	'schema_version',
	'status',
	'selection_policy',
	'candidate_betas',
	'beta_zero_parity',
	'source_hard_manifest',
	'source_posterior_manifest',
	'candidates',
	'selected_beta',
	'selected_candidate_manifest',
	'selected_manifest',
	'git',
)

LATERAL_TARGET_CANDIDATES_CSV = 'lateral_target_candidates.csv'
LATERAL_TARGET_CALIBRATION_SUMMARY_JSON = 'lateral_target_calibration_summary.json'
LATERAL_TARGET_CALIBRATION_SUMMARY_MARKDOWN = 'lateral_target_calibration_summary.md'
LATERAL_TARGET_CALIBRATION_HANDOFF_JSON = 'lateral_target_calibration_handoff.json'
LATERAL_SMOKE_SUMMARY_JSON = 'lateral_smoke_summary.json'
OUTPUT_NAMES = (
	LATERAL_TARGET_CANDIDATES_CSV,
	LATERAL_TARGET_CALIBRATION_SUMMARY_JSON,
	LATERAL_TARGET_CALIBRATION_SUMMARY_MARKDOWN,
	LATERAL_TARGET_CALIBRATION_HANDOFF_JSON,
)

_CANDIDATE_CSV_FIELDS = (
	'row_type',
	'candidate',
	'beta',
	'k',
	'candidate_eligible',
	'head_eligible',
	'eligibility_reasons',
	'manifest_path',
	'manifest_sha256',
	'head_hashes_sha256',
	'pairwise_strength_ratio',
	'affinity_scale',
	'emission_gap_scale',
	'valid_token_count',
	'invalid_token_count',
	'changed_token_count',
	'changed_fraction',
	'boundary_adjacent_changed_fraction',
	'interior_changed_fraction',
	'empty_state_count',
	'effective_k',
	'ordered_path_violation_count',
	'ordered_path_max_reverse_decrease',
	'source_transition_count',
	'lateral_transition_count',
	'affinity_weighted_source_disagreement',
	'affinity_weighted_lateral_disagreement',
	'highest_affinity_edge_count',
	'highest_affinity_source_disagreement',
	'highest_affinity_lateral_disagreement',
)


@dataclass(frozen=True)
class F3M5LateralSmoothingReviewConfig:
	"""Explicit inputs and output location for a target-only review publication."""

	artifact_root: Path
	workspace_root: Path
	calibration_handoff: Path
	calibration_report: Path
	output_dir: Path
	smoke_evidence: Path | None = None


@dataclass(frozen=True)
class F3M5LateralSmoothingReviewResult:
	"""Paths and statuses produced by one M5-LS review publication."""

	output_dir: Path
	calibration_status: str
	selected_beta: float | None
	smoke_status: str
	candidate_csv: Path
	summary_json: Path
	summary_markdown: Path
	calibration_handoff: Path
	smoke_summary: Path | None
	publish_manifest: PublishManifest | None


def f3_m5_lateral_smoothing_review_config_from_mapping(
	config: Mapping[str, object],
) -> F3M5LateralSmoothingReviewConfig:
	"""Resolve the closed, explicit review-publisher configuration mapping."""
	if not isinstance(config, Mapping):
		raise TypeError('M5-LS review config must be a mapping')
	unknown = set(config) - _CONFIG_KEYS
	missing = _REQUIRED_CONFIG_KEYS - set(config)
	if unknown:
		raise ValueError(f'unknown M5-LS review config keys: {sorted(unknown)!r}')
	if missing:
		raise ValueError(f'missing M5-LS review config keys: {sorted(missing)!r}')

	artifact_root = _directory_path(config, 'artifact_root')
	workspace_root = _directory_path(config, 'workspace_root')
	return F3M5LateralSmoothingReviewConfig(
		artifact_root=artifact_root,
		workspace_root=workspace_root,
		calibration_handoff=_file_path(config, 'calibration_handoff'),
		calibration_report=_file_path(config, 'calibration_report'),
		output_dir=_output_path(config, 'output_dir'),
		smoke_evidence=_optional_file_path(config, 'smoke_evidence'),
	)


def publish_f3_m5_lateral_smoothing_review(
	config: F3M5LateralSmoothingReviewConfig,
	*,
	dry_run: bool = False,
	smoke_evidence: Mapping[str, object] | None = None,
) -> F3M5LateralSmoothingReviewResult:
	"""Publish portable target-calibration review files and optional smoke summary.

	``smoke_evidence`` supports the in-memory evidence returned by the strict
	M5-LS validator.  The CLI path form is intended for its compact JSON form,
	which this function also accepts after loading it from ``config``.
	"""
	if smoke_evidence is not None and config.smoke_evidence is not None:
		raise ValueError('provide smoke evidence either in config or as a mapping')
	handoff = load_f3_m5_lateral_target_calibration_handoff(
		config.calibration_handoff
	)
	report = _load_json_mapping(config.calibration_report, 'calibration report')
	_validate_calibration_report(handoff, report)
	if handoff['status'] == 'M5_LS_TARGET_HOLD' and (
		smoke_evidence is not None or config.smoke_evidence is not None
	):
		raise ValueError('M5_LS_TARGET_HOLD cannot publish smoke evidence')
	resolved_smoke = _resolve_smoke_evidence(
		config,
		handoff=handoff,
		smoke_evidence=smoke_evidence,
	)
	evidence = _review_evidence(
		config,
		handoff=handoff,
		report=report,
		smoke=resolved_smoke,
	)
	result = _result_from_evidence(config, evidence)
	_validate_existing_smoke_output(config, smoke=resolved_smoke)
	if dry_run:
		return result
	manifest = _publish_evidence(
		config,
		evidence=evidence,
		handoff=handoff,
		smoke=resolved_smoke,
	)
	return replace(result, publish_manifest=manifest)


def compact_f3_m5_lateral_smoke_evidence(
	smoke_evidence: Mapping[str, object],
	*,
	calibration_handoff: Mapping[str, object],
) -> dict[str, object]:
	"""Reduce strict validator smoke evidence to safe lightweight diagnostics."""
	if smoke_evidence.get('artifact_type') == 'f3_m5_lateral_smoothing_smoke_summary':
		return _validate_compact_smoke_evidence(smoke_evidence, calibration_handoff)
	evidence = _mapping(smoke_evidence, 'smoke evidence')
	selected_beta = _selected_beta(calibration_handoff)
	_validate_raw_smoke_header(evidence, selected_beta=selected_beta)
	return _compact_raw_smoke_evidence(evidence, selected_beta=selected_beta)


def _validate_raw_smoke_header(
	evidence: Mapping[str, object], *, selected_beta: float
) -> None:
	_require_exact_values(
		evidence,
		{
			'status': 'PASS',
			'selected_beta': selected_beta,
			'target_representation': 'lateral_mean_field_hard_labels_v1',
			'target_semantics': 'ordered_hmm_edge_aware_lateral_mean_field_hard_v1',
		},
		label='smoke evidence',
	)


def _compact_raw_smoke_evidence(
	evidence: Mapping[str, object], *, selected_beta: float
) -> dict[str, object]:
	smoke = _mapping(evidence.get('smoke'), 'smoke evidence.smoke')
	latest = _mapping(smoke.get('latest'), 'smoke latest checkpoint')
	identity = _mapping(smoke.get('identity'), 'smoke checkpoint identity')
	_require_exact_values(
		latest,
		{'global_step': 2, 'epoch': 1},
		label='smoke latest checkpoint',
	)
	_require_exact_values(
		identity,
		{
			'schema_version': 4,
			'target_representation': evidence['target_representation'],
			'target_semantics': evidence['target_semantics'],
			'consistency_weight': 0.0,
		},
		label='smoke checkpoint identity',
	)
	_require_exact_values(
		smoke,
		{
			'hard_multi_head_loss_path_used': True,
			'posterior_loss_path_used': False,
			'consistency_contribution': 0.0,
			'gradients_finite': True,
		},
		label='smoke evidence',
	)
	return {
		'artifact_type': 'f3_m5_lateral_smoothing_smoke_summary',
		'schema_version': 1,
		'status': 'PASS',
		'selected_beta': selected_beta,
		'global_step': 2,
		'epoch': 1,
		'finite_losses': _finite_losses(latest),
		'gradients_finite': True,
		'target_representation': evidence['target_representation'],
		'target_semantics': evidence['target_semantics'],
		'checkpoint_schema_version': 4,
		'hard_multi_head_loss_path_used': True,
		'posterior_loss_path_used': False,
		'consistency_contribution': 0.0,
		'smoke_root': _path_text(smoke.get('root'), 'smoke root'),
		'hard_baseline_parity': {
			'initial_student_head': 'PASS',
			'trainability': 'PASS',
			'optimizer_groups': 'PASS',
			'smoke_root_isolated_from_full_root': 'PASS',
		},
	}


def render_f3_m5_lateral_smoothing_review_markdown(
	evidence: Mapping[str, object],
) -> str:
	"""Render a concise non-performance M5-LS calibration review summary."""
	calibration = _mapping(evidence.get('target_calibration'), 'target calibration')
	status = str(calibration['status'])
	selected_beta = calibration['selected_beta']
	selected = 'HOLD' if selected_beta is None else f'{float(selected_beta):.2f}'
	smoke = _mapping(evidence.get('smoke'), 'review smoke')
	rows = _mapping(calibration.get('candidates'), 'candidate evidence')
	lines = [
		'# F3 M5-LS target-only calibration',
		'',
		f'Target calibration: `{status}`',
		'',
		f'Selection policy: `{calibration["selection_policy"]}`',
		'',
		f'Selected beta: `{selected}`',
		'',
		f'Beta-zero parity: `{calibration["beta_zero_parity_status"]}`',
		'',
		'Candidate betas and the selection policy were fixed before diagnostics. '
		'No facies/lithology labels, decoder outputs, or downstream metrics are read.',
		'',
		'## Candidate eligibility',
		'',
		'| candidate | beta | eligible | reasons |',
		'| --- | ---: | :---: | --- |',
	]
	for name, beta in _CANONICAL_CANDIDATES:
		candidate = _mapping(rows.get(name), f'{name} candidate')
		eligibility = _mapping(candidate.get('eligibility'), f'{name} eligibility')
		reasons = eligibility.get('reasons')
		if not isinstance(reasons, list):
			raise TypeError(f'{name} eligibility reasons must be a list')
		reason_text = '; '.join(str(reason) for reason in reasons) or '—'
		escaped_reasons = reason_text.replace('|', '\\|')
		lines.append(
			f'| {name} | {beta:.2f} | {eligibility["eligible"]} | '
			f'{escaped_reasons} |'
		)
	lines.extend(
		[
			'',
			'## Execution status',
			'',
			f'Smoke: `{smoke["status"]}`',
			'',
			'Full pretraining: `NOT_EXECUTED`',
			'',
			'Embedding extraction: `NOT_EXECUTED`',
			'',
			'Downstream screening: `NOT_EXECUTED`',
			'',
			'This review records target-only technical diagnostics and makes no '
			'downstream performance conclusion.',
			'',
		]
	)
	return '\n'.join(lines)


def _directory_path(config: Mapping[str, object], key: str) -> Path:
	value = config[key]
	if not isinstance(value, str) or not value:
		raise TypeError(f'{key} must be a non-empty path string')
	path = Path(value).resolve()
	if not path.is_dir():
		raise FileNotFoundError(f'{key} directory is missing: {path}')
	return path


def _file_path(config: Mapping[str, object], key: str) -> Path:
	value = config[key]
	if not isinstance(value, str) or not value:
		raise TypeError(f'{key} must be a non-empty path string')
	path = Path(value).resolve()
	if not path.is_file():
		raise FileNotFoundError(f'{key} is missing: {path}')
	return path


def _optional_file_path(config: Mapping[str, object], key: str) -> Path | None:
	value = config.get(key)
	if value is None:
		return None
	if not isinstance(value, str) or not value:
		raise TypeError(f'{key} must be a non-empty path string when provided')
	path = Path(value).resolve()
	if not path.is_file():
		raise FileNotFoundError(f'{key} is missing: {path}')
	return path


def _output_path(config: Mapping[str, object], key: str) -> Path:
	value = config[key]
	if not isinstance(value, str) or not value:
		raise TypeError(f'{key} must be a non-empty path string')
	return Path(value).resolve()


def _load_json_mapping(path: Path, label: str) -> Mapping[str, object]:
	try:
		value = json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as error:
		raise ValueError(f'{label} is not valid JSON: {path}') from error
	return _mapping(value, label)


def _validate_calibration_report(
	handoff: Mapping[str, object], report: Mapping[str, object]
) -> None:
	if set(report) != set(_REPORT_KEYS):
		raise ValueError('M5-LS calibration report keys mismatch')
	expected = {key: handoff[key] for key in _REPORT_KEYS}
	expected['artifact_type'] = _CALIBRATION_REPORT_TYPE
	if dict(report) != expected:
		raise ValueError('M5-LS calibration report differs from calibration handoff')


def _resolve_smoke_evidence(
	config: F3M5LateralSmoothingReviewConfig,
	*,
	handoff: Mapping[str, object],
	smoke_evidence: Mapping[str, object] | None,
) -> dict[str, object] | None:
	if smoke_evidence is not None:
		return compact_f3_m5_lateral_smoke_evidence(
			smoke_evidence,
			calibration_handoff=handoff,
		)
	if config.smoke_evidence is None:
		return None
	return compact_f3_m5_lateral_smoke_evidence(
		_load_json_mapping(config.smoke_evidence, 'smoke evidence'),
		calibration_handoff=handoff,
	)


def _review_evidence(
	config: F3M5LateralSmoothingReviewConfig,
	*,
	handoff: Mapping[str, object],
	report: Mapping[str, object],
	smoke: Mapping[str, object] | None,
) -> dict[str, object]:
	status = str(handoff['status'])
	if status == 'M5_LS_TARGET_HOLD' and smoke is not None:
		raise ValueError('M5-LS TARGET_HOLD cannot publish smoke evidence')
	selected_beta = _selected_beta_or_none(handoff)
	if status == 'M5_LS_TARGET_SELECTED' and selected_beta is None:
		raise ValueError('selected calibration lacks a selected beta')
	if status == 'M5_LS_TARGET_HOLD' and selected_beta is not None:
		raise ValueError('HOLD calibration names a selected beta')
	smoke_status = _smoke_status(status=status, smoke=smoke)
	return {
		'artifact_type': 'f3_m5_lateral_smoothing_review_artifacts',
		'schema_version': 1,
		'target_calibration': {
			'status': status,
			'selection_policy': handoff['selection_policy'],
			'candidate_betas': handoff['candidate_betas'],
			'selected_beta': selected_beta,
			'beta_zero_parity_status': _mapping(
				handoff['beta_zero_parity'], 'beta-zero parity'
			)['status'],
			'beta_zero_parity': handoff['beta_zero_parity'],
			'source_hard_manifest': handoff['source_hard_manifest'],
			'source_posterior_manifest': handoff['source_posterior_manifest'],
			'candidates': handoff['candidates'],
			'selected_candidate_manifest': handoff['selected_candidate_manifest'],
			'selected_manifest': handoff['selected_manifest'],
		},
		'calibration_sources': {
			'calibration_handoff': _file_reference(config.calibration_handoff),
			'calibration_report': _file_reference(config.calibration_report),
			'calibration_report_artifact_type': report['artifact_type'],
		},
		'smoke': {
			'status': smoke_status,
			'evidence': None if smoke is None else dict(smoke),
		},
		'execution_scope': {
			'full_pretraining': 'NOT_EXECUTED',
			'embedding_extraction': 'NOT_EXECUTED',
			'downstream_screening': 'NOT_EXECUTED',
		},
	}


def _result_from_evidence(
	config: F3M5LateralSmoothingReviewConfig,
	evidence: Mapping[str, object],
) -> F3M5LateralSmoothingReviewResult:
	calibration = _mapping(evidence['target_calibration'], 'target calibration')
	smoke = _mapping(evidence['smoke'], 'review smoke')
	output = config.output_dir
	return F3M5LateralSmoothingReviewResult(
		output_dir=output,
		calibration_status=str(calibration['status']),
		selected_beta=_selected_beta_or_none(calibration),
		smoke_status=str(smoke['status']),
		candidate_csv=output / LATERAL_TARGET_CANDIDATES_CSV,
		summary_json=output / LATERAL_TARGET_CALIBRATION_SUMMARY_JSON,
		summary_markdown=output / LATERAL_TARGET_CALIBRATION_SUMMARY_MARKDOWN,
		calibration_handoff=output / LATERAL_TARGET_CALIBRATION_HANDOFF_JSON,
		smoke_summary=(
			None
			if smoke['evidence'] is None
			else output / LATERAL_SMOKE_SUMMARY_JSON
		),
		publish_manifest=None,
	)


def _validate_existing_smoke_output(
	config: F3M5LateralSmoothingReviewConfig,
	*,
	smoke: Mapping[str, object] | None,
) -> None:
	if smoke is None and (config.output_dir / LATERAL_SMOKE_SUMMARY_JSON).exists():
		raise ValueError(
			'existing lateral smoke summary requires explicit smoke evidence on rerun'
		)


def _publish_evidence(
	config: F3M5LateralSmoothingReviewConfig,
	*,
	evidence: Mapping[str, object],
	handoff: Mapping[str, object],
	smoke: Mapping[str, object] | None,
) -> PublishManifest:
	portable = _portable_value(
		evidence,
		artifact_root=config.artifact_root,
		workspace_root=config.workspace_root,
	)
	portable_handoff = _portable_value(
		handoff,
		artifact_root=config.artifact_root,
		workspace_root=config.workspace_root,
	)
	portable_calibration = _mapping(
		_mapping(portable, 'portable review evidence').get('target_calibration'),
		'portable target calibration',
	)
	items = [
		PublishItem(
			config.calibration_report,
			Path(LATERAL_TARGET_CANDIDATES_CSV),
			content_text=_candidate_csv_text(portable_calibration),
		),
		PublishItem(
			config.calibration_report,
			Path(LATERAL_TARGET_CALIBRATION_SUMMARY_JSON),
			content_text=_json_text(portable),
		),
		PublishItem(
			config.calibration_report,
			Path(LATERAL_TARGET_CALIBRATION_SUMMARY_MARKDOWN),
			content_text=render_f3_m5_lateral_smoothing_review_markdown(portable),
		),
		PublishItem(
			config.calibration_handoff,
			Path(LATERAL_TARGET_CALIBRATION_HANDOFF_JSON),
			content_text=_json_text(portable_handoff),
		),
	]
	if smoke is not None:
		smoke_source = (
			config.calibration_handoff
			if config.smoke_evidence is None
			else config.smoke_evidence
		)
		items.append(
			PublishItem(
				smoke_source,
				Path(LATERAL_SMOKE_SUMMARY_JSON),
				content_text=_json_text(
					_portable_value(
						smoke,
						artifact_root=config.artifact_root,
						workspace_root=config.workspace_root,
					)
				),
			),
		)
	manifest = publish_selected_results(items=items, output_dir=config.output_dir)
	_write_portable_publish_manifest(manifest, config=config)
	return manifest


def _candidate_csv_text(calibration: Mapping[str, object]) -> str:
	stream = io.StringIO(newline='')
	writer = csv.DictWriter(
		stream,
		fieldnames=_CANDIDATE_CSV_FIELDS,
		lineterminator='\n',
	)
	writer.writeheader()
	for row in _candidate_rows(calibration):
		writer.writerow(row)
	return stream.getvalue()


def _candidate_rows(calibration: Mapping[str, object]) -> list[dict[str, object]]:
	candidates = _mapping(calibration.get('candidates'), 'calibration candidates')
	rows: list[dict[str, object]] = []
	for name, beta in _CANONICAL_CANDIDATES:
		candidate = _mapping(candidates.get(name), f'{name} candidate')
		base = _candidate_row_base(candidate, name=name, beta=beta)
		rows.append({**base, 'row_type': 'candidate_aggregate', 'k': ''})
		head_summaries = _mapping_or_empty(candidate.get('heads'))
		checks_by_head = _mapping_or_empty(
			_mapping(candidate.get('eligibility'), f'{name} eligibility').get('checks')
		)
		for k in _CANONICAL_HEADS:
			head = _mapping_or_empty(head_summaries.get(str(k)))
			checks = _mapping_or_empty(checks_by_head.get(str(k)))
			rows.append(
				{
					**base,
					'row_type': 'candidate_head',
					'k': k,
					'head_eligible': _checks_status(checks),
					**_head_row_values(head),
				},
			)
	return rows


def _candidate_row_base(
	candidate: Mapping[str, object], *, name: str, beta: float
) -> dict[str, object]:
	eligibility = _mapping(candidate.get('eligibility'), f'{name} eligibility')
	reasons = eligibility.get('reasons')
	if not isinstance(reasons, list) or not all(
		isinstance(reason, str) for reason in reasons
	):
		raise TypeError(f'{name} eligibility reasons must be a list of strings')
	manifest = _mapping(candidate.get('manifest'), f'{name} manifest')
	head_hashes = candidate.get('head_hashes')
	smoothing = _mapping_or_empty(candidate.get('smoothing'))
	return dict.fromkeys(_CANDIDATE_CSV_FIELDS, '') | {
		'candidate': name,
		'beta': beta,
		'candidate_eligible': eligibility.get('eligible'),
		'eligibility_reasons': '; '.join(reasons),
		'manifest_path': manifest.get('path', ''),
		'manifest_sha256': manifest.get('sha256', ''),
		'head_hashes_sha256': _json_sha256(head_hashes),
		'pairwise_strength_ratio': smoothing.get('pairwise_strength_ratio', ''),
	}


def _head_row_values(head: Mapping[str, object]) -> dict[str, object]:
	values: dict[str, object] = {}
	resolved_scales = _mapping_or_empty(head.get('resolved_scales'))
	affinity = _mapping_or_empty(resolved_scales.get('affinity'))
	emission_gap = _mapping_or_empty(resolved_scales.get('emission_gap'))
	changed_regions = _mapping_or_empty(head.get('changed_fraction_by_source_region'))
	occupancy = _mapping_or_empty(head.get('state_occupancy'))
	ordered_path = _mapping_or_empty(head.get('ordered_path'))
	trace_paths = _mapping_or_empty(head.get('trace_paths'))
	disagreement = _mapping_or_empty(head.get('xy_edge_disagreement'))
	weighted = _mapping_or_empty(
		disagreement.get('affinity_weighted_normalized_order')
	)
	highest = _highest_affinity_quartile(disagreement.get('affinity_quartiles'))
	values.update(
		{
			'affinity_scale': affinity.get('resolved_scale', ''),
			'emission_gap_scale': emission_gap.get('resolved_scale', ''),
			'valid_token_count': head.get('valid_token_count', ''),
			'invalid_token_count': head.get('invalid_token_count', ''),
			'changed_token_count': head.get('changed_token_count', ''),
			'changed_fraction': head.get('changed_fraction', ''),
			'boundary_adjacent_changed_fraction': changed_regions.get(
				'boundary_adjacent', ''
			),
			'interior_changed_fraction': changed_regions.get('interior', ''),
			'empty_state_count': occupancy.get('empty_state_count', ''),
			'effective_k': occupancy.get('effective_k', ''),
			'ordered_path_violation_count': ordered_path.get('violation_count', ''),
			'ordered_path_max_reverse_decrease': ordered_path.get(
				'max_reverse_decrease', ''
			),
			'source_transition_count': head.get('source_transition_count', ''),
			'lateral_transition_count': trace_paths.get('transition_count', ''),
			'affinity_weighted_source_disagreement': weighted.get('source', ''),
			'affinity_weighted_lateral_disagreement': weighted.get('lateral', ''),
			'highest_affinity_edge_count': highest.get('edge_count', ''),
			'highest_affinity_source_disagreement': highest.get(
				'source_unweighted_mean', ''
			),
			'highest_affinity_lateral_disagreement': highest.get(
				'lateral_unweighted_mean', ''
			),
		},
	)
	return values


def _highest_affinity_quartile(value: object) -> Mapping[str, object]:
	if not isinstance(value, list) or len(value) != 4:
		return {}
	return _mapping_or_empty(value[-1])


def _checks_status(checks: Mapping[str, object]) -> bool | str:
	if not checks:
		return ''
	if not all(isinstance(value, bool) for value in checks.values()):
		raise TypeError('candidate eligibility checks must be booleans')
	return all(checks.values())


def _selected_beta(handoff: Mapping[str, object]) -> float:
	if handoff.get('status') != 'M5_LS_TARGET_SELECTED':
		raise ValueError('smoke evidence requires an M5_LS_TARGET_SELECTED handoff')
	return _canonical_beta(handoff.get('selected_beta'))


def _selected_beta_or_none(value: Mapping[str, object]) -> float | None:
	selected = value.get('selected_beta')
	if selected is None:
		return None
	return _canonical_beta(selected)


def _canonical_beta(value: object) -> float:
	if isinstance(value, bool) or not isinstance(value, int | float):
		raise TypeError('selected beta must be a canonical float')
	result = float(value)
	if result not in {beta for _, beta in _CANONICAL_CANDIDATES}:
		raise ValueError('selected beta is not one of the canonical candidates')
	return result


def _smoke_status(*, status: str, smoke: Mapping[str, object] | None) -> str:
	if smoke is not None:
		return 'PASS'
	if status == 'M5_LS_TARGET_SELECTED':
		return 'READY_NOT_RUN'
	return 'NOT_READY_HOLD'


def _validate_compact_smoke_evidence(
	evidence: Mapping[str, object], calibration_handoff: Mapping[str, object]
) -> dict[str, object]:
	required = {
		'artifact_type',
		'schema_version',
		'status',
		'selected_beta',
		'global_step',
		'epoch',
		'finite_losses',
		'gradients_finite',
		'target_representation',
		'target_semantics',
		'checkpoint_schema_version',
		'hard_multi_head_loss_path_used',
		'posterior_loss_path_used',
		'consistency_contribution',
		'smoke_root',
		'hard_baseline_parity',
	}
	if set(evidence) != required:
		raise ValueError('compact smoke evidence keys mismatch')
	_require_exact_values(
		evidence,
		{
			'artifact_type': 'f3_m5_lateral_smoothing_smoke_summary',
			'schema_version': 1,
			'status': 'PASS',
			'selected_beta': _selected_beta(calibration_handoff),
			'global_step': 2,
			'epoch': 1,
			'gradients_finite': True,
			'target_representation': 'lateral_mean_field_hard_labels_v1',
			'target_semantics': 'ordered_hmm_edge_aware_lateral_mean_field_hard_v1',
			'checkpoint_schema_version': 4,
			'hard_multi_head_loss_path_used': True,
			'posterior_loss_path_used': False,
			'consistency_contribution': 0.0,
		},
		label='compact smoke evidence',
	)
	_path_text(evidence.get('smoke_root'), 'compact smoke evidence smoke_root')
	_validate_compact_finite_losses(evidence.get('finite_losses'))
	_validate_compact_hard_baseline_parity(evidence.get('hard_baseline_parity'))
	return dict(evidence)


def _validate_compact_finite_losses(value: object) -> None:
	finite_losses = _mapping(value, 'compact finite losses')
	if not finite_losses:
		raise ValueError('compact smoke evidence has no finite losses')
	for name, loss in finite_losses.items():
		if not isinstance(name, str) or not _finite_number(loss):
			raise ValueError('compact smoke evidence loss is not finite')


def _validate_compact_hard_baseline_parity(value: object) -> None:
	parity = _mapping(value, 'compact hard baseline parity')
	expected = {
		'initial_student_head',
		'trainability',
		'optimizer_groups',
		'smoke_root_isolated_from_full_root',
	}
	if set(parity) != expected or any(item != 'PASS' for item in parity.values()):
		raise ValueError('compact smoke evidence hard baseline parity mismatch')


def _require_exact_values(
	value: Mapping[str, object],
	expected: Mapping[str, object],
	*,
	label: str,
) -> None:
	for key, expected_value in expected.items():
		if value.get(key) != expected_value:
			raise ValueError(f'{label} {key} mismatch')


def _finite_losses(latest: Mapping[str, object]) -> dict[str, float]:
	metrics = _mapping(latest.get('metrics'), 'smoke latest metrics')
	losses: dict[str, float] = {}
	for name, value in metrics.items():
		if not isinstance(name, str):
			raise TypeError('smoke metric names must be strings')
		if 'posterior' in name:
			raise ValueError('smoke metrics must not include posterior loss metrics')
		if name.startswith('loss'):
			if not _finite_number(value):
				raise ValueError(f'smoke loss is nonfinite: {name}')
			losses[name] = float(value)
	if not losses:
		raise ValueError('smoke metrics do not include a finite loss')
	return losses


def _finite_number(value: object) -> bool:
	return (
		not isinstance(value, bool)
		and isinstance(value, int | float)
		and math.isfinite(float(value))
	)


def _path_text(value: object, label: str) -> str:
	if isinstance(value, Path):
		return str(value)
	if not isinstance(value, str) or not value:
		raise TypeError(f'{label} must be a non-empty path string')
	return value


def _file_reference(path: Path) -> dict[str, str]:
	return {'path': str(path), 'sha256': _file_sha256(path)}


def _file_sha256(path: Path) -> str:
	hasher = hashlib.sha256()
	with path.open('rb') as stream:
		for block in iter(lambda: stream.read(1024 * 1024), b''):
			hasher.update(block)
	return hasher.hexdigest()


def _json_sha256(value: object) -> str:
	if value is None:
		return ''
	encoded = json.dumps(
		value,
		sort_keys=True,
		separators=(',', ':'),
		allow_nan=False,
	).encode('utf-8')
	return hashlib.sha256(
		encoded
	).hexdigest()


def _json_text(value: object) -> str:
	return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'


def _portable_value(
	value: object,
	*,
	artifact_root: Path,
	workspace_root: Path,
) -> object:
	if isinstance(value, Mapping):
		return {
			str(key): _portable_value(
				item,
				artifact_root=artifact_root,
				workspace_root=workspace_root,
			)
			for key, item in value.items()
		}
	if isinstance(value, list | tuple):
		items = [
			_portable_value(
				item,
				artifact_root=artifact_root,
				workspace_root=workspace_root,
			)
			for item in value
		]
		return tuple(items) if isinstance(value, tuple) else items
	if isinstance(value, Path):
		return _portable_path(
			str(value), artifact_root=artifact_root, workspace_root=workspace_root
		)
	if isinstance(value, str):
		return _portable_path(
			value, artifact_root=artifact_root, workspace_root=workspace_root
		)
	return value


def _portable_path(value: str, *, artifact_root: Path, workspace_root: Path) -> str:
	artifact = str(artifact_root.resolve())
	workspace = str(workspace_root.resolve())
	if value == artifact:
		return _ARTIFACT_ROOT_PLACEHOLDER
	if value.startswith(f'{artifact}/'):
		return f'{_ARTIFACT_ROOT_PLACEHOLDER}{value[len(artifact):]}'
	if value == workspace:
		return '.'
	if value.startswith(f'{workspace}/'):
		return value[len(workspace) + 1 :]
	return _replace_embedded_roots(
		value,
		artifact_root=artifact,
		workspace_root=workspace,
	)


def _replace_embedded_roots(
	value: str, *, artifact_root: str, workspace_root: str
) -> str:
	result = value.replace(
		f'{artifact_root}/', f'{_ARTIFACT_ROOT_PLACEHOLDER}/'
	)
	return result.replace(f'{workspace_root}/', '')


def _write_portable_publish_manifest(
	manifest: PublishManifest, *, config: F3M5LateralSmoothingReviewConfig
) -> None:
	payload = _portable_value(
		publish_manifest_to_dict(manifest),
		artifact_root=config.artifact_root,
		workspace_root=config.workspace_root,
	)
	if not isinstance(payload, dict):
		raise TypeError('portable publish manifest must be a mapping')
	payload['source_artifact_root'] = _ARTIFACT_ROOT_PLACEHOLDER
	manifest.manifest_path.write_text(_json_text(payload), encoding='utf-8')


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _mapping_or_empty(value: object) -> Mapping[str, object]:
	return {} if value is None else _mapping(value, 'optional mapping')


__all__ = [
	'LATERAL_SMOKE_SUMMARY_JSON',
	'LATERAL_TARGET_CALIBRATION_HANDOFF_JSON',
	'LATERAL_TARGET_CALIBRATION_SUMMARY_JSON',
	'LATERAL_TARGET_CALIBRATION_SUMMARY_MARKDOWN',
	'LATERAL_TARGET_CANDIDATES_CSV',
	'OUTPUT_NAMES',
	'F3M5LateralSmoothingReviewConfig',
	'F3M5LateralSmoothingReviewResult',
	'compact_f3_m5_lateral_smoke_evidence',
	'f3_m5_lateral_smoothing_review_config_from_mapping',
	'publish_f3_m5_lateral_smoothing_review',
	'render_f3_m5_lateral_smoothing_review_markdown',
]

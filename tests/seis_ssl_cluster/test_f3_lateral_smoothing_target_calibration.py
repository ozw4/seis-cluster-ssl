"""Focused target-only M5-LS calibration contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import seis_ssl_cluster.f3.lateral_smoothing_target_calibration as calibration


def _write(path: Path, value: bytes = b'{}') -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_bytes(value)
	return path


def _mapping(tmp_path: Path) -> dict[str, object]:
	artifact_root = tmp_path / 'artifacts'
	artifact_root.mkdir(parents=True)
	configs = tmp_path / 'configs'
	return {
		'artifact_root': str(artifact_root),
		'source_hard_manifest': str(_write(artifact_root / 'source' / 'hard.json')),
		'source_posterior_manifest': str(
			_write(artifact_root / 'source' / 'posterior.json')
		),
		'candidate_manifests': {
			'beta010': str(_write(artifact_root / 'candidates' / 'beta010.json')),
			'beta025': str(_write(artifact_root / 'candidates' / 'beta025.json')),
			'beta050': str(_write(artifact_root / 'candidates' / 'beta050.json')),
		},
		'selected_manifest': str(artifact_root / 'selected' / 'selected.json'),
		'calibration_handoff': str(artifact_root / 'selected' / 'handoff.json'),
		'calibration_report': str(artifact_root / 'selected' / 'report.json'),
		'hard_full_config': str(_write(configs / 'hard.yaml')),
		'lateral_smoke_config': str(_write(configs / 'smoke.yaml')),
		'lateral_full_config': str(_write(configs / 'full.yaml')),
	}


def _aggregate() -> dict[str, object]:
	return {
		'valid_token_count': 10,
		'invalid_token_count': 0,
		'changed_token_count': 2,
		'changed_fraction': 0.2,
		'changed_fraction_by_source_region': {
			'boundary_adjacent': 0.4,
			'interior': 0.2,
		},
		'state_occupancy': {'empty_state_count': 0},
		'ordered_path': {'violation_count': 0, 'max_reverse_decrease': 0},
		'trace_paths': {'transition_count': 3},
		'xy_edge_disagreement': {
			'affinity_weighted_normalized_order': {'source': 0.4, 'lateral': 0.3},
			'affinity_quartiles': [
				{},
				{},
				{},
				{
					'edge_count': 2,
					'source_unweighted_mean': 0.5,
					'lateral_unweighted_mean': 0.4,
				},
			],
		},
	}


def _parity() -> dict[str, object]:
	return {
		'status': 'PASS',
		'pairwise_strength_ratio': 0.0,
		'target_semantics': calibration.LATERAL_SMOOTHING_SEMANTICS,
		'heads': {
			str(k): {
				'survey_count': 1,
				'surveys': {
					'survey': {
						'trace_count': 1,
						'valid_trace_count': 1,
						'valid_token_count': 2,
						'labels_bitwise_identical': True,
						'valid_mask_exact': True,
					}
				},
				'labels_bitwise_identical': True,
				'valid_masks_exact': True,
			}
			for k in (6, 8, 10)
		},
	}


def _candidate_checks(*, eligible: bool) -> dict[str, dict[str, bool]]:
	return {
		str(k): {
			'diagnostics_well_formed': True,
			'ordered_path': True,
			'nonempty_state_occupancy': True,
			'changed_tokens_positive': eligible,
			'affinity_weighted_disagreement_reduced': True,
			'highest_affinity_quartile_reduced': True,
			'boundary_change_not_less_than_interior': True,
			'transition_count_not_increased': True,
		}
		for k in (6, 8, 10)
	}


def _evidence(
	config: calibration.F3M5LateralTargetCalibrationConfig,
	*,
	selected_beta: float | None = 0.10,
) -> dict[str, object]:
	candidates: dict[str, object] = {}
	for candidate in config.candidates:
		eligible = candidate.beta == selected_beta
		candidates[candidate.name] = {
			'beta': candidate.beta,
			'manifest': calibration._reference(candidate.manifest),  # noqa: SLF001
			'head_hashes': {str(k): {} for k in (6, 8, 10)},
			'smoothing': {},
			'heads': {str(k): {} for k in (6, 8, 10)},
			'eligibility': {
				'eligible': eligible,
				'checks': _candidate_checks(eligible=eligible),
				'reasons': []
				if eligible
				else [
					f'K={k}: lateral target changes zero tokens' for k in (6, 8, 10)
				],
			},
		}
	selected = next(
		(
			candidate
			for candidate in config.candidates
			if candidate.beta == selected_beta
		),
		None,
	)
	selected_reference = (
		None if selected is None else calibration._reference(selected.manifest)  # noqa: SLF001
	)
	return {
		'artifact_type': 'f3_m5_lateral_target_calibration',
		'schema_version': 1,
		'status': 'M5_LS_TARGET_SELECTED'
		if selected is not None
		else 'M5_LS_TARGET_HOLD',
		'selection_policy': 'target_only_smallest_eligible_beta_v1',
		'candidate_betas': [0.10, 0.25, 0.50],
		'beta_zero_parity': _parity(),
		'source_hard_manifest': calibration._reference(config.source_hard_manifest),  # noqa: SLF001
		'source_posterior_manifest': calibration._reference(  # noqa: SLF001
			config.source_posterior_manifest
		),
		'candidates': candidates,
		'selected_beta': selected_beta,
		'selected_candidate_manifest': selected_reference,
		'selected_manifest': None
		if selected_reference is None
		else {
			'path': str(config.selected_manifest),
			'sha256': selected_reference['sha256'],
		},
		'training_configs': {
			'hard_full_config': calibration._reference(config.hard_full_config),  # noqa: SLF001
			'lateral_smoke_config': calibration._reference(  # noqa: SLF001
				config.lateral_smoke_config
			),
			'lateral_full_config': calibration._reference(config.lateral_full_config),  # noqa: SLF001
		},
		'git': {
			'head': 'a' * 40,
			'dirty_status': [],
			'git_diff_sha256': 'b' * 64,
		},
	}


def test_calibration_config_requires_exact_ordered_canonical_candidates(
	tmp_path: Path,
) -> None:
	config = _mapping(tmp_path)
	resolved = calibration.f3_m5_lateral_target_calibration_config_from_mapping(config)
	assert [(item.name, item.beta) for item in resolved.candidates] == [
		('beta010', 0.10),
		('beta025', 0.25),
		('beta050', 0.50),
	]

	unknown_beta = _mapping(tmp_path / 'unknown')
	unknown_beta['candidate_manifests'] = {
		'beta012': unknown_beta['candidate_manifests']['beta010'],  # type: ignore[index]
		'beta025': unknown_beta['candidate_manifests']['beta025'],  # type: ignore[index]
		'beta050': unknown_beta['candidate_manifests']['beta050'],  # type: ignore[index]
	}
	with pytest.raises(ValueError, match='canonical'):
		calibration.f3_m5_lateral_target_calibration_config_from_mapping(unknown_beta)

	reordered = _mapping(tmp_path / 'reordered')
	reordered_candidates = reordered['candidate_manifests']
	assert isinstance(reordered_candidates, dict)
	reordered['candidate_manifests'] = {
		'beta025': reordered_candidates['beta025'],
		'beta010': reordered_candidates['beta010'],
		'beta050': reordered_candidates['beta050'],
	}
	with pytest.raises(ValueError, match='ordered canonical'):
		calibration.f3_m5_lateral_target_calibration_config_from_mapping(reordered)

	for forbidden in (
		'lithology_labels',
		'facies_labels',
		'evaluation_metrics',
		'head_ks',
	):
		invalid = _mapping(tmp_path / forbidden)
		invalid[forbidden] = '/not/allowed'
		with pytest.raises(ValueError, match='unknown'):
			calibration.f3_m5_lateral_target_calibration_config_from_mapping(invalid)


def test_head_eligibility_requires_every_preregistered_diagnostic() -> None:
	checks, reasons = calibration._head_eligibility_checks(  # noqa: SLF001
		_aggregate(), source_transition_count=3
	)
	assert all(checks.values())
	assert reasons == []

	mutations = {
		'zero_changed': lambda value: value.update(changed_token_count=0),
		'weighted_not_reduced': lambda value: value['xy_edge_disagreement'][
			'affinity_weighted_normalized_order'
		].update(lateral=0.4),
		'highest_not_reduced': lambda value: value['xy_edge_disagreement'][
			'affinity_quartiles'
		][-1].update(lateral_unweighted_mean=0.5),
		'boundary_below_interior': lambda value: value[
			'changed_fraction_by_source_region'
		].update(boundary_adjacent=0.1),
		'transition_increase': lambda value: value['trace_paths'].update(
			transition_count=4
		),
		'nonfinite': lambda value: value.update(changed_fraction=float('nan')),
	}
	for name, mutate in mutations.items():
		value = _aggregate()
		mutate(value)
		checks, reasons = calibration._head_eligibility_checks(  # noqa: SLF001
			value, source_transition_count=3
		)
		assert not all(checks.values()), name
		assert reasons, name

	malformed = _aggregate()
	malformed.pop('trace_paths')
	with pytest.raises(TypeError, match='trace path'):
		calibration._head_eligibility_checks(  # noqa: SLF001
			malformed, source_transition_count=3
		)


def test_smallest_eligible_candidate_is_selected_without_reading_labels(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	config = calibration.f3_m5_lateral_target_calibration_config_from_mapping(
		_mapping(tmp_path)
	)
	evidence = {
		'beta010': {'eligibility': {'eligible': False}},
		'beta025': {'eligibility': {'eligible': True}},
		'beta050': {'eligibility': {'eligible': True}},
	}
	monkeypatch.setattr(
		Path,
		'read_bytes',
		lambda _path: (_ for _ in ()).throw(AssertionError('selection read a file')),
	)
	selector = getattr(calibration, '_select' + '_smallest_eligible')
	assert selector(config.candidates, evidence).beta == 0.25


def test_all_ineligible_is_hold_and_beta_zero_failure_stops_decision(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	config = calibration.f3_m5_lateral_target_calibration_config_from_mapping(
		_mapping(tmp_path)
	)
	evidence = {
		candidate.name: {'eligibility': {'eligible': False}}
		for candidate in config.candidates
	}
	assert calibration._select_smallest_eligible(config.candidates, evidence) is None  # noqa: SLF001

	monkeypatch.setattr(
		calibration,
		'load_multi_head_target_manifest',
		lambda _path: {'head_ks': [6, 8, 10]},
	)
	monkeypatch.setattr(
		calibration,
		'load_multi_head_state_posterior_manifest',
		lambda _path: {'head_ks': [6, 8, 10]},
	)
	monkeypatch.setattr(calibration, '_source_contract', lambda *_args: None)
	monkeypatch.setattr(
		calibration,
		'_beta_zero_parity',
		lambda *_args: {**_parity(), 'status': 'FAIL'},
	)
	monkeypatch.setattr(
		calibration,
		'_source_transition_counts',
		lambda _hard: {'6': 0, '8': 0, '10': 0},
	)
	with pytest.raises(ValueError, match='beta-zero parity'):
		calibration._calibration_evidence(config)  # noqa: SLF001


def test_beta_zero_trace_parity_uses_lateral_core_and_rejects_mismatch(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	source = np.asarray([[[0, 1]]], dtype=np.int32)
	valid = np.asarray([[[True, True]]], dtype=bool)
	posterior = np.zeros((1, 1, 2, 6), dtype=np.float32)
	features = np.ones((1, 1, 2, 1), dtype=np.float32)
	zero_config = SimpleNamespace(pairwise_strength_ratio=0.0)
	context = SimpleNamespace(
		inputs={'survey': object()},
		models={6: {}},
		zero_config=zero_config,
		affinity_scale=1.0,
		gap_scales={6: 1.0},
	)
	calls: list[object] = []

	def exact_core(**kwargs):
		calls.append(kwargs['config'])
		labels = kwargs['source_labels'][0, 0, kwargs['z']].copy()
		return labels, SimpleNamespace(labels=labels)

	monkeypatch.setattr(calibration.lateral_targets, '_smooth_trace', exact_core)
	assert calibration._beta_zero_trace_parity(  # noqa: SLF001
		context,
		k=6,
		survey_id='survey',
		source_labels=source,
		valid=valid,
		posterior=posterior,
		features=features,
	) == (1, 1, 2)
	assert calls == [zero_config]

	def mismatched_core(**kwargs):
		labels = kwargs['source_labels'][0, 0, kwargs['z']].copy()
		return labels, SimpleNamespace(labels=np.asarray([1, 0], dtype=np.int32))

	monkeypatch.setattr(calibration.lateral_targets, '_smooth_trace', mismatched_core)
	with pytest.raises(ValueError, match='bitwise'):
		calibration._beta_zero_trace_parity(  # noqa: SLF001
			context,
			k=6,
			survey_id='survey',
			source_labels=source,
			valid=valid,
			posterior=posterior,
			features=features,
		)


def test_selected_manifest_is_byte_exact_and_reuses_without_mtime_change(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	mapping = _mapping(tmp_path)
	candidate_path = Path(mapping['candidate_manifests']['beta010'])  # type: ignore[index]
	candidate_path.write_bytes(b'{"candidate":"beta010","bytes":"exact"}\n')
	config = calibration.f3_m5_lateral_target_calibration_config_from_mapping(mapping)
	evidence = _evidence(config)
	monkeypatch.setattr(calibration, '_calibration_evidence', lambda _config: evidence)
	monkeypatch.setattr(
		calibration.lateral_targets,
		'load_multi_head_lateral_target_manifest',
		lambda _path: {},
	)
	first = calibration.calibrate_f3_m5_lateral_targets(config)
	assert first.status == 'M5_LS_TARGET_SELECTED'
	assert config.selected_manifest.read_bytes() == candidate_path.read_bytes()
	mtime = config.selected_manifest.stat().st_mtime_ns
	second = calibration.calibrate_f3_m5_lateral_targets(config, only_missing=True)
	assert second.published_selected_manifest is None
	assert config.selected_manifest.stat().st_mtime_ns == mtime
	assert (
		calibration.load_f3_m5_lateral_target_calibration_handoff(
			config.calibration_handoff
		)['selected_beta']
		== 0.10
	)


def test_incompatible_selection_is_not_silently_overwritten(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	config = calibration.f3_m5_lateral_target_calibration_config_from_mapping(
		_mapping(tmp_path)
	)
	config.selected_manifest.parent.mkdir(parents=True)
	config.selected_manifest.write_bytes(b'prior selected evidence')
	monkeypatch.setattr(
		calibration, '_calibration_evidence', lambda _config: _evidence(config)
	)
	with pytest.raises(ValueError, match='incompatible'):
		calibration.calibrate_f3_m5_lateral_targets(config)
	assert config.selected_manifest.read_bytes() == b'prior selected evidence'


def test_partial_evidence_never_publishes_selected_manifest(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	config = calibration.f3_m5_lateral_target_calibration_config_from_mapping(
		_mapping(tmp_path)
	)
	monkeypatch.setattr(
		calibration,
		'_calibration_evidence',
		lambda _config: (_ for _ in ()).throw(ValueError('parity mismatch')),
	)
	with pytest.raises(ValueError, match='parity mismatch'):
		calibration.calibrate_f3_m5_lateral_targets(config)
	assert not config.selected_manifest.exists()
	assert not config.calibration_handoff.exists()
	assert not config.calibration_report.exists()


def test_evidence_publication_failure_never_publishes_selected_manifest(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	"""The selected path is the final completion marker of publication."""
	config = calibration.f3_m5_lateral_target_calibration_config_from_mapping(
		_mapping(tmp_path)
	)
	monkeypatch.setattr(
		calibration, '_calibration_evidence', lambda _config: _evidence(config)
	)
	original_publish = calibration._publish_immutable_json  # noqa: SLF001

	def fail_report(
		path: Path,
		payload: object,
		*,
		label: str,
		quarantine_invalid: bool,
	) -> Path | None:
		if label == 'calibration report':
			raise OSError('injected report publication failure')
		return original_publish(
			path,
			payload,
			label=label,
			quarantine_invalid=quarantine_invalid,
		)

	monkeypatch.setattr(calibration, '_publish_immutable_json', fail_report)
	with pytest.raises(OSError, match='injected report publication failure'):
		calibration.calibrate_f3_m5_lateral_targets(config)
	assert config.calibration_handoff.is_file()
	assert not config.calibration_report.exists()
	assert not config.selected_manifest.exists()


def test_hold_quarantines_stale_selected_manifest_when_explicitly_allowed(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	config = calibration.f3_m5_lateral_target_calibration_config_from_mapping(
		_mapping(tmp_path)
	)
	config.selected_manifest.parent.mkdir(parents=True)
	config.selected_manifest.write_bytes(b'stale selected evidence')
	monkeypatch.setattr(
		calibration,
		'_calibration_evidence',
		lambda _config: _evidence(config, selected_beta=None),
	)
	calibration.calibrate_f3_m5_lateral_targets(
		config, quarantine_invalid=True
	)
	assert not config.selected_manifest.exists()
	assert list(config.selected_manifest.parent.glob('selected.json.quarantine.*'))


def test_candidate_manifest_drift_stops_selection_before_publication(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	mapping = _mapping(tmp_path)
	config = calibration.f3_m5_lateral_target_calibration_config_from_mapping(mapping)
	evidence = _evidence(config)
	selected_candidate = next(
		candidate for candidate in config.candidates if candidate.beta == 0.10
	)
	selected_candidate.manifest.write_bytes(b'{"changed":"after-validation"}\n')
	monkeypatch.setattr(calibration, '_calibration_evidence', lambda _config: evidence)
	with pytest.raises(ValueError, match='drifted'):
		calibration.calibrate_f3_m5_lateral_targets(config)
	assert not config.selected_manifest.exists()


def test_hold_publishes_target_only_evidence_without_selected_manifest(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	config = calibration.f3_m5_lateral_target_calibration_config_from_mapping(
		_mapping(tmp_path)
	)
	monkeypatch.setattr(
		calibration,
		'_calibration_evidence',
		lambda _config: _evidence(config, selected_beta=None),
	)
	result = calibration.calibrate_f3_m5_lateral_targets(config)
	assert result.status == 'M5_LS_TARGET_HOLD'
	assert result.selected_beta is None
	assert result.published_selected_manifest is None
	assert not config.selected_manifest.exists()
	assert config.calibration_handoff.is_file()
	assert config.calibration_report.is_file()
	assert (
		calibration.load_f3_m5_lateral_target_calibration_handoff(
			config.calibration_handoff
		)['status']
		== 'M5_LS_TARGET_HOLD'
	)


def test_handoff_rejects_selected_hash_mismatch(tmp_path: Path) -> None:
	config = calibration.f3_m5_lateral_target_calibration_config_from_mapping(
		_mapping(tmp_path)
	)
	payload = _evidence(config)
	payload['selected_manifest']['sha256'] = '0' * 64  # type: ignore[index]
	config.calibration_handoff.parent.mkdir(parents=True)
	config.calibration_handoff.write_text(json.dumps(payload), encoding='utf-8')
	with pytest.raises(ValueError, match='byte-exact'):
		calibration.load_f3_m5_lateral_target_calibration_handoff(
			config.calibration_handoff
		)


def test_handoff_rejects_non_smallest_eligible_selection(tmp_path: Path) -> None:
	config = calibration.f3_m5_lateral_target_calibration_config_from_mapping(
		_mapping(tmp_path)
	)
	payload = _evidence(config, selected_beta=0.10)
	beta025 = payload['candidates']['beta025']  # type: ignore[index]
	assert isinstance(beta025, dict)
	beta025['eligibility'] = {
		'eligible': True,
		'checks': _candidate_checks(eligible=True),
		'reasons': [],
	}
	selected = next(
		candidate for candidate in config.candidates if candidate.beta == 0.25
	)
	payload['selected_beta'] = 0.25
	payload['selected_candidate_manifest'] = calibration._reference(  # noqa: SLF001
		selected.manifest
	)
	payload['selected_manifest'] = {
		'path': str(config.selected_manifest),
		'sha256': payload['selected_candidate_manifest']['sha256'],  # type: ignore[index]
	}
	config.calibration_handoff.parent.mkdir(parents=True)
	config.calibration_handoff.write_text(json.dumps(payload), encoding='utf-8')
	with pytest.raises(ValueError, match='smallest eligible'):
		calibration.load_f3_m5_lateral_target_calibration_handoff(
			config.calibration_handoff
		)


def test_handoff_rejects_incomplete_beta_zero_survey_evidence(tmp_path: Path) -> None:
	config = calibration.f3_m5_lateral_target_calibration_config_from_mapping(
		_mapping(tmp_path)
	)
	payload = _evidence(config)
	head = payload['beta_zero_parity']['heads']['6']  # type: ignore[index]
	assert isinstance(head, dict)
	head['surveys'] = {}
	config.calibration_handoff.parent.mkdir(parents=True)
	config.calibration_handoff.write_text(json.dumps(payload), encoding='utf-8')
	with pytest.raises(ValueError, match='survey coverage'):
		calibration.load_f3_m5_lateral_target_calibration_handoff(
			config.calibration_handoff
		)


def test_handoff_rejects_incomplete_per_head_eligibility_checks(
	tmp_path: Path,
) -> None:
	config = calibration.f3_m5_lateral_target_calibration_config_from_mapping(
		_mapping(tmp_path)
	)
	payload = _evidence(config)
	checks = payload['candidates']['beta010']['eligibility']['checks']  # type: ignore[index]
	assert isinstance(checks, dict)
	head = checks['6']
	assert isinstance(head, dict)
	head.pop('transition_count_not_increased')
	config.calibration_handoff.parent.mkdir(parents=True)
	config.calibration_handoff.write_text(json.dumps(payload), encoding='utf-8')
	with pytest.raises(ValueError, match='eligibility checks are invalid'):
		calibration.load_f3_m5_lateral_target_calibration_handoff(
			config.calibration_handoff
		)

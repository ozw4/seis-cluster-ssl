"""Aggregate the center-trace masked original-split screening result."""
# ruff: noqa: SLF001, S603

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from seis_ssl_cluster.config import (
	f3_lithology_voxel_label_budget_center_trace_masked as center_config,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_multi_head import (
	f3_lithology_voxel_label_budget_multi_head_config_from_mapping,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology import (
	voxel_label_budget_center_trace_masked as center_runner,
)
from seis_ssl_cluster.f3.lithology import voxel_label_budget_control as control
from seis_ssl_cluster.f3.lithology import voxel_label_budget_multi_head as runner
from seis_ssl_cluster.f3.lithology import (
	voxel_label_budget_multi_head_results as shared_results,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_results import (
	load_f3_lithology_voxel_label_budget_evaluation_metrics,
)

COMPARISONS = (
	(center_config.CENTER_TRACE_MASKED_MODEL_ID, 'mh_nocons'),
	(center_config.CENTER_TRACE_MASKED_MODEL_ID, 'm1_current_k6'),
	(center_config.CENTER_TRACE_MASKED_MODEL_ID, 'mae'),
)
REPORT_OUTPUT_NAMES = (
	'center_trace_masked_original_job_metrics.csv',
	'center_trace_masked_original_paired_deltas.csv',
	'center_trace_masked_original_results_summary.json',
	'center_trace_masked_original_results_summary.md',
	'center_trace_masked_original_handoff.json',
)
AUDIT_OUTPUT_NAMES = (
	'center_trace_masked_screening_audit.json',
	'center_trace_masked_hard_baseline_parity.json',
)
OUTPUT_NAMES = REPORT_OUTPUT_NAMES
PUBLISHED_OUTPUT_NAMES = AUDIT_OUTPUT_NAMES + REPORT_OUTPUT_NAMES
_ARTIFACT_ROOT_PLACEHOLDER = '${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}'
_CANDIDATE_ROLE = center_config.CENTER_TRACE_MASKED_MODEL_ID
_CANDIDATE_TAG = center_config.CENTER_TRACE_MASKED_MODEL_TAG
_SCIENTIFIC_JOB_COUNT = 15
_GATE_METRICS = (
	'macro_f1',
	'mean_iou',
	'class_3_f1',
	'class_3_iou',
	'class_3_boundary_recall_t2',
	'class_3_boundary_recall_t4',
	'class_5_f1',
	'class_5_iou',
	'class_5_boundary_recall_t2',
	'class_5_boundary_recall_t4',
)
_PUBLISHED_ROOT = Path(
	'f3/facies_benchmark_v1/'
	'strat_hmm_multi_head_k6810_center_trace_masked_original_split_v1'
)


def inspect_f3_lithology_voxel_label_budget_center_trace_masked_results(
	config: object,
) -> Mapping[str, object]:
	"""Recompute the exact four-role, 60-row paired original-split matrix."""
	audit = _load_screening_audit(config)
	candidate_inspection = (
		center_runner.inspect_f3_lithology_voxel_label_budget_center_trace_masked(
			config
		)
	)
	candidate_rows = (
		center_runner.load_f3_lithology_voxel_label_budget_center_trace_masked_rows(
			config, inspection=candidate_inspection
		)
	)
	current_rows = tuple(candidate_inspection.current_rows.values())
	reference = candidate_inspection.historical_reference
	hard_config = f3_lithology_voxel_label_budget_multi_head_config_from_mapping(
		load_config(config.hard_multi_head_config)
	)
	hard_rows = candidate_inspection.hard_reference_rows
	if len(hard_rows) != config.job_count:
		raise ValueError('hard mh_nocons result matrix must contain 15 rows')
	members = _members(
		(*candidate_rows, *current_rows, *hard_rows), reference=reference
	)
	_expected_matrix(config, members)
	_validate_pairing(config, members)
	deltas = shared_results._paired_deltas(config, members, comparisons=COMPARISONS)
	summary = shared_results._summary(config, deltas, comparisons=COMPARISONS)
	decisions = decide_center_trace_masked_original_gate(
		summary, budgets=config.budgets
	)
	candidate_identity = candidate_inspection.candidate_identities.get(_CANDIDATE_ROLE)
	if not isinstance(candidate_identity, Mapping):
		raise TypeError('center-trace masked candidate lineage is missing')
	return {
		'job_metrics': tuple(
			control._member_metric_row(value) for value in members.values()
		),
		'paired_deltas': tuple(deltas),
		'summary_by_budget': tuple(summary),
		'decisions': decisions,
		'screening_audit': audit,
		'source_identities': {
			'screening_audit': _identity(Path(config.screening_audit)),
			'candidate_run_manifest': _identity(
				center_runner.center_trace_masked_run_manifest_path(config)
			),
			'candidate_provenance': _candidate_provenance(
				config, candidate_identity=candidate_identity
			),
			'hard_decoder_config': _identity(Path(config.hard_multi_head_config)),
			'reference_run_manifests': {
				'hard_multi_head': _identity(
					runner.multi_head_run_manifest_path(hard_config)
				),
				'current_k6': _identity(Path(config.current_k6_run_manifest)),
				'mae': _identity(Path(config.original_run_manifest)),
			},
			'paired_matrix_identity': {
				'roles': (
					'mae',
					'm1_current_k6',
					'mh_nocons',
					_CANDIDATE_ROLE,
				),
				'budgets': tuple(config.budgets),
				'subsample_seeds': tuple(config.subsample_seeds),
				'row_count': 4 * len(config.budgets) * len(config.subsample_seeds),
				'pair_identity_keys': tuple(control.PAIR_IDENTITY_KEYS),
			},
			'candidate_job_live_validation': {
				'status': 'PASS',
				'expected_count': config.job_count,
				'validated_count': len(candidate_rows),
			},
		},
	}


def decide_center_trace_masked_original_gate(
	summary: Sequence[Mapping[str, object]], *, budgets: Sequence[str]
) -> dict[str, object]:
	"""Apply the fixed center-trace masked original-split GO/HOLD/STOP gate."""
	comparison_id = control._comparison_id(_CANDIDATE_ROLE, 'mh_nocons')
	index = _validate_gate_summary(
		summary, budgets=budgets, comparison_id=comparison_id
	)
	positive, negative = [], []
	for budget in budgets:
		primary = [
			_summary_row(index, budget, metric) for metric in ('macro_f1', 'mean_iou')
		]
		if all(
			float(row['mean_delta']) > 0 and int(row['wins']) >= 4 for row in primary
		):
			positive.append(budget)
		if all(
			float(row['mean_delta']) < 0 and int(row['wins']) <= 1 for row in primary
		):
			negative.append(budget)
	degradations = []
	for class_id in (3, 5):
		for metric in ('f1', 'iou', 'boundary_recall_t2', 'boundary_recall_t4'):
			bad = [
				budget
				for budget in budgets
				if float(
					_summary_row(index, budget, f'class_{class_id}_{metric}')[
						'mean_delta'
					]
				)
				<= -0.05
			]
			if len(bad) >= 2:
				degradations.append(
					{'class_id': class_id, 'metric': metric, 'budgets': bad}
				)
	if len(positive) >= 2 and not degradations:
		status = 'CTMASK_ORIGINAL_GO'
	elif len(negative) >= 2 or degradations:
		status = 'CTMASK_ORIGINAL_STOP'
	else:
		status = 'CTMASK_ORIGINAL_HOLD'
	return {
		'artifact_type': 'f3_center_trace_masked_original_gate',
		'schema_version': 1,
		'overall_status': status,
		'hard_vs_center_trace_masked': {
			'positive_budgets': positive,
			'negative_budgets': negative,
			'systematic_major_degradation': degradations,
		},
		'gate': {
			'primary_comparison_id': comparison_id,
			'primary_metrics': ('macro_f1', 'mean_iou'),
			'positive_mean_delta': '> 0',
			'negative_mean_delta': '< 0',
			'minimum_primary_wins': 4,
			'maximum_negative_primary_wins': 1,
			'minimum_positive_budgets': 2,
			'minimum_negative_budgets': 2,
			'major_degradation_classes': (3, 5),
			'major_degradation_metrics': (
				'f1',
				'iou',
				'boundary_recall_t2',
				'boundary_recall_t4',
			),
			'major_degradation_delta': -0.05,
			'major_degradation_budget_count': 2,
		},
		'six_split_follow_up': {
			'ready': status == 'CTMASK_ORIGINAL_GO',
			'scientific_jobs_executed': 0,
			'six_split_jobs_executed': 0,
		},
		'scientific_jobs_executed': _SCIENTIFIC_JOB_COUNT,
		'six_split_jobs_executed': 0,
	}


def _validate_gate_summary(
	summary: Sequence[Mapping[str, object]],
	*,
	budgets: Sequence[str],
	comparison_id: str,
) -> dict[tuple[str, str], Mapping[str, object]]:
	"""Validate summary rows before indexing the fixed gate inputs."""
	seen: set[tuple[str, str, str]] = set()
	index: dict[tuple[str, str], Mapping[str, object]] = {}
	for row in summary:
		if not isinstance(row, Mapping):
			raise TypeError('center-trace masked gate summary rows must be mappings')
		try:
			key = (
				str(row['budget_id']),
				str(row['comparison_id']),
				str(row['metric']),
			)
		except KeyError as error:
			raise ValueError(
				'center-trace masked gate summary row is missing its identity'
			) from error
		if key in seen:
			raise ValueError(
				f'center-trace masked gate summary contains duplicate row: {key!r}'
			)
		seen.add(key)
		try:
			mean_delta = float(row['mean_delta'])
			wins = float(row['wins'])
		except (KeyError, TypeError, ValueError) as error:
			raise ValueError(
				f'center-trace masked gate summary has invalid metric row: {key!r}'
			) from error
		if not math.isfinite(mean_delta) or not math.isfinite(wins):
			raise ValueError(
				f'center-trace masked gate summary has non-finite metric row: {key!r}'
			)
		if wins != int(wins) or not 0 <= wins <= 5:
			raise ValueError(
				f'center-trace masked gate summary has invalid win count: {key!r}'
			)
		if key[1] == comparison_id:
			index[(key[0], key[2])] = row

	expected = {
		(str(budget), metric) for budget in budgets for metric in _GATE_METRICS
	}
	if set(index) != expected:
		missing = sorted(expected - set(index))
		extra = sorted(set(index) - expected)
		raise ValueError(
			'center-trace masked gate summary matrix is incomplete or duplicated: '
			f'missing={missing!r}, extra={extra!r}'
		)
	return index


def summarize_f3_lithology_voxel_label_budget_center_trace_masked(
	config: object, *, publish: bool = True
) -> Mapping[str, object]:
	"""Write lightweight reports and publish only the closed result contract."""
	inspection = inspect_f3_lithology_voxel_label_budget_center_trace_masked_results(
		config
	)
	execution = _execution_git_state(config)
	reports = Path(config.reports_dir)
	reports.mkdir(parents=True, exist_ok=True)
	published = _portable_payload(inspection, config=config)
	_write_audit_evidence(reports, audit=inspection['screening_audit'], config=config)
	_write_csv(reports / REPORT_OUTPUT_NAMES[0], published['job_metrics'])
	_write_csv(reports / REPORT_OUTPUT_NAMES[1], published['paired_deltas'])
	_write_json(reports / REPORT_OUTPUT_NAMES[2], published)
	decisions = inspection['decisions']
	(reports / REPORT_OUTPUT_NAMES[3]).write_text(
		'# Center-trace masked original-split screening\n\n'
		f'Status: `{decisions["overall_status"]}`\n\n'
		f'Six-split follow-up ready: `{decisions["six_split_follow_up"]["ready"]}`\n\n'
		'This is paired original-split facies-label evidence only.\n',
		encoding='utf-8',
	)
	handoff = _handoff_payload(inspection, execution=execution)
	_write_json(
		reports / REPORT_OUTPUT_NAMES[4], _portable_payload(handoff, config=config)
	)
	published_files: tuple[Path, ...] = ()
	if publish:
		output = config.base.publish.reports_root / _PUBLISHED_ROOT
		output.mkdir(parents=True, exist_ok=True)
		published_files = tuple(output / name for name in PUBLISHED_OUTPUT_NAMES)
		for source, destination in zip(
			(reports / name for name in PUBLISHED_OUTPUT_NAMES),
			published_files,
			strict=True,
		):
			shutil.copyfile(source, destination)
	return {
		'summary_json': reports / REPORT_OUTPUT_NAMES[2],
		'decisions': decisions,
		'published_files': published_files,
	}


def _summary_row(
	index: Mapping[tuple[str, str], Mapping[str, object]], budget: str, metric: str
) -> Mapping[str, object]:
	try:
		return index[(budget, metric)]
	except KeyError as error:
		raise ValueError(
			f'center-trace masked primary gate is missing {budget}/{metric}'
		) from error


def _members(
	rows: Sequence[Mapping[str, object]], *, reference: object
) -> dict[tuple[str, int, str], Mapping[str, object]]:
	members = {}
	for row in rows:
		role = str(row['model_role'])
		key = (str(row['budget_id']), int(row['subsample_seed']), role)
		if key in members:
			raise ValueError(f'center-trace masked duplicate member: {key!r}')
		members[key] = {
			'role': role,
			'model_tag': row['model_tag'],
			'row': row,
			'metrics': load_f3_lithology_voxel_label_budget_evaluation_metrics(
				metrics_path=control._identity_path(
					row['evaluation_metrics'], 'evaluation metrics'
				),
				boundary_metrics_path=control._identity_path(
					row['evaluation_boundary_metrics'], 'boundary metrics'
				),
				boundary_region_metrics_path=control._identity_path(
					row['evaluation_boundary_region_metrics'],
					'boundary region metrics',
				),
				label=role,
			),
		}
	for job in reference.jobs:
		key = (job.dataset.budget_id, job.dataset.subsample_seed, job.model_role)
		if key in members:
			raise ValueError(f'center-trace masked duplicate reference: {key!r}')
		members[key] = {
			'role': job.model_role,
			'model_tag': job.model_tag,
			'row': control._reference_member_row(job),
			'metrics': job.evaluation.metrics,
		}
	return members


def _expected_matrix(
	config: object, members: Mapping[tuple[str, int, str], object]
) -> None:
	expected = {
		(budget, seed, role)
		for budget in config.budgets
		for seed in config.subsample_seeds
		for role in ('mae', 'm1_current_k6', 'mh_nocons', _CANDIDATE_ROLE)
	}
	if set(members) != expected:
		raise ValueError(
			'center-trace masked comparison member matrix is incomplete or duplicated'
		)


def _validate_pairing(
	config: object, members: Mapping[tuple[str, int, str], Mapping[str, object]]
) -> None:
	for budget in config.budgets:
		for seed in config.subsample_seeds:
			rows = [
				members[(budget, seed, role)]['row']
				for role in (
					'mae',
					'm1_current_k6',
					'mh_nocons',
					_CANDIDATE_ROLE,
				)
			]
			for key in control.PAIR_IDENTITY_KEYS:
				if any(row.get(key) != rows[0].get(key) for row in rows[1:]):
					raise ValueError(
						'center-trace masked paired identity mismatch: '
						f'{budget}/seed{seed}/{key}'
					)


def _load_screening_audit(config: object) -> Mapping[str, object]:
	payload = center_config.validate_f3_center_trace_masked_screening_audit(config)
	if (
		payload.get('artifact_type')
		!= 'f3_center_trace_masked_original_screening_preflight'
		or payload.get('schema_version') != 1
		or payload.get('status') != 'PASS'
	):
		raise ValueError(
			'center-trace masked screening audit type/schema/status mismatch'
		)
	for key in (
		'checkpoint',
		'embedding',
		'valid_mask_parity',
		'hard_baseline_parity',
		'reference_run_manifests',
		'dataset_job_pairing',
	):
		if not isinstance(payload.get(key), Mapping):
			raise TypeError(f'center-trace masked screening audit {key} is missing')
	candidate = payload.get('candidate')
	if not isinstance(candidate, Mapping):
		raise TypeError('center-trace masked screening audit candidate is missing')
	if (
		candidate.get('model_id') != _CANDIDATE_ROLE
		or candidate.get('model_tag') != _CANDIDATE_TAG
	):
		raise ValueError('center-trace masked screening audit candidate mismatch')
	return payload


def _candidate_provenance(
	config: object, *, candidate_identity: Mapping[str, object]
) -> Mapping[str, object]:
	candidate = config.candidates[0]
	if candidate.model_id != _CANDIDATE_ROLE or candidate.model_tag != _CANDIDATE_TAG:
		raise ValueError('center-trace masked candidate identity mismatch')
	for key in (
		'checkpoint',
		'embeddings',
		'valid_tokens',
		'metadata',
		'pretraining_handoff',
	):
		if not isinstance(candidate_identity.get(key), Mapping):
			raise TypeError(f'center-trace masked candidate {key} lineage is missing')
	return {
		'model_id': candidate.model_id,
		'model_tag': candidate.model_tag,
		'pretraining_handoff': candidate_identity['pretraining_handoff'],
		'best_checkpoint': candidate_identity['checkpoint'],
		'embeddings': candidate_identity['embeddings'],
		'valid_tokens': candidate_identity['valid_tokens'],
		'embedding_metadata': candidate_identity['metadata'],
	}


def _handoff_payload(
	inspection: Mapping[str, object], *, execution: Mapping[str, object]
) -> dict[str, object]:
	identities = inspection['source_identities']
	decisions = inspection['decisions']
	if not isinstance(identities, Mapping) or not isinstance(decisions, Mapping):
		raise TypeError('center-trace masked summary evidence is incomplete')
	return {
		'artifact_type': 'f3_center_trace_masked_original_screening_handoff',
		'schema_version': 1,
		'status': 'PASS',
		'formal_status': decisions['overall_status'],
		'screening_audit': identities['screening_audit'],
		'candidate_run_manifest': identities['candidate_run_manifest'],
		'candidate_job_live_validation': identities['candidate_job_live_validation'],
		'candidate_provenance': identities['candidate_provenance'],
		'hard_decoder_config': identities['hard_decoder_config'],
		'reference_run_manifests': identities['reference_run_manifests'],
		'paired_matrix_identity': identities['paired_matrix_identity'],
		'gate_definition': decisions['gate'],
		'gate_result': decisions,
		'execution': dict(execution),
		'six_split_follow_up': decisions['six_split_follow_up'],
		'six_split_jobs_executed': 0,
		'scientific_jobs_executed': decisions['scientific_jobs_executed'],
	}


def _execution_git_state(config: object) -> Mapping[str, object]:
	workspace = Path(config.base.reports_root).parent
	git = shutil.which('git')
	if git is None:
		raise RuntimeError('git executable is unavailable')
	try:
		sha = subprocess.run(
			(git, 'rev-parse', 'HEAD'),
			cwd=workspace,
			check=True,
			capture_output=True,
			text=True,
		).stdout.strip()
		status = subprocess.run(
			(git, 'status', '--porcelain'),
			cwd=workspace,
			check=True,
			capture_output=True,
			text=True,
		).stdout
	except (OSError, subprocess.CalledProcessError) as error:
		raise RuntimeError(
			'unable to record center-trace masked execution git state'
		) from error
	if len(sha) != 40:
		raise ValueError('execution git SHA is invalid')
	return {'git_sha': sha, 'dirty': bool(status.strip())}


def _write_audit_evidence(reports: Path, *, audit: object, config: object) -> None:
	if not isinstance(audit, Mapping):
		raise TypeError('center-trace masked audit must be a mapping')
	_write_json(
		reports / AUDIT_OUTPUT_NAMES[0], _portable_payload(audit, config=config)
	)
	_write_json(
		reports / AUDIT_OUTPUT_NAMES[1],
		_portable_payload(
			{
				'artifact_type': 'f3_center_trace_masked_hard_baseline_parity',
				'schema_version': 1,
				'screening_audit': _identity(Path(config.screening_audit)),
				'evidence': audit['hard_baseline_parity'],
			},
			config=config,
		),
	)


def _identity(path: Path) -> Mapping[str, object]:
	if not path.is_file():
		raise FileNotFoundError(path)
	return {
		'path': str(path),
		'sha256': file_sha256(path),
		'byte_size': path.stat().st_size,
	}


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
	if not rows:
		raise ValueError(f'cannot write empty table: {path.name}')
	fields = list(rows[0])
	if any(list(row) != fields for row in rows):
		raise ValueError(f'inconsistent CSV columns: {path.name}')
	with path.open('w', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
		writer.writeheader()
		writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
		encoding='utf-8',
	)


def _portable_payload(payload: object, *, config: object) -> object:
	return _portable_value(
		payload,
		artifact_root=Path(config.base.artifact_root),
		workspace_root=Path(config.base.reports_root).parent,
	)


def _portable_value(  # noqa: PLR0911
	value: object, *, artifact_root: Path, workspace_root: Path
) -> object:
	if isinstance(value, Mapping):
		return {
			key: _portable_value(
				item, artifact_root=artifact_root, workspace_root=workspace_root
			)
			for key, item in value.items()
		}
	if isinstance(value, (tuple, list)):
		values = [
			_portable_value(
				item, artifact_root=artifact_root, workspace_root=workspace_root
			)
			for item in value
		]
		return tuple(values) if isinstance(value, tuple) else values
	if not isinstance(value, str):
		return value
	path = Path(value)
	if not path.is_absolute():
		return value
	for root, replacement in (
		(artifact_root, _ARTIFACT_ROOT_PLACEHOLDER),
		(workspace_root, ''),
	):
		try:
			relative = path.relative_to(root)
		except ValueError:
			continue
		relative_text = relative.as_posix()
		if replacement:
			return (
				replacement
				if relative_text == '.'
				else f'{replacement}/{relative_text}'
			)
		return '.' if relative_text == '.' else relative_text
	return value


inspect_results = inspect_f3_lithology_voxel_label_budget_center_trace_masked_results
summarize_results = summarize_f3_lithology_voxel_label_budget_center_trace_masked


__all__ = [
	'AUDIT_OUTPUT_NAMES',
	'COMPARISONS',
	'OUTPUT_NAMES',
	'PUBLISHED_OUTPUT_NAMES',
	'REPORT_OUTPUT_NAMES',
	'decide_center_trace_masked_original_gate',
	'inspect_f3_lithology_voxel_label_budget_center_trace_masked_results',
	'summarize_f3_lithology_voxel_label_budget_center_trace_masked',
]

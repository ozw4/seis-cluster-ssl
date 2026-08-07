"""Aggregate the unanimous XY-neighbour original-split screening result."""
# ruff: noqa: SLF001

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from seis_ssl_cluster.config import (
	f3_lithology_voxel_label_budget_xy_neighbor_unanimous as unanimous_config,
)
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_multi_head import (
	f3_lithology_voxel_label_budget_multi_head_config_from_mapping,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology import voxel_label_budget_control as control
from seis_ssl_cluster.f3.lithology import voxel_label_budget_multi_head as runner
from seis_ssl_cluster.f3.lithology import (
	voxel_label_budget_multi_head_results as shared,
)
from seis_ssl_cluster.f3.lithology import (
	voxel_label_budget_xy_neighbor_consensus_results as consensus_results,
)
from seis_ssl_cluster.f3.lithology import (
	voxel_label_budget_xy_neighbor_unanimous as unanimous_runner,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_results import (
	load_f3_lithology_voxel_label_budget_evaluation_metrics,
)
from seis_ssl_cluster.f3.xy_neighbor_unanimous_target_audit import (
	replay_f3_xy_neighbor_unanimous_target_audit,
)

COMPARISONS = (
	('mh_xyunanim1_nocons', 'mh_nocons'),
	('mh_xyunanim1_nocons', 'mh_xycons1_nocons'),
	('mh_xyunanim1_nocons', 'm1_current_k6'),
	('mh_xyunanim1_nocons', 'mae'),
)
REPORT_OUTPUT_NAMES = (
	'xy_neighbor_unanimous_original_job_metrics.csv',
	'xy_neighbor_unanimous_original_paired_deltas.csv',
	'xy_neighbor_unanimous_original_results_summary.json',
	'xy_neighbor_unanimous_original_results_summary.md',
	'xy_neighbor_unanimous_original_handoff.json',
)
AUDIT_OUTPUT_NAMES = (
	'xy_neighbor_unanimous_target_audit.json',
	'xy_neighbor_unanimous_spatial_audit.csv',
	'xy_neighbor_unanimous_hard_baseline_parity.json',
	'xy_neighbor_unanimous_screening_audit.json',
)
OUTPUT_NAMES = REPORT_OUTPUT_NAMES
PUBLISHED_OUTPUT_NAMES = AUDIT_OUTPUT_NAMES + OUTPUT_NAMES
_CANDIDATE_ROLE = 'mh_xyunanim1_nocons'
_CANDIDATE_TAG = (
	'strat_hmm_pretext_mh_k6810_xyunanim1_nocons_topblock1_distill_v1'
)
_XY_CONSENSUS_ROLE = 'mh_xycons1_nocons'
_XY_CONSENSUS_TAG = (
	'strat_hmm_pretext_mh_k6810_xycons1_nocons_topblock1_distill_v1'
)
_PUBLISHED_ROOT = Path(
	'f3/facies_benchmark_v1/'
	'strat_hmm_multi_head_k6810_xy_neighbor_unanimous_original_split_v1'
)


def inspect_f3_lithology_voxel_label_budget_xy_neighbor_unanimous_results(
	config: object,
) -> Mapping[str, object]:
	"""Live-revalidate the exact five-role, 75-row paired original-split matrix."""
	audit = _load_screening_audit(config)
	candidate_inspection = (
		unanimous_runner.inspect_f3_lithology_voxel_label_budget_xy_neighbor_unanimous(
			config
		)
	)
	candidate_rows = (
		unanimous_runner.load_f3_lithology_voxel_label_budget_xy_neighbor_unanimous_rows(
			config
		)
	)
	dataset_rows = runner._dataset_rows(config)
	current_rows = tuple(runner._current_k6_rows(config, dataset_rows).values())
	reference = runner._mae_reference(config, dataset_rows)
	hard_config = f3_lithology_voxel_label_budget_multi_head_config_from_mapping(
		load_config(config.hard_multi_head_config)
	)
	hard_rows = [
		row
		for row in runner.load_f3_lithology_voxel_label_budget_multi_head_rows(
			hard_config
		)
		if row['model_role'] == 'mh_nocons'
	]
	consensus_rows = _load_xy_neighbor_consensus_reference_rows(audit)
	members = _members(
		(*candidate_rows, *consensus_rows, *current_rows, *hard_rows),
		reference=reference,
	)
	_expected_matrix(config, members)
	_validate_pairing(config, members)
	deltas = shared._paired_deltas(config, members, comparisons=COMPARISONS)
	summary = shared._summary(config, deltas, comparisons=COMPARISONS)
	decisions = decide_xy_neighbor_unanimous_original_gate(
		summary, budgets=config.budgets
	)
	candidate_identity = candidate_inspection.candidate_identities.get(_CANDIDATE_ROLE)
	if not isinstance(candidate_identity, Mapping):
		raise TypeError('XY-neighbour-unanimous candidate lineage is missing')
	references = _source_reference_identities(
		config, audit=audit, hard_config=hard_config
	)
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
			'target_audit': _target_audit_identity(audit),
			'candidate_run_manifest': _identity(
				runner.multi_head_run_manifest_path(config)
			),
			'candidate_provenance': _candidate_provenance(
				config, candidate_identity=candidate_identity
			),
			'hard_decoder_config': _identity(Path(config.hard_multi_head_config)),
			'reference_run_manifests': references,
			'paired_matrix_identity': {
				'roles': (
					'mae',
					'm1_current_k6',
					'mh_nocons',
					_XY_CONSENSUS_ROLE,
					_CANDIDATE_ROLE,
				),
				'budgets': tuple(config.budgets),
				'subsample_seeds': tuple(config.subsample_seeds),
				'row_count': 5 * len(config.budgets) * len(config.subsample_seeds),
				'pair_identity_keys': tuple(control.PAIR_IDENTITY_KEYS),
			},
		},
	}


def decide_xy_neighbor_unanimous_original_gate(
	summary: Sequence[Mapping[str, object]], *, budgets: Sequence[str]
) -> dict[str, object]:
	"""Apply the unchanged original-split gate using hard as the primary baseline."""
	comparison_id = 'mh_xyunanim1_nocons_vs_mh_nocons'
	index = {
		(str(row['budget_id']), str(row['comparison_id']), str(row['metric'])): row
		for row in summary
	}
	positive, negative = [], []
	for budget in budgets:
		primary = [
			_summary_row(index, budget, comparison_id, metric)
			for metric in ('macro_f1', 'mean_iou')
		]
		if all(
			float(row['mean_delta']) > 0 and int(row['wins']) >= 4
			for row in primary
		):
			positive.append(budget)
		if all(
			float(row['mean_delta']) < 0 and int(row['wins']) <= 1
			for row in primary
		):
			negative.append(budget)
	degradations = []
	for class_id in (3, 5):
		for metric in ('f1', 'iou', 'boundary_recall_t2', 'boundary_recall_t4'):
			bad = [
				budget
				for budget in budgets
				if float(
					_summary_row(
						index,
						budget,
						comparison_id,
						f'class_{class_id}_{metric}',
					)['mean_delta']
				)
				<= -0.05
			]
			if len(bad) >= 2:
				degradations.append(
					{'class_id': class_id, 'metric': metric, 'budgets': bad}
				)
	if len(positive) >= 2 and not degradations:
		status = 'XYUNANIM_ORIGINAL_GO'
	elif len(negative) >= 2 or degradations:
		status = 'XYUNANIM_ORIGINAL_STOP'
	else:
		status = 'XYUNANIM_ORIGINAL_HOLD'
	diagnostic_id = 'mh_xyunanim1_nocons_vs_mh_xycons1_nocons'
	return {
		'artifact_type': 'f3_xy_neighbor_unanimous_original_gate',
		'schema_version': 1,
		'overall_status': status,
		'hard_vs_xy_neighbor_unanimous': {
			'comparison_id': comparison_id,
			'positive_budgets': positive,
			'negative_budgets': negative,
			'systematic_major_degradation': degradations,
		},
		'diagnostic_unanimous_vs_xy_neighbor_consensus': {
			'comparison_id': diagnostic_id,
			'gate_effect': 'none',
		},
		'gate': {
			'primary_comparison_id': comparison_id,
			'diagnostic_comparison_id': diagnostic_id,
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
			'ready': status == 'XYUNANIM_ORIGINAL_GO',
			'scientific_jobs_executed': 0,
			'six_split_jobs_executed': 0,
		},
	}


def summarize_f3_lithology_voxel_label_budget_xy_neighbor_unanimous(
	config: object, *, publish: bool = True
) -> Mapping[str, object]:
	"""Write only lightweight unanimous reports and publication evidence."""
	inspection = inspect_f3_lithology_voxel_label_budget_xy_neighbor_unanimous_results(
		config
	)
	execution = _execution_git_state(config)
	reports = Path(config.reports_dir)
	reports.mkdir(parents=True, exist_ok=True)
	published = _portable_payload(inspection, config=config)
	_write_audit_evidence(reports, audit=inspection['screening_audit'], config=config)
	_write_csv(
		reports / REPORT_OUTPUT_NAMES[0],
		published['job_metrics'],
		lineterminator='\n',
	)
	_write_csv(
		reports / REPORT_OUTPUT_NAMES[1],
		published['paired_deltas'],
		lineterminator='\n',
	)
	_write_json(reports / REPORT_OUTPUT_NAMES[2], published)
	status = inspection['decisions']['overall_status']
	ready = inspection['decisions']['six_split_follow_up']['ready']
	(reports / REPORT_OUTPUT_NAMES[3]).write_text(
		'# XY-neighbour unanimous original-split screening\n\n'
		f'Status: `{status}`\n\n'
		f'Six-split follow-up ready: `{ready}`\n\n'
		'The hard `mh_nocons` comparison is the preregistered primary gate. '
		'The 3-of-4 XY-consensus comparison is diagnostic only.\n',
		encoding='utf-8',
	)
	handoff = _handoff_payload(inspection, execution=execution)
	_write_json(
		reports / REPORT_OUTPUT_NAMES[4], _portable_payload(handoff, config=config)
	)
	published_files: tuple[Path, ...] = ()
	if publish:
		output = config.base.publish.results_root / _PUBLISHED_ROOT
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
		'decisions': inspection['decisions'],
		'published_files': published_files,
	}


def _summary_row(
	index: Mapping[tuple[str, str, str], Mapping[str, object]],
	budget: str,
	comparison_id: str,
	metric: str,
) -> Mapping[str, object]:
	try:
		return index[(budget, comparison_id, metric)]
	except KeyError as error:
		raise ValueError(
			f'XY-neighbour-unanimous primary gate is missing '
			f'{budget}/{comparison_id}/{metric}'
		) from error


def _members(
	rows: Sequence[Mapping[str, object]], *, reference: object
) -> dict[tuple[str, int, str], Mapping[str, object]]:
	members = {}
	for row in rows:
		role = str(row['model_role'])
		key = (str(row['budget_id']), int(row['subsample_seed']), role)
		if key in members:
			raise ValueError(f'duplicate XY-neighbour-unanimous member: {key!r}')
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
			raise ValueError(f'duplicate XY-neighbour-unanimous member: {key!r}')
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
		for role in (
			'mae',
			'm1_current_k6',
			'mh_nocons',
			_XY_CONSENSUS_ROLE,
			_CANDIDATE_ROLE,
		)
	}
	if set(members) != expected:
		raise ValueError(
			'XY-neighbour-unanimous comparison member matrix is incomplete or '
			'duplicated'
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
					_XY_CONSENSUS_ROLE,
					_CANDIDATE_ROLE,
				)
			]
			for key in control.PAIR_IDENTITY_KEYS:
				if any(row.get(key) != rows[0].get(key) for row in rows[1:]):
					raise ValueError(
						'XY-neighbour-unanimous paired identity mismatch: '
						f'{budget}/seed{seed}/{key}'
					)


def _load_xy_neighbor_consensus_reference_rows(
	audit: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
	"""Read the diagnostic 3-of-4 manifest without creating or mutating jobs."""
	references = _mapping(
		audit.get('reference_run_manifests'), 'screening audit reference manifests'
	)
	reference = _mapping(
		references.get('xy_neighbor_consensus'), '3-of-4 reference manifest'
	)
	_identity_matches_live(reference, label='3-of-4 reference manifest')
	payload = _read_json(Path(str(reference['path'])))
	if (
		payload.get('artifact_type')
		!= 'f3_lithology_voxel_label_budget_xy_neighbor_consensus'
		or payload.get('schema_version') != 1
	):
		raise ValueError('3-of-4 reference run manifest type/schema mismatch')
	rows = payload.get('rows')
	if not isinstance(rows, list) or len(rows) != 15:
		raise ValueError('3-of-4 reference run manifest must contain exactly 15 rows')
	if payload.get('row_count') != 15 or payload.get('complete_count') != 15:
		raise ValueError('3-of-4 reference run manifest is incomplete')
	result: list[Mapping[str, object]] = []
	for row in rows:
		if not isinstance(row, Mapping):
			raise TypeError('3-of-4 reference row must be a mapping')
		if (
			row.get('model_role') != _XY_CONSENSUS_ROLE
			or row.get('model_tag') != _XY_CONSENSUS_TAG
			or row.get('status') != 'complete'
		):
			raise ValueError('3-of-4 reference row identity/status mismatch')
		result.append(row)
	return tuple(result)


def _load_screening_audit(config: object) -> Mapping[str, object]:
	payload = unanimous_config.validate_f3_xy_neighbor_unanimous_screening_audit(
		config
	)
	if (
		payload.get('artifact_type')
		!= 'f3_xy_neighbor_unanimous_original_screening_preflight'
		or payload.get('schema_version') != 1
		or payload.get('status') != 'PASS'
	):
		raise ValueError(
			'XY-neighbour-unanimous screening audit type/schema/status mismatch'
		)
	if not isinstance(payload.get('hard_baseline_parity'), Mapping):
		raise TypeError('XY-neighbour-unanimous audit parity evidence is missing')
	if not isinstance(payload.get('xy_neighbor_unanimous_spatial_smoothness'), Mapping):
		raise TypeError('XY-neighbour-unanimous audit spatial evidence is missing')
	candidate = payload.get('candidate')
	if not isinstance(candidate, Mapping):
		raise TypeError('XY-neighbour-unanimous audit candidate binding is missing')
	if (
		candidate.get('model_id') != _CANDIDATE_ROLE
		or candidate.get('model_tag') != _CANDIDATE_TAG
	):
		raise ValueError('XY-neighbour-unanimous audit candidate identity mismatch')
	target_audit = _mapping(payload.get('target_audit'), 'screening target audit')
	if target_audit.get('status') != 'XYUNANIM_TARGET_GO':
		raise ValueError('XY-neighbour-unanimous target audit is not GO')
	return payload


def _source_reference_identities(
	config: object,
	*,
	audit: Mapping[str, object],
	hard_config: object,
) -> Mapping[str, object]:
	references = _mapping(
		audit.get('reference_run_manifests'), 'screening audit reference manifests'
	)
	expected = {
		'xy_neighbor_consensus': None,
		'hard_multi_head': runner.multi_head_run_manifest_path(hard_config),
		'current_k6': Path(config.current_k6_run_manifest),
		'mae': Path(config.original_run_manifest),
	}
	result = {}
	for name, expected_path in expected.items():
		value = _mapping(references.get(name), f'{name} reference identity')
		_identity_matches_live(value, label=f'{name} reference identity')
		if (
			expected_path is not None
			and Path(str(value['path'])).resolve() != expected_path.resolve()
		):
			raise ValueError(f'{name} reference path differs from screening audit')
		result[name] = _identity(Path(str(value['path'])))
	return result


def _candidate_provenance(
	config: object, *, candidate_identity: Mapping[str, object]
) -> Mapping[str, object]:
	candidate = config.candidates[0]
	if candidate.model_id != _CANDIDATE_ROLE or candidate.model_tag != _CANDIDATE_TAG:
		raise ValueError('XY-neighbour-unanimous candidate identity mismatch')
	for key in (
		'checkpoint',
		'embeddings',
		'valid_tokens',
		'metadata',
		'pretraining_handoff',
	):
		if not isinstance(candidate_identity.get(key), Mapping):
			raise TypeError(
				f'XY-neighbour-unanimous candidate {key} lineage is missing'
			)
	return {
		'model_id': candidate.model_id,
		'model_tag': candidate.model_tag,
		'pretraining_handoff': candidate_identity['pretraining_handoff'],
		'best_checkpoint': candidate_identity['checkpoint'],
		'embeddings': candidate_identity['embeddings'],
		'valid_tokens': candidate_identity['valid_tokens'],
		'embedding_metadata': candidate_identity['metadata'],
	}


def _target_audit_identity(audit: Mapping[str, object]) -> Mapping[str, object]:
	path, payload = _replayed_target_audit(audit)
	return {**_identity(path), 'status': payload['status']}


def _handoff_payload(
	inspection: Mapping[str, object],
	*,
	execution: Mapping[str, object],
) -> dict[str, object]:
	identities = _mapping(inspection['source_identities'], 'source identities')
	decisions = _mapping(inspection['decisions'], 'gate result')
	status = str(decisions.get('overall_status', ''))
	scientific_status = {
		'XYUNANIM_ORIGINAL_GO': 'GO',
		'XYUNANIM_ORIGINAL_HOLD': 'HOLD',
		'XYUNANIM_ORIGINAL_STOP': 'STOP',
	}.get(status)
	if scientific_status is None:
		raise ValueError('unanimous scientific gate status is invalid')
	return {
		'artifact_type': 'f3_xy_neighbor_unanimous_original_screening_handoff',
		'schema_version': 1,
		'pipeline_status': 'PASS',
		'scientific_gate_status': scientific_status,
		'status': 'PASS',
		'screening_audit': identities['screening_audit'],
		'target_audit': identities['target_audit'],
		'source_hard_manifest': inspection['screening_audit']['source_hard_manifest'],
		'xy_neighbor_consensus_target_manifest': inspection['screening_audit'][
			'xy_neighbor_consensus_target_manifest'
		],
		'xy_neighbor_unanimous_target_manifest': inspection['screening_audit'][
			'xy_neighbor_unanimous_target_manifest'
		],
		'candidate_run_manifest': identities['candidate_run_manifest'],
		'candidate_pretraining_handoff': identities['candidate_provenance'][
			'pretraining_handoff'
		],
		'candidate_best_checkpoint': identities['candidate_provenance'][
			'best_checkpoint'
		],
		'candidate_embeddings': {
			'embeddings': identities['candidate_provenance']['embeddings'],
			'valid_tokens': identities['candidate_provenance']['valid_tokens'],
			'metadata': identities['candidate_provenance']['embedding_metadata'],
		},
		'candidate_provenance': identities['candidate_provenance'],
		'hard_decoder_config': identities['hard_decoder_config'],
		'reference_run_manifests': identities['reference_run_manifests'],
		'paired_matrix_identity': identities['paired_matrix_identity'],
		'gate_definition': decisions['gate'],
		'gate_result': decisions,
		'execution': dict(execution),
		'six_split_follow_up': decisions['six_split_follow_up'],
	}


def _execution_git_state(config: object) -> Mapping[str, object]:
	workspace = Path(config.base.results_root).parent
	try:
		sha = subprocess.run(
			('git', 'rev-parse', 'HEAD'),
			cwd=workspace,
			check=True,
			capture_output=True,
			text=True,
		).stdout.strip()
		status = subprocess.run(
			('git', 'status', '--porcelain'),
			cwd=workspace,
			check=True,
			capture_output=True,
			text=True,
		).stdout
	except (OSError, subprocess.CalledProcessError) as error:
		raise RuntimeError(
			'unable to record XY-neighbour-unanimous execution git state'
		) from error
	if len(sha) != 40:
		raise ValueError('execution git SHA is invalid')
	return {'git_sha': sha, 'dirty': bool(status.strip())}


def _write_audit_evidence(reports: Path, *, audit: object, config: object) -> None:
	audit_mapping = _mapping(audit, 'unanimous screening audit')
	_path, target_payload = _replayed_target_audit(audit_mapping)
	_write_json(
		reports / AUDIT_OUTPUT_NAMES[0],
		_portable_payload(target_payload, config=config),
	)
	_write_csv(
		reports / AUDIT_OUTPUT_NAMES[1],
		_portable_payload(
			_spatial_audit_rows(audit_mapping), config=config
		),
		lineterminator='\n',
	)
	_write_json(
		reports / AUDIT_OUTPUT_NAMES[2],
		_portable_payload(
			{
				'artifact_type': 'f3_xy_neighbor_unanimous_hard_baseline_parity',
				'schema_version': 1,
				'screening_audit': _identity(Path(config.screening_audit)),
				'evidence': audit_mapping['hard_baseline_parity'],
			},
			config=config,
		),
	)
	_write_json(
		reports / AUDIT_OUTPUT_NAMES[3],
		_portable_payload(audit_mapping, config=config),
	)


def _replayed_target_audit(
	audit: Mapping[str, object],
) -> tuple[Path, Mapping[str, object]]:
	"""Replay the immutable target-only decision recorded by screening."""
	value = _mapping(audit.get('target_audit'), 'screening target audit')
	_identity_matches_live(value, label='screening target audit')
	path = Path(str(value['path'])).resolve()
	artifact_root_value = audit.get('artifact_root')
	if not isinstance(artifact_root_value, str) or not artifact_root_value:
		raise TypeError('screening audit artifact_root is missing')
	artifact_root = Path(artifact_root_value).resolve()
	if not artifact_root.is_dir():
		raise FileNotFoundError('screening audit artifact_root is missing')
	payload = replay_f3_xy_neighbor_unanimous_target_audit(
		path, artifact_root=artifact_root
	)
	if payload.get('status') != 'XYUNANIM_TARGET_GO':
		raise ValueError('target audit status changed after screening preflight')
	return path, payload


def _spatial_audit_rows(
	audit: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
	adapted = dict(audit)
	adapted['xy_spatial_smoothness'] = audit.get(
		'xy_neighbor_unanimous_spatial_smoothness'
	)
	return consensus_results._spatial_audit_rows(adapted)


def _identity(path: Path) -> Mapping[str, object]:
	if not path.is_file():
		raise FileNotFoundError(path)
	return {
		'path': str(path),
		'sha256': file_sha256(path),
		'byte_size': path.stat().st_size,
	}


def _identity_matches_live(value: Mapping[str, object], *, label: str) -> None:
	path, sha = value.get('path'), value.get('sha256')
	if not isinstance(path, str) or not isinstance(sha, str):
		raise TypeError(f'{label} identity is incomplete')
	actual = Path(path)
	if not actual.is_file() or file_sha256(actual) != sha:
		raise ValueError(f'{label} SHA-256 mismatch')


def _mapping(value: object, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _read_json(path: Path) -> Mapping[str, object]:
	try:
		payload = json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as error:
		raise ValueError(f'JSON object required: {path}') from error
	if not isinstance(payload, Mapping):
		raise TypeError(f'JSON object required: {path}')
	return payload


# Keep the shared formatting and portable-path behavior identical to the
# established consensus publisher without widening its public contract.
_write_csv = consensus_results._write_csv
_write_json = consensus_results._write_json
_portable_payload = consensus_results._portable_payload
__all__ = [
	'AUDIT_OUTPUT_NAMES',
	'COMPARISONS',
	'OUTPUT_NAMES',
	'PUBLISHED_OUTPUT_NAMES',
	'REPORT_OUTPUT_NAMES',
	'decide_xy_neighbor_unanimous_original_gate',
	'inspect_f3_lithology_voxel_label_budget_xy_neighbor_unanimous_results',
	'summarize_f3_lithology_voxel_label_budget_xy_neighbor_unanimous',
]

"""Aggregate the XY-neighbour consensus original-split screening result."""
# ruff: noqa: SLF001

from __future__ import annotations

import csv
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_multi_head import (
	f3_lithology_voxel_label_budget_multi_head_config_from_mapping,
)
from seis_ssl_cluster.config.f3_lithology_voxel_label_budget_xy_neighbor_consensus import (  # noqa: E501
	validate_f3_xy_neighbor_consensus_screening_audit,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology import voxel_label_budget_control as control
from seis_ssl_cluster.f3.lithology import voxel_label_budget_multi_head as runner
from seis_ssl_cluster.f3.lithology import (
	voxel_label_budget_multi_head_results as shared,
)
from seis_ssl_cluster.f3.lithology import (
	voxel_label_budget_xy_neighbor_consensus as xy_runner,
)
from seis_ssl_cluster.f3.lithology.voxel_label_budget_results import (
	load_f3_lithology_voxel_label_budget_evaluation_metrics,
)
from seis_ssl_cluster.results import (
	PublishItem,
	publish_manifest_to_dict,
	publish_selected_results,
)

COMPARISONS = (
	('mh_xycons1_nocons', 'mh_nocons'),
	('mh_xycons1_nocons', 'm1_current_k6'),
	('mh_xycons1_nocons', 'mae'),
)
REPORT_OUTPUT_NAMES = (
	'xy_neighbor_consensus_original_job_metrics.csv',
	'xy_neighbor_consensus_original_paired_deltas.csv',
	'xy_neighbor_consensus_original_results_summary.json',
	'xy_neighbor_consensus_original_results_summary.md',
	'xy_neighbor_consensus_original_handoff.json',
)
AUDIT_OUTPUT_NAMES = (
	'xy_neighbor_consensus_screening_audit.json',
	'xy_neighbor_consensus_spatial_audit.csv',
	'xy_neighbor_consensus_hard_baseline_parity.json',
)
# The candidate-owned report contract is exactly these five names. Audit
# evidence is copied in addition to this contract at publication time.
OUTPUT_NAMES = REPORT_OUTPUT_NAMES
PUBLISHED_OUTPUT_NAMES = AUDIT_OUTPUT_NAMES + OUTPUT_NAMES
_ARTIFACT_ROOT_PLACEHOLDER = '${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}'
_CANDIDATE_ROLE = 'mh_xycons1_nocons'
_CANDIDATE_TAG = 'strat_hmm_pretext_mh_k6810_xycons1_nocons_topblock1_distill_v1'
_PUBLISHED_ROOT = Path(
	'f3/facies_benchmark_v1/'
	'strat_hmm_multi_head_k6810_xy_neighbor_consensus_original_split_v1'
)


def inspect_f3_lithology_voxel_label_budget_xy_neighbor_consensus_results(
	config: object,
) -> Mapping[str, object]:
	"""Live-revalidate the exact four-role paired original-split matrix."""
	audit = _load_screening_audit(config)
	candidate_inspection = (
		xy_runner.inspect_f3_lithology_voxel_label_budget_xy_neighbor_consensus(config)
	)
	candidate_rows = (
		xy_runner.load_f3_lithology_voxel_label_budget_xy_neighbor_consensus_rows(
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
	members = _members(
		(*candidate_rows, *current_rows, *hard_rows), reference=reference
	)
	_expected_matrix(config, members)
	_validate_pairing(config, members)
	deltas = shared._paired_deltas(config, members, comparisons=COMPARISONS)
	summary = shared._summary(config, deltas, comparisons=COMPARISONS)
	decisions = decide_xy_neighbor_consensus_original_gate(
		summary, budgets=config.budgets
	)
	candidate_identity = candidate_inspection.candidate_identities.get(_CANDIDATE_ROLE)
	if not isinstance(candidate_identity, Mapping):
		raise TypeError('XY-consensus candidate lineage is missing')
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
				runner.multi_head_run_manifest_path(config)
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
				'roles': ('mae', 'm1_current_k6', 'mh_nocons', _CANDIDATE_ROLE),
				'budgets': tuple(config.budgets),
				'subsample_seeds': tuple(config.subsample_seeds),
				'row_count': 4 * len(config.budgets) * len(config.subsample_seeds),
				'pair_identity_keys': tuple(control.PAIR_IDENTITY_KEYS),
			},
		},
	}


def decide_xy_neighbor_consensus_original_gate(
	summary: Sequence[Mapping[str, object]], *, budgets: Sequence[str]
) -> dict[str, object]:
	"""Apply the preregistered original-split XY-consensus GO/HOLD/STOP gate."""
	comparison_id = 'mh_xycons1_nocons_vs_mh_nocons'
	index = {
		(str(row['budget_id']), str(row['metric'])): row
		for row in summary
		if row['comparison_id'] == comparison_id
	}
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
		status = 'XYCONS_ORIGINAL_GO'
	elif len(negative) >= 2 or degradations:
		status = 'XYCONS_ORIGINAL_STOP'
	else:
		status = 'XYCONS_ORIGINAL_HOLD'
	return {
		'artifact_type': 'f3_xy_neighbor_consensus_original_gate',
		'schema_version': 1,
		'overall_status': status,
		'hard_vs_xy_neighbor_consensus': {
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
			'ready': status == 'XYCONS_ORIGINAL_GO',
			'scientific_jobs_executed': 0,
		},
	}


def summarize_f3_lithology_voxel_label_budget_xy_neighbor_consensus(
	config: object, *, publish: bool = True
) -> Mapping[str, object]:
	"""Write only lightweight XY-consensus reports and publication evidence."""
	inspection = inspect_f3_lithology_voxel_label_budget_xy_neighbor_consensus_results(
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
		'# XY-neighbour consensus original-split screening\n\n'
		f'Status: `{status}`\n\n'
		f'Six-split follow-up ready: `{ready}`\n\n'
		'This is paired original-split facies-label evidence only; it does not '
		'claim general downstream superiority.\n',
		encoding='utf-8',
	)
	handoff = _handoff_payload(inspection, execution=execution)
	_write_json(
		reports / REPORT_OUTPUT_NAMES[4], _portable_payload(handoff, config=config)
	)
	manifest = None
	if publish:
		manifest = publish_selected_results(
			items=[
				PublishItem(reports / name, Path(name))
				for name in PUBLISHED_OUTPUT_NAMES
			],
			output_dir=config.base.publish.results_root / _PUBLISHED_ROOT,
			max_file_size_bytes=10 * 1024 * 1024,
		)
		_write_portable_publish_manifest(manifest, config=config)
	return {
		'summary_json': reports / REPORT_OUTPUT_NAMES[2],
		'decisions': inspection['decisions'],
		'publish_manifest': manifest,
	}


def _summary_row(
	index: Mapping[tuple[str, str], Mapping[str, object]], budget: str, metric: str
) -> Mapping[str, object]:
	try:
		return index[(budget, metric)]
	except KeyError as error:
		raise ValueError(
			f'XY-consensus primary gate is missing {budget}/{metric}'
		) from error


def _members(
	rows: Sequence[Mapping[str, object]], *, reference: object
) -> dict[tuple[str, int, str], Mapping[str, object]]:
	members = {}
	for row in rows:
		role = str(row['model_role'])
		key = (str(row['budget_id']), int(row['subsample_seed']), role)
		if key in members:
			raise ValueError(f'duplicate XY-consensus member: {key!r}')
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
			raise ValueError(f'duplicate XY-consensus member: {key!r}')
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
			'XY-consensus comparison member matrix is incomplete or duplicated'
		)


def _validate_pairing(
	config: object, members: Mapping[tuple[str, int, str], Mapping[str, object]]
) -> None:
	for budget in config.budgets:
		for seed in config.subsample_seeds:
			rows = [
				members[(budget, seed, role)]['row']
				for role in ('mae', 'm1_current_k6', 'mh_nocons', _CANDIDATE_ROLE)
			]
			for key in control.PAIR_IDENTITY_KEYS:
				if any(row.get(key) != rows[0].get(key) for row in rows[1:]):
					raise ValueError(
						'XY-consensus paired identity mismatch: '
						f'{budget}/seed{seed}/{key}'
					)


def _load_screening_audit(config: object) -> Mapping[str, object]:
	payload = validate_f3_xy_neighbor_consensus_screening_audit(config)
	if (
		payload.get('artifact_type')
		!= 'f3_xy_neighbor_consensus_original_screening_preflight'
		or payload.get('schema_version') != 1
		or payload.get('status') != 'PASS'
	):
		raise ValueError('XY-consensus screening audit type/schema/status mismatch')
	if not isinstance(payload.get('hard_baseline_parity'), Mapping):
		raise TypeError('XY-consensus screening audit parity evidence is missing')
	if not isinstance(payload.get('xy_spatial_smoothness'), Mapping):
		raise TypeError('XY-consensus screening audit spatial evidence is missing')
	candidate = payload.get('candidate')
	if not isinstance(candidate, Mapping):
		raise TypeError('XY-consensus screening audit candidate binding is missing')
	if (
		candidate.get('model_id') != _CANDIDATE_ROLE
		or candidate.get('model_tag') != _CANDIDATE_TAG
	):
		raise ValueError('XY-consensus screening audit candidate identity mismatch')
	return payload


def _candidate_provenance(
	config: object, *, candidate_identity: Mapping[str, object]
) -> Mapping[str, object]:
	candidate = config.candidates[0]
	if candidate.model_id != _CANDIDATE_ROLE or candidate.model_tag != _CANDIDATE_TAG:
		raise ValueError('XY-consensus candidate identity mismatch')
	for key in (
		'checkpoint',
		'embeddings',
		'valid_tokens',
		'metadata',
		'pretraining_handoff',
	):
		if not isinstance(candidate_identity.get(key), Mapping):
			raise TypeError(f'XY-consensus candidate {key} lineage is missing')
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
	inspection: Mapping[str, object],
	*,
	execution: Mapping[str, object],
) -> dict[str, object]:
	identities = inspection['source_identities']
	if not isinstance(identities, Mapping):
		raise TypeError('XY-consensus source identities are missing')
	decisions = inspection['decisions']
	if not isinstance(decisions, Mapping):
		raise TypeError('XY-consensus gate result is missing')
	return {
		'artifact_type': 'f3_xy_neighbor_consensus_original_screening_handoff',
		'schema_version': 1,
		'status': 'PASS',
		'screening_audit': identities['screening_audit'],
		'candidate_run_manifest': identities['candidate_run_manifest'],
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
			'unable to record XY-consensus execution git state'
		) from error
	if len(sha) != 40:
		raise ValueError('execution git SHA is invalid')
	return {'git_sha': sha, 'dirty': bool(status.strip())}


def _write_audit_evidence(reports: Path, *, audit: object, config: object) -> None:
	if not isinstance(audit, Mapping):
		raise TypeError('XY-consensus audit must be a mapping')
	_write_json(
		reports / AUDIT_OUTPUT_NAMES[0], _portable_payload(audit, config=config)
	)
	_write_csv(
		reports / AUDIT_OUTPUT_NAMES[1],
		_portable_payload(_spatial_audit_rows(audit), config=config),
		lineterminator='\n',
	)
	_write_json(
		reports / AUDIT_OUTPUT_NAMES[2],
		_portable_payload(
			{
				'artifact_type': 'f3_xy_neighbor_consensus_hard_baseline_parity',
				'schema_version': 1,
				'screening_audit': _identity(Path(config.screening_audit)),
				'evidence': audit['hard_baseline_parity'],
			},
			config=config,
		),
	)


def _spatial_audit_rows(
	audit: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
	spatial = audit['xy_spatial_smoothness']
	if not isinstance(spatial, Mapping):
		raise TypeError('XY-consensus spatial audit must be a mapping')
	per_k = spatial.get('per_k')
	if not isinstance(per_k, Mapping) or set(per_k) != {'6', '8', '10'}:
		raise ValueError('XY-consensus spatial audit must contain K=6/8/10')
	rows = []
	for head_k in ('6', '8', '10'):
		evidence = per_k[head_k]
		if not isinstance(evidence, Mapping):
			raise TypeError(f'XY-consensus spatial audit K={head_k} is invalid')
		for edge_name, axis in (
			('x_edges', 'x'),
			('y_edges', 'y'),
			('combined', 'combined'),
		):
			edge = evidence.get(edge_name)
			if not isinstance(edge, Mapping):
				raise TypeError(
					f'XY-consensus spatial audit K={head_k}/{edge_name} is invalid'
				)
			rows.append(
				{
					'head_k': int(head_k),
					'axis': axis,
					'valid_token_count': evidence.get('valid_token_count'),
					'changed_token_count': evidence.get('changed_token_count'),
					'changed_token_fraction': evidence.get('changed_fraction'),
					'source_state_occupancy_json': json.dumps(
						evidence.get('source_state_occupancy'), separators=(',', ':')
					),
					'output_state_occupancy_json': json.dumps(
						evidence.get('output_state_occupancy'), separators=(',', ':')
					),
					'empty_output_state_count': evidence.get(
						'empty_output_state_count'
					),
					'source_temporal_transition_count': evidence.get(
						'source_temporal_transition_count'
					),
					'output_temporal_transition_count': evidence.get(
						'output_temporal_transition_count'
					),
					'ordered_path_violations_source': _ordered_violations(
						evidence, 'source'
					),
					'ordered_path_violations_output': _ordered_violations(
						evidence, 'output'
					),
					**dict(edge),
				}
			)
	return tuple(rows)


def _ordered_violations(evidence: Mapping[str, object], side: str) -> object:
	violations = evidence.get('ordered_path_violations')
	if not isinstance(violations, Mapping):
		raise TypeError('XY-consensus spatial audit ordered-path evidence is invalid')
	return violations.get(side)


def _identity(path: Path) -> Mapping[str, object]:
	if not path.is_file():
		raise FileNotFoundError(path)
	return {
		'path': str(path),
		'sha256': file_sha256(path),
		'byte_size': path.stat().st_size,
	}


def _write_csv(
	path: Path,
	rows: Sequence[Mapping[str, object]],
	*,
	lineterminator: str = '\r\n',
) -> None:
	if not rows:
		raise ValueError(f'no rows to write: {path.name}')
	with path.open('w', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(
			handle,
			fieldnames=list(rows[0]),
			lineterminator=lineterminator,
		)
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
		workspace_root=Path(config.base.results_root).parent,
	)


def _portable_value(
	value: object, *, artifact_root: Path, workspace_root: Path
) -> object:
	if isinstance(value, Mapping):
		return {
			key: _portable_value(
				item,
				artifact_root=artifact_root,
				workspace_root=workspace_root,
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
	return _portable_path(
		value, artifact_root=artifact_root, workspace_root=workspace_root
	)


def _portable_path(value: str, *, artifact_root: Path, workspace_root: Path) -> str:
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


def _write_portable_publish_manifest(manifest: object, *, config: object) -> None:
	payload = _portable_payload(publish_manifest_to_dict(manifest), config=config)
	if not isinstance(payload, Mapping):
		raise TypeError('portable publish manifest must be a mapping')
	manifest.manifest_path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
	)


__all__ = [
	'AUDIT_OUTPUT_NAMES',
	'COMPARISONS',
	'OUTPUT_NAMES',
	'PUBLISHED_OUTPUT_NAMES',
	'REPORT_OUTPUT_NAMES',
	'decide_xy_neighbor_consensus_original_gate',
	'inspect_f3_lithology_voxel_label_budget_xy_neighbor_consensus_results',
	'summarize_f3_lithology_voxel_label_budget_xy_neighbor_consensus',
]

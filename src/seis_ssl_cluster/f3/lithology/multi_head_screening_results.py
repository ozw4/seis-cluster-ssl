"""Paired screening of completed Multi-Head sources against canonical HMM."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	DATA_SIZES,
	LAYOUT_IDS,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.candidate_benchmark import (
	audit_f3_lithology_candidate_source,
	f3_lithology_candidate_config_from_mapping,
	load_f3_lithology_candidate_canonical_config,
)
from seis_ssl_cluster.f3.lithology.hmm_final_source_audit import (
	audit_f3_hmm_final_source,
)
from seis_ssl_cluster.f3.lithology.paired_candidate_results import (
	COMPARISON_FIELDNAMES,
	F3PairedCandidateModel,
	F3PairedCandidateSummaryConfig,
	_absolute_path,
	_aggregate_layout_clustered,
	_aggregate_scope,
	_audit_job_rows,
	_benchmark_provenance,
	_comparison_csv,
	_ensure_summary_absent,
	_nonempty_string,
	_paths_overlap,
	_require_exact_keys,
	_shared_evaluation_inputs,
	_shared_validation_identity,
	_validate_source_pair,
	_write_atomic_summary,
)
from seis_ssl_cluster.training.strat_hmm_source_audit import audit_multi_head_source

if TYPE_CHECKING:
	from collections.abc import Mapping


def screening_config_from_mapping(raw: Mapping[str, object]) -> dict[str, object]:
	"""Resolve recipe references without reading live artifacts."""
	_require_exact_keys(
		raw,
		{'candidates', 'control_id', 'control_training_config', 'summary_root'},
		'config',
	)
	entries = raw['candidates']
	if not isinstance(entries, list) or not entries:
		raise ValueError('candidates must be a nonempty list')
	candidates = []
	for entry in entries:
		_require_exact_keys(
			entry, {'downstream_config', 'training_config', 'head_ks'}, 'candidate'
		)
		ks = entry['head_ks']
		if (
			not isinstance(ks, list)
			or len(ks) < 2
			or any(type(k) is not int or k < 2 for k in ks)
			or ks != sorted(set(ks))
		):
			raise ValueError(
				'head_ks must contain at least two ascending unique K values'
			)
		candidates.append(
			{
				**{
					key: _absolute_path(entry[key], key)
					for key in ('downstream_config', 'training_config')
				},
				'head_ks': ks,
			}
		)
	for key in ('downstream_config', 'training_config'):
		if len({entry[key] for entry in candidates}) != len(candidates):
			raise ValueError(f'duplicate candidate {key}')
	return {
		'candidates': candidates,
		'control_id': _nonempty_string(raw['control_id'], 'control_id'),
		'control_training_config': _absolute_path(
			raw['control_training_config'], 'control_training_config'
		),
		'summary_root': _absolute_path(raw['summary_root'], 'summary_root'),
	}


def inspect_multi_head_screening(config: Mapping[str, object]) -> dict[str, object]:
	"""Audit every source and paired cell; return no partial screening result."""
	control_audit = audit_f3_hmm_final_source(config['control_training_config'])
	rows = []
	results = []
	shared = None
	ids = set()
	for entry in config['candidates']:
		candidate = f3_lithology_candidate_config_from_mapping(
			load_config(entry['downstream_config'])
		)
		canonical = load_f3_lithology_candidate_canonical_config(candidate)
		control = canonical.model_by_id(config['control_id'])
		for root in (
			candidate.runs_root,
			canonical.runs_root,
			canonical.summary_root,
			candidate.embeddings_dir,
			control.embeddings_dir,
		):
			if _paths_overlap(config['summary_root'], root):
				raise ValueError('screening summary overlaps an input/output namespace')
		if candidate.candidate_id in ids:
			raise ValueError('duplicate candidate ID')
		ids.add(candidate.candidate_id)
		lineage = audit_multi_head_source(entry['training_config'])
		if (
			lineage['head_ks'] != entry['head_ks']
			or lineage['model_tag'] != candidate.candidate_id
		):
			raise ValueError('candidate head set or model ID differs from source audit')
		control_config = replace(
			candidate,
			candidate_id=control.model_id,
			checkpoint=control.checkpoint,
			embeddings_dir=control.embeddings_dir,
		)
		sources = {
			'candidate': audit_f3_lithology_candidate_source(candidate, canonical),
			'control': audit_f3_lithology_candidate_source(control_config, canonical),
		}
		for role, audit in (('candidate', lineage), ('control', control_audit)):
			if audit['checkpoint'] != {
				'path': sources[role]['checkpoint_path'],
				'sha256': sources[role]['checkpoint_sha256'],
			}:
				raise ValueError(f'{role} source audit and embedding checkpoint differ')
			sources[role]['semantic_lineage'] = audit
		_validate_source_pair(sources)
		pair = F3PairedCandidateSummaryConfig(
			canonical_config=candidate.canonical_config,
			runs_root=candidate.runs_root,
			summary_root=config['summary_root'],
			candidate=F3PairedCandidateModel(
				candidate.candidate_id, candidate.checkpoint, candidate.embeddings_dir
			),
			control=F3PairedCandidateModel(
				control.model_id, control.checkpoint, control.embeddings_dir
			),
		)
		comparison = _audit_job_rows(
			pair, canonical, sources, control_runs_root=canonical.runs_root
		)
		_add_primary_metric(comparison)
		identity = {
			'benchmark': _benchmark_provenance(pair),
			'control': sources['control'],
			'source_embedding': lineage['source_embedding'],
			'validation': _shared_validation_identity(comparison),
			'evaluation_inputs': _shared_evaluation_inputs(comparison),
			'control_cells': [
				{
					key: value
					for key, value in row.items()
					if key.startswith('control_')
					or key
					in (
						'layout_id',
						'data_size',
						'supervision_identity_sha256',
						'decoder_contract_identity_sha256',
					)
				}
				for row in comparison
			],
		}
		if shared is not None and shared != identity:
			raise ValueError(
				'candidates do not share benchmark, baseline cells, or source identity'
			)
		shared = identity
		results.append(
			{
				'candidate_id': candidate.candidate_id,
				'head_ks': entry['head_ks'],
				'sources': sources,
				'by_size': {
					size: _aggregate_scope(
						[row for row in comparison if row['data_size'] == size],
						expected_count=len(LAYOUT_IDS),
						label=size,
						descriptive_unit='layout_id',
						paired_t_unit='layout_id',
						metric='mean_iou',
					)
					for size in DATA_SIZES
				},
				'secondary_macro_f1_by_size': {
					size: _aggregate_scope(
						[
							{
								**row,
								'candidate_minus_control': row[
									'macro_f1_candidate_minus_control'
								],
							}
							for row in comparison
							if row['data_size'] == size
						],
						expected_count=len(LAYOUT_IDS),
						label=size,
						descriptive_unit='layout_id',
						paired_t_unit='layout_id',
					)
					for size in DATA_SIZES
				},
				'all_15_descriptive': _aggregate_scope(
					comparison,
					expected_count=len(LAYOUT_IDS) * len(DATA_SIZES),
					label='all_15',
					descriptive_unit='layout_id_x_data_size_cell',
					paired_t_unit=None,
					metric='mean_iou',
				),
				'overall_layout_clustered': _aggregate_layout_clustered(comparison),
			}
		)
		rows.extend(comparison)
	return {
		'schema_version': 1,
		'status': 'complete',
		'primary_metric': 'mean_iou',
		'secondary_metric': 'macro_f1',
		'aggregation_unit': 'unique_validation_voxel',
		'delta_convention': 'candidate minus control; positive favors candidate',
		'interpretation': (
			'Exploratory F3 screening; no multiplicity-adjusted significance '
			'or automatic promotion.'
		),
		'control_id': config['control_id'],
		'shared_identity': shared,
		'candidate_cells': len(rows),
		'control_cells': len(LAYOUT_IDS) * len(DATA_SIZES),
		'candidates': results,
		'comparison': rows,
	}


def summarize_multi_head_screening(
	config: Mapping[str, object],
) -> tuple[Path, Path, Path]:
	"""Publish exactly three files after all candidates pass the live audit."""
	_ensure_summary_absent(config['summary_root'])
	payload = inspect_multi_head_screening(config)
	lines = [
		'# F3 Multi-Head paired screening',
		'',
		(
			f'Control: `{payload["control_id"]}`. '
			'Primary: mean IoU; secondary: macro F1; '
			'delta = candidate minus control.'
		),
		'',
		(
			'Each size uses five paired layouts. Overall statistics reduce the three '
			'sizes within each layout (n=5). The 15 cells are descriptive, '
			'not independent replicates.'
		),
		'',
		str(payload['interpretation']),
		'',
		'| Candidate | Size | Mean delta | SD | Wins / ties / losses |',
		'| --- | --- | ---: | ---: | --- |',
	]
	for candidate in payload['candidates']:
		for size, aggregate in candidate['by_size'].items():
			delta = aggregate['candidate_minus_control']
			lines.append(
				f'| {candidate["candidate_id"]} | {size} | {delta["mean"]:.6f} | '
				f'{delta["sample_standard_deviation"]:.6f} | {delta["wins"]} / '
				f'{delta["ties"]} / {delta["losses"]} |'
			)
	return _write_atomic_summary(
		config['summary_root'],
		{
			'comparison.csv': _comparison_csv(
				payload['comparison'],
				fieldnames=(
					*COMPARISON_FIELDNAMES,
					'candidate_mean_iou',
					'control_mean_iou',
					'macro_f1_candidate_minus_control',
				),
			),
			'summary.json': json.dumps(
				payload, indent=2, sort_keys=True, allow_nan=False
			)
			+ '\n',
			'summary.md': '\n'.join(lines) + '\n',
		},
	)


def _add_primary_metric(rows: list[dict[str, object]]) -> None:
	for row in rows:
		for role in ('candidate', 'control'):
			path = Path(row[f'{role}_metrics_path'])
			before = file_sha256(path)
			metrics = json.loads(path.read_text())
			value = metrics.get('mean_iou')
			if (
				type(value) not in (int, float)
				or not math.isfinite(value)
				or not 0 <= value <= 1
			):
				raise ValueError(f'{path}: mean_iou must be finite and in [0, 1]')
			if before != row[f'{role}_metrics_sha256'] or file_sha256(path) != before:
				raise ValueError(f'{path}: metrics changed after paired evidence audit')
			row[f'{role}_mean_iou'] = float(value)
		row['macro_f1_candidate_minus_control'] = row['candidate_minus_control']
		row['candidate_minus_control'] = (
			row['candidate_mean_iou'] - row['control_mean_iou']
		)

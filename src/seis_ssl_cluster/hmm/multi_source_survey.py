"""Audit complete K6810 arms and pair them with frozen source-family controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.f3.lithology.paired_candidate_results import (
	validate_f3_paired_completed_jobs,
)
from seis_ssl_cluster.hmm.multi_source_aggregate import (
	arm_statistics,
	comparison_row,
	publish_summary,
)
from seis_ssl_cluster.hmm.multi_source_cell import audit_completed_cell
from seis_ssl_cluster.hmm.multi_source_definition import (
	validate_multi_source_experiment_definition,
)
from seis_ssl_cluster.hmm.multi_source_receipts import (
	CANDIDATE_IDS,
	CELL_IDENTITIES,
	DATA_SIZES,
	METRICS,
	SOURCE_FAMILIES,
	SURVEYS,
	artifact_path,
	canonical_cells_sha256,
	control_receipt_for,
	file_sha256,
	load_control_receipts,
	load_matrix,
	load_reuse_receipt,
	object_sha256,
	read_json,
	validate_fixed_training_config,
	verify_control,
	verify_reuse_receipt,
)
from seis_ssl_cluster.training.strat_hmm_source_audit import audit_multi_head_source
from seis_ssl_cluster.volve.horizon_five_way_config import (
	volve_horizon_five_way_config_from_mapping,
)
from seis_ssl_cluster.volve.horizon_five_way_results import (
	inspect_volve_horizon_five_way_cell,
)


def _paths(raw: dict[str, Any], survey: str, model_id: str) -> tuple[Path, Path]:
	if survey == 'f3':
		if raw['candidate']['id'] != model_id:
			raise ValueError('downstream candidate identity drift')
		checkpoint = raw['candidate']['checkpoint']
	elif survey == 'parihaka':
		checkpoint = raw['embeddings']['models'][model_id]['checkpoint']
	elif 'arm' in raw:
		if raw['arm']['arm_id'] != model_id:
			raise ValueError('downstream arm identity drift')
		checkpoint = raw['arm']['checkpoint']
	else:
		checkpoint = raw['models'][model_id]['checkpoint']
	return Path(raw['outputs']['runs_root']), Path(checkpoint)


def _check_directories(root: Path, model_id: str) -> None:
	model_root = root / f'model={model_id}'
	if not model_root.is_dir():
		raise FileNotFoundError(f'missing downstream arm: {model_root}')
	layouts = {
		p.name
		for p in model_root.iterdir()
		if p.is_dir() and p.name.startswith('layout=')
	}
	if layouts != {f'layout=layout_{i:03d}' for i in range(5)}:
		raise ValueError('unexpected or missing downstream layout directories')
	for layout in layouts:
		sizes = {
			p.name
			for p in (model_root / layout).iterdir()
			if p.is_dir() and p.name.startswith('size=')
		}
		if sizes != {f'size={size}' for size in DATA_SIZES}:
			raise ValueError('unexpected or missing downstream size directories')


def _audit_cell(
	survey: str, raw: dict[str, Any], model_id: str, layout: str, size: str
) -> dict[str, Any]:
	if survey == 'volve' and 'arm' not in raw:
		return inspect_volve_horizon_five_way_cell(
			volve_horizon_five_way_config_from_mapping(raw),
			model_id=model_id,
			layout_id=layout,
			data_size=size,
		)
	return audit_completed_cell(survey, raw, model_id, layout, size)


def _paired_identity(
	survey: str, candidate: dict[str, Any], control: dict[str, Any]
) -> None:
	if survey == 'f3':
		validate_f3_paired_completed_jobs(candidate, control, label='K6810/matching K6')
	elif survey == 'parihaka':

		def shared(evidence: dict[str, Any]) -> dict[str, Any]:
			identity = evidence['benchmark_identity']
			return {
				k: v for k, v in identity.items() if k not in ('model', 'embedding')
			}

		if shared(candidate) != shared(control):
			raise ValueError(
				'paired Channel decoder, supervision, or test identity drift'
			)
	elif any(
		candidate[k] != control[k]
		for k in (
			'shared_run_identity',
			'support_identity',
			'split_plan_sha256',
			'decoder_initial_state_sha256',
			'valid_tokens_sha256',
		)
	):
		raise ValueError('paired horizon decoder or support identity drift')


def _metrics(path: Path, survey: str) -> tuple[float, float, str]:
	before = file_sha256(path)
	payload = read_json(path)
	if survey == 'f3':
		primary, secondary = payload['mean_iou'], payload['macro_f1']
	elif survey == 'parihaka':
		primary, secondary = (
			payload['test']['channel_iou'],
			payload['test']['channel_f1'],
		)
	else:
		primary = payload['test']['primary_common']['macro_mae_samples']
		secondary = payload['test']['primary_common']['macro_within_2_samples']
	if file_sha256(path) != before:
		raise ValueError('metrics changed during audit')
	return primary, secondary, before


def _audited_arm(
	config: dict[str, Any], family: str, receipt: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
	survey = config['survey']
	entry = config['arms'][family]
	for field in ('training_config', 'downstream_config', 'control_downstream_config'):
		if 'reports' in Path(entry[field]).resolve().parts:
			raise ValueError('reports cannot be pipeline inputs')
	candidate_id = CANDIDATE_IDS[family]
	if entry['candidate_id'] != candidate_id:
		raise ValueError('summary candidate identity drift')
	candidate = load_config(entry['downstream_config'])
	control = load_config(entry['control_downstream_config'])
	candidate_root, candidate_checkpoint = _paths(candidate, survey, candidate_id)
	control_root, control_checkpoint = _paths(control, survey, receipt['checkpoint_id'])
	if control_checkpoint.resolve() != artifact_path(
		receipt['checkpoint_path'], Path(config['artifact_root'])
	):
		raise ValueError('control checkpoint path identity drift')
	validate_fixed_training_config(
		yaml.safe_load(Path(entry['training_config']).read_text())
	)
	lineage = audit_multi_head_source(Path(entry['training_config']))
	if (
		lineage['model_tag'] != candidate_id
		or lineage['head_ks'] != [6, 8, 10]
		or lineage['epoch'] != 25
		or lineage['checkpoint']
		!= {
			'path': str(candidate_checkpoint),
			'sha256': file_sha256(candidate_checkpoint),
		}
	):
		raise ValueError('K6810 source lineage drift')
	for root, model in (
		(candidate_root, candidate_id),
		(control_root, receipt['checkpoint_id']),
	):
		_check_directories(root, model)
	cells, control_cells, audits = [], [], []
	for layout, size in sorted(CELL_IDENTITIES):
		candidate_dir = (
			candidate_root
			/ f'model={candidate_id}'
			/ f'layout={layout}'
			/ f'size={size}'
		)
		control_dir = (
			control_root
			/ f'model={receipt["checkpoint_id"]}'
			/ f'layout={layout}'
			/ f'size={size}'
		)
		name = 'evaluation/metrics.json' if survey == 'f3' else 'metrics.json'
		candidate_metrics, control_metrics = candidate_dir / name, control_dir / name
		before = (file_sha256(candidate_metrics), file_sha256(control_metrics))
		candidate_audit = _audit_cell(survey, candidate, candidate_id, layout, size)
		control_audit = _audit_cell(
			survey, control, receipt['checkpoint_id'], layout, size
		)
		_paired_identity(survey, candidate_audit, control_audit)
		primary, secondary, digest = _metrics(candidate_metrics, survey)
		control_primary, control_secondary, control_digest = _metrics(
			control_metrics, survey
		)
		if before != (digest, control_digest):
			raise ValueError('metrics changed after completion audit')
		cells.append(
			{
				'source_family': family,
				'candidate_id': candidate_id,
				'layout_id': layout,
				'data_size': size,
				'primary': primary,
				'secondary': secondary,
				'control_primary': control_primary,
				'control_secondary': control_secondary,
				'reused_existing': (survey, family) == ('f3', 'mae'),
				'metrics_path': str(candidate_metrics),
				'metrics_sha256': digest,
				'control_metrics_path': str(control_metrics),
				'control_metrics_sha256': control_digest,
			}
		)
		control_cells.append(
			{
				'layout_id': layout,
				'data_size': size,
				'primary': control_primary,
				'secondary': control_secondary,
			}
		)
		audits.append(
			{
				'layout_id': layout,
				'data_size': size,
				'candidate': candidate_audit,
				'control': control_audit,
			}
		)
	verify_control(receipt, file_sha256(control_checkpoint), control_cells)
	canonical_cells_sha256(survey, cells)
	return cells, {'source': lineage, 'paired_cells': audits}


def _validate_experiment_inputs(config: dict[str, Any]) -> None:
	root = Path(config['experiment_root']).resolve()
	validate_multi_source_experiment_definition(root)
	if root.parents[1].name != config['survey']:
		raise ValueError('summary survey differs from experiment definition')
	for family, entry in config['arms'].items():
		arm_root = root
		if (config['survey'], family) == ('f3', 'mae'):
			arm_root = root.parent / '127_hmm_v2_multi_head_screening_v1'
		cid = CANDIDATE_IDS[family]
		expected = {
			'training_config': arm_root / '30_pretraining' / cid / '02_full_25ep.yaml',
			'downstream_config': arm_root / '50_downstream' / f'{cid}.yaml',
		}
		if entry['candidate_id'] != cid or any(
			Path(entry[field]).resolve() != path.resolve()
			for field, path in expected.items()
		):
			raise ValueError(
				'summary inputs differ from validated experiment definition'
			)


def inspect_survey(config: dict[str, Any]) -> dict[str, Any]:
	"""Return a complete 45-cell payload only after source and paired-cell audits."""
	if (
		config.get('schema_version') != 1
		or config.get('survey') not in SURVEYS
		or set(config['arms']) != set(SOURCE_FAMILIES)
	):
		raise ValueError('expected one survey and all three fixed source families')
	_validate_experiment_inputs(config)
	survey = config['survey']
	load_matrix(Path(config['matrix']))
	for field in ('matrix', 'control_receipt', 'selection_receipt', 'summary_root'):
		if 'reports' in Path(config[field]).resolve().parts:
			raise ValueError('reports cannot be pipeline inputs or outputs')
	controls = load_control_receipts(Path(config['control_receipt']))
	reuse = load_reuse_receipt(Path(config['selection_receipt']))
	if survey == 'f3':
		verify_reuse_receipt(reuse, Path(config['artifact_root']))
	cells, rows, arms, evidence = [], [], [], {}
	for family in SOURCE_FAMILIES:
		selected, audit = _audited_arm(
			config, family, control_receipt_for(controls, survey, family)
		)
		if (survey, family) == ('f3', 'mae') and canonical_cells_sha256(
			survey, selected
		) != reuse['canonical_cells_sha256']:
			raise ValueError('live F3 reused cells drift from receipt')
		cells.extend(selected)
		rows.extend(
			comparison_row(survey, family, size, selected) for size in DATA_SIZES
		)
		arms.append(
			{
				'source_family': family,
				'candidate_id': CANDIDATE_IDS[family],
				**arm_statistics(survey, selected),
			}
		)
		evidence[family] = audit
	payload = {
		'schema_version': 1,
		'status': 'complete',
		'survey': survey,
		**METRICS[survey],
		'cells': cells,
		'rows': rows,
		'arms': arms,
		'evidence': evidence,
		**{
			name: {'path': str(config[name]), 'sha256': file_sha256(Path(config[name]))}
			for name in ('matrix', 'control_receipt', 'selection_receipt')
		},
	}
	payload['summary_sha256'] = object_sha256(payload)
	return payload


def summarize_survey(config: dict[str, Any]) -> tuple[Path, Path, Path]:
	"""Publish exactly three review files under the dedicated artifact namespace."""
	output = Path(config['summary_root'])
	if output.exists() or output.is_symlink():
		raise FileExistsError('survey summary output already exists')
	payload = inspect_survey(config)
	lines = [
		f'# {payload["survey"]} K6810 multi-source comparison',
		'',
		(
			f'Evaluation: {payload["evaluation_split"]}. '
			f'Primary: {payload["primary_metric"]}; '
			f'secondary: {payload["secondary_metric"]}.'
		),
		'',
		(
			'Positive primary improvement favors K6810. '
			'All-15 results are descriptive; '
			'overall statistics use five layout clusters. '
			'No automatic selection or promotion.'
		),
		'',
		'| Source | Size | K6810 | K6 | Improvement | SD | Wins/ties/losses |',
		'| --- | --- | ---: | ---: | ---: | ---: | --- |',
	]
	lines.extend(
		f'| {r["source_family"]} | {r["data_size"]} | '
		f'{r["k6810_mean"]:.6f} | {r["k6_mean"]:.6f} | '
		f'{r["primary_improvement"]:.6f} | '
		f'{r["primary_improvement_sample_sd"]:.6f} | '
		f'{r["wins"]}/{r["ties"]}/{r["losses"]} |'
		for r in payload['rows']
	)
	return publish_summary(output, payload, payload['rows'], '\n'.join(lines) + '\n')


def main() -> None:
	"""Read-only validation or explicit publication of a complete survey result."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--config', type=Path, required=True)
	parser.add_argument('--dry-run', action='store_true')
	args = parser.parse_args()
	config = load_config(args.config)
	if args.dry_run:
		print(json.dumps({'status': inspect_survey(config)['status']}))
	else:
		summarize_survey(config)


if __name__ == '__main__':
	main()

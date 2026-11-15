"""Recover v1 distillation and v2 consistency comparisons as separate tables."""

from __future__ import annotations

import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from seis_ssl_cluster.clustering.features import file_sha256
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.hmm.distillation_sensitivity import read_rows, write_rows
from seis_ssl_cluster.hmm.multi_source_survey import _metrics, _paired_identity
from seis_ssl_cluster.training.strat_hmm_source_audit import audit_multi_head_source

PUBLISHED_FILES = frozenset(
	{
		'v1_pairs.csv',
		'v2_consistency_pairs.csv',
		'comparison.csv',
		'summary.json',
		'summary.md',
	}
)


def _pair(  # noqa: PLR0913, PLR0917
	survey: str,
	source: str,
	heads: str,
	size: str,
	layout: str,
	control: float,
	candidate: float,
	*,
	comparison: str,
) -> dict[str, Any]:
	return {
		'comparison': comparison,
		'survey': survey,
		'source': source,
		'head_ks': heads,
		'layout_id': layout,
		'data_size': size,
		'control': control,
		'candidate': candidate,
		'improvement': control - candidate
		if survey == 'volve'
		else candidate - control,
		'evaluation_split': 'common_test' if survey == 'f3' else 'test',
	}


def collect_auxiliary(root: Path) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915
	"""Use live artifacts, retaining completion evidence despite stale runbooks."""
	common = (
		root
		/ 'surveys/f3/facies_benchmark_v2/evaluation'
		/ 'lithology_common_test_v1/summary/per_layout.csv'
	)
	f3 = read_rows(common)
	original = read_rows(
		root / 'studies/hmm_v2/distillation_sensitivity_v1/v1_recovered/cells.csv'
	)
	v1, v2, evidence, missing = [], [], [], []
	for row in f3:
		if row['version'] == 'v1' and float(row['distillation_weight']) == 0.1:
			control = next(
				c
				for c in f3
				if (
					c['version'],
					c['source'],
					c['layout_id'],
					c['data_size'],
					c['distillation_weight'],
				)
				== ('v1', row['source'], row['layout_id'], row['data_size'], '0.2')
			)
			v1.append(
				_pair(
					'f3',
					row['source'],
					'6',
					row['data_size'],
					row['layout_id'],
					float(control['mean_iou']),
					float(row['mean_iou']),
					comparison='v1_dist010_vs020',
				)
			)
		elif (
			row['version'] == 'v2'
			and row['head_ks'] in ('4|6|8', '6|8|10')
			and float(row['consistency_weight']) == 0.1
		):
			control = next(
				c
				for c in f3
				if (
					c['version'],
					c['source'],
					c['layout_id'],
					c['data_size'],
					c['head_ks'],
					c['consistency_weight'],
				)
				== (
					'v2',
					row['source'],
					row['layout_id'],
					row['data_size'],
					row['head_ks'],
					'0',
				)
			)
			v2.append(
				_pair(
					'f3',
					row['source'],
					row['head_ks'],
					row['data_size'],
					row['layout_id'],
					float(control['mean_iou']),
					float(row['mean_iou']),
					comparison='v2_cons010_vs000',
				)
			)
	ids = {'2a': ('mae', '2b'), '4a': ('local_bt', '4b'), '6a': ('random', '6b')}
	for row in original:
		if row['dataset'] == 'f3' or row['condition_id'] not in ids:
			continue
		source, control_id = ids[row['condition_id']]
		control = next(
			c
			for c in original
			if (c['dataset'], c['condition_id'], c['layout_id'], c['data_size'])
			== (row['dataset'], control_id, row['layout_id'], row['data_size'])
		)
		v1.append(
			_pair(
				row['dataset'],
				source,
				'6',
				row['data_size'],
				row['layout_id'],
				float(control['primary_value']),
				float(row['primary_value']),
				comparison='v1_dist010_vs020',
			)
		)
	for heads in ('k6810', 'k468'):
		aggregate = load_config(
			f'experiments/hmm_v2/{heads}_multi_source_evaluation_v1/aggregate.yaml'
		)
		cons = load_config(
			f'experiments/hmm_v2/{heads}_cons010_multi_source_evaluation_v1/aggregate.yaml'
		)
		for survey in ('parihaka', 'volve'):
			paths = [Path(c['summaries'][survey]['path']) for c in (aggregate, cons)]
			if any(not p.is_file() for p in paths):
				missing.append(
					{
						'survey': survey,
						'heads': heads,
						'files': [str(p) for p in paths if not p.is_file()],
					}
				)
				continue
			summaries = [json.loads(p.read_text()) for p in paths]
			for path, summary in zip(paths, summaries, strict=True):
				if summary['status'] != 'complete' or len(summary['cells']) != 45:
					raise ValueError(f'incomplete auxiliary source: {path}')
				sources = {}
				for family, ev in summary['evidence'].items():
					source = ev['source']
					if source.get('status') == 'reused_existing':
						continue
					live = audit_multi_head_source(Path(source['config']))
					if live['checkpoint'] != source['checkpoint']:
						raise ValueError('auxiliary checkpoint lineage changed')
					sources[family] = live
				for cell in summary['cells']:
					primary, secondary, digest = _metrics(
						Path(cell['metrics_path']), survey
					)
					if (primary, secondary, digest) != (
						cell['primary'],
						cell['secondary'],
						cell['metrics_sha256'],
					):
						raise ValueError('auxiliary source metrics changed')
				evidence.append(
					{'path': str(path), 'sha256': file_sha256(path), 'sources': sources}
				)
			for candidate in summaries[1]['cells']:
				source, layout, size = (
					candidate[k] for k in ('source_family', 'layout_id', 'data_size')
				)
				control = next(
					c
					for c in summaries[0]['cells']
					if (c['source_family'], c['layout_id'], c['data_size'])
					== (source, layout, size)
				)
				audits = []
				for summary in summaries:
					ev = summary['evidence'][source]['paired_cells']
					matched = (
						next(
							a['candidate']
							for a in ev
							if (
								a['candidate']['benchmark_identity']['layout_id'],
								a['candidate']['benchmark_identity']['data_size'],
							)
							== (layout, size)
						)
						if survey == 'parihaka'
						else next(
							a['candidate']
							for a in ev
							if (a['layout_id'], a['data_size']) == (layout, size)
						)
					)
					audits.append(matched)
				_paired_identity(survey, *audits)
				v2.append(
					_pair(
						survey,
						source,
						'6|8|10' if heads == 'k6810' else '4|6|8',
						size,
						layout,
						control['primary'],
						candidate['primary'],
						comparison='v2_cons010_vs000',
					)
				)
	groups = defaultdict(list)
	for row in [*v1, *v2]:
		groups[
			(
				row['comparison'],
				row['survey'],
				row['source'],
				row['head_ks'],
				row['data_size'],
			)
		].append(row)
	rows = []
	for key, group in sorted(groups.items()):
		if len(group) != 5 or len({r['layout_id'] for r in group}) != 5:
			raise ValueError(
				'auxiliary comparisons require five distinct paired layouts'
			)
		row = dict(
			zip(
				('comparison', 'survey', 'source', 'head_ks', 'data_size'),
				key,
				strict=True,
			)
		)
		for metric in ('control', 'candidate', 'improvement'):
			row[metric + '_mean'] = statistics.mean(r[metric] for r in group)
			row[metric + '_sample_sd'] = statistics.stdev(r[metric] for r in group)
		rows.append(row)
	return {
		'v1': v1,
		'v2': v2,
		'rows': rows,
		'evidence': evidence,
		'missing': missing,
		'f3_common_test': {'path': str(common), 'sha256': file_sha256(common)},
		'interpretation': (
			'v1 and v2 are separate comparisons. '
			'F3 v1 has execution differences including workers; paired layouts '
			'do not establish identical training RNG streams. '
			'local_bt denotes the current asymmetric Barlow Twins setting.'
		),
	}


def publish_auxiliary(output: Path, payload: dict[str, Any]) -> None:
	"""Publish the exact auxiliary review file set without overwriting outputs."""
	output.mkdir(parents=True, exist_ok=False)
	for name, key in [
		('v1_pairs.csv', 'v1'),
		('v2_consistency_pairs.csv', 'v2'),
		('comparison.csv', 'rows'),
	]:
		write_rows(output / name, payload[key])
	(output / 'summary.json').write_text(
		json.dumps(payload, indent=2, allow_nan=False) + '\n'
	)
	(output / 'summary.md').write_text(
		'# Existing loss comparisons\n\n'
		+ payload['interpretation']
		+ f'\n\nv1 pairs: {len(payload["v1"])}; '
		+ f'v2 pairs: {len(payload["v2"])}.\n\nMissing: '
		+ json.dumps(payload['missing'])
		+ '\n'
	)


def main() -> None:
	"""Recover and publish this study's auxiliary results."""
	root = Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
	publish_auxiliary(
		root / 'studies/hmm_v2/distillation_sensitivity_v1/auxiliary',
		collect_auxiliary(root),
	)


if __name__ == '__main__':
	main()

"""Retain every saved HMM evaluation beside verified baseline comparisons."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from seis_ssl_cluster.clustering.features import file_sha256
from seis_ssl_cluster.results.hmm_baseline_comparison import (
	BASE_KEYS,
	DETAIL_KEYS,
	METRICS,
	evaluation,
	pair,
	rows,
)

FAMILIES = {'MAE': 'mae', 'Barlow Twins': 'local_bt', 'Random': 'random'}
SUPPORTS = {
	'f3': 'unique_test_voxel',
	'parihaka': 'test_labels',
	'volve': 'primary_common',
}


def artifact_source(artifact: Path, recorded: str) -> Path:
	"""Resolve catalog paths without accepting report or external numerical inputs."""
	path = Path(recorded)
	if not path.is_absolute():
		prefix = 'artifacts/seis_ssl_cluster/'
		path = artifact / recorded.removeprefix(prefix)
	path = path.resolve()
	if not path.is_relative_to(artifact.resolve()) or 'reports' in path.parts:
		raise ValueError('all saved numerical sources must remain in artifacts')
	return path


def version(record: dict[str, Any]) -> str:
	"""Retain an explicit version or clearly label historical unknown versions."""
	return record.get('hmm_version') or (
		record['protocol'].removeprefix('HMM ')
		if record['protocol'].startswith('HMM v')
		else 'legacy'
	)


def details(record: dict[str, Any]) -> dict[str, Any]:
	"""Expose the version and full saved evaluation identity in every comparison."""
	return dict(
		zip(
			DETAIL_KEYS,
			(
				version(record),
				record['head_ks'],
				record['model'],
				record['record_id'],
				record['support'],
				record['split'],
				record['source_cohort'],
				record['checkpoint_selection'],
				record['archived'],
				record['status'],
			),
			strict=True,
		)
	)


def number(value: str) -> float | str:
	"""Keep missing observations empty rather than converting them to zero."""
	return float(value) if value != '' else ''


def expand_saved(  # noqa: C901, PLR0912, PLR0915
	paired: list[dict[str, Any]],
	baselines: list[dict[str, Any]],
	artifact: Path,
	catalog: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
	"""Keep all HMM endpoints, pairing only scientifically compatible test scores."""
	path = artifact_source(artifact, catalog['path'])
	if file_sha256(path) != catalog['sha256']:
		raise ValueError('saved evaluation catalog changed')
	records = [r for r in rows(path) if r['hmm_version']]
	if not records:
		raise ValueError('saved evaluation catalog has no HMM results')
	lookup: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
	for b in baselines:
		lookup[tuple(b[k] for k in BASE_KEYS)].append(b)
	existing: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
	for row in paired:
		source = artifact_source(artifact, row['with_hmm_metrics_file'])
		existing[source, row['evaluation_split'], SUPPORTS[row['survey']]].append(row)
	covered = set()
	hashes: dict[Path, str] = {}
	additional = []
	for record in records:
		source = artifact_source(artifact, record['source_file'])
		if source not in hashes:
			hashes[source] = file_sha256(source)
			additional.append(
				{
					'path': str(source.relative_to(artifact.resolve())),
					'sha256': hashes[source],
				}
			)
		if hashes[source] != record['source_sha256']:
			raise ValueError(f'saved metric source changed: {source}')
		key = (source, record['split'], record['support'])
		if key in covered:
			raise ValueError('duplicate saved HMM evaluation identity')
		covered.add(key)
		info = details(record)
		if key in existing:
			for row in existing[key]:
				row.update(info)
			continue
		survey = record['survey']
		metric, unit, direction, secondary, secondary_unit = METRICS[survey]
		c = dict(
			survey=survey,
			source=FAMILIES.get(record['initialization'], record['initialization']),
			data_size=record['label_budget'],
			layout_id=record['replicate_id'],
			distillation_weight=record['distillation_weight'],
			consistency_weight=record['consistency_weight'],
			evaluation_split=record['split'],
			decoder_seed=record['decoder_seed'],
			primary=number(record.get(metric, '')),
			secondary=number(record.get(secondary, '')),
			metrics_file=str(source.relative_to(artifact.resolve())),
			**info,
		)
		matches = []
		reason = 'no verified no-HMM baseline for this split/support'
		if (
			record['split'] in ('test', 'common_test')
			and record['support'] == SUPPORTS[survey]
		):
			try:
				measured = evaluation(source, survey, expected_hmm=True)
			except (ValueError, KeyError, FileNotFoundError) as error:
				reason = {
					FileNotFoundError: 'missing evaluation metadata',
					KeyError: 'missing decoder/support identity fields',
					ValueError: 'evaluation support or HMM-source check failed',
				}[type(error)]
			else:
				c.update(measured)
				c['evaluation_split'] = 'common_test' if survey == 'f3' else 'test'
				candidates = lookup[tuple(c[k] for k in BASE_KEYS)]
				reason = 'no compatible no-HMM decoder/support identity'
				for baseline in candidates:
					identity = (
						'shared_identity'
						if c['decoder_seed'] == baseline['decoder_seed']
						else 'comparison_identity'
					)
					if c[identity] == baseline[identity]:
						matches.extend(pair([c], [baseline]))
		if matches:
			paired.extend(matches)
		else:
			row = dict.fromkeys(paired[0], '')
			row.update(
				info,
				survey=survey,
				source=c['source'],
				data_size=c['data_size'],
				layout_id=c['layout_id'],
				distillation_weight=c['distillation_weight'],
				consistency_weight=c['consistency_weight'],
				evaluation_split=record['split'],
				primary_metric=metric,
				primary_unit=unit,
				primary_direction=direction,
				secondary_metric=secondary,
				secondary_unit=secondary_unit,
				with_hmm_primary=c['primary'],
				with_hmm_secondary=c['secondary'],
				with_hmm_decoder_seed=c['decoder_seed'],
				with_hmm_metrics_file=c['metrics_file'],
				comparison_status='unmatched_baseline',
				comparison_type='unpaired',
				comparison_note=reason,
			)
			paired.append(row)
	if not set(existing).issubset(covered):
		raise ValueError('original comparison missing from saved HMM catalog')
	observed = Counter(version(r) for r in records)
	if not {'v1', 'v2', 'v3'}.issubset(observed):
		raise ValueError('saved HMM catalog must retain v1, v2 and v3')
	return paired, {
		'saved_catalog': catalog,
		'saved_hmm_records': len(records),
		'saved_hmm_by_version': dict(sorted(observed.items())),
		'comparison_rows_by_version': dict(
			sorted(Counter(r['hmm_version'] for r in paired).items())
		),
		'matched_rows': sum(r['comparison_status'] == 'paired' for r in paired),
		'unmatched_rows': sum(r['comparison_status'] != 'paired' for r in paired),
		'additional_sources': additional,
	}

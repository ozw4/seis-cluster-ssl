"""Join every saved evaluation to its verified controls, when available."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from seis_ssl_cluster.results.experiment_catalog import sha256
from seis_ssl_cluster.results.hmm_baseline_comparison import METRICS, rows, write_csv
from seis_ssl_cluster.results.saved_hmm_comparison import SUPPORTS, artifact_source

if TYPE_CHECKING:
	from collections.abc import Iterable, Mapping
	from typing import Any

ROW = dict[str, str]
FILES = frozenset({'all_experiment_comparison.csv', 'provenance.json', 'README.md'})
CATALOG = 'studies/all_experiment_metrics_20261115_v1/catalog'
BASELINES = 'studies/hmm_v2/missing_controls_completion_v1/hmm_vs_no_hmm_v1'
KMEANS_STUDIES = ('kmeans_k6810_multi_source_v1', 'kmeans_k6810_dis0_con0_v1')
COMPARISON_FIELDS = (
	'comparison_status',
	'comparison_study',
	'same_decoder_seed',
	'baseline_reference_status',
	'comparison_primary_metric',
	'comparison_primary_unit',
	'comparison_primary_direction',
	'comparison_primary_value',
	'comparison_secondary_metric',
	'comparison_secondary_unit',
	'comparison_secondary_value',
	'reference_record_id',
	'reference_model',
	'reference_protocol',
	'reference_distillation_weight',
	'reference_consistency_weight',
	'reference_encoder_seed',
	'reference_decoder_seed',
	'reference_split',
	'reference_support',
	'reference_primary_value',
	'reference_secondary_value',
	'reference_source_file',
	'reference_source_sha256',
	'primary_improvement',
	'secondary_improvement',
)


def _same_number(saved: str, expected: object) -> bool:
	return math.isclose(float(saved), float(str(expected)), rel_tol=0, abs_tol=1e-12)


def _split(record: ROW) -> str:
	if (
		record['survey'] == 'f3'
		and record['support'] == SUPPORTS['f3']
		and record['split'] in ('test', 'common_test')
	):
		return 'common_test'
	return record['split']


def _compare(
	observed: ROW, reference: ROW, *, study: str, reference_status: str = ''
) -> ROW:
	for key in (
		'survey',
		'initialization',
		'label_budget',
		'replicate_id',
		'support',
	):
		if observed[key] != reference[key]:
			raise ValueError(f'incompatible comparison {key}')
	if _split(observed) != _split(reference):
		raise ValueError('incompatible comparison split')
	primary, unit, direction, secondary, second_unit = METRICS[observed['survey']]
	primary_delta = float(observed[primary]) - float(reference[primary])
	secondary_delta = float(observed[secondary]) - float(reference[secondary])
	return dict(
		comparison_status='paired',
		comparison_study=study,
		same_decoder_seed=str(observed['decoder_seed'] == reference['decoder_seed']),
		baseline_reference_status=reference_status,
		comparison_primary_metric=primary,
		comparison_primary_unit=unit,
		comparison_primary_direction=direction,
		comparison_primary_value=observed[primary],
		comparison_secondary_metric=secondary,
		comparison_secondary_unit=second_unit,
		comparison_secondary_value=observed[secondary],
		**{
			f'reference_{key}': reference[key]
			for key in (
				'record_id',
				'model',
				'protocol',
				'distillation_weight',
				'consistency_weight',
				'encoder_seed',
				'decoder_seed',
				'split',
				'support',
				'source_file',
				'source_sha256',
			)
		},
		reference_primary_value=reference[primary],
		reference_secondary_value=reference[secondary],
		primary_improvement=str(
			-primary_delta if direction == 'lower' else primary_delta
		),
		secondary_improvement=str(secondary_delta),
	)


class ComparisonIndex:
	"""Identify a saved score by its file, split, and evaluation support."""

	def __init__(self, artifact: Path, records: list[ROW]) -> None:
		"""Index both raw metrics and F3 common-test cohort tables."""
		self.artifact = artifact
		self.records = {row['record_id']: row for row in records}
		self.files = {
			(artifact_source(artifact, r['source_file']), _split(r), r['support']): r
			for r in records
		}
		if len(self.records) != len(records) or len(self.files) != len(records):
			raise ValueError('duplicate catalog evaluation identity')
		self.cohorts: dict[tuple[Path, str, str, str], list[ROW]] = defaultdict(list)
		for r in records:
			if r['survey'] != 'f3' or _split(r) != 'common_test':
				continue
			key = (
				artifact_source(artifact, r['source_cohort']),
				r['model'],
				r['replicate_id'],
				r['label_budget'],
			)
			self.cohorts[key].append(r)
		self.hashes: dict[Path, str] = {}

	def baseline_pairs(self, pairs: Iterable[ROW]) -> Iterable[tuple[str, ROW]]:
		"""Use the existing verified HMM/no-HMM relationships and both seeds."""
		for pair in pairs:
			if pair['comparison_status'] != 'paired':
				continue
			observed = self.records[pair['with_hmm_record_id']]
			path = artifact_source(self.artifact, pair['without_hmm_metrics_file'])
			reference = self.files[path, _split(observed), observed['support']]
			if not observed['hmm_version'] or reference['protocol'] != 'none':
				raise ValueError('HMM/no-HMM pair has incompatible protocols')
			if (observed['decoder_seed'], reference['decoder_seed']) != (
				pair['with_hmm_decoder_seed'],
				pair['without_hmm_decoder_seed'],
			):
				raise ValueError('HMM/no-HMM pair has inconsistent decoder seeds')
			result = _compare(
				observed,
				reference,
				study='hmm_vs_no_hmm',
				reference_status=pair['baseline_reference_status'],
			)
			for side, prefix in (
				('with_hmm', 'comparison'),
				('without_hmm', 'reference'),
			):
				for metric in ('primary', 'secondary'):
					if not _same_number(
						result[f'{prefix}_{metric}_value'], pair[f'{side}_{metric}']
					):
						raise ValueError(
							'HMM/no-HMM comparison score differs from catalog'
						)
			yield observed['record_id'], result

	def _kmeans_record(self, cell: Mapping[str, Any], side: str) -> ROW:
		ref = cell[f'{side}_metrics']
		path = artifact_source(self.artifact, ref['path'])
		if path not in self.hashes:
			self.hashes[path] = sha256(path)
		if self.hashes[path] != ref['sha256']:
			raise ValueError('k-means comparison source hash drift')
		survey = cell['survey']
		if survey == 'f3' and path.name == 'per_layout.csv':
			matches = self.cohorts[
				path.parent.parent,
				cell[f'{side}_id'],
				cell['layout_id'],
				cell['data_size'],
			]
			if len(matches) != 1:
				raise ValueError('F3 k-means cell requires one common-test result')
			return matches[0]
		split = 'validation' if survey == 'f3' else 'test'
		support = 'unique_validation_voxel' if survey == 'f3' else SUPPORTS[survey]
		return self.files[path, split, support]

	def kmeans_pairs(
		self, study: str, cells: Iterable[Mapping[str, Any]]
	) -> Iterable[tuple[str, ROW]]:
		"""Bind both published k-means ablations to exact catalog scores."""
		for cell in cells:
			observed = self._kmeans_record(cell, 'control')
			reference = self._kmeans_record(cell, 'candidate')
			if not observed['hmm_version'] or reference['protocol'] != 'k-means':
				raise ValueError('HMM/k-means pair has incompatible protocols')
			if (
				observed['decoder_seed'] != '42000'
				or reference['decoder_seed'] != '42000'
			):
				raise ValueError('HMM/k-means pair requires decoder seed 42000')
			result = _compare(observed, reference, study=study)
			for metric in ('primary', 'secondary'):
				if not _same_number(
					result[f'comparison_{metric}_value'], cell[f'control_{metric}']
				) or not _same_number(
					result[f'reference_{metric}_value'], cell[metric]
				):
					raise ValueError(
						'HMM/k-means comparison score differs from catalog'
					)
			yield observed['record_id'], result


def assemble(records: list[ROW], pairs: Iterable[tuple[str, ROW]]) -> list[ROW]:
	"""Retain every endpoint and expand only where a verified reference exists."""
	by_id: dict[str, list[ROW]] = defaultdict(list)
	seen = set()
	for record_id, pair in pairs:
		key = (record_id, pair['reference_record_id'], pair['comparison_study'])
		if key in seen:
			raise ValueError('duplicate verified comparison')
		seen.add(key)
		by_id[record_id].append(pair)
	if not set(by_id).issubset(r['record_id'] for r in records):
		raise ValueError('comparison refers to missing catalog evaluation')
	result = []
	for record in records:
		for pair in by_id.get(record['record_id'], [{}]):
			fields = dict.fromkeys(COMPARISON_FIELDS, '')
			fields.update(comparison_status='saved_evaluation')
			fields.update(pair)
			result.append({**record, **fields})
	return result


def export(artifact: Path, output: Path) -> dict[str, Any]:
	"""Publish an exact three-file comparison from saved artifact sources."""
	output = artifact_source(artifact, str(output.resolve()))
	if output.exists() and {p.name for p in output.iterdir()} != FILES:
		raise ValueError('refusing to overwrite unowned comparison files')
	catalog = artifact / CATALOG / 'experiment_metrics.csv'
	baseline = artifact / BASELINES / 'factorial_cells.csv'
	provenance = json.loads((baseline.parent / 'provenance.json').read_text())
	if sha256(catalog) != provenance['saved_catalog']['sha256']:
		raise ValueError('HMM/no-HMM comparison was generated from a different catalog')
	records = rows(catalog)
	index = ComparisonIndex(artifact, records)
	pairs = list(index.baseline_pairs(rows(baseline)))
	inputs = [catalog, baseline]
	for study in KMEANS_STUDIES:
		path = artifact / 'studies/kmeans' / study / 'summary/summary.json'
		payload = json.loads(path.read_text())
		if (
			payload['study'] != study
			or payload['status'] != 'complete'
			or len(payload['cells']) != 135
		):
			raise ValueError(f'incomplete k-means comparison: {study}')
		pairs.extend(index.kmeans_pairs(study, payload['cells']))
		inputs.append(path)
	result = assemble(records, pairs)
	metadata = {
		'saved_evaluations': len(records),
		'comparison_csv_rows': len(result),
		'paired_rows': len(pairs),
		'paired_by_study': dict(Counter(r['comparison_study'] for _, r in pairs)),
		'evaluations_by_protocol': dict(Counter(r['protocol'] for r in records)),
		'evaluations_by_hmm_version': dict(
			Counter(r['hmm_version'] for r in records if r['hmm_version'])
		),
		'inputs_sha256': {str(p.relative_to(artifact)): sha256(p) for p in inputs},
		'published_files': sorted(FILES),
	}
	output.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.TemporaryDirectory(
		prefix='.all-comparison-', dir=output.parent
	) as tmp:
		stage = Path(tmp)
		path = stage / 'all_experiment_comparison.csv'
		write_csv(path, result)
		metadata['comparison_csv_sha256'] = sha256(path)
		(stage / 'provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
		(stage / 'README.md').write_text(
			'# All saved experiment comparisons\n\n'
			f'{len(records)} saved evaluations, {len(result)} rows, '
			f'{len(pairs)} verified pairs.\n\n'
			'Use `record_id` to count distinct evaluations. Each verified reference '
			'has its own row. `comparison_study` identifies the source comparison. '
			'`saved_evaluation` rows have blank references.\n\n'
			'Positive improvement favors the row model. For Volve MAE, '
			'improvement is reference minus observed error. The other improvements are '
			'observed minus reference. Seeds, splits, and supports are explicit.\n\n'
			'The older k-means study compares against HMM with consistency 0.1. '
			'The new study sets both auxiliary loss weights to zero on both sides.\n'
		)
		if {p.name for p in stage.iterdir()} != FILES:
			raise ValueError('unexpected comparison publication files')
		output.mkdir(exist_ok=True)
		for name in ('all_experiment_comparison.csv', 'README.md', 'provenance.json'):
			(stage / name).replace(output / name)
	return metadata


def main() -> None:
	"""Write the current all-experiment comparison CSV."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		'--artifact-root', type=Path, default=Path('artifacts/seis_ssl_cluster')
	)
	parser.add_argument('--output', type=Path, required=True)
	args = parser.parse_args()
	print(json.dumps(export(args.artifact_root.resolve(), args.output), indent=2))


if __name__ == '__main__':
	main()

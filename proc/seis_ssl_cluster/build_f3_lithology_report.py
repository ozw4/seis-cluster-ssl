"""Build F3 token-level lithology probe reports."""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.f3 import (
	F3LithologyComparisonReportConfig,
	F3LithologyReportConfig,
	build_f3_lithology_report,
)

STAGE = 'build_f3_lithology_report'
DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'f3'
	/ 'facies_benchmark_v1'
	/ '50_lithology'
	/ 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
	/ 'overlap_x16'
	/ 'png_slices_segy_labels_v1'
	/ '06_build_lithology_report.yaml'
)


def main() -> None:
	"""Build an F3 lithology probe report or print a dry-run summary."""
	parser = ArgumentParser(description='Build an F3 lithology probe report.')
	parser.add_argument(
		'--config',
		type=Path,
		default=DEFAULT_CONFIG,
		help='Path to a YAML configuration file.',
	)
	parser.add_argument(
		'--dry-run',
		action='store_true',
		help='Validate the config and print a run summary without writing reports.',
	)
	args = parser.parse_args()

	raw_config = load_config(args.config)
	config = f3_lithology_report_config_from_mapping(raw_config)
	if args.dry_run:
		_print_summary(config)
		print('execution: dry-run; F3 lithology report skipped')
		return

	result = build_f3_lithology_report(config)
	warnings = result.payload.get('warnings', [])
	warning_count = len(warnings) if isinstance(warnings, Sequence) else 0
	print(f'f3_lithology_report.warning_count: {warning_count}')
	print(f'f3_lithology_report.markdown: {result.report_markdown}')
	print(f'f3_lithology_report.json: {result.report_json}')
	if result.comparison_csv is not None:
		print(f'f3_lithology_report.comparison_csv: {result.comparison_csv}')
	if result.comparison_markdown is not None:
		print(
			'f3_lithology_report.comparison_markdown: '
			f'{result.comparison_markdown}',
		)


def f3_lithology_report_config_from_mapping(
	config: Mapping[str, object],
) -> F3LithologyReportConfig:
	"""Validate and normalize the F3 lithology report config."""
	_validate_allowed_keys(
		config,
		frozenset(
			{
				'paths',
				'dataset',
				'model',
				'labels',
				'lithology',
				'probe',
				'predictions',
				'visualizations',
				'reports',
				'comparison',
			},
		),
		prefix='config',
	)
	paths = _required_mapping(config, 'paths')
	artifact_root = _required_absolute_path(paths, 'artifact_root', prefix='paths')
	f3_root = _required_absolute_path(paths, 'f3_root', prefix='paths')
	dataset = _required_mapping(config, 'dataset')
	model = _required_mapping(config, 'model')
	labels = _required_mapping(config, 'labels')
	lithology = _required_mapping(config, 'lithology')
	probe = _required_mapping(config, 'probe')
	predictions = _optional_mapping(config, 'predictions')
	visualizations = _optional_mapping(config, 'visualizations')
	reports = _required_mapping(config, 'reports')
	metrics_json = _required_absolute_path(probe, 'metrics_json', prefix='probe')
	probe_config_json = _optional_absolute_path(
		probe,
		'probe_config_resolved_json',
		prefix='probe',
	)
	output_dir = _required_absolute_path(reports, 'output_dir', prefix='reports')
	output_markdown = _required_absolute_path(
		reports,
		'output_markdown',
		prefix='reports',
	)
	output_json = _required_absolute_path(reports, 'output_json', prefix='reports')
	prediction_metadata_json = _optional_absolute_path(
		predictions,
		'metadata_json',
		prefix='predictions',
	)
	visualization_metadata_json = _optional_absolute_path(
		visualizations,
		'metadata_json',
		prefix='visualizations',
	)
	token_dataset_metadata_json = _optional_absolute_path(
		reports,
		'token_dataset_metadata_json',
		prefix='reports',
	)
	comparison = _comparison_config(
		_optional_mapping(config, 'comparison'),
		artifact_root=artifact_root,
		dataset=dataset,
	)
	for label, path in _report_paths(
		metrics_json=metrics_json,
		probe_config_json=probe_config_json,
		token_dataset_metadata_json=token_dataset_metadata_json,
		prediction_metadata_json=prediction_metadata_json,
		visualization_metadata_json=visualization_metadata_json,
		output_dir=output_dir,
		output_markdown=output_markdown,
		output_json=output_json,
		comparison=comparison,
	):
		_validate_artifact_path(
			path,
			label,
			artifact_root=artifact_root,
			f3_root=f3_root,
		)
	return F3LithologyReportConfig(
		output_dir=output_dir,
		output_markdown=output_markdown,
		output_json=output_json,
		metrics_json=metrics_json,
		probe_config_json=probe_config_json,
		token_dataset_metadata_json=token_dataset_metadata_json,
		prediction_metadata_json=prediction_metadata_json,
		visualization_metadata_json=visualization_metadata_json,
		dataset=dataset,
		model=model,
		labels=labels,
		lithology=lithology,
		probe=probe,
		comparison=comparison,
	)


def _comparison_config(
	comparison: Mapping[str, object],
	*,
	artifact_root: Path,
	dataset: Mapping[str, object],
) -> F3LithologyComparisonReportConfig:
	version = _optional_str(
		dataset,
		'version',
		default='facies_benchmark_v1',
		prefix='dataset',
	)
	default_search_root = artifact_root / 'lithology' / 'f3' / version
	default_output_dir = default_search_root / 'reports'
	search_root = _optional_absolute_path(
		comparison,
		'search_root',
		prefix='comparison',
		default=default_search_root,
	)
	output_dir = _optional_absolute_path(
		comparison,
		'output_dir',
		prefix='comparison',
		default=default_output_dir,
	)
	return F3LithologyComparisonReportConfig(
		search_root=search_root,
		output_csv=_optional_absolute_path(
			comparison,
			'output_csv',
			prefix='comparison',
			default=output_dir / 'comparison_table.csv',
		),
		output_markdown=_optional_absolute_path(
			comparison,
			'output_markdown',
			prefix='comparison',
			default=output_dir / 'comparison_report.md',
		),
	)


def _report_paths(  # noqa: PLR0913
	*,
	metrics_json: Path,
	probe_config_json: Path | None,
	token_dataset_metadata_json: Path | None,
	prediction_metadata_json: Path | None,
	visualization_metadata_json: Path | None,
	output_dir: Path,
	output_markdown: Path,
	output_json: Path,
	comparison: F3LithologyComparisonReportConfig,
) -> tuple[tuple[str, Path], ...]:
	paths = [
		('probe.metrics_json', metrics_json),
		('reports.output_dir', output_dir),
		('reports.output_markdown', output_markdown),
		('reports.output_json', output_json),
		('comparison.search_root', comparison.search_root),
		('comparison.output_csv', comparison.output_csv),
		('comparison.output_markdown', comparison.output_markdown),
	]
	optional_paths = (
		('probe.probe_config_resolved_json', probe_config_json),
		('reports.token_dataset_metadata_json', token_dataset_metadata_json),
		('predictions.metadata_json', prediction_metadata_json),
		('visualizations.metadata_json', visualization_metadata_json),
	)
	paths.extend((label, path) for label, path in optional_paths if path is not None)
	return tuple(paths)


def _print_summary(config: F3LithologyReportConfig) -> None:
	print(f'stage: {STAGE}')
	print(f'model.tag: {config.model.get("tag")}')
	print(f'model.checkpoint: {config.model.get("checkpoint")}')
	print(f'lithology.root: {config.lithology.get("root")}')
	print(f'probe.spec: {config.probe.get("spec")}')
	print(f'probe.metrics_json: {config.metrics_json}')
	print(f'probe.probe_config_resolved_json: {config.probe_config_json}')
	print(f'reports.output_dir: {config.output_dir}')
	print(f'reports.output_markdown: {config.output_markdown}')
	print(f'reports.output_json: {config.output_json}')
	print(f'predictions.metadata_json: {config.prediction_metadata_json}')
	print(f'visualizations.metadata_json: {config.visualization_metadata_json}')
	if config.comparison is not None:
		print(f'comparison.search_root: {config.comparison.search_root}')
		print(f'comparison.output_csv: {config.comparison.output_csv}')
		print(f'comparison.output_markdown: {config.comparison.output_markdown}')


def _required_mapping(
	parent: Mapping[str, object],
	key: str,
) -> Mapping[str, Any]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_mapping(
	parent: Mapping[str, object],
	key: str,
) -> Mapping[str, Any]:
	value = parent.get(key)
	if value is None:
		return {}
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping; got {value!r}'
		raise TypeError(msg)
	return value


def _validate_allowed_keys(
	parent: Mapping[str, object],
	allowed: frozenset[str],
	*,
	prefix: str,
) -> None:
	unexpected = sorted(set(parent) - allowed)
	if unexpected:
		msg = (
			f'{prefix} key(s) not allowed: {unexpected!r}; '
			f'allowed keys are {sorted(allowed)!r}'
		)
		raise ValueError(msg)


def _required_absolute_path(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> Path:
	path = Path(_required_str(parent, key, prefix=prefix))
	if not path.is_absolute():
		msg = f'{prefix}.{key} must be an absolute path; got {path}'
		raise ValueError(msg)
	return path


def _optional_absolute_path(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
	default: Path | None = None,
) -> Path | None:
	value = parent.get(key)
	if value is None:
		return default
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	path = Path(value)
	if not path.is_absolute():
		msg = f'{prefix}.{key} must be an absolute path; got {path}'
		raise ValueError(msg)
	return path


def _required_str(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> str:
	value = parent.get(key)
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_str(
	parent: Mapping[str, object],
	key: str,
	*,
	default: str,
	prefix: str,
) -> str:
	value = parent.get(key, default)
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return value


def _validate_artifact_path(
	path: Path,
	label: str,
	*,
	artifact_root: Path,
	f3_root: Path,
) -> None:
	if 'runs' in path.parts:
		msg = f'{label} must not use runs/ paths; got {path}'
		raise ValueError(msg)
	if _is_relative_to(path, f3_root):
		msg = f'{label} must not be under paths.f3_root; got {path}'
		raise ValueError(msg)
	if not _is_relative_to(path, artifact_root):
		msg = f'{label} must be under paths.artifact_root ({artifact_root}); got {path}'
		raise ValueError(msg)


def _is_relative_to(path: Path, root: Path) -> bool:
	try:
		path.resolve(strict=False).relative_to(root.resolve(strict=False))
	except ValueError:
		return False
	return True


if __name__ == '__main__':
	main()

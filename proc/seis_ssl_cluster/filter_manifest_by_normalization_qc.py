"""Generate clean amplitude-only NOPIMS manifest/path-list files from QC."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from seis_ssl_cluster.config import (
	load_config,
	resolve_normalization_qc_config,
)
from seis_ssl_cluster.data.manifest_filter import (
	FilteredManifestStatsQcResult,
	filter_manifests_by_stats_qc,
)
from seis_ssl_cluster.data.normalization_qc import (
	NormalizationStatsQcThresholds,
	normalization_qc_report_to_dict,
)
from seis_ssl_cluster.data.path_list import load_npy_path_list
from seis_ssl_cluster.data.schema import (
	read_manifest_json,
	write_manifest_json,
)
from seis_ssl_cluster.utils.cli import (
	parse_config_args,
	print_config_summary,
)

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[1]
	/ 'configs'
	/ 'seis_ssl_cluster'
	/ 'filter_manifest_by_normalization_qc.yaml'
)


def main() -> None:
	"""Run normalization stats QC and write clean outputs."""
	args = parse_config_args(
		'Filter amplitude-only manifests by normalization QC.',
		DEFAULT_CONFIG,
	)
	config = resolve_normalization_qc_config(load_config(args.config))
	paths = _required_mapping(config, 'paths')
	nopims_root = Path(_required_str(paths, 'nopims_root'))
	manifest_path = _manifest_input_path(config)
	split_path = _split_input_path(config)
	output_manifest_path = _manifest_output_path(config)
	output_split_path = _split_output_path(config)
	qc_json_path = _qc_json_path(config)
	excluded_surveys_path = _excluded_surveys_path(config)
	thresholds = _qc_thresholds(config)

	if args.dry_run:
		print_config_summary(config)

	if not manifest_path.is_file() or not split_path.is_file():
		if args.dry_run:
			print(f'normalization_qc.manifest_path: {manifest_path}')
			print(
				'normalization_qc.manifest_exists: '
				f'{str(manifest_path.is_file()).lower()}',
			)
			print(f'normalization_qc.source_split_path: {split_path}')
			print(
				'normalization_qc.source_split_exists: '
				f'{str(split_path.is_file()).lower()}',
			)
			print('normalization_qc.write: false')
			print('normalization_qc.compute: skipped')
			return
		missing = manifest_path if not manifest_path.is_file() else split_path
		msg = f'normalization QC input does not exist: {missing}'
		raise FileNotFoundError(msg)

	manifests = read_manifest_json(manifest_path)
	path_entries = load_npy_path_list(split_path)
	result = filter_manifests_by_stats_qc(
		manifests,
		path_entries,
		nopims_root=nopims_root,
		thresholds=thresholds,
	)

	write_outputs = not args.dry_run
	_print_summary(
		manifest_path=manifest_path,
		split_path=split_path,
		output_manifest_path=output_manifest_path,
		output_split_path=output_split_path,
		result=result,
		write=write_outputs,
	)
	if not write_outputs:
		return

	_write_qc_json(
		result,
		qc_json_path,
		source_manifest_path=manifest_path,
		source_split_path=split_path,
	)
	_write_excluded_surveys(result, excluded_surveys_path)
	output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
	write_manifest_json(result.clean_manifests, output_manifest_path)
	_write_clean_path_list(result, output_split_path)


def _manifest_input_path(config: Mapping[str, object]) -> Path:
	manifests = _required_mapping(config, 'manifests')
	return Path(_required_str(manifests, 'input'))


def _manifest_output_path(config: Mapping[str, object]) -> Path:
	manifests = _required_mapping(config, 'manifests')
	return Path(_required_str(manifests, 'output'))


def _split_input_path(config: Mapping[str, object]) -> Path:
	splits = _required_mapping(config, 'splits')
	return Path(_required_str(splits, 'input'))


def _split_output_path(config: Mapping[str, object]) -> Path:
	splits = _required_mapping(config, 'splits')
	return Path(_required_str(splits, 'output'))


def _qc_json_path(config: Mapping[str, object]) -> Path:
	qc = _required_mapping(config, 'qc')
	return Path(_required_str(qc, 'output_json'))


def _excluded_surveys_path(config: Mapping[str, object]) -> Path:
	qc = _required_mapping(config, 'qc')
	return Path(_required_str(qc, 'excluded_surveys'))


def _qc_thresholds(config: Mapping[str, object]) -> NormalizationStatsQcThresholds:
	qc = _required_mapping(config, 'qc')
	return NormalizationStatsQcThresholds(
		min_iqr=_optional_positive_float(qc, 'min_iqr', default=1.0e-4),
		max_normalized_abs=_optional_positive_float(
			qc,
			'max_normalized_abs',
			default=1.0e6,
		),
	)


def _print_summary(  # noqa: PLR0913
	*,
	manifest_path: Path,
	split_path: Path,
	output_manifest_path: Path,
	output_split_path: Path,
	result: FilteredManifestStatsQcResult,
	write: bool,
) -> None:
	print(f'normalization_qc.manifest_path: {manifest_path}')
	print(f'normalization_qc.source_split_path: {split_path}')
	print(f'normalization_qc.total_surveys: {len(result.report.items)}')
	print(
		'normalization_qc.passed_surveys: '
		f'{sum(item.status == "pass" for item in result.report.items)}',
	)
	print(f'normalization_qc.excluded_surveys: {len(result.excluded_surveys)}')
	print(f'normalization_qc.clean_manifest: {output_manifest_path}')
	print(f'normalization_qc.clean_path_list: {output_split_path}')
	print(f'normalization_qc.write: {str(write).lower()}')


def _write_qc_json(
	result: FilteredManifestStatsQcResult,
	output_path: Path,
	*,
	source_manifest_path: Path,
	source_split_path: Path,
) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(
		json.dumps(
			normalization_qc_report_to_dict(
				result.report,
				source_manifest_path=source_manifest_path,
				source_split_path=source_split_path,
			),
			indent=2,
			sort_keys=True,
			allow_nan=False,
		)
		+ '\n',
		encoding='utf-8',
	)


def _write_excluded_surveys(
	result: FilteredManifestStatsQcResult,
	output_path: Path,
) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	text = ''.join(f'{survey_id}\n' for survey_id in result.excluded_surveys)
	output_path.write_text(text, encoding='utf-8')


def _write_clean_path_list(
	result: FilteredManifestStatsQcResult,
	output_path: Path,
) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	text = ''.join(f'{entry}\n' for entry in result.clean_path_entries)
	output_path.write_text(text, encoding='utf-8')


def _required_mapping(parent: Mapping[str, object], key: str) -> Mapping[str, Any]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return value


def _required_str(parent: Mapping[str, object], key: str) -> str:
	value = parent.get(key)
	if not isinstance(value, str) or not value:
		msg = f'{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_positive_float(
	parent: Mapping[str, object],
	key: str,
	*,
	default: float,
) -> float:
	value = parent.get(key, default)
	if isinstance(value, bool) or not isinstance(value, int | float):
		msg = f'{key} must be numeric; got {value!r}'
		raise TypeError(msg)
	value = float(value)
	if value <= 0.0:
		msg = f'{key} must be positive; got {value!r}'
		raise ValueError(msg)
	return value


if __name__ == '__main__':
	main()

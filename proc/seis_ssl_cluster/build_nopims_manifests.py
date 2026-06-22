"""Build amplitude-only NOPIMS manifests from configured `.npy` path lists."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[2] / 'src'
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from seis_ssl_cluster.config import (  # noqa: E402
	load_config,
	resolve_manifest_build_config,
)
from seis_ssl_cluster.data import (  # noqa: E402
	ManifestBuildSummary,
	scan_nopims_amplitude_manifests_from_path_list,
	write_manifest_json,
)
from seis_ssl_cluster.utils.cli import (  # noqa: E402
	parse_config_args,
	print_config_summary,
)

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[1]
	/ 'configs'
	/ 'seis_ssl_cluster'
	/ 'build_nopims_manifests.yaml'
)


def main() -> None:
	"""Build NOPIMS manifests or print a dry-run summary."""
	args = parse_config_args(
		'Build amplitude-only NOPIMS manifests.',
		DEFAULT_CONFIG,
	)
	config = resolve_manifest_build_config(load_config(args.config))
	paths = _required_mapping(config, 'paths')
	manifest_cfg = _required_mapping(config, 'manifest')
	nopims_root = Path(_required_str(paths, 'nopims_root'))
	output_path = _manifest_output_path(manifest_cfg)
	input_path_list = Path(_required_str(manifest_cfg, 'input_path_list'))
	normalization_stats_dir = Path(
		_required_str(manifest_cfg, 'normalization_stats_dir'),
	)
	_require_absolute_path(
		normalization_stats_dir,
		'manifest.normalization_stats_dir',
	)

	if args.dry_run:
		print_config_summary(config)
		_print_manifest_target(
			nopims_root,
			input_path_list,
			output_path,
			normalization_stats_dir,
		)
		print('manifest scan: skipped')
		return

	result = scan_nopims_amplitude_manifests_from_path_list(
		nopims_root=nopims_root,
		input_path_list=input_path_list,
		normalization_stats_dir=normalization_stats_dir,
	)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	write_manifest_json(result.manifests, output_path)
	_print_manifest_summary(result.summary(output_path=output_path))
	print(f'wrote manifest: {output_path}')


def _manifest_output_path(manifest_cfg: Mapping[str, Any]) -> Path:
	output_dir = Path(_required_str(manifest_cfg, 'output_dir'))
	output_name = _required_str(manifest_cfg, 'output_name')
	_require_bare_filename(output_name, 'manifest.output_name')
	return output_dir / output_name


def _required_mapping(parent: Mapping[str, object], key: str) -> Mapping[str, Any]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping'
		raise TypeError(msg)
	return value


def _required_str(parent: Mapping[str, object], key: str) -> str:
	value = parent.get(key)
	if not isinstance(value, str):
		msg = f'{key} must be a string; got {value!r}'
		raise TypeError(msg)
	return value


def _require_absolute_path(path: Path, label: str) -> None:
	if not path.is_absolute():
		msg = f'{label} must be an absolute artifact-registry path; got {path}'
		raise ValueError(msg)


def _require_bare_filename(value: str, label: str) -> None:
	path = Path(value)
	if (
		not value
		or path.is_absolute()
		or path.name != value
		or value in {'.', '..'}
		or '/' in value
		or '\\' in value
	):
		msg = f'{label} must be a bare filename; got {value!r}'
		raise ValueError(msg)


def _print_manifest_target(
	nopims_root: Path,
	input_path_list: Path,
	output_path: Path,
	normalization_stats_dir: Path,
) -> None:
	print(f'manifest.nopims_root: {nopims_root}')
	print(f'manifest.input_path_list: {input_path_list}')
	print(f'manifest.output_path: {output_path}')
	print(f'manifest.normalization_stats_dir: {normalization_stats_dir}')


def _print_manifest_summary(summary: ManifestBuildSummary) -> None:
	print(f'manifest.survey_count: {summary.survey_count}')
	print(f'manifest.amplitude_volume_count: {summary.amplitude_volume_count}')
	if summary.output_path is not None:
		print(f'manifest.output_path: {summary.output_path}')


if __name__ == '__main__':
	main()

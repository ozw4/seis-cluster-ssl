"""Compute amplitude-only NOPIMS normalization stats sidecars."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[2] / 'src'
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from seis_ssl_cluster.config import load_config, validate_config  # noqa: E402
from seis_ssl_cluster.data.normalization import (  # noqa: E402
	compute_normalization_stats,
	load_normalization_stats,
	write_normalization_stats,
)
from seis_ssl_cluster.data.schema import (  # noqa: E402
	GRID_ORDER_XYZ,
	AmplitudeVolumeRecord,
	SurveyManifest,
	read_manifest_json,
)
from seis_ssl_cluster.utils.cli import print_config_summary  # noqa: E402

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[1]
	/ 'configs'
	/ 'seis_ssl_cluster'
	/ 'prepare_nopims_normalization_stats.yaml'
)
MANIFEST_BUILD_HINT = (
	'hint: build the manifest with `python proc/seis_ssl_cluster/'
	'build_nopims_manifests.py --config proc/configs/seis_ssl_cluster/'
	'build_nopims_manifests.yaml`'
)


@dataclass(frozen=True)
class NormalizationTarget:
	"""Validated manifest target for one sidecar stats file."""

	survey_id: str
	amplitude: AmplitudeVolumeRecord
	output_path: Path


def main() -> None:
	"""Compute missing NOPIMS normalization stats sidecars."""
	args = _parse_args()
	config = validate_config(load_config(args.config))
	paths = _required_mapping(config, 'paths')
	artifact_root = Path(_required_str(paths, 'artifact_root'))
	nopims_root = Path(_required_str(paths, 'nopims_root'))
	manifest_path = _manifest_path(config)
	normalization_cfg = _required_mapping(config, 'normalization')
	clip_low, clip_high = _optional_percentiles(normalization_cfg)
	eps = _optional_positive_float(normalization_cfg, 'epsilon', default=1.0e-6)
	max_samples = _optional_positive_int(
		normalization_cfg,
		'max_samples',
		default=1_000_000,
	)
	seed = _optional_int(normalization_cfg, 'seed', default=42)
	_validate_disabled_normalization_options(normalization_cfg)

	if not manifest_path.is_file():
		if args.dry_run:
			print_config_summary(config)
			_print_missing_manifest_summary(
				manifest_path,
				max_samples=max_samples,
				seed=seed,
				overwrite=args.overwrite,
			)
			return
		msg = f'manifests.train does not exist: {manifest_path}. {MANIFEST_BUILD_HINT}'
		raise FileNotFoundError(msg)

	manifests = read_manifest_json(manifest_path)
	targets = [
		_normalization_target(
			manifest,
			artifact_root=artifact_root,
			nopims_root=nopims_root,
		)
		for manifest in manifests
	]
	if not args.overwrite:
		_validate_existing_stats(targets)
	existing_count = sum(target.output_path.is_file() for target in targets)
	missing_count = len(targets) - existing_count

	if args.dry_run:
		print_config_summary(config)
		print(f'normalization_stats.manifest_path: {manifest_path}')
		print('normalization_stats.manifest_exists: true')
		print(f'normalization_stats.manifest_entries: {len(targets)}')
		print(f'normalization_stats.existing_files: {existing_count}')
		print(f'normalization_stats.missing_files: {missing_count}')
		print(f'normalization_stats.max_samples: {max_samples}')
		print(f'normalization_stats.seed: {seed}')
		print(f'normalization_stats.overwrite: {str(args.overwrite).lower()}')
		print('normalization_stats.compute: skipped')
		return

	written_count = 0
	skipped_count = 0
	for target in targets:
		if target.output_path.is_file() and not args.overwrite:
			skipped_count += 1
			continue
		stats = compute_normalization_stats(
			target.amplitude.path,
			survey_id=target.survey_id,
			grid_order=target.amplitude.grid_order,
			clip_low_percentile=clip_low,
			clip_high_percentile=clip_high,
			max_samples=max_samples,
			seed=seed,
			eps=eps,
		)
		_validate_stats_belong_to_target(stats, target)
		write_normalization_stats(stats, target.output_path)
		written_count += 1

	print(f'normalization_stats.manifest_path: {manifest_path}')
	print(f'normalization_stats.manifest_entries: {len(targets)}')
	print(f'normalization_stats.written_files: {written_count}')
	print(f'normalization_stats.skipped_existing_files: {skipped_count}')


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description='Prepare amplitude-only NOPIMS normalization statistics.',
	)
	parser.add_argument(
		'--config',
		type=Path,
		default=DEFAULT_CONFIG,
		help='Path to a YAML configuration file.',
	)
	parser.add_argument(
		'--dry-run',
		action='store_true',
		help='Validate inputs and print a run summary without writing stats.',
	)
	parser.add_argument(
		'--overwrite',
		action='store_true',
		help='Recompute stats files even when sidecars already exist.',
	)
	return parser.parse_args()


def _manifest_path(config: Mapping[str, object]) -> Path:
	manifests = _required_mapping(config, 'manifests')
	return Path(_required_str(manifests, 'train'))


def _normalization_target(
	manifest: SurveyManifest,
	*,
	artifact_root: Path,
	nopims_root: Path,
) -> NormalizationTarget:
	amplitude = manifest.amplitude
	if not amplitude.path.is_file():
		msg = f'amplitude.path does not exist: {amplitude.path}'
		raise FileNotFoundError(msg)
	if amplitude.grid_order != GRID_ORDER_XYZ:
		msg = (
			f'amplitude.grid_order must be {list(GRID_ORDER_XYZ)!r}; '
			f'got {list(amplitude.grid_order)!r}'
		)
		raise ValueError(msg)
	output_path = amplitude.normalization_stats_path
	if not output_path.is_absolute():
		msg = (
			'amplitude.normalization_stats_path must be an absolute '
			f'artifact-registry path for {manifest.survey_id!r}; got '
			f'{output_path}'
		)
		raise ValueError(msg)
	if _is_relative_to(output_path, nopims_root):
		msg = (
			'amplitude.normalization_stats_path must not be under '
			f'paths.nopims_root for {manifest.survey_id!r}; got {output_path}'
		)
		raise ValueError(msg)
	if not _is_relative_to(output_path, artifact_root):
		msg = (
			'amplitude.normalization_stats_path must be under '
			f'paths.artifact_root ({artifact_root}) for {manifest.survey_id!r}; '
			f'got {output_path}'
		)
		raise ValueError(msg)
	return NormalizationTarget(
		survey_id=manifest.survey_id,
		amplitude=amplitude,
		output_path=output_path,
	)


def _is_relative_to(path: Path, root: Path) -> bool:
	try:
		path.resolve(strict=False).relative_to(root.resolve(strict=False))
	except ValueError:
		return False
	return True


def _validate_existing_stats(targets: list[NormalizationTarget]) -> None:
	for target in targets:
		if not target.output_path.is_file():
			continue
		stats = load_normalization_stats(target.output_path)
		_validate_stats_belong_to_target(stats, target)


def _validate_stats_belong_to_target(
	stats: object,
	target: NormalizationTarget,
) -> None:
	if not hasattr(stats, 'survey_id'):
		msg = f'invalid normalization stats object: {target.output_path}'
		raise TypeError(msg)
	if stats.survey_id != target.survey_id:
		msg = (
			f'normalization stats survey_id {stats.survey_id!r} does not match '
			f'manifest survey_id {target.survey_id!r}: {target.output_path}'
		)
		raise ValueError(msg)
	if stats.grid_order != target.amplitude.grid_order:
		msg = (
			'normalization stats grid_order does not match manifest '
			f'grid_order for {target.survey_id!r}: {target.output_path}'
		)
		raise ValueError(msg)
	if stats.source_path.resolve(strict=False) != target.amplitude.path.resolve(
		strict=False,
	):
		msg = (
			'normalization stats source_path does not match manifest '
			f'amplitude.path for {target.survey_id!r}: {target.output_path}'
		)
		raise ValueError(msg)


def _print_missing_manifest_summary(
	manifest_path: Path,
	*,
	max_samples: int | None,
	seed: int,
	overwrite: bool,
) -> None:
	print(f'normalization_stats.manifest_path: {manifest_path}')
	print('normalization_stats.manifest_exists: false')
	print('normalization_stats.manifest_entries: 0')
	print(f'normalization_stats.max_samples: {max_samples}')
	print(f'normalization_stats.seed: {seed}')
	print(f'normalization_stats.overwrite: {str(overwrite).lower()}')
	print('normalization_stats.compute: skipped')
	print(f'normalization_stats.message: manifest does not exist: {manifest_path}')
	print(MANIFEST_BUILD_HINT)


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


def _optional_int(
	parent: Mapping[str, object],
	key: str,
	*,
	default: int,
) -> int:
	value = parent.get(key, default)
	if isinstance(value, bool) or not isinstance(value, int):
		msg = f'{key} must be an integer; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_positive_int(
	parent: Mapping[str, object],
	key: str,
	*,
	default: int | None,
) -> int | None:
	value = parent.get(key, default)
	if value is None:
		return None
	if isinstance(value, bool) or not isinstance(value, int):
		msg = f'{key} must be an integer or null; got {value!r}'
		raise TypeError(msg)
	if value <= 0:
		msg = f'{key} must be positive when provided; got {value!r}'
		raise ValueError(msg)
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


def _optional_percentiles(parent: Mapping[str, object]) -> tuple[float, float]:
	value = parent.get('clipping_percentiles', [0.5, 99.5])
	if (
		not isinstance(value, list)
		or len(value) != 2
		or any(
			isinstance(item, bool) or not isinstance(item, int | float)
			for item in value
		)
	):
		msg = f'clipping_percentiles must contain two numbers; got {value!r}'
		raise TypeError(msg)
	low, high = (float(value[0]), float(value[1]))
	if not 0.0 <= low < high <= 100.0:
		msg = f'clipping_percentiles must satisfy 0 <= low < high <= 100; got {value!r}'
		raise ValueError(msg)
	return low, high


def _validate_disabled_normalization_options(
	normalization_cfg: Mapping[str, object],
) -> None:
	for key in (
		'smooth_time_depth_trend_correction',
		'trace_wise_agc',
		'patch_wise_zscore',
	):
		value = normalization_cfg.get(key, False)
		if value is not False:
			msg = f'normalization.{key} must be false for the amplitude-only MVP'
			raise ValueError(msg)


if __name__ == '__main__':
	main()

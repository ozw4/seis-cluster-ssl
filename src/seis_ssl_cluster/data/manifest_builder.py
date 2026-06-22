"""Build amplitude-only NOPIMS manifests from explicit `.npy` path lists."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from seis_ssl_cluster.data.path_list import (
	make_survey_id_from_path,
	resolve_npy_path_list,
)
from seis_ssl_cluster.data.schema import (
	GRID_ORDER_XYZ,
	AmplitudeVolumeRecord,
	SurveyManifest,
	write_manifest_json,
)
from seis_ssl_cluster.data.volume_store import inspect_npy_volume


@dataclass(frozen=True)
class ManifestBuildSummary:
	"""Compact summary of an amplitude manifest build."""

	survey_count: int
	amplitude_volume_count: int
	output_path: Path | None = None


@dataclass(frozen=True)
class ManifestBuildResult:
	"""Amplitude manifest build result."""

	manifests: list[SurveyManifest]

	def summary(self, output_path: Path | None = None) -> ManifestBuildSummary:
		"""Return aggregate counts."""
		return summarize_manifests(self.manifests, output_path=output_path)


def build_nopims_amplitude_manifests(
	nopims_root: str | Path,
	output_path: str | Path,
	input_path_list: str | Path,
	normalization_stats_dir: str | Path,
) -> list[SurveyManifest]:
	"""Build and write amplitude-only NOPIMS manifests from an explicit path-list."""
	result = scan_nopims_amplitude_manifests_from_path_list(
		nopims_root=nopims_root,
		input_path_list=input_path_list,
		normalization_stats_dir=normalization_stats_dir,
	)
	destination = Path(output_path)
	destination.parent.mkdir(parents=True, exist_ok=True)
	write_manifest_json(result.manifests, destination)
	return result.manifests


def build_nopims_manifests(
	nopims_root: str | Path,
	output_path: str | Path,
	input_path_list: str | Path,
	normalization_stats_dir: str | Path,
) -> list[SurveyManifest]:
	"""Build and write amplitude-only NOPIMS manifests from an explicit path-list."""
	return build_nopims_amplitude_manifests(
		nopims_root=nopims_root,
		output_path=output_path,
		input_path_list=input_path_list,
		normalization_stats_dir=normalization_stats_dir,
	)


def scan_nopims_amplitude_manifests_from_path_list(
	nopims_root: str | Path,
	input_path_list: str | Path,
	normalization_stats_dir: str | Path,
) -> ManifestBuildResult:
	"""Build amplitude manifests from a user-maintained `.npy` path-list."""
	root = Path(nopims_root)
	stats_dir = Path(normalization_stats_dir)
	if not stats_dir.is_absolute():
		msg = (
			'normalization_stats_dir must be an absolute artifact-registry path; '
			f'got {stats_dir}'
		)
		raise ValueError(msg)
	if _is_relative_to(stats_dir, root):
		msg = (
			'normalization_stats_dir must not be under nopims_root; '
			f'got {stats_dir}'
		)
		raise ValueError(msg)
	paths = resolve_npy_path_list(input_path_list, root)

	manifests: list[SurveyManifest] = []
	survey_ids: dict[str, Path] = {}
	for path in paths:
		survey_id = make_survey_id_from_path(path, root)
		if survey_id in survey_ids:
			msg = (
				f'duplicate generated survey_id {survey_id!r}: '
				f'{survey_ids[survey_id]} and {path}'
			)
			raise ValueError(msg)
		survey_ids[survey_id] = path

		info = inspect_npy_volume(path)
		record = AmplitudeVolumeRecord(
			survey_id=survey_id,
			path=path,
			shape_xyz=info.shape_xyz,
			dtype=info.dtype,
			grid_order=GRID_ORDER_XYZ,
			normalization_stats_path=stats_dir
			/ f'{survey_id}.normalization_stats.json',
		)
		manifest = SurveyManifest(
			survey_id=survey_id,
			root=path.parent,
			amplitude=record,
		)
		manifest.validate()
		manifests.append(manifest)

	return ManifestBuildResult(manifests=manifests)


def summarize_manifests(
	manifests: list[SurveyManifest],
	output_path: Path | None = None,
) -> ManifestBuildSummary:
	"""Summarize manifest coverage in path-list order."""
	return ManifestBuildSummary(
		survey_count=len(manifests),
		amplitude_volume_count=len(manifests),
		output_path=output_path,
	)


def _is_relative_to(path: Path, root: Path) -> bool:
	try:
		path.resolve(strict=False).relative_to(root.resolve(strict=False))
	except ValueError:
		return False
	return True


__all__ = [
	'ManifestBuildResult',
	'ManifestBuildSummary',
	'build_nopims_amplitude_manifests',
	'build_nopims_manifests',
	'scan_nopims_amplitude_manifests_from_path_list',
	'summarize_manifests',
]

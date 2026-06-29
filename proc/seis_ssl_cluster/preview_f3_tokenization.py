"""Create F3 facies benchmark teacher-slice tokenization previews."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from seis_ssl_cluster.config import (
	load_config,
	resolve_f3_facies_inspection_config,
)
from seis_ssl_cluster.config.schema import STAGE_F3_TOKENIZATION_PREVIEW
from seis_ssl_cluster.f3 import (
	F3TokenizationConfig,
	F3TokenizationFigureConfig,
	F3TokenizationOutputConfig,
	inspect_f3_png_labels,
	load_f3_label_consistency_alignments,
	write_f3_tokenization_preview_outputs,
)
from seis_ssl_cluster.utils.cli import parse_config_args

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'f3'
	/ 'facies_benchmark_v1'
	/ '00_inspection'
	/ '06_make_tokenization_preview.yaml'
)


def main() -> None:
	"""Write F3 teacher label tokenization preview figures and summaries."""
	args = parse_config_args(
		'Create F3 facies benchmark tokenization preview figures.',
		DEFAULT_CONFIG,
	)
	config = resolve_f3_facies_inspection_config(
		load_config(args.config),
		stage=STAGE_F3_TOKENIZATION_PREVIEW,
	)
	paths = _required_mapping(config, 'paths')
	inspection = _required_mapping(config, 'inspection')
	f3_root = Path(_required_str(paths, 'f3_root'))
	outputs = _output_config(inspection)
	tokenization = _tokenization_config(
		_required_mapping(inspection, 'tokenization'),
	)
	figure = _figure_config(_required_mapping(inspection, 'figure'))
	label_consistency_json = Path(_required_str(inspection, 'label_consistency_json'))
	png_extensions = _string_sequence(
		inspection.get('png_candidate_extensions', ['.png']),
		'inspection.png_candidate_extensions',
	)
	allow_unknown_colors = _optional_bool(
		inspection.get('allow_unknown_colors', False),
		'inspection.allow_unknown_colors',
	)

	if args.dry_run:
		_print_summary(
			config=config,
			f3_root=f3_root,
			outputs=outputs,
			label_consistency_json=label_consistency_json,
			tokenization=tokenization,
			figure=figure,
			png_extensions=png_extensions,
			allow_unknown_colors=allow_unknown_colors,
		)
		print('execution: dry-run; F3 tokenization preview skipped')
		return

	png_labels = inspect_f3_png_labels(
		f3_root,
		candidate_extensions=png_extensions,
		allow_unknown_colors=allow_unknown_colors,
	)
	alignments = load_f3_label_consistency_alignments(label_consistency_json)
	result = write_f3_tokenization_preview_outputs(
		png_labels,
		outputs,
		tokenization,
		alignments,
		figure,
	)
	print(f'f3_tokenization_preview.png_count: {len(result.png_paths)}')
	print(f'f3_tokenization_preview.sidecar_count: {len(result.sidecar_paths)}')
	print(f'wrote F3 tokenization metadata: {result.metadata_json}')
	print(f'wrote F3 tokenization summary csv: {result.summary_csv}')
	print(f'wrote F3 tokenization summary markdown: {result.summary_markdown}')
	print(f'wrote F3 tokenization directory: {outputs.tokenization_dir}')


def _output_config(inspection: Mapping[str, object]) -> F3TokenizationOutputConfig:
	return F3TokenizationOutputConfig(
		tokenization_dir=Path(_required_str(inspection, 'tokenization_dir')),
		metadata_json=Path(_required_str(inspection, 'metadata_json')),
		summary_csv=Path(_required_str(inspection, 'summary_csv')),
		summary_markdown=Path(_required_str(inspection, 'summary_markdown')),
	)


def _tokenization_config(
	tokenization: Mapping[str, object],
) -> F3TokenizationConfig:
	return F3TokenizationConfig(
		patch_size_xyz=_int_triplet(
			tokenization.get('patch_size_xyz'),
			'inspection.tokenization.patch_size_xyz',
		),
		min_labeled_fraction=_optional_fraction(
			tokenization.get('min_labeled_fraction', 0.5),
			'inspection.tokenization.min_labeled_fraction',
		),
		min_majority_fraction=_optional_fraction(
			tokenization.get('min_majority_fraction', 0.7),
			'inspection.tokenization.min_majority_fraction',
		),
	)


def _figure_config(figure: Mapping[str, object]) -> F3TokenizationFigureConfig:
	output_formats = _string_sequence(
		figure.get('output_formats', ['png']),
		'inspection.figure.output_formats',
	)
	if output_formats != ('png',):
		msg = 'inspection.figure.output_formats must be ["png"]'
		raise ValueError(msg)
	return F3TokenizationFigureConfig(
		dpi=_optional_positive_int(figure.get('dpi', 300), 'inspection.figure.dpi'),
		background=_optional_str(
			figure.get('background', 'white'),
			'inspection.figure.background',
		),
	)


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


def _optional_str(value: object, label: str) -> str:
	if not isinstance(value, str) or not value:
		msg = f'{label} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_bool(value: object, label: str) -> bool:
	if not isinstance(value, bool):
		msg = f'{label} must be a boolean; got {value!r}'
		raise TypeError(msg)
	return value


def _optional_positive_int(value: object, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
		msg = f'{label} must be a positive integer; got {value!r}'
		raise ValueError(msg)
	return value


def _optional_fraction(value: object, label: str) -> float:
	if not isinstance(value, int | float) or isinstance(value, bool):
		msg = f'{label} must be a number in [0, 1]; got {value!r}'
		raise TypeError(msg)
	fraction = float(value)
	if not 0.0 <= fraction <= 1.0:
		msg = f'{label} must be in [0, 1]; got {value!r}'
		raise ValueError(msg)
	return fraction


def _int_triplet(value: object, label: str) -> tuple[int, int, int]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		msg = f'{label} must be a three-item sequence; got {value!r}'
		raise TypeError(msg)
	values = tuple(value)
	if len(values) != 3:
		msg = f'{label} must contain three values; got {value!r}'
		raise ValueError(msg)
	for item in values:
		if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
			msg = f'{label} values must be positive integers; got {value!r}'
			raise ValueError(msg)
	return int(values[0]), int(values[1]), int(values[2])


def _string_sequence(value: object, label: str) -> tuple[str, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		msg = f'{label} must be a sequence of strings; got {value!r}'
		raise TypeError(msg)
	values = tuple(value)
	if not values or not all(isinstance(item, str) and item for item in values):
		msg = f'{label} must contain non-empty strings; got {value!r}'
		raise TypeError(msg)
	return values


def _print_summary(  # noqa: PLR0913
	*,
	config: Mapping[str, object],
	f3_root: Path,
	outputs: F3TokenizationOutputConfig,
	label_consistency_json: Path,
	tokenization: F3TokenizationConfig,
	figure: F3TokenizationFigureConfig,
	png_extensions: Sequence[str],
	allow_unknown_colors: bool,
) -> None:
	paths = _required_mapping(config, 'paths')
	output_root = _required_mapping(config, 'outputs')
	print(f'stage: {config.get("stage")}')
	print(f'paths.f3_root: {f3_root}')
	print(f'paths.artifact_root: {paths.get("artifact_root")}')
	print(f'outputs.inspection_dir: {output_root.get("inspection_dir")}')
	print(f'inspection.label_consistency_json: {label_consistency_json}')
	print(f'inspection.tokenization_dir: {outputs.tokenization_dir}')
	print(f'inspection.metadata_json: {outputs.metadata_json}')
	print(f'inspection.summary_csv: {outputs.summary_csv}')
	print(f'inspection.summary_markdown: {outputs.summary_markdown}')
	print(f'inspection.png_candidate_extensions: {", ".join(png_extensions)}')
	print(f'inspection.allow_unknown_colors: {allow_unknown_colors}')
	print(
		'inspection.tokenization.patch_size_xyz: '
		f'{list(tokenization.patch_size_xyz)}',
	)
	print(
		'inspection.tokenization.min_labeled_fraction: '
		f'{tokenization.min_labeled_fraction}',
	)
	print(
		'inspection.tokenization.min_majority_fraction: '
		f'{tokenization.min_majority_fraction}',
	)
	print(f'inspection.figure.dpi: {figure.dpi}')
	print(f'inspection.figure.background: {figure.background}')
	print('inspection.figure.output_formats: png')


if __name__ == '__main__':
	main()

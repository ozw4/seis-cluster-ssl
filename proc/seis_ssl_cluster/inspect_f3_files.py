"""Inspect the raw F3 facies benchmark file inventory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from seis_ssl_cluster.config import (
	load_config,
	resolve_f3_facies_inspection_config,
)
from seis_ssl_cluster.config.schema import STAGE_F3_INSPECT_FILES
from seis_ssl_cluster.f3 import (
	F3InventoryOutputConfig,
	scan_f3_file_inventory,
	write_f3_file_inventory_outputs,
)
from seis_ssl_cluster.utils.cli import parse_config_args

DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'f3'
	/ 'facies_benchmark_v1'
	/ '00_inspection'
	/ '01_inspect_files.yaml'
)


def main() -> None:
	"""Inspect the F3 raw directory and write inventory artifacts."""
	args = parse_config_args(
		'Inspect raw F3 facies benchmark files.',
		DEFAULT_CONFIG,
	)
	config = resolve_f3_facies_inspection_config(
		load_config(args.config),
		stage=STAGE_F3_INSPECT_FILES,
	)
	paths = _required_mapping(config, 'paths')
	inspection = _required_mapping(config, 'inspection')
	f3_root = Path(_required_str(paths, 'f3_root'))
	include_globs = _string_sequence(
		inspection.get('include_globs', ['**/*']),
		'inspection.include_globs',
	)
	exclude_globs = _string_sequence(
		inspection.get('exclude_globs', []),
		'inspection.exclude_globs',
	)
	outputs = _output_config(inspection)
	_validate_hash_files_disabled(inspection)

	if args.dry_run:
		_print_summary(
			config=config,
			f3_root=f3_root,
			outputs=outputs,
			include_globs=include_globs,
			exclude_globs=exclude_globs,
		)
		print('execution: dry-run; F3 file inspection skipped')
		return

	inventory = scan_f3_file_inventory(
		f3_root,
		include_globs=include_globs,
		exclude_globs=exclude_globs,
	)
	write_f3_file_inventory_outputs(inventory, outputs)
	print(f'f3_inventory.file_count: {len(inventory.files)}')
	print(
		'f3_inventory.label_png_count: '
		f'{inventory.category_counts()["label_png"]}',
	)
	print(f'f3_inventory.class_count: {len(inventory.classes)}')
	print(f'wrote file inventory: {outputs.file_inventory_json}')
	print(f'wrote markdown summary: {outputs.file_inventory_markdown}')
	print(f'wrote class info: {outputs.class_info_json}')
	print(f'wrote label PNG inventory: {outputs.label_png_inventory_csv}')


def _output_config(inspection: Mapping[str, object]) -> F3InventoryOutputConfig:
	return F3InventoryOutputConfig(
		file_inventory_json=Path(_required_str(inspection, 'output_json')),
		file_inventory_csv=Path(_required_str(inspection, 'output_csv')),
		file_inventory_markdown=Path(_required_str(inspection, 'output_markdown')),
		class_info_json=Path(_required_str(inspection, 'class_info_json')),
		label_png_inventory_csv=Path(
			_required_str(inspection, 'label_png_inventory_csv'),
		),
	)


def _validate_hash_files_disabled(inspection: Mapping[str, object]) -> None:
	hash_files = inspection.get('hash_files', False)
	if not isinstance(hash_files, bool):
		msg = f'inspection.hash_files must be a boolean; got {hash_files!r}'
		raise TypeError(msg)
	if hash_files:
		msg = 'inspection.hash_files is not supported by inspect_f3_files'
		raise ValueError(msg)


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


def _string_sequence(value: object, label: str) -> tuple[str, ...]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		msg = f'{label} must be a sequence of strings; got {value!r}'
		raise TypeError(msg)
	values = tuple(value)
	if not all(isinstance(item, str) for item in values):
		msg = f'{label} must be a sequence of strings; got {value!r}'
		raise TypeError(msg)
	return values


def _print_summary(
	*,
	config: Mapping[str, object],
	f3_root: Path,
	outputs: F3InventoryOutputConfig,
	include_globs: Sequence[str],
	exclude_globs: Sequence[str],
) -> None:
	paths = _required_mapping(config, 'paths')
	output_root = _required_mapping(config, 'outputs')
	print(f'stage: {config.get("stage")}')
	print(f'paths.f3_root: {f3_root}')
	print(f'paths.artifact_root: {paths.get("artifact_root")}')
	print(f'outputs.inspection_dir: {output_root.get("inspection_dir")}')
	print(f'inspection.include_globs: {", ".join(include_globs)}')
	print(f'inspection.exclude_globs: {", ".join(exclude_globs)}')
	print(f'inspection.output_json: {outputs.file_inventory_json}')
	print(f'inspection.output_csv: {outputs.file_inventory_csv}')
	print(f'inspection.output_markdown: {outputs.file_inventory_markdown}')
	print(f'inspection.class_info_json: {outputs.class_info_json}')
	print(
		'inspection.label_png_inventory_csv: '
		f'{outputs.label_png_inventory_csv}',
	)


if __name__ == '__main__':
	main()

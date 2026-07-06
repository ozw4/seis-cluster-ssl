from __future__ import annotations

from pathlib import Path

import seis_ssl_cluster.config.results as results_config
import seis_ssl_cluster.config.validate as validate_config
from seis_ssl_cluster.config.results import (
	DEFAULT_ALLOWED_SUFFIXES,
	DEFAULT_MAX_FILE_SIZE_BYTES,
	PublishItem,
	publish_selected_results,
	validate_results_artifacts,
)


def test_results_publish_exports_resolve_from_config_module(tmp_path: Path) -> None:
	source = tmp_path / 'artifacts' / 'seis_ssl_cluster' / 'report.md'
	source.parent.mkdir(parents=True)
	source.write_text('# report\n', encoding='utf-8')
	output_dir = tmp_path / 'results' / 'f3'

	manifest = publish_selected_results(
		items=(PublishItem(source, Path('report.md')),),
		output_dir=output_dir,
		allowed_suffixes=DEFAULT_ALLOWED_SUFFIXES,
		max_file_size_bytes=DEFAULT_MAX_FILE_SIZE_BYTES,
	)

	assert manifest.manifest_path.is_file()
	assert (output_dir / 'report.md').read_text(encoding='utf-8') == '# report\n'


def test_results_validation_exports_resolve_from_config_module(tmp_path: Path) -> None:
	root = tmp_path / 'results'
	report = root / 'f3' / 'facies_benchmark_v1' / 'inspection' / 'report.md'
	report.parent.mkdir(parents=True)
	report.write_text('# report\n', encoding='utf-8')

	result = validate_results_artifacts(
		root,
		required_files=(report.relative_to(root),),
	)

	assert result.ok
	assert result.file_count == 1


def test_results_entrypoints_reexport_from_validate_module() -> None:
	for name in (
		'DEFAULT_ALLOWED_SUFFIXES',
		'DEFAULT_MAX_FILE_SIZE_BYTES',
		'PublishItem',
		'publish_selected_results',
		'validate_results_artifacts',
	):
		assert getattr(validate_config, name) is getattr(results_config, name)

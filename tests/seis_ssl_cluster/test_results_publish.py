from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from seis_ssl_cluster.results import (
	DEFAULT_ALLOWED_SUFFIXES,
	DEFAULT_MAX_FILE_SIZE_BYTES,
	FORBIDDEN_SUFFIXES,
	PublishItem,
	publish_selected_results,
)


def test_publish_selected_results_copies_allowed_files_and_writes_manifest(
	tmp_path: Path,
) -> None:
	source_root = tmp_path / 'artifacts' / 'seis_ssl_cluster'
	report = _write_file(source_root / 'reports' / 'summary.md', b'# summary\n')
	metrics = _write_file(source_root / 'reports' / 'metrics.json', b'{"f1": 0.5}\n')
	output_dir = tmp_path / 'results' / 'f3'

	manifest = publish_selected_results(
		items=(
			PublishItem(report, Path('summary.md')),
			PublishItem(metrics, Path('tables/metrics.json')),
		),
		output_dir=output_dir,
		allowed_suffixes=DEFAULT_ALLOWED_SUFFIXES,
		max_file_size_bytes=DEFAULT_MAX_FILE_SIZE_BYTES,
	)

	assert (output_dir / 'summary.md').read_bytes() == report.read_bytes()
	assert (output_dir / 'tables' / 'metrics.json').read_bytes() == (
		metrics.read_bytes()
	)
	assert manifest.manifest_path == output_dir.resolve() / 'publish_manifest.json'
	assert len(manifest.items) == 2

	payload = json.loads(manifest.manifest_path.read_text(encoding='utf-8'))
	assert payload['source_artifact_root'] == str(source_root)
	assert payload['output_dir'] == str(output_dir)
	assert payload['skipped_optional_items'] == []
	assert payload['warnings'] == []
	assert payload['items'][0] == {
		'source': str(report),
		'target': 'summary.md',
		'size_bytes': report.stat().st_size,
		'sha256': hashlib.sha256(report.read_bytes()).hexdigest(),
	}


def test_publish_selected_results_rejects_forbidden_suffix(
	tmp_path: Path,
) -> None:
	checkpoint = _write_file(tmp_path / 'artifacts' / 'model.pt', b'heavy')

	with pytest.raises(ValueError, match='forbidden suffix'):
		publish_selected_results(
			items=(PublishItem(checkpoint, Path('model.pt')),),
			output_dir=tmp_path / 'results',
		)

	for suffix in FORBIDDEN_SUFFIXES:
		assert suffix not in DEFAULT_ALLOWED_SUFFIXES


def test_publish_selected_results_rejects_suffix_not_on_allowlist(
	tmp_path: Path,
) -> None:
	report = _write_file(tmp_path / 'artifacts' / 'report.html', b'<p>heavy</p>')

	with pytest.raises(ValueError, match='suffix is not allowed'):
		publish_selected_results(
			items=(PublishItem(report, Path('report.html')),),
			output_dir=tmp_path / 'results',
		)

	assert not (tmp_path / 'results' / 'report.html').exists()


def test_publish_selected_results_rejects_size_over_limit(
	tmp_path: Path,
) -> None:
	report = _write_file(tmp_path / 'artifacts' / 'large.txt', b'12345')

	with pytest.raises(ValueError, match='exceeds max_file_size_bytes'):
		publish_selected_results(
			items=(PublishItem(report, Path('large.txt')),),
			output_dir=tmp_path / 'results',
			max_file_size_bytes=4,
		)


def test_publish_selected_results_records_missing_optional_item(
	tmp_path: Path,
) -> None:
	missing = tmp_path / 'artifacts' / 'optional.csv'
	output_dir = tmp_path / 'results'

	manifest = publish_selected_results(
		items=(PublishItem(missing, Path('optional.csv'), required=False),),
		output_dir=output_dir,
	)

	assert manifest.items == []
	assert len(manifest.skipped_optional_items) == 1
	assert manifest.skipped_optional_items[0].reason == 'source_missing'
	payload = json.loads(manifest.manifest_path.read_text(encoding='utf-8'))
	assert payload['skipped_optional_items'] == [
		{
			'source': str(missing),
			'target': 'optional.csv',
			'reason': 'source_missing',
		}
	]
	assert payload['warnings'] == [f'optional publish source does not exist: {missing}']


def test_publish_selected_results_rejects_missing_required_item(
	tmp_path: Path,
) -> None:
	missing = tmp_path / 'artifacts' / 'required.md'

	with pytest.raises(FileNotFoundError, match='required publish source'):
		publish_selected_results(
			items=(PublishItem(missing, Path('required.md')),),
			output_dir=tmp_path / 'results',
		)


def test_publish_selected_results_rejects_target_outside_output_dir(
	tmp_path: Path,
) -> None:
	report = _write_file(tmp_path / 'artifacts' / 'summary.md', b'ok')

	with pytest.raises(ValueError, match='within output_dir'):
		publish_selected_results(
			items=(PublishItem(report, Path('../outside.md')),),
			output_dir=tmp_path / 'results',
		)


def test_publish_selected_results_rejects_existing_target_when_overwrite_disabled(
	tmp_path: Path,
) -> None:
	report = _write_file(tmp_path / 'artifacts' / 'summary.md', b'new')
	target = _write_file(tmp_path / 'results' / 'summary.md', b'existing')

	with pytest.raises(FileExistsError, match='already exists'):
		publish_selected_results(
			items=(PublishItem(report, Path('summary.md')),),
			output_dir=tmp_path / 'results',
			overwrite=False,
		)

	assert target.read_bytes() == b'existing'


def test_publish_selected_results_rejects_existing_target_directory(
	tmp_path: Path,
) -> None:
	report = _write_file(tmp_path / 'artifacts' / 'summary.md', b'new')
	target_dir = tmp_path / 'results' / 'summary.md'
	target_dir.mkdir(parents=True)

	with pytest.raises(IsADirectoryError, match='must not be a directory'):
		publish_selected_results(
			items=(PublishItem(report, Path('summary.md')),),
			output_dir=tmp_path / 'results',
			overwrite=True,
		)

	assert target_dir.is_dir()
	assert not (target_dir / report.name).exists()


def test_publish_selected_results_rejects_symlink_source(tmp_path: Path) -> None:
	real_report = _write_file(tmp_path / 'artifacts' / 'real.md', b'ok')
	link = tmp_path / 'artifacts' / 'linked.md'
	link.symlink_to(real_report)

	with pytest.raises(ValueError, match='must not be a symlink'):
		publish_selected_results(
			items=(PublishItem(link, Path('linked.md')),),
			output_dir=tmp_path / 'results',
		)


def _write_file(path: Path, content: bytes) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_bytes(content)
	return path

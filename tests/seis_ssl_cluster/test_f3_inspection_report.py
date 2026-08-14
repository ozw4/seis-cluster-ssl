from __future__ import annotations

import json
from pathlib import Path

import pytest

from seis_ssl_cluster.f3 import (
	READINESS_CAUTION,
	READINESS_PROCEED,
	READINESS_STOP,
	F3InspectionPublishConfig,
	F3InspectionReportConfig,
	build_f3_inspection_report,
	publish_f3_inspection_report,
)


def test_f3_inspection_report_outputs_markdown_json_and_missing_warning(
	tmp_path: Path,
) -> None:
	config = _report_config(tmp_path)
	_write_report_components(config, write_tokenization=False)
	_touch_figures(config)

	result = build_f3_inspection_report(config)

	report = json.loads(config.output_json.read_text(encoding='utf-8'))
	markdown = config.output_markdown.read_text(encoding='utf-8')

	assert result.report_markdown == config.output_markdown
	assert result.report_json == config.output_json
	assert config.output_markdown.is_file()
	assert config.output_json.is_file()
	assert report['downstream_readiness']['status'] == READINESS_CAUTION
	assert any(
		'missing input report component: tokenization_preview' in warning
		for warning in report['warnings']
	)
	for section in (
		'## 1. Dataset files',
		'## 2. Volume geometry',
		'## 3. Seismic amplitude statistics',
		'## 4. Facies classes',
		'## 5. Train/validation labels',
		'## 6. PNG vs SEGY label consistency',
		'## 7. Quicklook figures',
		'## 8. Tokenization preview',
		'## 9. Readiness for downstream',
	):
		assert section in markdown
	assert '[quicklook/seismic/seismic_xz_y_mid.png]' in markdown
	assert '(quicklook/overlays/train_inline_0250_overlay.png)' in markdown
	assert str(tmp_path) not in _quicklook_section(markdown)


def test_f3_inspection_report_readiness_uses_consistency_and_tokenization(
	tmp_path: Path,
) -> None:
	config = _report_config(tmp_path)
	_write_report_components(
		config,
		consistency_passed=True,
		retained_tokens=10,
		dropped_tokens=0,
		ambiguous_tokens=0,
	)
	_touch_figures(config)

	proceed = build_f3_inspection_report(config).payload

	_write_json(
		config.label_consistency_json,
		_label_consistency_payload(passed=True, border_only_mismatch=True),
	)
	caution = build_f3_inspection_report(config).payload

	_write_json(
		config.label_consistency_json,
		_label_consistency_payload(passed=False),
	)
	stop = build_f3_inspection_report(config).payload

	_write_json(
		config.label_consistency_json,
		_label_consistency_payload(passed=True),
	)
	_write_json(
		config.tokenization_preview_json,
		_tokenization_payload(
			retained_tokens=0,
			dropped_tokens=10,
			ambiguous_tokens=10,
		),
	)
	stop_from_tokens = build_f3_inspection_report(config).payload

	_write_json(
		config.tokenization_preview_json,
		_tokenization_payload(
			retained_tokens=0,
			dropped_tokens=0,
			ambiguous_tokens=0,
		),
	)
	stop_from_zero_total_tokens = build_f3_inspection_report(config).payload

	assert proceed['downstream_readiness']['status'] == READINESS_PROCEED
	assert caution['downstream_readiness']['status'] == READINESS_CAUTION
	assert any(
		'ignored z-border samples' in reason
		for reason in caution['downstream_readiness']['reasons']
	)
	assert stop['downstream_readiness']['status'] == READINESS_STOP
	assert stop_from_tokens['downstream_readiness']['status'] == READINESS_STOP
	assert (
		stop_from_zero_total_tokens['downstream_readiness']['status']
		== READINESS_STOP
	)


def test_f3_inspection_report_publish_enabled_writes_lightweight_results(
	tmp_path: Path,
) -> None:
	config = _report_config(tmp_path / 'artifacts' / 'seis_ssl_cluster')
	_write_report_components(config)
	_touch_figures(config)
	consistency_figure = _write_file(
		config.inspection_dir
		/ 'quicklook'
		/ 'consistency'
		/ 'train_inline_0250_mismatch.png',
		b'consistency-png',
	)
	_write_file(
		config.inspection_dir / 'quicklook' / 'overlays' / 'extra_overlay.png',
		b'extra-png',
	)
	_write_file(config.inspection_dir / 'checkpoint.pt', b'heavy')
	_write_file(config.inspection_dir / 'embeddings.npy', b'heavy')
	output_dir = (
		tmp_path / 'reports' / 'f3' / 'facies_benchmark_v1' / 'inspection'
	)

	result = build_f3_inspection_report(
		config,
		publish_config=F3InspectionPublishConfig(
			enabled=True,
			output_dir=output_dir,
			include_figures=True,
			max_file_size_bytes=10 * 1024 * 1024,
		),
	)

	assert result.published_files
	expected_files = {
		Path('report.md'),
		Path('report.json'),
		Path('figures/seismic_xz_y_mid.png'),
		Path('figures/train_inline_0250_overlay.png'),
		Path('figures/train_inline_0250_tokenization.png'),
		Path('figures/label_consistency_example.png'),
	}
	published_files = {
		path.relative_to(output_dir)
		for path in output_dir.rglob('*')
		if path.is_file()
	}
	assert published_files == expected_files
	assert {
		path.relative_to(output_dir) for path in result.published_files
	} == expected_files
	assert (
		output_dir / 'figures' / 'label_consistency_example.png'
	).read_bytes() == consistency_figure.read_bytes()
	assert not any(path.suffix in {'.pt', '.npy'} for path in output_dir.rglob('*'))

	_assert_published_markdown_is_lightweight(output_dir, tmp_path)
	_assert_published_json_is_lightweight(output_dir, tmp_path)

def test_f3_inspection_report_publish_disabled_writes_no_results(
	tmp_path: Path,
) -> None:
	config = _report_config(tmp_path)
	_write_report_components(config)
	_touch_figures(config)
	output_dir = tmp_path / 'reports'

	result = build_f3_inspection_report(
		config,
		publish_config=F3InspectionPublishConfig(
			enabled=False,
			output_dir=output_dir,
		),
	)

	assert result.published_files == ()
	assert not output_dir.exists()


def test_f3_inspection_report_publish_include_figures_false_omits_links(
	tmp_path: Path,
) -> None:
	config = _report_config(tmp_path)
	_write_report_components(config)
	_touch_figures(config)
	output_dir = tmp_path / 'reports'

	result = build_f3_inspection_report(
		config,
		publish_config=F3InspectionPublishConfig(
			enabled=True,
			output_dir=output_dir,
			include_figures=False,
		),
	)

	assert {path.relative_to(output_dir) for path in result.published_files} == {
		Path('report.md'),
		Path('report.json'),
	}
	assert (output_dir / 'report.md').is_file()
	assert (output_dir / 'report.json').is_file()
	assert not (output_dir / 'figures').exists()
	markdown = (output_dir / 'report.md').read_text(encoding='utf-8')
	assert 'quicklook/' not in _quicklook_section(markdown)
	published_payload = json.loads(
		(output_dir / 'report.json').read_text(encoding='utf-8'),
	)
	assert published_payload['quicklook_figures'] == []


def test_f3_inspection_report_publish_missing_optional_figure_warns(
	tmp_path: Path,
) -> None:
	config = _report_config(tmp_path)
	_write_report_components(config)
	output_dir = tmp_path / 'reports'

	result = build_f3_inspection_report(
		config,
		publish_config=F3InspectionPublishConfig(
			enabled=True,
			output_dir=output_dir,
			include_figures=True,
		),
	)

	assert {path.relative_to(output_dir) for path in result.published_files} == {
		Path('report.md'),
		Path('report.json'),
	}
	assert (output_dir / 'report.md').is_file()
	assert (output_dir / 'report.json').is_file()
	assert not (output_dir / 'figures').exists()


def test_f3_inspection_report_publish_requires_report_files(tmp_path: Path) -> None:
	config = _report_config(tmp_path)
	output_dir = tmp_path / 'reports'

	with pytest.raises(FileNotFoundError, match='required publish source'):
		publish_f3_inspection_report(
			config,
			F3InspectionPublishConfig(
				enabled=True,
				output_dir=output_dir,
			),
		)
	assert not output_dir.exists()


def test_f3_inspection_publish_validates_all_required_sources_before_writing(
	tmp_path: Path,
) -> None:
	config = _report_config(tmp_path)
	_write_file(config.output_markdown, b'artifact report\n')
	output_dir = tmp_path / 'reports'

	with pytest.raises(FileNotFoundError, match='required publish source'):
		publish_f3_inspection_report(
			config,
			F3InspectionPublishConfig(enabled=True, output_dir=output_dir),
			payload={},
		)

	assert not output_dir.exists()


def test_f3_inspection_publish_rejects_optional_figure_symlink_before_writing(
	tmp_path: Path,
) -> None:
	config = _report_config(tmp_path)
	_write_file(config.output_markdown, b'artifact report\n')
	_write_json(config.output_json, {})
	figure = config.inspection_dir / config.figure_paths[0]
	figure.parent.mkdir(parents=True, exist_ok=True)
	figure.symlink_to(_write_file(tmp_path / 'outside.png', b'png'))
	output_dir = tmp_path / 'reports'

	with pytest.raises(FileNotFoundError, match='regular file'):
		publish_f3_inspection_report(
			config,
			F3InspectionPublishConfig(enabled=True, output_dir=output_dir),
			payload={},
		)

	assert not output_dir.exists()


def test_f3_inspection_publish_rejects_size_before_writing(tmp_path: Path) -> None:
	config = _report_config(tmp_path)
	_write_file(config.output_markdown, b'artifact report\n')
	_write_json(config.output_json, {})
	output_dir = tmp_path / 'reports'

	with pytest.raises(ValueError, match='exceeds max_file_size_bytes'):
		publish_f3_inspection_report(
			config,
			F3InspectionPublishConfig(
				enabled=True,
				output_dir=output_dir,
				max_file_size_bytes=1,
			),
			payload={},
		)

	assert not output_dir.exists()


@pytest.mark.parametrize('target_kind', ['symlink', 'directory'])
def test_f3_inspection_publish_rejects_unsafe_target_before_writing(
	tmp_path: Path,
	target_kind: str,
) -> None:
	config = _report_config(tmp_path)
	_write_file(config.output_markdown, b'artifact report\n')
	_write_json(config.output_json, {})
	output_dir = tmp_path / 'reports'
	unsafe_target = output_dir / 'report.json'
	unsafe_target.parent.mkdir(parents=True)
	if target_kind == 'symlink':
		unsafe_target.symlink_to(_write_file(tmp_path / 'outside.json', b'{}\n'))
	else:
		unsafe_target.mkdir()

	exception = ValueError if target_kind == 'symlink' else IsADirectoryError
	with pytest.raises(exception):
		publish_f3_inspection_report(
			config,
			F3InspectionPublishConfig(enabled=True, output_dir=output_dir),
			payload={},
		)

	assert not (output_dir / 'report.md').exists()


@pytest.mark.parametrize('output_kind', ['symlink', 'file'])
def test_f3_inspection_publish_rejects_unsafe_output_dir_before_writing(
	tmp_path: Path,
	output_kind: str,
) -> None:
	config = _report_config(tmp_path)
	_write_file(config.output_markdown, b'artifact report\n')
	_write_json(config.output_json, {})
	outside = tmp_path / 'outside'
	outside.mkdir()
	output_dir = tmp_path / 'reports'
	if output_kind == 'symlink':
		output_dir.symlink_to(outside, target_is_directory=True)
	else:
		output_dir.write_text('not a directory\n', encoding='utf-8')

	exception = ValueError if output_kind == 'symlink' else NotADirectoryError
	with pytest.raises(exception):
		publish_f3_inspection_report(
			config,
			F3InspectionPublishConfig(enabled=True, output_dir=output_dir),
			payload={},
		)

	assert not any(outside.iterdir())


@pytest.mark.parametrize('parent_kind', ['symlink', 'file'])
def test_f3_inspection_publish_rejects_unsafe_figure_parent_before_writing(
	tmp_path: Path,
	parent_kind: str,
) -> None:
	config = _report_config(tmp_path)
	_write_file(config.output_markdown, b'artifact report\n')
	_write_json(config.output_json, {})
	_write_file(config.inspection_dir / config.figure_paths[0], b'png')
	output_dir = tmp_path / 'reports'
	output_dir.mkdir()
	figures_dir = output_dir / 'figures'
	outside = tmp_path / 'outside'
	outside.mkdir()
	if parent_kind == 'symlink':
		figures_dir.symlink_to(outside, target_is_directory=True)
	else:
		figures_dir.write_text('not a directory\n', encoding='utf-8')

	exception = ValueError if parent_kind == 'symlink' else NotADirectoryError
	with pytest.raises(exception):
		publish_f3_inspection_report(
			config,
			F3InspectionPublishConfig(enabled=True, output_dir=output_dir),
			payload={},
		)

	assert not (output_dir / 'report.md').exists()
	assert not (output_dir / 'report.json').exists()
	assert not any(outside.iterdir())


def test_f3_inspection_publish_rejects_source_target_collision(tmp_path: Path) -> None:
	config = _report_config(tmp_path)
	_write_file(config.output_markdown, b'artifact report\n')
	_write_json(config.output_json, {})
	before = config.output_markdown.read_bytes()

	with pytest.raises(ValueError, match='publish target must differ from source'):
		publish_f3_inspection_report(
			config,
			F3InspectionPublishConfig(
				enabled=True,
				output_dir=config.output_markdown.parent,
			),
			payload={},
		)

	assert config.output_markdown.read_bytes() == before


def _report_config(root: Path) -> F3InspectionReportConfig:
	inspection_dir = root / 'inspection' / 'f3' / 'facies_benchmark_v1'
	return F3InspectionReportConfig(
		inspection_dir=inspection_dir,
		file_inventory_json=inspection_dir / 'inventory' / 'file_inventory.json',
		class_info_json=inspection_dir / 'inventory' / 'class_info.json',
		segy_geometry_json=inspection_dir / 'segy' / 'segy_geometry.json',
		seismic_amplitude_stats_json=(
			inspection_dir / 'segy' / 'seismic_amplitude_stats.json'
		),
		label_unique_values_json=inspection_dir / 'segy' / 'label_unique_values.json',
		png_label_summary_json=inspection_dir / 'labels' / 'png_label_summary.json',
		png_label_inventory_json=(
			inspection_dir / 'labels' / 'png_label_inventory.json'
		),
		quicklook_metadata_json=inspection_dir / 'stats' / 'quicklook_metadata.json',
		label_consistency_json=inspection_dir / 'stats' / 'label_consistency.json',
		tokenization_preview_json=(
			inspection_dir / 'stats' / 'tokenization_preview.json'
		),
		output_markdown=inspection_dir / 'report.md',
		output_json=inspection_dir / 'report.json',
		figure_paths=(
			Path('quicklook/seismic/seismic_xz_y_mid.png'),
			Path('quicklook/overlays/train_inline_0250_overlay.png'),
			Path('quicklook/tokenization/train_inline_0250_tokenization.png'),
		),
	)


def _assert_published_markdown_is_lightweight(
	output_dir: Path,
	tmp_path: Path,
) -> None:
	published_markdown = (output_dir / 'report.md').read_text(encoding='utf-8')
	assert '(figures/seismic_xz_y_mid.png)' in published_markdown
	assert '(figures/train_inline_0250_overlay.png)' in published_markdown
	assert '(figures/train_inline_0250_tokenization.png)' in published_markdown
	assert '(quicklook/seismic/seismic_xz_y_mid.png)' not in published_markdown
	assert 'publish用の軽量reportでは省略' in published_markdown
	assert 'interpretation/train/' not in published_markdown
	assert 'interpretation/validation/' not in published_markdown
	assert 'f3_seismic.segy' not in published_markdown
	assert 'f3_labels.segy' not in published_markdown
	assert '/data/F3' not in published_markdown
	assert str(tmp_path) not in _quicklook_section(published_markdown)


def _assert_published_json_is_lightweight(
	output_dir: Path,
	tmp_path: Path,
) -> None:
	published_json_text = (output_dir / 'report.json').read_text(encoding='utf-8')
	published_json = json.loads(published_json_text)
	assert 'inspection_dir' not in published_json
	assert 'outputs' not in published_json
	assert all(
		'path' not in item
		for item in published_json['component_status']
	)
	assert 'seismic_segy' not in published_json['dataset_files']
	assert 'label_segy' not in published_json['dataset_files']
	assert 'class_info' not in published_json['dataset_files']
	assert 'seismic_path' not in published_json['volume_geometry']
	assert 'label_path' not in published_json['volume_geometry']
	assert 'slices' not in published_json['train_validation_labels']
	assert published_json['train_validation_labels']['slice_list_omitted'] is True
	assert published_json['quicklook_figures'][0]['path'] == (
		'figures/seismic_xz_y_mid.png'
	)
	assert 'interpretation/train/' not in published_json_text
	assert 'interpretation/validation/' not in published_json_text
	assert 'f3_seismic.segy' not in published_json_text
	assert 'f3_labels.segy' not in published_json_text
	assert '/data/F3' not in published_json_text
	assert str(tmp_path) not in published_json_text


def _write_report_components(  # noqa: PLR0913
	config: F3InspectionReportConfig,
	*,
	write_tokenization: bool = True,
	consistency_passed: bool = True,
	retained_tokens: int = 8,
	dropped_tokens: int = 2,
	ambiguous_tokens: int = 1,
) -> None:
	_write_json(config.file_inventory_json, _file_inventory_payload())
	_write_json(config.class_info_json, _class_info_payload())
	_write_json(config.segy_geometry_json, _geometry_payload())
	_write_json(config.seismic_amplitude_stats_json, _amplitude_payload())
	_write_json(config.label_unique_values_json, _label_unique_values_payload())
	_write_json(config.png_label_summary_json, _png_label_summary_payload())
	_write_json(config.png_label_inventory_json, {'files': []})
	_write_json(config.quicklook_metadata_json, {'outputs': []})
	_write_json(
		config.label_consistency_json,
		_label_consistency_payload(passed=consistency_passed),
	)
	if write_tokenization:
		_write_json(
			config.tokenization_preview_json,
			_tokenization_payload(
				retained_tokens=retained_tokens,
				dropped_tokens=dropped_tokens,
				ambiguous_tokens=ambiguous_tokens,
			),
		)


def _touch_figures(config: F3InspectionReportConfig) -> None:
	for path in config.figure_paths:
		figure_path = config.inspection_dir / path
		figure_path.parent.mkdir(parents=True, exist_ok=True)
		figure_path.write_bytes(b'fake-png')


def _file_inventory_payload() -> dict[str, object]:
	return {
		'category_counts': {
			'seismic_segy': 1,
			'label_segy': 1,
			'class_info': 1,
			'label_png': 2,
		},
		'split_counts': {'train': 1, 'validation': 1},
		'files': [
			{
				'relative_path': 'seismic/f3_seismic.segy',
				'category': 'seismic_segy',
			},
			{'relative_path': 'seismic/f3_labels.segy', 'category': 'label_segy'},
			{
				'relative_path': 'interpretation/class_info.json',
				'category': 'class_info',
			},
			{
				'relative_path': 'interpretation/train/train_inline_0250.png',
				'category': 'label_png',
				'split': 'train',
				'slice_type': 'inline',
				'slice_index': 250,
			},
			{
				'relative_path': (
					'interpretation/validation/validation_crossline_0300.png'
				),
				'category': 'label_png',
				'split': 'validation',
				'slice_type': 'crossline',
				'slice_index': 300,
			},
		],
		'warnings': [],
	}


def _class_info_payload() -> dict[str, object]:
	return {
		'classes': [
			{
				'class_id': 0,
				'class_name': 'Background',
				'rgb': [1, 2, 3],
				'hex_color': '#010203',
			},
			{
				'class_id': 1,
				'class_name': 'Sand',
				'rgb': [4, 5, 6],
				'hex_color': '#040506',
			},
		],
	}


def _geometry_payload() -> dict[str, object]:
	geometry = {
		'path': '/data/F3/f3_seismic.segy',
		'cube_shape': [4, 5, 6],
		'iline_count': 4,
		'iline_min': 100,
		'iline_max': 103,
		'xline_count': 5,
		'xline_min': 200,
		'xline_max': 204,
		'sample_count': 6,
		'sample_min': 0,
		'sample_max': 20,
	}
	return {
		'shape_consistency': {
			'seismic_cube_shape': [4, 5, 6],
			'label_cube_shape': [4, 5, 6],
			'matches': True,
		},
		'segy_files': {'seismic': geometry, 'label': geometry},
	}


def _amplitude_payload() -> dict[str, object]:
	return {
		'stats': {
			'finite_count': 120,
			'nonfinite_count': 0,
			'zero_count': 3,
			'min': -2.0,
			'p1': -1.0,
			'p50': 0.0,
			'p99': 1.0,
			'max': 2.0,
		},
	}


def _label_unique_values_payload() -> dict[str, object]:
	return {
		'stats': {
			'class_info': {
				'classes': [
					{'class_id': 0, 'count': 70},
					{'class_id': 1, 'count': 50},
				],
			},
		},
	}


def _png_label_summary_payload() -> dict[str, object]:
	class_counts = [
		{
			'class_id': 0,
			'class_name': 'Background',
			'pixel_count': 20,
			'fraction': 0.5,
		},
		{'class_id': 1, 'class_name': 'Sand', 'pixel_count': 20, 'fraction': 0.5},
	]
	return {
		'file_count': 2,
		'total_pixels': 40,
		'total_unknown_pixels': 0,
		'overall_class_counts': class_counts,
		'splits': {
			'train': {
				'file_count': 1,
				'total_pixels': 20,
				'unknown_pixel_count': 0,
				'class_counts': class_counts,
			},
			'validation': {
				'file_count': 1,
				'total_pixels': 20,
				'unknown_pixel_count': 0,
				'class_counts': class_counts,
			},
		},
		'files': [
			{
				'relative_path': 'interpretation/train/train_inline_0250.png',
				'split': 'train',
				'slice_type': 'inline',
				'slice_index': 250,
			},
			{
				'relative_path': (
					'interpretation/validation/validation_crossline_0300.png'
				),
				'split': 'validation',
				'slice_type': 'crossline',
				'slice_index': 300,
			},
		],
		'warnings': [],
	}


def _label_consistency_payload(
	*,
	passed: bool,
	border_only_mismatch: bool = False,
) -> dict[str, object]:
	files = []
	if border_only_mismatch:
		files.append(
			{
				'border_only_mismatch': True,
				'mismatch_rate': 0.2,
				'effective_mismatch_rate': 0.0,
			},
		)
	return {
		'passed': passed,
		'png_label_file_count': 2,
		'max_mismatch_rate': 0.001,
		'ignore_border_samples_z': 1 if border_only_mismatch else 0,
		'max_observed_mismatch_rate': 0.0 if passed else 0.2,
		'max_observed_effective_mismatch_rate': (
			0.0 if passed else 0.2
		),
		'total_mismatch_pixel_count': 0 if passed else 8,
		'warnings': [],
		'files': files,
	}


def _tokenization_payload(
	*,
	retained_tokens: int,
	dropped_tokens: int,
	ambiguous_tokens: int,
) -> dict[str, object]:
	total_tokens = retained_tokens + dropped_tokens
	return {
		'tokenization_config': {
			'patch_size_xyz': [8, 8, 8],
			'min_labeled_fraction': 0.5,
			'min_majority_fraction': 0.7,
		},
		'overall_summary': {
			'total_tokens': total_tokens,
			'retained_tokens': retained_tokens,
			'dropped_tokens': dropped_tokens,
			'ambiguous_token_count': ambiguous_tokens,
			'empty_token_count': max(0, dropped_tokens - ambiguous_tokens),
			'class_counts_retained': {'0': retained_tokens},
		},
		'outputs': [],
	}


def _write_json(path: Path, payload: dict[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n',
		encoding='utf-8',
	)


def _write_file(path: Path, content: bytes) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_bytes(content)
	return path


def _quicklook_section(markdown: str) -> str:
	start = markdown.index('## 7. Quicklook figures')
	stop = markdown.index('## 8. Tokenization preview')
	return markdown[start:stop]

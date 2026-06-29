"""Aggregate F3 inspection artifacts into a Japanese report."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

READINESS_PROCEED = 'proceed'
READINESS_CAUTION = 'caution'
READINESS_STOP = 'stop'

_DEFAULT_FIGURE_PATHS = (
	Path('quicklook/seismic/seismic_xz_y_mid.png'),
	Path('quicklook/overlays/train_inline_0250_overlay.png'),
	Path('quicklook/tokenization/train_inline_0250_tokenization.png'),
)
_REPORT_COMPONENTS = (
	'file_inventory',
	'class_info',
	'segy_geometry',
	'seismic_amplitude_stats',
	'label_unique_values',
	'png_label_summary',
	'png_label_inventory',
	'quicklook_metadata',
	'label_consistency',
	'tokenization_preview',
)


@dataclass(frozen=True)
class F3InspectionReportConfig:
	"""Input and output paths for the F3 inspection report builder."""

	inspection_dir: Path
	file_inventory_json: Path
	class_info_json: Path
	segy_geometry_json: Path
	seismic_amplitude_stats_json: Path
	label_unique_values_json: Path
	png_label_summary_json: Path
	png_label_inventory_json: Path
	quicklook_metadata_json: Path
	label_consistency_json: Path
	tokenization_preview_json: Path
	output_markdown: Path
	output_json: Path
	figure_paths: tuple[Path, ...] = _DEFAULT_FIGURE_PATHS


@dataclass(frozen=True)
class F3InspectionReportResult:
	"""Paths and payload written by the F3 inspection report builder."""

	report_markdown: Path
	report_json: Path
	payload: dict[str, object]


def build_f3_inspection_report(
	config: F3InspectionReportConfig,
) -> F3InspectionReportResult:
	"""Build and write the F3 inspection Markdown and JSON reports."""
	components, component_status, warnings = _load_report_components(config)
	payload = _report_payload(
		config,
		components=components,
		component_status=component_status,
		warnings=warnings,
	)
	_write_json(config.output_json, payload)
	_write_text(config.output_markdown, render_f3_inspection_report_markdown(payload))
	return F3InspectionReportResult(
		report_markdown=config.output_markdown,
		report_json=config.output_json,
		payload=payload,
	)


def render_f3_inspection_report_markdown(
	payload: Mapping[str, object],
) -> str:
	"""Render the machine-readable report payload as Japanese Markdown."""
	dataset_files = _mapping(payload.get('dataset_files'))
	geometry = _mapping(payload.get('volume_geometry'))
	amplitude = _mapping(payload.get('seismic_amplitude_statistics'))
	classes = _sequence_of_mappings(payload.get('facies_classes'))
	labels = _mapping(payload.get('train_validation_labels'))
	consistency = _mapping(payload.get('label_consistency'))
	figures = _sequence_of_mappings(payload.get('quicklook_figures'))
	tokenization = _mapping(payload.get('tokenization_preview'))
	readiness = _mapping(payload.get('downstream_readiness'))
	warnings = _string_list(payload.get('warnings'))

	lines = [
		'# F3 facies benchmark inspection report',
		'',
		'このreportはF3 facies benchmark inspectionの出力を統合し、'
		'少量教師の岩相判別MVPへ進むための判断材料をまとめる。',
		'',
		'## 1. Dataset files',
		'',
		*_render_dataset_files(dataset_files),
		'',
		'## 2. Volume geometry',
		'',
		*_render_geometry(geometry),
		'',
		'## 3. Seismic amplitude statistics',
		'',
		*_render_amplitude(amplitude),
		'',
		'## 4. Facies classes',
		'',
		'| class ID | class name | RGB color | pixel count | voxel count |',
		'|---:|---|---|---:|---:|',
	]
	lines.extend(_render_class_rows(classes))
	lines.extend(
		[
			'',
			'## 5. Train/validation labels',
			'',
			*_render_train_validation_labels(labels),
			'',
			'## 6. PNG vs SEGY label consistency',
			'',
			*_render_consistency(consistency),
			'',
			'## 7. Quicklook figures',
			'',
			*_render_figures(figures),
			'',
			'## 8. Tokenization preview',
			'',
			*_render_tokenization(tokenization),
			'',
			'## 9. Readiness for downstream',
			'',
			*_render_readiness(readiness),
			'',
			'## Warnings',
			'',
		],
	)
	if warnings:
		lines.extend(f'- {warning}' for warning in warnings)
	else:
		lines.append('- none')
	return '\n'.join(lines) + '\n'


def _load_report_components(
	config: F3InspectionReportConfig,
) -> tuple[
	dict[str, Mapping[str, object] | None],
	list[dict[str, object]],
	list[str],
]:
	path_by_component = {
		'file_inventory': config.file_inventory_json,
		'class_info': config.class_info_json,
		'segy_geometry': config.segy_geometry_json,
		'seismic_amplitude_stats': config.seismic_amplitude_stats_json,
		'label_unique_values': config.label_unique_values_json,
		'png_label_summary': config.png_label_summary_json,
		'png_label_inventory': config.png_label_inventory_json,
		'quicklook_metadata': config.quicklook_metadata_json,
		'label_consistency': config.label_consistency_json,
		'tokenization_preview': config.tokenization_preview_json,
	}
	components: dict[str, Mapping[str, object] | None] = {}
	status: list[dict[str, object]] = []
	warnings: list[str] = []
	for name in _REPORT_COMPONENTS:
		path = path_by_component[name]
		payload = _read_json_component(name, path, warnings)
		components[name] = payload
		status.append(
			{
				'name': name,
				'path': str(path),
				'available': payload is not None,
			},
		)
	return components, status, warnings


def _read_json_component(
	name: str,
	path: Path,
	warnings: list[str],
) -> Mapping[str, object] | None:
	if not path.is_file():
		warnings.append(f'missing input report component: {name} ({path})')
		return None
	try:
		payload = json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as exc:
		warnings.append(
			f'invalid input report component JSON: {name} ({path}): {exc.msg}',
		)
		return None
	if not isinstance(payload, Mapping):
		warnings.append(f'input report component is not a JSON object: {name} ({path})')
		return None
	return payload


def _report_payload(
	config: F3InspectionReportConfig,
	*,
	components: Mapping[str, Mapping[str, object] | None],
	component_status: Sequence[Mapping[str, object]],
	warnings: Sequence[str],
) -> dict[str, object]:
	dataset_files = _dataset_files_summary(components)
	geometry = _geometry_summary(components)
	amplitude = _amplitude_summary(components)
	classes = _class_summary(components)
	labels = _label_summary(components)
	consistency = _consistency_summary(components)
	figures, figure_warnings = _figure_summary(config)
	tokenization = _tokenization_summary(components)
	report_warnings = [
		*warnings,
		*_component_warnings(components),
		*figure_warnings,
	]
	readiness = _readiness_summary(
		component_status,
		amplitude=amplitude,
		labels=labels,
		consistency=consistency,
		tokenization=tokenization,
	)
	return {
		'artifact_type': 'f3_inspection_report',
		'inspection_dir': str(config.inspection_dir),
		'outputs': {
			'markdown': str(config.output_markdown),
			'json': str(config.output_json),
		},
		'component_status': [dict(item) for item in component_status],
		'warnings': report_warnings,
		'dataset_files': dataset_files,
		'volume_geometry': geometry,
		'seismic_amplitude_statistics': amplitude,
		'facies_classes': classes,
		'train_validation_labels': labels,
		'label_consistency': consistency,
		'quicklook_figures': figures,
		'tokenization_preview': tokenization,
		'downstream_readiness': readiness,
	}


def _dataset_files_summary(
	components: Mapping[str, Mapping[str, object] | None],
) -> dict[str, object]:
	inventory = _mapping(components.get('file_inventory'))
	files = _sequence_of_mappings(inventory.get('files'))
	by_category: dict[str, list[str]] = {
		'seismic_segy': [],
		'label_segy': [],
		'class_info': [],
		'label_png': [],
	}
	for record in files:
		category = record.get('category')
		relative_path = record.get('relative_path')
		if category in by_category and isinstance(relative_path, str):
			by_category[category].append(relative_path)
	png_records = [
		record for record in files if record.get('category') == 'label_png'
	]
	return {
		'available': bool(inventory),
		'category_counts': dict(_mapping(inventory.get('category_counts'))),
		'split_counts': dict(_mapping(inventory.get('split_counts'))),
		'seismic_segy': by_category['seismic_segy'],
		'label_segy': by_category['label_segy'],
		'class_info': by_category['class_info'],
		'train_png_count': _count_split(png_records, 'train'),
		'validation_png_count': _count_split(png_records, 'validation'),
	}


def _geometry_summary(
	components: Mapping[str, Mapping[str, object] | None],
) -> dict[str, object]:
	geometry = _mapping(components.get('segy_geometry'))
	shape = _mapping(geometry.get('shape_consistency'))
	segy_files = _mapping(geometry.get('segy_files'))
	seismic = _mapping(segy_files.get('seismic'))
	label = _mapping(segy_files.get('label'))
	return {
		'available': bool(geometry),
		'shape': shape.get('seismic_cube_shape'),
		'label_shape': shape.get('label_cube_shape'),
		'shape_matches': shape.get('matches'),
		'inline_range': _range_dict(seismic, 'iline'),
		'crossline_range': _range_dict(seismic, 'xline'),
		'sample_range': _range_dict(seismic, 'sample'),
		'seismic_path': seismic.get('path'),
		'label_path': label.get('path'),
		'axis_assumption': (
			'repo internal axis assumption: x=inline, '
			'y=crossline, z=sample/time'
		),
		'z_display_convention': (
			'XZ/YZ断面ではz/sample/time方向が下向きに増える表示にする。'
		),
	}


def _amplitude_summary(
	components: Mapping[str, Mapping[str, object] | None],
) -> dict[str, object]:
	payload = _mapping(components.get('seismic_amplitude_stats'))
	stats = _mapping(payload.get('stats'))
	return {
		'available': bool(payload),
		'min': stats.get('min'),
		'p1': stats.get('p1'),
		'p50': stats.get('p50'),
		'p99': stats.get('p99'),
		'max': stats.get('max'),
		'finite_count': stats.get('finite_count'),
		'nonfinite_count': stats.get('nonfinite_count'),
		'zero_count': stats.get('zero_count'),
	}


def _class_summary(
	components: Mapping[str, Mapping[str, object] | None],
) -> list[dict[str, object]]:
	classes = _class_info_classes(components)
	voxel_counts = _label_voxel_counts(components)
	pixel_counts = _png_pixel_counts(components)
	return [
		{
			'class_id': item.get('class_id'),
			'class_name': item.get('class_name'),
			'rgb': item.get('rgb'),
			'hex_color': item.get('hex_color'),
			'pixel_count': pixel_counts.get(item.get('class_id'), 0),
			'voxel_count': voxel_counts.get(item.get('class_id'), 0),
		}
		for item in classes
	]


def _label_summary(
	components: Mapping[str, Mapping[str, object] | None],
) -> dict[str, object]:
	payload = _mapping(components.get('png_label_summary'))
	files = _sequence_of_mappings(payload.get('files'))
	splits = _mapping(payload.get('splits'))
	train = _mapping(splits.get('train'))
	validation = _mapping(splits.get('validation'))
	return {
		'available': bool(payload),
		'file_count': payload.get('file_count'),
		'total_pixels': payload.get('total_pixels'),
		'total_unknown_pixels': payload.get('total_unknown_pixels'),
		'slices': [
			{
				'split': item.get('split'),
				'slice_type': item.get('slice_type'),
				'slice_index': item.get('slice_index'),
				'relative_path': item.get('relative_path'),
			}
			for item in files
		],
		'splits': {
			'train': _split_label_summary(train),
			'validation': _split_label_summary(validation),
		},
		'imbalance_notes': _imbalance_notes(payload),
	}


def _consistency_summary(
	components: Mapping[str, Mapping[str, object] | None],
) -> dict[str, object]:
	payload = _mapping(components.get('label_consistency'))
	return {
		'available': bool(payload),
		'passed': payload.get('passed'),
		'png_label_file_count': payload.get('png_label_file_count'),
		'max_mismatch_rate': payload.get('max_mismatch_rate'),
		'max_observed_mismatch_rate': payload.get('max_observed_mismatch_rate'),
		'total_mismatch_pixel_count': payload.get('total_mismatch_pixel_count'),
		'warnings': _string_list(payload.get('warnings')),
	}


def _figure_summary(
	config: F3InspectionReportConfig,
) -> tuple[list[dict[str, object]], list[str]]:
	warnings: list[str] = []
	figures: list[dict[str, object]] = []
	report_dir = config.output_markdown.parent
	for path in config.figure_paths:
		source_path = path if path.is_absolute() else config.inspection_dir / path
		relative_path = _relative_path_for_markdown(source_path, report_dir)
		exists = source_path.is_file()
		if not exists:
			warnings.append(f'missing quicklook figure: {relative_path}')
		figures.append(
			{
				'path': relative_path,
				'exists': exists,
			},
		)
	return figures, warnings


def _tokenization_summary(
	components: Mapping[str, Mapping[str, object] | None],
) -> dict[str, object]:
	payload = _mapping(components.get('tokenization_preview'))
	overall = _mapping(payload.get('overall_summary'))
	total_tokens = _int_or_none(overall.get('total_tokens'))
	retained_tokens = _int_or_none(overall.get('retained_tokens'))
	dropped_tokens = _int_or_none(overall.get('dropped_tokens'))
	ambiguous_tokens = _int_or_none(overall.get('ambiguous_token_count'))
	return {
		'available': bool(payload),
		'patch_size_xyz': _mapping(payload.get('tokenization_config')).get(
			'patch_size_xyz',
		),
		'total_tokens': total_tokens,
		'retained_tokens': retained_tokens,
		'dropped_tokens': dropped_tokens,
		'retained_token_ratio': _fraction_or_none(retained_tokens, total_tokens),
		'dropped_token_ratio': _fraction_or_none(dropped_tokens, total_tokens),
		'ambiguous_token_count': ambiguous_tokens,
		'ambiguous_token_ratio': _fraction_or_none(ambiguous_tokens, total_tokens),
		'empty_token_count': overall.get('empty_token_count'),
		'warnings': [
			'patch単位の代表classは粗い教師ラベルであり、境界付近では'
			'facies混合を含む可能性がある。',
		],
	}


def _readiness_summary(  # noqa: C901, PLR0915
	component_status: Sequence[Mapping[str, object]],
	*,
	amplitude: Mapping[str, object],
	labels: Mapping[str, object],
	consistency: Mapping[str, object],
	tokenization: Mapping[str, object],
) -> dict[str, object]:
	status = READINESS_PROCEED
	reasons: list[str] = []
	required_fixes: list[str] = []
	missing = [
		item['name']
		for item in component_status
		if item.get('available') is False
	]
	if missing:
		status = READINESS_CAUTION
		reasons.append(f'未生成のinspection componentがある: {", ".join(missing)}')
		required_fixes.append('missing componentを生成してreportを再作成する。')
	if consistency.get('available') is not True:
		status = _max_readiness(status, READINESS_CAUTION)
		reasons.append('PNG vs SEGY label consistencyが未確認。')
		required_fixes.append('label consistency checkを完了する。')
	elif consistency.get('passed') is not True:
		status = READINESS_STOP
		reasons.append('PNG vs SEGY label consistencyがFAIL。')
		required_fixes.append('PNG/SEGY label対応の不一致を修正する。')
	if tokenization.get('available') is not True:
		status = _max_readiness(status, READINESS_CAUTION)
		reasons.append('tokenization previewが未生成。')
		required_fixes.append('tokenization previewを生成する。')
	else:
		retained = _int_or_none(tokenization.get('retained_tokens'))
		total = _int_or_none(tokenization.get('total_tokens'))
		retained_ratio = tokenization.get('retained_token_ratio')
		dropped_ratio = tokenization.get('dropped_token_ratio')
		if total == 0:
			status = READINESS_STOP
			reasons.append('tokenization対象tokenが0で、教師tokenを作れない。')
			required_fixes.append('patch sizeまたはtoken化対象sliceを見直す。')
		elif retained == 0:
			status = READINESS_STOP
			reasons.append('retained tokenが0で、教師tokenを作れない。')
			required_fixes.append('patch sizeまたはtoken採用閾値を見直す。')
		elif isinstance(retained_ratio, float) and retained_ratio < 0.5:
			status = _max_readiness(status, READINESS_CAUTION)
			reasons.append('retained token ratioが0.5未満。')
		if isinstance(dropped_ratio, float) and dropped_ratio > 0.25:
			status = _max_readiness(status, READINESS_CAUTION)
			reasons.append('dropped/ambiguous token比率が高い。')
	unknown_pixels = _int_or_none(labels.get('total_unknown_pixels'))
	if unknown_pixels and unknown_pixels > 0:
		status = _max_readiness(status, READINESS_CAUTION)
		reasons.append('PNG labelにclass_info外の色が含まれる。')
		required_fixes.append('unknown PNG label colorをclass_infoと照合する。')
	nonfinite_count = _int_or_none(amplitude.get('nonfinite_count'))
	if nonfinite_count and nonfinite_count > 0:
		status = _max_readiness(status, READINESS_CAUTION)
		reasons.append('seismic amplitudeにnonfinite値がある。')
		required_fixes.append('nonfinite amplitudeの扱いをtraining前に決める。')
	if not reasons:
		reasons.append('consistencyはPASSで、tokenizationも教師tokenを保持している。')
		required_fixes.append('なし。')
	return {
		'status': status,
		'recommendation_ja': _readiness_recommendation(status),
		'reasons': reasons,
		'required_fixes_before_training': required_fixes,
	}


def _component_warnings(
	components: Mapping[str, Mapping[str, object] | None],
) -> list[str]:
	warnings: list[str] = []
	for name, payload in components.items():
		if payload is None:
			continue
		warnings.extend(
			f'{name}: {warning}'
			for warning in _string_list(payload.get('warnings'))
		)
	return warnings


def _class_info_classes(
	components: Mapping[str, Mapping[str, object] | None],
) -> list[Mapping[str, object]]:
	class_info = _mapping(components.get('class_info'))
	classes = _sequence_of_mappings(class_info.get('classes'))
	if classes:
		return classes
	png_summary = _mapping(components.get('png_label_summary'))
	classes = _sequence_of_mappings(
		_mapping(png_summary.get('class_info')).get('classes'),
	)
	if classes:
		return classes
	label_values = _mapping(components.get('label_unique_values'))
	stats = _mapping(label_values.get('stats'))
	return _sequence_of_mappings(_mapping(stats.get('class_info')).get('classes'))


def _label_voxel_counts(
	components: Mapping[str, Mapping[str, object] | None],
) -> dict[object, int]:
	label_values = _mapping(components.get('label_unique_values'))
	stats = _mapping(label_values.get('stats'))
	classes = _sequence_of_mappings(_mapping(stats.get('class_info')).get('classes'))
	return {
		item.get('class_id'): int(item.get('count', 0))
		for item in classes
		if isinstance(item.get('count'), int)
	}


def _png_pixel_counts(
	components: Mapping[str, Mapping[str, object] | None],
) -> dict[object, int]:
	png_summary = _mapping(components.get('png_label_summary'))
	counts = _sequence_of_mappings(png_summary.get('overall_class_counts'))
	return {
		item.get('class_id'): int(item.get('pixel_count', 0))
		for item in counts
		if isinstance(item.get('pixel_count'), int)
	}


def _split_label_summary(split: Mapping[str, object]) -> dict[str, object]:
	return {
		'file_count': split.get('file_count'),
		'total_pixels': split.get('total_pixels'),
		'unknown_pixel_count': split.get('unknown_pixel_count'),
		'class_counts': [
			{
				'class_id': item.get('class_id'),
				'class_name': item.get('class_name'),
				'pixel_count': item.get('pixel_count'),
				'fraction': item.get('fraction'),
			}
			for item in _sequence_of_mappings(split.get('class_counts'))
		],
	}


def _imbalance_notes(payload: Mapping[str, object]) -> list[str]:
	notes: list[str] = []
	for split_name, split_payload in _mapping(payload.get('splits')).items():
		split = _mapping(split_payload)
		counts = [
			_int_or_none(item.get('pixel_count'))
			for item in _sequence_of_mappings(split.get('class_counts'))
		]
		positive = [count for count in counts if count is not None and count > 0]
		zero_count = len([count for count in counts if count == 0])
		if zero_count:
			notes.append(f'{split_name}: 出現pixelが0のclassが{zero_count}件ある。')
		if len(positive) >= 2:
			ratio = max(positive) / min(positive)
			if ratio >= 5.0:
				notes.append(f'{split_name}: class imbalance ratioが{ratio:.3g}。')
	if not notes:
		notes.append('train/validationのclass分布に重大な欠落は未検出。')
	return notes


def _render_dataset_files(dataset_files: Mapping[str, object]) -> list[str]:
	return [
		'- seismic SEGY: '
		f'{_count_and_paths(dataset_files.get("seismic_segy"))}',
		'- label SEGY: '
		f'{_count_and_paths(dataset_files.get("label_segy"))}',
		'- class_info: '
		f'{_count_and_paths(dataset_files.get("class_info"))}',
		f'- train PNG labels: {_display(dataset_files.get("train_png_count"))}件',
		'- validation PNG labels: '
		f'{_display(dataset_files.get("validation_png_count"))}件',
	]


def _render_geometry(geometry: Mapping[str, object]) -> list[str]:
	return [
		f'- shape: {_display(geometry.get("shape"))}',
		f'- label shape: {_display(geometry.get("label_shape"))}',
		f'- shape一致: {_display(geometry.get("shape_matches"))}',
		f'- inline range: {_display(geometry.get("inline_range"))}',
		f'- crossline range: {_display(geometry.get("crossline_range"))}',
		f'- sample range: {_display(geometry.get("sample_range"))}',
		f'- {geometry.get("axis_assumption", "未確認")}',
		f'- z display convention: {geometry.get("z_display_convention", "未確認")}',
	]


def _render_amplitude(amplitude: Mapping[str, object]) -> list[str]:
	return [
		(
			'- min / p1 / p50 / p99 / max: '
			f'{_display(amplitude.get("min"))} / '
			f'{_display(amplitude.get("p1"))} / '
			f'{_display(amplitude.get("p50"))} / '
			f'{_display(amplitude.get("p99"))} / '
			f'{_display(amplitude.get("max"))}'
		),
		(
			'- finite / nonfinite / zero count: '
			f'{_display(amplitude.get("finite_count"))} / '
			f'{_display(amplitude.get("nonfinite_count"))} / '
			f'{_display(amplitude.get("zero_count"))}'
		),
	]


def _render_class_rows(classes: Sequence[Mapping[str, object]]) -> list[str]:
	if not classes:
		return ['| - | 未確認 | - | - | - |']
	return [
		(
			f'| {_display(item.get("class_id"))} | '
			f'{_display(item.get("class_name"))} | '
			f'{_display(item.get("rgb"))} | '
			f'{_display(item.get("pixel_count"))} | '
			f'{_display(item.get("voxel_count"))} |'
		)
		for item in classes
	]


def _render_train_validation_labels(labels: Mapping[str, object]) -> list[str]:
	lines = [
		f'- PNG label files: {_display(labels.get("file_count"))}',
		f'- total pixels: {_display(labels.get("total_pixels"))}',
		f'- unknown pixels: {_display(labels.get("total_unknown_pixels"))}',
		'- slice list:',
	]
	slices = _sequence_of_mappings(labels.get('slices'))
	if slices:
		lines.extend(
			(
				f'  - {item.get("split")} '
				f'{item.get("slice_type")} {item.get("slice_index")}: '
				f'`{item.get("relative_path")}`'
			)
			for item in slices
		)
	else:
		lines.append('  - 未確認')
	lines.extend(['', '### Class distribution by split', ''])
	for split_name, split in _mapping(labels.get('splits')).items():
		split_mapping = _mapping(split)
		lines.append(
			f'- {split_name}: files={_display(split_mapping.get("file_count"))}',
		)
		lines.extend(
			(
				f'  - class {count.get("class_id")} '
				f'{count.get("class_name")}: '
				f'{_display(count.get("pixel_count"))} pixels '
				f'({_display(count.get("fraction"))})'
			)
			for count in _sequence_of_mappings(split_mapping.get('class_counts'))
		)
	lines.extend(['', '### Imbalance notes', ''])
	lines.extend(f'- {note}' for note in _string_list(labels.get('imbalance_notes')))
	return lines


def _render_consistency(consistency: Mapping[str, object]) -> list[str]:
	status = 'PASS' if consistency.get('passed') is True else 'FAIL/未確認'
	lines = [
		f'- status: {status}',
		f'- PNG labels: {_display(consistency.get("png_label_file_count"))}',
		'- max mismatch threshold: '
		f'{_display(consistency.get("max_mismatch_rate"))}',
		'- max observed mismatch rate: '
		f'{_display(consistency.get("max_observed_mismatch_rate"))}',
		'- total mismatch pixels: '
		f'{_display(consistency.get("total_mismatch_pixel_count"))}',
	]
	warnings = _string_list(consistency.get('warnings'))
	lines.append('- warnings: ' + ('なし' if not warnings else '; '.join(warnings)))
	return lines


def _render_figures(figures: Sequence[Mapping[str, object]]) -> list[str]:
	if not figures:
		return ['- 未確認']
	return [
		(
			f'- [{item.get("path")}]({item.get("path")}) '
			f'(exists={item.get("exists")})'
		)
		for item in figures
	]


def _render_tokenization(tokenization: Mapping[str, object]) -> list[str]:
	lines = [
		f'- patch size: {_display(tokenization.get("patch_size_xyz"))}',
		'- retained token ratio: '
		f'{_display(tokenization.get("retained_token_ratio"))}',
		'- ambiguous/dropped token ratio: '
		f'{_display(tokenization.get("ambiguous_token_ratio"))} / '
		f'{_display(tokenization.get("dropped_token_ratio"))}',
		(
			'- total / retained / dropped tokens: '
			f'{_display(tokenization.get("total_tokens"))} / '
			f'{_display(tokenization.get("retained_tokens"))} / '
			f'{_display(tokenization.get("dropped_tokens"))}'
		),
		'- warnings:',
	]
	lines.extend(
		f'  - {warning}' for warning in _string_list(tokenization.get('warnings'))
	)
	return lines


def _render_readiness(readiness: Mapping[str, object]) -> list[str]:
	lines = [
		f'- 判定: `{_display(readiness.get("status"))}`',
		f'- 推奨: {readiness.get("recommendation_ja", "未確認")}',
		'- 理由:',
	]
	lines.extend(f'  - {item}' for item in _string_list(readiness.get('reasons')))
	lines.append('- training前のrequired fixes:')
	lines.extend(
		f'  - {item}'
		for item in _string_list(readiness.get('required_fixes_before_training'))
	)
	return lines


def _count_and_paths(value: object) -> str:
	paths = _string_list(value)
	if not paths:
		return '0件'
	if len(paths) <= 3:
		return f'{len(paths)}件 (' + ', '.join(f'`{path}`' for path in paths) + ')'
	return f'{len(paths)}件 (例: `{paths[0]}`)'


def _range_dict(payload: Mapping[str, object], prefix: str) -> dict[str, object]:
	return {
		'count': payload.get(f'{prefix}_count'),
		'min': payload.get(f'{prefix}_min'),
		'max': payload.get(f'{prefix}_max'),
	}


def _count_split(records: Sequence[Mapping[str, object]], split: str) -> int:
	return sum(1 for record in records if record.get('split') == split)


def _relative_path_for_markdown(path: Path, report_dir: Path) -> str:
	try:
		return os.path.relpath(path, start=report_dir)
	except ValueError:
		return path.as_posix()


def _max_readiness(current: str, candidate: str) -> str:
	order = {
		READINESS_PROCEED: 0,
		READINESS_CAUTION: 1,
		READINESS_STOP: 2,
	}
	return candidate if order[candidate] > order[current] else current


def _readiness_recommendation(status: str) -> str:
	if status == READINESS_PROCEED:
		return '少量教師岩相判別MVPへ進める。'
	if status == READINESS_CAUTION:
		return 'MVPへ進む前にwarningと不足componentを確認する。'
	return 'trainingへ進まず、必須修正を先に行う。'


def _fraction_or_none(numerator: int | None, denominator: int | None) -> float | None:
	if numerator is None or denominator is None or denominator == 0:
		return None
	return float(numerator / denominator)


def _int_or_none(value: object) -> int | None:
	if isinstance(value, bool):
		return None
	if isinstance(value, int):
		return value
	if isinstance(value, float) and value.is_integer():
		return int(value)
	return None


def _mapping(value: object) -> Mapping[str, object]:
	return value if isinstance(value, Mapping) else {}


def _sequence_of_mappings(value: object) -> list[Mapping[str, object]]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		return []
	return [item for item in value if isinstance(item, Mapping)]


def _string_list(value: object) -> list[str]:
	if not isinstance(value, Sequence) or isinstance(value, str | bytes):
		return []
	return [item for item in value if isinstance(item, str)]


def _display(value: object) -> str:
	if value is None:
		return '未確認'
	if isinstance(value, float):
		return f'{value:.6g}'
	if isinstance(value, list | tuple):
		return json.dumps(value, ensure_ascii=False)
	if isinstance(value, Mapping):
		return json.dumps(dict(value), ensure_ascii=False, sort_keys=True)
	return str(value)


def _write_json(path: str | Path, payload: Mapping[str, object]) -> None:
	json_path = Path(path)
	json_path.parent.mkdir(parents=True, exist_ok=True)
	json_path.write_text(
		json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + '\n',
		encoding='utf-8',
	)


def _write_text(path: str | Path, text: str) -> None:
	text_path = Path(path)
	text_path.parent.mkdir(parents=True, exist_ok=True)
	text_path.write_text(text, encoding='utf-8')


__all__ = [
	'READINESS_CAUTION',
	'READINESS_PROCEED',
	'READINESS_STOP',
	'F3InspectionReportConfig',
	'F3InspectionReportResult',
	'build_f3_inspection_report',
	'render_f3_inspection_report_markdown',
]

"""Markdown rendering helpers for F3 lithology reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from seis_ssl_cluster.f3.lithology.report._common import (
	_BASELINE_FEATURE_KINDS,
	_class_metric_sort_key,
	_class_name,
	_display,
	_float_or_none,
	_mapping,
	_relative_path_for_markdown,
	_sequence_of_mappings,
	_string_list,
)


def render_f3_lithology_report_markdown(payload: Mapping[str, object]) -> str:
	"""Render a lithology probe report payload as Japanese Markdown."""
	dataset = _mapping(payload.get('dataset'))
	pretrained = _mapping(payload.get('pretrained_encoder'))
	token_dataset = _mapping(payload.get('token_dataset'))
	probe = _mapping(payload.get('probe'))
	metrics = _mapping(payload.get('metrics'))
	figures = _sequence_of_mappings(payload.get('figures'))
	interpretation = _mapping(payload.get('interpretation'))
	warnings = _string_list(payload.get('warnings'))
	lines = [
		'# F3 token-level lithology probe report',
		'',
		'このreportはF3 token-level lithology probeの既存artifactを統合し、'
		'pretrained model、AGC有無、probe種別の比較に使う。',
		'',
		'## Dataset',
		'',
		*_render_dataset(dataset),
		'',
		'## Pretrained encoder',
		'',
		*_render_pretrained(pretrained),
		'',
		'## Token dataset',
		'',
		*_render_token_dataset(token_dataset),
		'',
		'## Probe',
		'',
		*_render_probe(probe),
		'',
		'## Metrics',
		'',
		*_render_metrics(metrics),
		'',
		'## Figures',
		'',
		*_render_figures(figures),
		'',
		'## Interpretation',
		'',
		*_render_interpretation(interpretation),
		'',
		'## Warnings',
		'',
	]
	lines.extend((f'- {warning}' for warning in warnings),)
	if not warnings:
		lines.append('- none')
	return '\n'.join(lines) + '\n'

def _render_comparison_markdown(
	rows: Sequence[Mapping[str, object]],
	fieldnames: Sequence[str],
	figure_paths: Mapping[str, Path],
	warnings: Sequence[str],
) -> str:
	lines = [
		'# F3 lithology probe comparison report',
		'',
		f'集約run数: {len(rows)}',
		'',
		'## Comparison table',
		'',
		'| ' + ' | '.join(fieldnames) + ' |',
		'|' + '|'.join('---' for _ in fieldnames) + '|',
	]
	lines.extend(
		(
			'| '
			+ ' | '.join(_display(row.get(field, '')) for field in fieldnames)
			+ ' |'
		)
		for row in rows
	)
	lines.extend(['', '## Figures', ''])
	report_dir = (
		next(iter(figure_paths.values())).parent.parent
		if figure_paths
		else Path()
	)
	lines.extend(
		f'- [{name}]({_relative_path_for_markdown(path, report_dir)})'
		for name, path in figure_paths.items()
	)
	lines.extend(['', '## Interpretation', ''])
	lines.extend(_comparison_interpretation(rows))
	lines.extend(['', '## Warnings', ''])
	if warnings:
		lines.extend(f'- {warning}' for warning in warnings)
	else:
		lines.append('- none')
	return '\n'.join(lines) + '\n'

def _comparison_interpretation(
	rows: Sequence[Mapping[str, object]],
) -> list[str]:
	pretrained = _best_comparison_row(rows, 'pretrained_encoder', metric='macro_f1')
	z_only = _best_comparison_row(rows, 'z_only', metric='macro_f1')
	xyz_coordinates = _best_comparison_row(
		rows,
		'xyz_coordinates',
		metric='macro_f1',
	)
	amplitude = _best_comparison_row(rows, 'amplitude_stats', metric='macro_f1')
	random_encoder = _best_comparison_row(
		rows,
		'random_encoder',
		metric='macro_f1',
	)
	return [
		(
			'- pretrained encoderがz-onlyを上回るか: '
			f'{_comparison_delta_sentence(pretrained, z_only)}'
		),
		(
			'- pretrained encoderがxyz-coordinateを上回るか: '
			f'{_comparison_delta_sentence(pretrained, xyz_coordinates)}'
		),
		(
			'- pretrained encoderがamplitude-onlyを上回るか: '
			f'{_comparison_delta_sentence(pretrained, amplitude)}'
		),
		(
			'- pretrained encoderがrandom encoderを上回るか: '
			f'{_comparison_delta_sentence(pretrained, random_encoder)}'
		),
		(
			'- class 3/5など弱いclassで改善があるか: '
			f'{_weak_class_delta_sentence(rows, pretrained)}'
		),
		(
			'- F3 faciesが深度だけで説明できる程度: '
			f'{_depth_only_sentence(pretrained, z_only)}'
		),
	]

def _best_comparison_row(
	rows: Sequence[Mapping[str, object]],
	feature_kind: str,
	*,
	metric: str,
) -> Mapping[str, object] | None:
	candidates = [
		row
		for row in rows
		if row.get('feature_kind') == feature_kind
		and _float_or_none(row.get(metric)) is not None
	]
	if not candidates:
		return None
	return max(candidates, key=lambda row: _float_or_none(row.get(metric)) or 0.0)

def _comparison_delta_sentence(
	pretrained: Mapping[str, object] | None,
	baseline: Mapping[str, object] | None,
) -> str:
	if pretrained is None or baseline is None:
		return '比較対象のmetricsが不足しているため未確認。'
	macro_delta = _metric_delta(pretrained, baseline, 'macro_f1')
	iou_delta = _metric_delta(pretrained, baseline, 'mean_iou')
	if macro_delta is None:
		return 'macro F1が不足しているため未確認。'
	if macro_delta > 0.0:
		relation = '上回る'
	elif macro_delta == 0.0:
		relation = '同等'
	else:
		relation = '下回る'
	iou_text = (
		'mean IoU差分 未確認'
		if iou_delta is None
		else f'mean IoU差分 {iou_delta:+.4f}'
	)
	return f'{relation} (macro F1差分 {macro_delta:+.4f}, {iou_text})。'

def _weak_class_delta_sentence(
	rows: Sequence[Mapping[str, object]],
	pretrained: Mapping[str, object] | None,
) -> str:
	if pretrained is None:
		return 'pretrained encoder metricsが不足しているため未確認。'
	class_columns = [
		column
		for column in sorted(
			{
				key
				for row in rows
				for key in row
				if key.startswith('class_') and key.endswith('_f1')
			},
			key=_class_metric_sort_key,
		)
		if column.startswith('class_') and column.endswith('_f1')
	]
	priority = [
		column for column in ('class_3_f1', 'class_5_f1') if column in class_columns
	]
	targets = priority or class_columns[:2]
	if not targets:
		return 'per-class F1が不足しているため未確認。'
	parts = []
	for column in targets:
		pretrained_value = _float_or_none(pretrained.get(column))
		baseline_value = _best_baseline_class_f1(rows, column)
		if pretrained_value is None or baseline_value is None:
			continue
		class_label = column.removeprefix('class_').removesuffix('_f1')
		parts.append(
			f'class {class_label}: '
			f'F1差分 {pretrained_value - baseline_value:+.4f}',
		)
	return '、'.join(parts) + '。' if parts else '比較可能なclass別F1が不足している。'

def _best_baseline_class_f1(
	rows: Sequence[Mapping[str, object]],
	column: str,
) -> float | None:
	values = [
		value
		for row in rows
		if row.get('feature_kind') in _BASELINE_FEATURE_KINDS
		for value in (_float_or_none(row.get(column)),)
		if value is not None
	]
	return max(values) if values else None

def _depth_only_sentence(
	pretrained: Mapping[str, object] | None,
	z_only: Mapping[str, object] | None,
) -> str:
	if pretrained is None or z_only is None:
		return 'z-onlyまたはpretrained encoder metricsが不足しているため未確認。'
	pretrained_macro = _float_or_none(pretrained.get('macro_f1'))
	z_macro = _float_or_none(z_only.get('macro_f1'))
	if pretrained_macro is None or z_macro is None:
		return 'macro F1が不足しているため未確認。'
	delta = pretrained_macro - z_macro
	if delta <= 0.02:
		return (
			f'z-onlyとの差が小さい (macro F1差分 {delta:+.4f}) ため、'
			'深度で説明できる寄与が大きい。'
		)
	return (
		f'z-onlyとの差がある (macro F1差分 {delta:+.4f}) ため、'
		'深度以外の特徴が効いている可能性がある。'
	)

def _metric_delta(
	left: Mapping[str, object],
	right: Mapping[str, object],
	metric: str,
) -> float | None:
	left_value = _float_or_none(left.get(metric))
	right_value = _float_or_none(right.get(metric))
	if left_value is None or right_value is None:
		return None
	return left_value - right_value

def _render_dataset(dataset: Mapping[str, object]) -> list[str]:
	classes = _sequence_of_mappings(dataset.get('classes'))
	lines = [
		f'- F3 shape: {_display(dataset.get("f3_shape"))}',
		f'- classes: {len(classes)}',
		f'- label source of truth: {_display(dataset.get("label_source_of_truth"))}',
		f'- PNG label role: {_display(dataset.get("png_label_role"))}',
		(
			'- train/validation slices: '
			f'{_display(dataset.get("train_validation_slices"))}'
		),
		(
			'- tokenization thresholds: '
			f'{_display(dataset.get("tokenization_thresholds"))}'
		),
		f'- class imbalance: {_display(dataset.get("class_imbalance"))}',
		'',
		'| class_id | class_name | rgb |',
		'|---:|---|---|',
	]
	lines.extend(
		(
			f'| {_display(item.get("class_id"))} | '
			f'{_display(_class_name(item))} | {_display(item.get("rgb"))} |'
		)
		for item in classes
	)
	return lines

def _render_pretrained(pretrained: Mapping[str, object]) -> list[str]:
	return [
		f'- MODEL_TAG: {_display(pretrained.get("MODEL_TAG"))}',
		f'- checkpoint path: {_display(pretrained.get("checkpoint_path"))}',
		f'- EMBED_SPEC: {_display(pretrained.get("EMBED_SPEC"))}',
		f'- AGC有無: {_display(pretrained.get("agc_enabled"))}',
		f'- visible loss有無: {_display(pretrained.get("visible_loss_enabled"))}',
		f'- mask ratio: {_display(pretrained.get("mask_ratio"))}',
		(
			'- encoder fine-tuning: '
			f'{_display(pretrained.get("freeze_encoder") is not True)}'
		),
	]

def _render_token_dataset(token_dataset: Mapping[str, object]) -> list[str]:
	return [
		f'- train token count: {_display(token_dataset.get("train_token_count"))}',
		(
			'- validation token count: '
			f'{_display(token_dataset.get("validation_token_count"))}'
		),
		f'- class counts: {_display(token_dataset.get("class_counts"))}',
		(
			'- dropped token ratio: '
			f'{_display(token_dataset.get("dropped_token_ratio"))}'
		),
		(
			'- ambiguous token ratio: '
			f'{_display(token_dataset.get("ambiguous_token_ratio"))}'
		),
	]

def _render_probe(probe: Mapping[str, object]) -> list[str]:
	return [
		f'- PROBE_SPEC: {_display(probe.get("PROBE_SPEC"))}',
		f'- classifier type: {_display(probe.get("classifier_type"))}',
		f'- feature scaling: {_display(probe.get("feature_scaling"))}',
		f'- class weighting: {_display(probe.get("class_weighting"))}',
		f'- hyperparameters: {_display(probe.get("hyperparameters"))}',
	]

def _render_metrics(metrics: Mapping[str, object]) -> list[str]:
	overall = _mapping(metrics.get('overall'))
	per_class = _sequence_of_mappings(metrics.get('per_class'))
	lines = [
		f'- accuracy: {_display(overall.get("accuracy"))}',
		f'- balanced accuracy: {_display(overall.get("balanced_accuracy"))}',
		f'- macro F1: {_display(overall.get("macro_f1"))}',
		f'- weighted F1: {_display(overall.get("weighted_f1"))}',
		f'- mean IoU: {_display(overall.get("mean_iou"))}',
		'',
		'| class_id | class_name | F1 | IoU | support |',
		'|---:|---|---:|---:|---:|',
	]
	lines.extend(
		(
			f'| {_display(item.get("class_id"))} | {_display(item.get("class_name"))} '
			f'| {_display(item.get("f1"))} | {_display(item.get("iou"))} '
			f'| {_display(item.get("support"))} |'
		)
		for item in per_class
	)
	lines.extend(['', '- confusion matrix:', '', '```text'])
	matrix = metrics.get('confusion_matrix')
	lines.append(_display(matrix))
	lines.append('```')
	return lines

def _render_figures(figures: Sequence[Mapping[str, object]]) -> list[str]:
	if not figures:
		return ['- none']
	return [
		f'- [{_display(item.get("type"))}]({_display(item.get("path"))})'
		for item in figures
	]

def _render_interpretation(interpretation: Mapping[str, object]) -> list[str]:
	lines: list[str] = []
	for key in (
		'良い点',
		'失敗しているclass',
		'class imbalanceの影響',
		'AGCあり/なし比較',
		'次の改善候補',
	):
		lines.append(f'### {key}')
		lines.append('')
		value = interpretation.get(key)
		if isinstance(value, Sequence) and not isinstance(value, str | bytes):
			lines.extend(f'- {item}' for item in value)
		else:
			lines.append(f'- {_display(value)}')
		lines.append('')
	return lines[:-1]

def _interpretation_summary(
	*,
	pretrained: Mapping[str, object],
	token_dataset: Mapping[str, object],
	metrics: Mapping[str, object],
) -> dict[str, object]:
	overall = _mapping(metrics.get('overall'))
	per_class = _sequence_of_mappings(metrics.get('per_class'))
	failures = [
		item
		for item in sorted(
			per_class,
			key=lambda entry: (
				float('inf')
				if _float_or_none(entry.get('f1')) is None
				else float(entry['f1'])
			),
		)
		if _float_or_none(item.get('f1')) is not None
	][:2]
	good_points = [
		(
			'weighted F1は'
			f"{_display(overall.get('weighted_f1'))}で、頻出classの性能を確認できる。"
		),
		(
			'balanced accuracyは'
			f"{_display(overall.get('balanced_accuracy'))}で、"
			'class imbalanceを考慮した比較指標になる。'
		),
	]
	return {
		'良い点': good_points,
		'失敗しているclass': [
			(
				f"class {item.get('class_id')} {item.get('class_name')}: "
				f"F1={_display(item.get('f1'))}, IoU={_display(item.get('iou'))}"
			)
			for item in failures
		]
		or ['metricsが不足しているため特定できない。'],
		'class imbalanceの影響': _imbalance_interpretation(token_dataset),
		'AGCあり/なし比較': _agc_interpretation(pretrained),
		'次の改善候補': [
			'comparison_table.csvでMODEL_TAG、EMBED_SPEC、PROBE_SPECごとの'
			'macro F1とmean IoUを比較する。',
			'低F1 classは教師slice追加、tokenization閾値、class weightingの'
			'影響を切り分ける。',
			'linear probeで頭打ちなら同じfrozen encoder上でMLP probeを比較する。',
		],
	}

def _imbalance_interpretation(token_dataset: Mapping[str, object]) -> str:
	imbalance = _mapping(token_dataset.get('class_imbalance'))
	ratio = _float_or_none(imbalance.get('max_to_min_positive_ratio'))
	if ratio is None:
		return 'class count情報が不足しているため影響を評価できない。'
	if ratio > 5.0:
		return (
			f'class countの最大/最小比が{ratio:.3g}で、minor classのF1低下に注意する。'
		)
	return f'class countの最大/最小比は{ratio:.3g}で、極端な偏りは限定的。'

def _agc_interpretation(pretrained: Mapping[str, object]) -> str:
	agc = pretrained.get('agc_enabled')
	state = 'AGCあり' if agc is True else 'AGCなし' if agc is False else 'AGC不明'
	return (
		f'このrunは{state}として集計される。AGCあり/なしの優劣は'
		'comparison_table.csvで同じEMBED_SPEC、LABEL_SET、PROBE_SPECを揃えて比較する。'
	)

__all__ = [
	'_agc_interpretation',
	'_best_baseline_class_f1',
	'_best_comparison_row',
	'_comparison_delta_sentence',
	'_comparison_interpretation',
	'_depth_only_sentence',
	'_imbalance_interpretation',
	'_interpretation_summary',
	'_metric_delta',
	'_render_comparison_markdown',
	'_render_dataset',
	'_render_figures',
	'_render_interpretation',
	'_render_metrics',
	'_render_pretrained',
	'_render_probe',
	'_render_token_dataset',
	'_weak_class_delta_sentence',
	'render_f3_lithology_report_markdown',
]

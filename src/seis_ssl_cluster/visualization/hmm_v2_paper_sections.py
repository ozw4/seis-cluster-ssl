"""Canonical HMM v2 figures on the exact frozen HMM v1 paper sections."""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology import hmm_v1_comparison_visualization as facies
from seis_ssl_cluster.f3.lithology.common_test import inspect_test_cell
from seis_ssl_cluster.f3.lithology.voxel_prediction_artifact import (
	validate_f3_voxel_prediction_artifact,
)
from seis_ssl_cluster.parihaka import paper_prediction as channel_prediction
from seis_ssl_cluster.parihaka import paper_sections as channel_figures
from seis_ssl_cluster.parihaka.channel_completion import inspect_completed_channel_job
from seis_ssl_cluster.volve import paper_prediction as horizon_prediction
from seis_ssl_cluster.volve import paper_sections as horizon_figures
from seis_ssl_cluster.volve.horizon_data import load_volve_horizon_data

EXPERIMENT = 'hmm_v2_canonical_paper_sections_v1'
DIS0_EXPERIMENT = 'hmm_v2_zero_distillation_paper_sections_v1'
SURVEYS = ('f3', 'parihaka', 'volve')
MODEL_ORDER = tuple(
	(f, t) for f in ('mae', 'local_bt', 'random') for t in ('none', 'v2')
)
DIS0_REVIEW_FILES = frozenset({'README.md'})


def _read(path: str | Path) -> dict[str, Any]:
	return json.loads(Path(path).read_text())


def verify_bound(record: dict[str, str]) -> Path:
	"""Reject source drift and report inputs before rendering or inference."""
	path = Path(record['path'])
	if not path.is_absolute() or 'reports' in path.resolve().parts:
		raise ValueError(f'expected absolute non-report input: {path}')
	if file_sha256(path) != record['sha256']:
		raise ValueError(f'input hash drift: {path}')
	return path


def validate_pretext(metadata: dict[str, Any], treatment: str) -> None:
	"""Require the declared v2 or v2_dis0 recipe in actual embedding metadata."""
	pretext = metadata.get('stratigraphy_pretext')
	if treatment == 'none':
		if pretext:
			raise ValueError('non-HMM parent has HMM provenance')
		return
	if (
		treatment not in ('v2', 'v2_dis0')
		or not pretext
		or any(
			pretext.get(k) != v
			for k, v in {
				'head_spec': 'multi_resolution_ordered_prototypes_v1',
				'head_ks': [6, 8, 10],
				'distillation_weight': 0.0 if treatment == 'v2_dis0' else 0.2,
				'consistency_weight': 0.1,
				'method': 'strat_hmm_multi_head_pretext',
			}.items()
		)
	):
		raise ValueError(f'expected canonical {treatment} K6810/consistency0.1 recipe')


def fixed_rows(
	reference: dict[str, Any], survey: str, *, treatment: str = 'v2'
) -> list[dict[str, Any]]:
	"""Reuse original section identities, without selecting by new predictions."""
	if treatment not in ('v2', 'v2_dis0'):
		raise ValueError('unsupported figure treatment')
	if survey == 'f3':
		rows = [
			r
			for d in facies.DOMAINS
			for r in reference['selection'][f'selected_{d}_slices']
		]
	else:
		rows = reference['selected_slices']
	rows = copy.deepcopy(rows)
	expected_domains = ('inline', 'crossline') if survey == 'volve' else facies.DOMAINS
	if (
		len(rows) != 10 * len(expected_domains)
		or len({(r['domain'], r['array_index']) for r in rows}) != len(rows)
		or any(sum(r['domain'] == d for r in rows) != 10 for d in expected_domains)
		or any(
			r['active_training_line'] is not False
			and not (
				r['domain'] == 'time' and r['active_training_line'] == 'not_applicable'
			)
			for r in rows
		)
	):
		raise ValueError(
			'reference must contain ten distinct non-training sections per domain'
		)
	for row in rows:
		old = row['output_png']
		if '_hmm_v1_' not in old or '..' in Path(old).parts or Path(old).is_absolute():
			raise ValueError('unsafe or non-v1 reference output name')
		row['output_png'] = old.replace('_hmm_v1_', f'_hmm_{treatment}_')
	return rows


def _verify_reference(record: dict[str, str]) -> dict[str, Any]:
	path = verify_bound(record)
	reference = _read(path)
	for output in reference['outputs']['pngs']:
		relative = output.get('relative_path', output.get('path'))
		verify_bound({'path': str(path.parent / relative), 'sha256': output['sha256']})
	return reference


def _bind_dis0_source(
	config: dict[str, Any], model: dict[str, Any], survey: str
) -> None:
	family = model['source_family']
	summary = _read(config['inputs'][f'{family}_sources']['path'])
	if summary['status'] != 'complete':
		raise ValueError('zero-distillation study is incomplete')
	records = [
		r
		for r in summary['evidence']
		if r['survey'] == survey
		and r['weight'] == '0.0'
		and r['layout'] == 'layout_000'
		and r['size'] == 'large'
	]
	if len(records) != 1 or (
		records[0]['candidate_id'] != model['model_id']
		or records[0]['metrics_sha256'][0] != model['metrics']['sha256']
		or Path(records[0]['metrics_paths'][0]).resolve()
		!= Path(model['metrics']['path']).resolve()
	):
		raise ValueError('source not bound to the completed v2_dis0 study')


def _dis0_model(
	model: dict[str, Any],
	survey: str,
	config: dict[str, Any],
	baseline: dict[str, Any],
	reference: dict[str, Any],
) -> dict[str, Any]:
	if model['treatment'] == 'v2_dis0':
		_bind_dis0_source(config, model, survey)
		return model
	parent = next(
		m
		for m in baseline['surveys'][survey]['models']
		if m['condition_id'] == model['condition_id']
	)
	if model != parent:
		raise ValueError('v2_dis0 non-HMM parent differs from the original v2 figure')
	if survey == 'f3':
		return model
	previous = next(
		m for m in reference['models'] if m['condition_id'] == model['condition_id']
	)
	metadata = verify_bound(previous['render_prediction_metadata'])
	return dict(model, reuse_prediction_dir=str(metadata.parent))


def expected_figure_files(rows: list[dict[str, Any]]) -> set[str]:
	"""Exact per-survey publication, excluding arrays and operational records."""
	return {'manifest.json', 'selected_slices.csv', *(r['output_png'] for r in rows)}


def _inspect_source(model: dict[str, Any], survey: str) -> dict[str, Any]:  # noqa: C901, PLR0912
	for key in (
		'metrics',
		'decoder_checkpoint',
		'encoder_checkpoint',
		'embedding_metadata',
		'embeddings',
		'valid_tokens',
	):
		verify_bound(model[key])
	run, emb = Path(model['run_dir']), Path(model['embedding_dir'])
	metadata = _read(model['embedding_metadata']['path'])
	validate_pretext(metadata, model['treatment'])
	if metadata['checkpoint_sha256'] != model['encoder_checkpoint']['sha256'] or (
		Path(metadata['checkpoint_path']).resolve()
		!= Path(model['encoder_checkpoint']['path']).resolve()
	):
		raise ValueError('encoder and embedding identity mismatch')
	if Path(model['embedding_metadata']['path']).parent.resolve() != emb.resolve():
		raise ValueError('embedding directory mismatch')
	if survey == 'f3':
		condition = {
			'audit_id': model['condition_id'],
			'checkpoint_sha256': model['encoder_checkpoint']['sha256'],
			'embedding_metadata_sha256': model['embedding_metadata']['sha256'],
		}
		cell = inspect_test_cell(run, condition, 'layout_000', 'large')
		if (
			cell.evidence['decoder_checkpoint']['sha256']
			!= model['decoder_checkpoint']['sha256']
		):
			raise ValueError('F3 selected decoder differs from pinned checkpoint')
		return dict(model, evidence=cell.evidence, decoder_seed=42000)
	metrics = _read(model['metrics']['path'])
	identity = metrics['benchmark_identity']
	if (identity['model'], identity['layout_id'], identity['data_size']) != (
		model['model_id'],
		'layout_000',
		'large',
	):
		raise ValueError('decoder model/layout/budget mismatch')
	if survey == 'parihaka':
		completion = inspect_completed_channel_job(run, metrics=metrics)
		emeta = identity['embedding']
		for key, value in emeta['common_metadata'].items():
			if metadata.get(key) != value:
				raise ValueError(f'embedding common metadata mismatch: {key}')
		if metadata.get('stratigraphy_pretext') != emeta['model_source'].get(
			'stratigraphy_pretext'
		):
			raise ValueError('decoder and HMM provenance mismatch')
		if metadata['checkpoint_sha256'] != emeta['checkpoint_sha256']:
			raise ValueError('decoder encoder mismatch')
		if (
			file_sha256(emb / 'parihaka.valid_tokens.npy')
			!= identity['geometry']['valid_tokens_sha256']
		):
			raise ValueError('Parihaka valid-token mismatch')
		return dict(
			model,
			identity=identity,
			completion=completion,
			checkpoint_sha256=model['decoder_checkpoint']['sha256'],
			cube_shape=identity['geometry']['volume_shape_xyz'],
			decoder_seed=identity['training']['seed'],
		)
	best = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
	latest = torch.load(run / 'latest.pt', map_location='cpu', weights_only=False)
	horizon_prediction.validate_checkpoint(
		run, model['model_id'], metrics, best, latest
	)
	for key in ('embeddings', 'metadata'):
		bound = model['embedding_metadata' if key == 'metadata' else 'embeddings']
		if identity['embedding'][key + '_sha256'] != bound['sha256']:
			raise ValueError('Volve checkpoint embedding identity mismatch')
	if identity['embedding']['valid_tokens_sha256'] != model['valid_tokens']['sha256']:
		raise ValueError('Volve valid-token mismatch')
	if metadata['checkpoint_sha256'] != identity['embedding']['checkpoint_sha256']:
		raise ValueError('Volve decoder encoder mismatch')
	return dict(
		model,
		identity=identity,
		best_checkpoint_sha256=model['decoder_checkpoint']['sha256'],
		valid_tokens_path=model['valid_tokens']['path'],
		decoder_seed=identity['training']['seed'],
	)


def inspect_plan(config_path: Path) -> dict[str, Any]:  # noqa: C901, PLR0912
	"""Hash and audit all eighteen sources and all reference figures, read-only."""
	config = load_config(config_path)
	root = Path(config['artifact_root'])
	is_dis0 = config['experiment_id'] == DIS0_EXPERIMENT
	treatment = 'v2_dis0' if is_dis0 else 'v2'
	namespace = (
		'zero_distillation_paper_sections_v1'
		if is_dis0
		else 'canonical_paper_sections_v1'
	)
	if (
		config['experiment_id'] not in (EXPERIMENT, DIS0_EXPERIMENT)
		or Path(config['output_root']).resolve()
		!= (root / 'studies/hmm_v2' / namespace).resolve()
	):
		raise ValueError('unexpected experiment/output namespace')
	for record in config['inputs'].values():
		verify_bound(record)
	selection = load_config(Path(config['inputs']['canonical_selection']['path']))
	contract = selection['scientific_contract']
	if (
		contract['selected_head_ks'],
		contract['distillation_weight'],
		contract['consistency_weight'],
	) != ([6, 8, 10], 0.2, 0.1):
		raise ValueError('canonical selection changed')
	aggregate = _read(config['inputs']['canonical_summary']['path'])
	baseline_config = (
		load_config(config['inputs']['v2_execution']['path']) if is_dis0 else None
	)
	plans = {}
	for survey in SURVEYS:
		sc = config['surveys'][survey]
		if tuple((m['source_family'], m['treatment']) for m in sc['models']) != tuple(
			(f, treatment if t == 'v2' else t) for f, t in MODEL_ORDER
		):
			raise ValueError('six-model comparison order changed')
		verify_bound(sc['reference_csv'])
		reference = _verify_reference(sc['reference_manifest'])
		v2_reference = None
		if is_dis0:
			v2_reference = _verify_reference(sc['v2_manifest'])
			verify_bound(sc['v2_csv'])
			if fixed_rows(reference, survey) != v2_reference['selected_slices']:
				raise ValueError(
					'v2 reference sections differ from the fixed v1 sections'
				)
		sources = []
		for model in sc['models']:
			selected = (
				_dis0_model(model, survey, config, baseline_config, v2_reference)
				if is_dis0
				else model
			)
			if model['treatment'] == 'v2':
				cell = next(
					x
					for x in aggregate['cells']
					if x['survey'] == survey
					and x['source_family'] == model['source_family']
					and x['layout_id'] == 'layout_000'
					and x['data_size'] == 'large'
				)
				if model['metrics']['sha256'] != cell['metrics_sha256']:
					raise ValueError('source not bound to the selected v2 aggregate')
			print(f'Preflight {survey}/{model["condition_id"]}', flush=True)
			sources.append(_inspect_source(selected, survey))
		if survey == 'parihaka':
			baseline_selection = load_config(
				Path(config['inputs']['parihaka_mae_selection']['path'])
			)
			if (
				sources[0]['metrics']['sha256']
				!= baseline_selection['evidence']['metrics_sha256']['layout_000']
				or sources[0]['encoder_checkpoint']['sha256']
				!= baseline_selection['scientific_contract']['frozen_encoder_sha256']
			):
				raise ValueError(
					'Parihaka MAE differs from the designated canonical reference'
				)
			if any(
				s['identity']['geometry'] != sources[0]['identity']['geometry']
				or s['identity']['train_lines'] != sources[0]['identity']['train_lines']
				for s in sources
			):
				raise ValueError('Parihaka cross-model geometry/training lines differ')
			if sources[0]['decoder_seed'] != 42001 or any(
				s['decoder_seed'] != 42000 for s in sources[1:]
			):
				raise ValueError(
					'Parihaka must use the designated MAE seed-42001 reference'
				)
		plans[survey] = {
			'sources': sources,
			'reference': reference,
			'rows': fixed_rows(reference, survey, treatment=treatment),
		}
	return {
		'config': config,
		'config_path': str(config_path.resolve()),
		'config_sha256': file_sha256(config_path),
		'surveys': plans,
	}


def _prediction_root(plan: dict[str, Any], survey: str) -> Path:
	return (
		Path(plan['config']['output_root'])
		/ 'predictions'
		/ survey
		/ 'layout_000/large'
	)


def _predictions(plan: dict[str, Any], survey: str) -> list[Any]:  # noqa: C901, PLR0912
	arrays = []
	for source in plan['surveys'][survey]['sources']:
		if survey == 'f3':
			artifact = validate_f3_voxel_prediction_artifact(
				Path(source['prediction_dir']), mmap_mode='r'
			)
			arrays.append(artifact.arrays)
			continue
		reused = source.get('reuse_prediction_dir')
		root = (
			Path(reused)
			if reused
			else _prediction_root(plan, survey) / f'model={source["model_id"]}'
		)
		if not root.exists():
			if not torch.cuda.is_available():
				raise RuntimeError('CUDA required for new frozen-decoder predictions')
			root.parent.mkdir(parents=True, exist_ok=True)
			with tempfile.TemporaryDirectory(prefix='.export-', dir=root.parent) as tmp:
				if survey == 'parihaka':
					staged = channel_prediction.export_model(
						source, Path(tmp), plan['config']['inference_device']
					)
				else:
					export_config = {
						'outputs': {'predictions': tmp},
						'inference': {
							'device': plan['config']['inference_device'],
							'precision': 'float16',
							'window_start': 552,
							'window_stop': 768,
						},
					}
					staged = horizon_prediction.export_prediction(export_config, source)
				staged.rename(root)
		meta = _read(root / 'prediction_metadata.json')
		if not meta['complete'] or any(
			meta[k] for k in ('smoothing', 'ensemble', 'postprocessing')
		):
			raise ValueError('only complete raw predictions are allowed')
		if survey == 'parihaka':
			if (
				Path(meta['run_dir']).resolve() != Path(source['run_dir']).resolve()
				or Path(meta['embedding_dir']).resolve()
				!= Path(source['embedding_dir']).resolve()
			):
				raise ValueError('Parihaka exported input directories differ')
			if meta['checkpoint_sha256'] != source['checkpoint_sha256']:
				raise ValueError(
					'Parihaka exported decoder differs from selected checkpoint'
				)
			channel_prediction.validate_prediction(root, tuple(source['cube_shape']))
			arrays.append(
				SimpleNamespace(
					predictions=np.load(
						root / 'parihaka_voxel_predictions.npy', mmap_mode='r'
					),
					valid_mask=np.load(
						root / 'parihaka_valid_voxel_mask.npy', mmap_mode='r'
					),
				)
			)
		else:
			if meta['source']['identity'] != source['identity']:
				raise ValueError('Volve exported source identity differs')
			if (
				meta['source']['best_checkpoint_sha256']
				!= source['best_checkpoint_sha256']
			):
				raise ValueError(
					'Volve exported decoder differs from selected checkpoint'
				)
			prediction = np.load(root / 'horizon_sample_predictions.npy', mmap_mode='r')
			valid = np.load(root / 'valid_lateral_mask.npy', mmap_mode='r')
			horizon_prediction.validate_prediction(prediction, valid)
			if not np.array_equal(valid, horizon_prediction.model_valid_mask(source)):
				raise ValueError('Volve exported validity differs from current source')
			arrays.append(prediction)
		for filename, digest in meta['array_sha256'].items():
			verify_bound({'path': str(root / filename), 'sha256': digest})
		source['render_prediction_metadata'] = {
			'path': str(root / 'prediction_metadata.json'),
			'sha256': file_sha256(root / 'prediction_metadata.json'),
		}
	return arrays


def _manifest(plan: dict[str, Any], survey: str) -> dict[str, Any]:
	sp = plan['surveys'][survey]
	treatment = (
		'v2_dis0' if plan['config'].get('experiment_id') == DIS0_EXPERIMENT else 'v2'
	)
	manifest = copy.deepcopy(sp['reference'])
	for key in (
		'outputs',
		'models',
		'source_freeze_manifest',
		'source_git_commit',
		'comparison_policy',
		'selection',
		'prediction_metadata',
		'generation_git_commit',
	):
		manifest.pop(key, None)
	manifest.update(
		artifact_type=(
			f'{survey}_hmm_v2_dis0_paper_sections'
			if treatment == 'v2_dis0'
			else f'{survey}_hmm_v2_canonical_paper_sections'
		),
		hmm_version=f'HMM_{treatment}',
		generated_at_utc=datetime.now(timezone.utc).isoformat(),
		comparison={
			'layout_id': 'layout_000',
			'data_size': 'large',
			'head_ks': [6, 8, 10],
			'distillation_weight': 0.0 if treatment == 'v2_dis0' else 0.2,
			'consistency_weight': 0.1,
		},
		models=[{k: v for k, v in s.items() if k != 'identity'} for s in sp['sources']],
		reference_manifest=plan['config']['surveys'][survey]['reference_manifest'],
		selection_policy=(
			'exact frozen v1 section indices and display crops; no reselection'
		),
		selected_slices=sp['rows'],
		canonical_selection=plan['config']['inputs']['canonical_selection'],
		execution_config_sha256=plan['config_sha256'],
		config_path=plan['config_path'],
		config_sha256=plan['config_sha256'],
		generator_sha256=file_sha256(Path(__file__)),
		output_root=str(
			Path(plan['config']['output_root'])
			/ 'figures'
			/ survey
			/ 'layout_000/large'
		),
		training=False,
		metric_evaluation=False,
		prediction_postprocessing=False,
	)
	if treatment == 'v2_dis0':
		manifest['v2_reference_manifest'] = plan['config']['surveys'][survey][
			'v2_manifest'
		]
		manifest['source_studies'] = {
			family: plan['config']['inputs'][f'{family}_sources']
			for family in ('mae', 'local_bt', 'random')
		}
	if survey == 'parihaka':
		manifest['baseline_note'] = (
			'MAE/no-HMM uses designated decoder seed 42001; '
			'other decoders use 42000. Original v1 figure used 42000 for MAE.'
		)
	return manifest


def _facies_prepared(
	plan: dict[str, Any], survey: str, arrays: list[Any]
) -> facies.PreparedComparison:
	sp = plan['surveys'][survey]
	manifest = _manifest(plan, survey)
	if survey == 'f3':
		inputs = {
			k.removesuffix('_path'): v
			for k, v in sp['reference']['inputs'].items()
			if k.endswith('_path')
		}
		seismic = np.load(inputs['seismic_volume'], mmap_mode='r')
		labels = np.load(inputs['label_volume'], mmap_mode='r')
		coords, coordinate_report = facies.load_coordinates(inputs, seismic.shape)
		manifest['coordinates'] = coordinate_report
		classes = _read(inputs['class_info'])['classes']
	else:
		data = (
			Path(plan['config']['artifact_root']) / 'data/parihaka/facies_benchmark_v1'
		)
		seismic = np.load(data / 'parihaka_amplitude.npy', mmap_mode='r')
		labels = channel_figures.BinaryLabels(
			np.load(data / 'parihaka_labels.npy', mmap_mode='r')
		)
		coords = {
			d: np.arange(1, n + 1)
			for d, n in zip(facies.DOMAINS, seismic.shape, strict=True)
		}
		classes = sp['reference']['classes']
	config = {'figure': sp['reference']['figure'], 'models': sp['sources']}
	return facies.PreparedComparison(
		config, seismic, labels, arrays, classes, coords, manifest, sp['rows']
	)


def verify_publication(root: Path, rows: list[dict[str, Any]]) -> None:
	"""Verify the exact published set, readable PNGs and original 300-dpi sizing."""
	actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
	if actual != expected_figure_files(rows):
		raise ValueError('unexpected/missing figure publication files')
	for row in rows:
		with Image.open(root / row['output_png']) as image:
			image.load()
			if image.size != (2125, 1240):
				raise ValueError('figure pixel dimensions differ from v1')


def render_survey(plan: dict[str, Any], survey: str) -> Path:
	"""Reuse the original renderers with new checked data and fixed v1 sections."""
	sp = plan['surveys'][survey]
	root = Path(plan['config']['output_root']) / 'figures' / survey / 'layout_000/large'
	if root.exists():
		raise FileExistsError(f'refusing to replace completed figures: {root}')
	arrays = _predictions(plan, survey)
	if survey == 'volve':
		manifest = _manifest(plan, survey)
		data = load_volve_horizon_data(plan['config']['volve_root'])
		canonical = _read(manifest['canonical_input_metadata_path'])
		prepared = {
			'config': {'figure': sp['reference']['figure']},
			'data': data,
			'seismic': np.load(
				canonical['provenance']['amplitude']['path'], mmap_mode='r'
			),
			'coords': {
				'inline': data.inline_values,
				'crossline': data.crossline_values,
				'time': data.time_ms.astype(np.float64) / 1000,
			},
			'models': sp['sources'],
			'manifest': manifest,
		}
		horizon_figures.validate_shared_sources(prepared, canonical)
	else:
		prepared = _facies_prepared(plan, survey, arrays)
		manifest = prepared.manifest
	root.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.TemporaryDirectory(prefix='.figures-', dir=root.parent) as tmp:
		stage = Path(tmp)
		for row in sp['rows']:
			if survey == 'volve':
				fig = horizon_figures.render_figure(prepared, row, arrays)
			else:
				fig = (
					channel_figures.render
					if survey == 'parihaka'
					else facies.render_figure
				)(prepared, row)
			path = stage / row['output_png']
			path.parent.mkdir(parents=True, exist_ok=True)
			try:
				fig.savefig(path, dpi=300, facecolor='white', bbox_inches=None)
			finally:
				plt.close(fig)
			print(f'Rendered {survey}/{row["output_png"]}', flush=True)
		facies.write_selected_csv(stage / 'selected_slices.csv', sp['rows'])
		manifest['outputs'] = (
			horizon_figures.verify_outputs
			if survey == 'volve'
			else facies.verify_outputs
		)(stage, sp['rows'])
		manifest['published_files'] = sorted(expected_figure_files(sp['rows']))
		(stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
		verify_publication(stage, sp['rows'])
		stage.rename(root)
	return root


def publish_dis0_review(plan: dict[str, Any], root: Path) -> Path:
	"""Publish only a Markdown index linking all eighty artifact-side figures."""
	if plan['config']['experiment_id'] != DIS0_EXPERIMENT:
		raise ValueError('review index requires a v2_dis0 plan')
	if root.exists():
		raise FileExistsError(f'refusing to replace completed review: {root}')
	lines = [
		'# v2_dis0: 全断面比較図',
		'',
		'F3 30枚・Parihaka 30枚・Volve 20枚、計80枚。',
		'',
		'v2と同じlayout_000/large、断面位置・表示範囲・振幅スケール・配色・',
		'2x4配置・180x105 mm/300 dpiです。蒸留係数だけを0にした',
		'MAE・Barlow Twins・Randomを、それぞれHMMなしの親モデルと並べています。',
		'',
		'Barlow Twinsは各データセットで25 epoch学習・特徴抽出と、図に使用する',
		'layout_000/largeのdecoder学習を追加しました。他のレイアウト・教師量は',
		'今回の追加実験の対象に含めていません。MAE・Randomは既存結果を再利用しました。',
		'',
		'K=[6,8,10]・整合性0.1。元のv2モデル・図は保持しています。',
		'ParihakaのHMMなしMAEは指定済みdecoder seed 42001、他は42000です。',
		'全断面の定性的比較であり、新たな独立test領域を示す図ではありません。',
		'',
		'公開ファイルはREADME.mdの1点のみ。予測配列と全図はartifactsに保存し、',
		'このレビュー文書はパイプライン入力には使用しません。',
		'',
	]
	for survey in SURVEYS:
		folder = (
			Path(plan['config']['output_root'])
			/ 'figures'
			/ survey
			/ 'layout_000/large'
		)
		rows = plan['surveys'][survey]['rows']
		verify_publication(folder, rows)
		lines += [f'## {survey.upper() if survey == "f3" else survey.title()}', '']
		for name in ('manifest.json', 'selected_slices.csv'):
			lines += [f'[{name}]({os.path.relpath(folder / name, root)})', '']
		for domain in facies.DOMAINS:
			selected = [r for r in rows if r['domain'] == domain]
			if not selected:
				continue
			lines += [f'### {domain.title()}', '']
			lines += [
				f'- [{r["actual_coordinate"]}]'
				f'({os.path.relpath(folder / r["output_png"], root)})'
				for r in selected
			]
			lines += ['']
	root.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.TemporaryDirectory(prefix='.review-', dir=root.parent) as tmp:
		stage = Path(tmp)
		(stage / 'README.md').write_text('\n'.join(lines))
		if {p.name for p in stage.iterdir()} != DIS0_REVIEW_FILES:
			raise ValueError('unexpected review publication files')
		stage.rename(root)
	return root / 'README.md'


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI without reading artifacts or starting execution."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--config', type=Path, required=True)
	mode = parser.add_mutually_exclusive_group()
	mode.add_argument('--dry-run', action='store_true')
	mode.add_argument('--execute', action='store_true')
	parser.add_argument('--survey', choices=SURVEYS)
	return parser


def main() -> None:
	"""Preflight by default; explicitly execute inference and figure creation."""
	args = build_parser().parse_args()
	plan = inspect_plan(args.config)
	print(
		json.dumps(
			{
				'mode': 'execute' if args.execute else 'dry_run',
				'figures': {s: len(p['rows']) for s, p in plan['surveys'].items()},
				'output_root': plan['config']['output_root'],
			}
		),
		flush=True,
	)
	if args.execute:
		torch.set_num_threads(1)
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True
		for survey in (args.survey,) if args.survey else SURVEYS:
			render_survey(plan, survey)
		if plan['config'].get('experiment_id') == DIS0_EXPERIMENT and not args.survey:
			publish_dis0_review(plan, Path(plan['config']['review_root']))

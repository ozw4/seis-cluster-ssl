"""Definitions and completion audit for the matched zero/zero k-means control."""

# ruff: noqa: PLC0415

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from seis_ssl_cluster.clustering.kmeans_definition import (
	LAYOUTS,
	SIZES,
	SOURCES,
	SURVEYS,
	_replace,
	read_yaml,
)
from seis_ssl_cluster.clustering.kmeans_targets import file_reference
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.hmm.distillation_sensitivity import read_rows, write_rows

STUDY = 'kmeans_k6810_dis0_con0_v1'
MATRIX = Path('experiments/kmeans') / STUDY / 'matrix.yaml'
TEST_NAMESPACE = 'lithology_common_test_kmeans_dis0_con0_v1'
SUMMARY_FILES = frozenset({'cells.csv', 'comparison.csv', 'summary.json', 'summary.md'})
REVIEW_FILES = frozenset(
	{'README.md', 'cells.csv', 'comparison.csv', 'provenance.json'}
)


def validate_live_inputs(matrix: dict[str, Any], workspace: Path) -> None:
	"""Verify frozen targets match HMM support and the original encoder."""
	from seis_ssl_cluster.clustering.kmeans_targets import (
		load_unordered_target_manifest,
	)

	for arms in matrix['surveys'].values():
		for arm in arms.values():
			raw = read_yaml(workspace / arm['configs']['training'])
			path = Path(os.path.expandvars(raw['pseudo_targets']['manifest']))
			if file_sha256(path) != arm['target_manifest_sha256']:
				raise ValueError('frozen k-means target manifest hash drift')
			target = load_unordered_target_manifest(path)
			parent = Path(os.path.expandvars(raw['student']['init_checkpoint']))
			if file_sha256(parent) != arm['parent_checkpoint_sha256']:
				raise ValueError('original encoder checkpoint hash drift')
			for survey in target['source_embedding']['surveys'].values():
				metadata = json.loads(Path(survey['metadata']['path']).read_text())
				if metadata['checkpoint_sha256'] != arm['parent_checkpoint_sha256']:
					raise ValueError('k-means features came from a different encoder')
			hmm = load_config(workspace / arm['references']['training']['path'])
			manifest = json.loads(Path(hmm['pseudo_targets']['manifest']).read_text())
			if (
				target['common']['valid_tokens_sha256']
				!= manifest['common']['valid_tokens_sha256']
			):
				raise ValueError('k-means and HMM pseudo-target support differs')


def arm_configs(workspace: Path, arm: dict[str, Any]) -> dict[str, dict[str, Any]]:
	"""Derive only target representation and new identities from zero/zero HMM."""
	baseline = {}
	for stage, ref in arm['references'].items():
		path = workspace / ref['path']
		if file_sha256(path) != ref['sha256']:
			raise ValueError('zero/zero reference hash drift')
		baseline[stage] = read_yaml(path)
	cid, control = arm['candidate_id'], arm['control_id']
	if cid != control.replace('hmm_v2', 'kmeans'):
		raise ValueError('zero/zero candidate identity drift')
	result = {
		stage: _replace(
			copy.deepcopy(raw),
			{
				arm['reference_experiment']: arm['experiment'],
				control: cid,
				'v2_dis0_con0_v1': STUDY,
			},
		)
		for stage, raw in baseline.items()
		if stage != 'targets'
	}
	full = result['training']
	if (
		full['head']['ks'] != [6, 8, 10]
		or full['loss']['distillation_weight'] != 0.0
		or full['loss']['consistency_weight'] != 0.0
		or full['train']['feature_diagnostics_samples'] != 16
		or full['train']['seed'] != 42
		or full['train']['epochs'] != 25
	):
		raise ValueError('expected original K6810 zero/zero HMM recipe')
	full['pseudo_targets'] = {
		'manifest': baseline['targets']['output_root']
		+ '/multi_head_target_manifest.json',
		'target_representation': 'unordered_kmeans_labels_v1',
		'min_confidence': 0.0,
	}
	full['identity']['scientific_identity'] = {
		'experiment_role': 'multi_head_unordered_kmeans_pretext',
		'variant': 'kmeans_nocons',
		'head_spec': full['head']['spec'],
		'head_ks': [6, 8, 10],
		'target_representation': 'unordered_kmeans_labels_v1',
		'target_manifest_sha256': (
			'${SEIS_SSL_CLUSTER_MULTI_HEAD_TARGET_MANIFEST_SHA256}'
		),
		'consistency_policy': 'disabled_for_unordered_kmeans_v1',
	}
	smoke = copy.deepcopy(full)
	smoke['paths']['output_root'] = full['paths']['output_root'].replace(
		'/full_25ep', '/smoke_1step'
	)
	smoke['train'].update(epochs=1, max_steps=1)
	result.update(smoke=smoke, targets=baseline['targets'])
	return result


def validate_definition(matrix: dict[str, Any], workspace: Path) -> dict[str, Any]:
	"""Reject missing sources, stale references, scientific changes and collisions."""
	if (
		matrix.get('schema_version') != 1
		or matrix.get('study') != STUDY
		or set(matrix['surveys']) != set(SURVEYS)
	):
		raise ValueError('invalid zero/zero k-means matrix')
	outputs = set()
	for survey in SURVEYS:
		if set(matrix['surveys'][survey]) != set(SOURCES):
			raise ValueError('zero/zero requires all three sources')
		for arm in matrix['surveys'][survey].values():
			expected = arm_configs(workspace, arm)
			if set(expected) != set(arm['configs']):
				raise ValueError('zero/zero stage set drift')
			for stage, raw in expected.items():
				if read_yaml(workspace / arm['configs'][stage]) != raw:
					raise ValueError('zero/zero definition drift')
			for stage in ('training', 'smoke'):
				output = expected[stage]['paths']['output_root']
				if output in outputs or f'/{STUDY}/' not in output:
					raise ValueError('zero/zero output collision')
				outputs.add(output)
	return matrix


def validate_smoke(arm: dict[str, Any], workspace: Path) -> dict[str, Any]:
	"""Require zero weighted constraints and identical initial HMM parameters."""
	import torch

	cfg = read_yaml(workspace / arm['configs']['smoke'])
	root = Path(os.path.expandvars(cfg['paths']['output_root']))
	payload = torch.load(root / 'latest.pt', map_location='cpu', weights_only=False)
	metrics = payload['metrics']
	if (
		payload['global_step'] != 1
		or metrics['loss_distillation_weighted'] != 0.0
		or metrics['loss_consistency_weighted'] != 0.0
		or metrics['loss_prototype_weighted'] <= 0.0
	):
		raise ValueError('zero/zero smoke losses violate the experiment contract')
	meta = json.loads((root / 'run_metadata.json').read_text())
	ref = load_config(workspace / arm['references']['training']['path'])
	old = json.loads(
		(Path(ref['paths']['output_root']) / 'run_metadata.json').read_text()
	)
	if (
		meta['control_identity']['initial_parameter_sha256']
		!= old['control_identity']['initial_parameter_sha256']
	):
		raise ValueError('initial parameters differ from zero/zero HMM')
	return {
		'candidate': arm['candidate_id'],
		'zero_constraints': True,
		'initial_parameters_match': True,
		'prototype': metrics['loss_prototype_weighted'],
	}


def validate_full_history(arm: dict[str, Any], workspace: Path) -> None:
	"""Require all 25 epochs and zero auxiliary contributions in saved history."""
	raw = read_yaml(workspace / arm['configs']['training'])
	root = Path(os.path.expandvars(raw['paths']['output_root']))
	rows = read_rows(root / 'multi_head_epoch_metrics.csv')
	if [int(r['epoch']) for r in rows] != list(range(1, 26)) or any(
		float(r[k]) != 0.0
		for r in rows
		for k in ('loss_distillation_weighted', 'loss_consistency_weighted')
	):
		raise ValueError('25 complete epochs with zero constraints required')


def prepare_test(matrix: dict[str, Any]) -> Path:
	"""Bind the three unordered conditions before scoring their 45 F3 test cells."""
	import yaml

	artifact = Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
	output = artifact / 'studies/kmeans' / STUDY / 'test_definition'
	if (output / 'evaluate.yaml').is_file():
		return output / 'evaluate.yaml'
	rows = []
	for source, arm in matrix['surveys']['f3'].items():
		raw = load_config(arm['configs']['downstream'])
		cp = Path(raw['candidate']['checkpoint'])
		embedding = Path(raw['candidate']['embeddings_dir'])
		rows.append(
			{
				'audit_id': 'kmeans_dis0_con0__' + source,
				'version': 'kmeans',
				'source': source,
				'head_ks': '6|8|10',
				'distillation_weight': '0.0',
				'consistency_weight': '0',
				'historical_condition_id': arm['candidate_id'],
				'layouts': '|'.join(LAYOUTS),
				'sizes': '|'.join(SIZES),
				'existing_cells': 15,
				'checkpoint_sha256': file_sha256(cp),
				'embedding_metadata_sha256': file_sha256(
					next(embedding.glob('*.embedding_metadata.json'))
				),
				'checkpoint_artifact_relative': str(cp.relative_to(artifact)),
				'embeddings_artifact_relative': str(embedding.relative_to(artifact)),
				'historical_runs_artifact_relative': str(
					(
						Path(raw['outputs']['runs_root'])
						/ f'model={arm["candidate_id"]}'
					).relative_to(artifact)
				),
			}
		)
	output.mkdir(parents=True, exist_ok=False)
	inventory = output / 'condition_inventory.csv'
	write_rows(inventory, rows)
	config = load_config(
		'experiments/f3/facies_benchmark_v2/136_lithology_common_test_v3_v1/evaluation.yaml'
	)
	config.update(
		experiment_id=TEST_NAMESPACE,
		expected_conditions=3,
		expected_cells=45,
		output_root=str(
			artifact / 'surveys/f3/facies_benchmark_v2/evaluation' / TEST_NAMESPACE
		),
	)
	config['inputs']['inventory'] = {
		'path': str(inventory),
		'sha256': file_sha256(inventory),
	}
	path = output / 'evaluate.yaml'
	path.write_text(yaml.safe_dump(config, sort_keys=False))
	return path


def finalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
	"""Replace F3 validation with verified common-test values on both sides."""
	from seis_ssl_cluster.clustering.kmeans_study import (
		comparison_rows,
	)

	artifact = Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
	base = artifact / 'surveys/f3/facies_benchmark_v2/evaluation'
	indices = []
	mask_identity = None
	for namespace in (TEST_NAMESPACE, 'lithology_common_test_v2_dis0_con0_v1'):
		definition = json.loads(
			(base / namespace / 'inputs/test_definition.json').read_text()
		)
		identity = (
			definition['mask_sha256'],
			definition['voxel_count'],
			definition['split'],
		)
		if identity[1:] != (113979624, 'test') or (
			mask_identity is not None and identity != mask_identity
		):
			raise ValueError('F3 comparisons require identical common-test masks')
		mask_identity = identity
		rows = read_rows(base / namespace / 'summary/per_layout.csv')
		index = {(r['source'], r['layout_id'], r['data_size']): r for r in rows}
		if len(rows) != 45 or len(index) != 45:
			raise ValueError('all 45 F3 common-test cells required')
		indices.append(index)
	for cell in payload['cells']:
		cell['evaluation_split'] = 'common_test' if cell['survey'] == 'f3' else 'test'
		cell['decoder_seed'] = 42000
		cell['primary_difference'] = cell['primary'] - cell['control_primary']
		cell['secondary_difference'] = cell['secondary'] - cell['control_secondary']
		if cell['survey'] == 'f3':
			key = (cell['source_family'], cell['layout_id'], cell['data_size'])
			for prefix, index, namespace in zip(
				('', 'control_'),
				indices,
				(TEST_NAMESPACE, 'lithology_common_test_v2_dis0_con0_v1'),
				strict=True,
			):
				row = index[key]
				# Use the immutable per-layout table as the score source; its audit
				# receipts bind predictions, decoder and the shared test mask.
				cell[prefix + 'primary'] = float(row['mean_iou'])
				cell[prefix + 'secondary'] = float(row['macro_f1'])
				cell['candidate_metrics' if not prefix else 'control_metrics'] = (
					file_reference(base / namespace / 'summary/per_layout.csv')
				)
			cell['primary_difference'] = cell['primary'] - cell['control_primary']
			cell['secondary_difference'] = cell['secondary'] - cell['control_secondary']
		cell['primary_improvement'] = cell['primary_difference'] * (
			-1 if cell['survey'] == 'volve' else 1
		)
	payload['rows'] = comparison_rows(payload['cells'])
	for row in payload['rows']:
		row['evaluation_split'] = 'common_test' if row['survey'] == 'f3' else 'test'
	payload['interpretation'] = (
		'Fixed unordered k-means versus HMM target generation, with distillation=0 '
		'and consistency=0 on both sides. Raw differences are kmeans minus HMM; '
		'lower horizon MAE is better.'
	)
	return payload


def publish_review(summary: Path, output: Path) -> None:
	"""Publish the exact four portable review files without execution payloads."""
	if output.exists():
		raise FileExistsError(output)
	output.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.TemporaryDirectory(
		prefix='.kmeans-review-', dir=output.parent
	) as tmp:
		stage = Path(tmp)
		for name in ('cells.csv', 'comparison.csv'):
			rows = read_rows(summary / name)
			for row in rows:
				for key in ('candidate_metrics', 'control_metrics'):
					row.pop(key, None)
			write_rows(stage / name, rows)
		(stage / 'README.md').write_text((summary / 'summary.md').read_text())
		(stage / 'provenance.json').write_text(
			json.dumps(
				{
					'study': STUDY,
					'status': 'complete',
					'cells': 135,
					'published_files': sorted(REVIEW_FILES),
					'summary_sha256': file_sha256(summary / 'summary.json'),
					'review_files_sha256': {
						p.name: file_sha256(p) for p in stage.iterdir()
					},
				},
				indent=2,
			)
			+ '\n'
		)
		if {p.name for p in stage.iterdir()} != REVIEW_FILES:
			raise ValueError('unexpected zero/zero review files')
		shutil.move(str(stage), str(output))


def execute_recorded(  # noqa: PLR0913, PLR0917
	matrix: dict[str, Any],
	workspace: Path,
	root: Path,
	index: int,
	total: int,
	stage: str,
	command: list[str],
	log: Path,
) -> None:
	"""Record admissions, verify all smokes before full runs, and stop on failure."""
	from datetime import datetime, timezone

	from seis_ssl_cluster.clustering.kmeans_study import run_command

	state = {
		'status': 'running',
		'stage': stage,
		'step': index + 1,
		'total_steps': total,
		'command': command,
		'pid': os.getpid(),
		'log': str(log),
		'updated_at_utc': datetime.now(timezone.utc).isoformat(),
	}
	path = root / 'status.json'
	path.write_text(json.dumps(state, indent=2) + '\n')
	try:
		if stage == 'smoke':
			config = Path(command[command.index('--config') + 1])
			arm = next(
				a
				for arms in matrix['surveys'].values()
				for a in arms.values()
				if workspace / a['configs']['smoke'] == config
			)
			raw = read_yaml(config)
			if (
				Path(os.path.expandvars(raw['paths']['output_root'])) / 'latest.pt'
			).exists():
				validate_smoke(arm, workspace)
				log.write_text('Reused verified completed zero/zero smoke\n')
				return
		wait_for_gpu(stage, state, path)
		if stage == 'full':
			verified = [
				validate_smoke(arm, workspace)
				for arms in matrix['surveys'].values()
				for arm in arms.values()
			]
			(root / 'all_smokes_verified.json').write_text(
				json.dumps(verified, indent=2) + '\n'
			)
		run_command(stage, command, dry_run=False, log=log)
		if stage == 'smoke':
			config = Path(command[command.index('--config') + 1])
			arm = next(
				a
				for arms in matrix['surveys'].values()
				for a in arms.values()
				if workspace / a['configs']['smoke'] == config
			)
			result = validate_smoke(arm, workspace)
			(
				root
				/ ('smoke_' + arm['candidate_id'] + '_' + config.parts[-5] + '.json')
			).write_text(json.dumps(result, indent=2) + '\n')
	except BaseException as exc:
		state.update(
			status='failed',
			error=str(exc),
			updated_at_utc=datetime.now(timezone.utc).isoformat(),
		)
		path.write_text(json.dumps(state, indent=2) + '\n')
		raise


def refresh_catalog(workspace: Path) -> None:
	"""Refresh CSV/HTML/ZIP while preserving the previous catalog."""
	import zipfile
	from datetime import datetime, timezone

	from seis_ssl_cluster.results.experiment_catalog import (
		build_catalog,
	)

	artifact = Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
	parent = artifact / 'studies/all_experiment_metrics_20261115_v1'
	staging = parent / ('catalog_' + STUDY)
	build_catalog(workspace, artifact, staging)
	if (parent / 'catalog/experiment_metrics.csv').exists():
		from seis_ssl_cluster.results.experiment_catalog import MAIN_METRICS

		old = read_rows(parent / 'catalog/experiment_metrics.csv')
		new = read_rows(staging / 'experiment_metrics.csv')
		fields = ('source_file', 'split', 'support', 'checkpoint_selection')
		index = {tuple(r[k] for k in fields): r for r in new}
		if any(
			tuple(r[k] for k in fields) not in index
			or any(
				r.get(metric, '') != index[tuple(r[k] for k in fields)].get(metric, '')
				for metric in MAIN_METRICS
			)
			for r in old
		):
			raise ValueError('catalog refresh omitted or changed a saved evaluation')
	manifest = json.loads((staging / 'manifest.json').read_text())
	archive = staging.with_suffix('.zip')
	with zipfile.ZipFile(
		archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6
	) as stream:
		for name in manifest['published_files']:
			if (
				name != 'manifest.json'
				and file_sha256(staging / name) != manifest['sha256'][name]
			):
				raise ValueError('catalog publication checksum drift')
			stream.write(staging / name, 'catalog/' + name)
	with zipfile.ZipFile(archive) as stream:
		if stream.testzip() is not None:
			raise ValueError('catalog archive integrity failed')
	backup = (
		artifact
		/ 'operations/kmeans'
		/ STUDY
		/ ('previous_catalog_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
	)
	backup.mkdir(parents=True)
	for name in ('catalog', 'catalog.zip'):
		if (parent / name).exists():
			shutil.move(str(parent / name), str(backup / name))
	staging.rename(parent / 'catalog')
	archive.rename(parent / 'catalog.zip')


def wait_for_gpu(stage: str, state: dict[str, Any], path: Path) -> None:
	"""Require two idle GPU readings before admitting a fixed-size job."""
	if stage in ('smoke', 'full', 'embeddings', 'downstream') and os.environ.get(
		'CUDA_VISIBLE_DEVICES'
	):
		import time

		from seis_ssl_cluster.hmm.v2_dis0_con0 import gpu_available

		device = os.environ['CUDA_VISIBLE_DEVICES']
		required = 38 * 1024 if stage in ('smoke', 'full', 'embeddings') else 12 * 1024
		while True:
			if gpu_available(device, required):
				time.sleep(5)
				if gpu_available(device, required):
					break
			state['status'] = 'waiting_for_gpu'
			path.write_text(json.dumps(state, indent=2) + '\n')
			time.sleep(30)
		state['status'] = 'running'
		path.write_text(json.dumps(state, indent=2) + '\n')

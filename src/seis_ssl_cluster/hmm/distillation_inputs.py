"""Live input and initial sampling audit for the fixed distillation study."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from seis_ssl_cluster.clustering.features import file_sha256
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.config.pretraining import resolve_strat_hmm_pretext_config
from seis_ssl_cluster.data import NopimsStratMultiHeadTargetDataset, read_manifest_json
from seis_ssl_cluster.hmm.distillation_sensitivity import (
	MATRIX,
	read_rows,
	validate_matrix,
)
from seis_ssl_cluster.hmm.multi_source_cell import audit_completed_cell
from seis_ssl_cluster.training import load_checkpoint
from seis_ssl_cluster.training.dataloaders import (
	build_strat_multi_head_target_dataloader,
)
from seis_ssl_cluster.training.strat_hmm.components import build_strat_hmm_components
from seis_ssl_cluster.training.strat_hmm.feature_diagnostics import (
	FixedCropFeatureDiagnostics,
)
from seis_ssl_cluster.training.strat_hmm.runner import _state_dict_sha256
from seis_ssl_cluster.training.strat_hmm.runtime import _zero_mask_from_config
from seis_ssl_cluster.training.strat_hmm_source_audit import audit_multi_head_source


def build_dataset(config: dict[str, Any]) -> NopimsStratMultiHeadTargetDataset:
	"""Use every inherited sampling and preprocessing value."""
	data = config['data']
	return NopimsStratMultiHeadTargetDataset(
		read_manifest_json(Path(config['manifests']['train'])),
		config['pseudo_targets']['manifest'],
		local_crop_size_xyz=data['local_crop_size'],
		patch_size_xyz=config['model']['patch_size'],
		seed=config['train']['seed'],
		samples_per_epoch=config['train']['samples_per_epoch'],
		zero_mask=_zero_mask_from_config(config),
		min_valid_fraction=data['min_valid_fraction'],
		max_resample_attempts=data['max_resample_attempts'],
		normalized_clip_abs=data['normalized_clip_abs'],
		amplitude_agc=data['amplitude_agc'],
		finite_check_mode=data['finite_check_mode'],
		min_confidence=config['pseudo_targets']['min_confidence'],
	)


def audit_inputs(*, include_baseline_cells: bool) -> dict[str, Any]:  # noqa: C901
	"""Compare actual initial tensors and the first two eight-worker batches."""
	matrix = validate_matrix()
	records = []
	for survey, arms in matrix['surveys'].items():
		initial_reference, batch_reference = None, None
		for weight, arm in arms.items():
			path = Path(arm['configs']['training']['path'])
			config = resolve_strat_hmm_pretext_config(load_config(path))
			references = {
				name: {
					'path': config[section][key],
					'sha256': file_sha256(config[section][key]),
				}
				for name, section, key in [
					('student', 'student', 'init_checkpoint'),
					('teacher', 'teacher', 'checkpoint'),
					('target', 'pseudo_targets', 'manifest'),
					('train_manifest', 'manifests', 'train'),
				]
			}
			if references['student'] != references['teacher']:
				raise ValueError('student and teacher must start at the same MAE100')
			torch.manual_seed(42)
			c = build_strat_hmm_components(config, device='cpu')
			initial = {
				name: _state_dict_sha256(module.state_dict())
				for name, module in [
					('student', c.student),
					('head', c.heads),
					('teacher', c.teacher),
				]
			}
			initial['torch_rng'] = hashlib.sha256(
				torch.get_rng_state().numpy().tobytes()
			).hexdigest()
			if initial_reference is None:
				initial_reference = initial
			if initial != initial_reference:
				raise ValueError('initial parameters/RNG differ across coefficients')
			dataset = build_dataset(config)
			loader = build_strat_multi_head_target_dataloader(
				dataset,
				batch_size=config['train']['batch_size'],
				num_workers=config['train']['num_workers'],
				shuffle=config['train']['shuffle'],
				seed=42,
				device='cpu',
			)
			iterator = iter(loader)
			hashes, coords = [], []
			try:
				for _ in range(2):
					batch = next(iterator)
					h = hashlib.sha256()
					for value in [
						batch['x'],
						batch['local_valid_mask'],
						*(
							v
							for target in batch['strat_multi_targets'].values()
							for v in target.values()
						),
					]:
						h.update(value.numpy().tobytes())
					hashes.append(h.hexdigest())
					coords.append(batch['coords'])
			finally:
				iterator._shutdown_workers()  # noqa: SLF001
			if batch_reference is None:
				batch_reference = hashes
			if hashes != batch_reference:
				raise ValueError('initial sample/crop sequence differs')
			record = {
				'survey': survey,
				'weight': weight,
				'references': references,
				'initial': initial,
				'first_two_batches_sha256': hashes,
				'coords': coords,
				'config': str(path),
				'config_sha256': file_sha256(path),
			}
			if weight == '0.2' and include_baseline_cells:
				record['source_audit'] = audit_multi_head_source(path)
				historical = json.loads(
					(
						Path(record['source_audit']['checkpoint']['path']).parent
						/ 'run_metadata.json'
					).read_text()
				)['control_identity']['initial_state_sha256']
				if historical != {k: initial[k] for k in ('student', 'head')}:
					raise ValueError('historical 0.2 initialization differs')
				raw = load_config(arm['configs']['downstream']['path'])
				record['downstream_audits'] = [
					{
						'layout': layout,
						'size': size,
						'audit': audit_completed_cell(
							survey, raw, arm['candidate_id'], layout, size
						),
					}
					for layout in [f'layout_{i:03d}' for i in range(5)]
					for size in ('small', 'medium', 'large')
				]
			records.append(record)
			print(survey, weight, 'OK', flush=True)
			del c, dataset, loader, iterator
			gc.collect()
	return {'status': 'complete', 'records': records}


def diagnose_baselines(matrix_path: Path = MATRIX) -> None:
	"""Evaluate retained 0.2 initial/final weights in separate outputs."""
	root = Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
	matrix = validate_matrix(matrix_path)
	for survey, arms in matrix['surveys'].items():
		output = (
			root
			/ 'studies/hmm_v2'
			/ matrix.get('analysis_id', 'distillation_sensitivity_v1')
			/ 'baseline_diagnostics'
			/ survey
		)
		resume = matrix.get('source') == 'random'
		if (
			resume
			and (output / 'feature_diagnostics.csv').exists()
			and {int(r['epoch']) for r in read_rows(output / 'feature_diagnostics.csv')}
			== {0, 25}
		):
			continue
		config = resolve_strat_hmm_pretext_config(
			load_config(arms['0.2']['configs']['training']['path'])
		)
		# Positive-weight construction follows exactly the original initialization.
		torch.manual_seed(42)
		components = build_strat_hmm_components(config, device='cuda')
		output.mkdir(parents=True, exist_ok=resume)
		diagnostic = FixedCropFeatureDiagnostics(
			build_dataset(config), samples=16, output_root=output
		)
		diagnostic.record(
			components.student, components.teacher, epoch=0, global_step=0
		)
		payload = load_checkpoint(
			Path(config['paths']['output_root']) / 'latest.pt', map_location='cpu'
		)
		components.student.load_state_dict(payload['model_state_dict'])
		diagnostic.record(
			components.student,
			components.teacher,
			epoch=25,
			global_step=payload['global_step'],
		)
		(output / 'availability.json').write_text(
			json.dumps(
				{
					'available_epochs': [0, 25],
					'unavailable_fixed_crop_epochs': list(range(1, 25)),
					'reason': (
						'Intermediate 0.2 encoder checkpoints were not retained; '
						'Existing per-epoch training-crop distances '
						'remain in original metrics.'
					),
				},
				indent=2,
			)
			+ '\n'
		)
		del components, payload, diagnostic
		torch.cuda.empty_cache()


def main() -> None:
	"""Write study-owned audit evidence or initial/final baseline diagnostics."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--matrix', type=Path, default=MATRIX)
	parser.add_argument('--with-baseline-cells', action='store_true')
	parser.add_argument('--diagnose-baselines', action='store_true')
	args = parser.parse_args()
	if args.diagnose_baselines:
		diagnose_baselines(args.matrix)
	else:
		if args.matrix != MATRIX:
			parser.error('--matrix is supported only with --diagnose-baselines')
		payload = audit_inputs(include_baseline_cells=args.with_baseline_cells)
		output = (
			Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
			/ 'operations/hmm_v2/distillation_sensitivity_v1'
		)
		output.mkdir(parents=True, exist_ok=True)
		with (
			output
			/ (
				'preflight_recheck.json'
				if args.with_baseline_cells
				else 'initial_alignment_recheck.json'
			)
		).open('x') as stream:
			json.dump(payload, stream, indent=2, default=str)
			stream.write('\n')


if __name__ == '__main__':
	main()

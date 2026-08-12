# ruff: noqa: CPY001
"""Portable, lightweight review results for completed Parihaka MAE training."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

from seis_ssl_cluster.config import load_config, resolve_mae_training_config
from seis_ssl_cluster.parihaka.mae_validation import (
	MODEL_TAG,
	ParihakaMaeValidationResult,
	validate_parihaka_mae,
)
from seis_ssl_cluster.parihaka.prepare_volume import (
	parihaka_prepare_volume_config_from_mapping,
)

SUMMARY_JSON_NAME = 'parihaka_mae_pretraining_summary.json'
SUMMARY_MARKDOWN_NAME = 'parihaka_mae_pretraining_summary.md'
CHECKPOINT_JSON_NAME = 'parihaka_mae_checkpoint_summary.json'
PARIHAKA_MAE_RESULT_FILES = (
	SUMMARY_JSON_NAME,
	SUMMARY_MARKDOWN_NAME,
	CHECKPOINT_JSON_NAME,
)
_MAX_RESULT_BYTES = 10 * 1024 * 1024
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class ParihakaMaeResults:
	"""The three producer-owned review files and whether all were reused."""

	output_dir: Path
	paths: tuple[Path, ...]
	reused: bool


def summarize_parihaka_mae(
	*,
	prepare_config_path: str | Path,
	full_config_path: str | Path,
	output_dir: str | Path,
	overwrite: bool = False,
) -> ParihakaMaeResults:
	"""Validate live evidence and write exactly three portable review files."""
	prepare_path = Path(prepare_config_path)
	full_path = Path(full_config_path)
	smoke_path = full_path.parent / '01_smoke_2step.yaml'
	validation = validate_parihaka_mae(
		prepare_config_path=prepare_path,
		smoke_config_path=smoke_path,
		full_config_path=full_path,
		check='full',
	)
	evidence = _build_evidence(
		prepare_config_path=prepare_path,
		full_config_path=full_path,
		validation=validation,
	)
	checkpoint_summary = cast(
		'Mapping[str, object]', evidence['checkpoints']
	)
	contents = {
		SUMMARY_JSON_NAME: _json_bytes(evidence),
		SUMMARY_MARKDOWN_NAME: _render_markdown(evidence).encode(),
		CHECKPOINT_JSON_NAME: _json_bytes(checkpoint_summary),
	}
	for name, content in contents.items():
		if len(content) >= _MAX_RESULT_BYTES:
			raise ValueError(f'result file must be smaller than 10 MiB: {name}')
	root = Path(output_dir)
	targets = tuple(root / name for name in PARIHAKA_MAE_RESULT_FILES)
	conflicts = [
		path
		for path in targets
		if path.exists() and path.read_bytes() != contents[path.name]
	]
	if conflicts and not overwrite:
		joined = ', '.join(map(str, conflicts))
		raise FileExistsError(
			f'producer-owned result content differs; use --overwrite: {joined}'
		)
	root.mkdir(parents=True, exist_ok=True)
	reused = True
	for target in targets:
		content = contents[target.name]
		if target.exists() and target.read_bytes() == content:
			continue
		target.write_bytes(content)
		reused = False
	return ParihakaMaeResults(output_dir=root, paths=targets, reused=reused)


def _build_evidence(
	*,
	prepare_config_path: Path,
	full_config_path: Path,
	validation: ParihakaMaeValidationResult,
) -> dict[str, object]:
	prepare = parihaka_prepare_volume_config_from_mapping(
		load_config(prepare_config_path)
	)
	full = resolve_mae_training_config(load_config(full_config_path))
	metadata = _read_json(prepare.outputs.metadata)
	provenance = _mapping(metadata, 'provenance')
	source = _mapping(metadata, 'source')
	outputs = _mapping(metadata, 'outputs')
	amplitude = _mapping(outputs, 'amplitude_npy')
	latest_path = _required_path(validation.latest_checkpoint, 'latest checkpoint')
	best_path = _required_path(validation.best_checkpoint, 'best checkpoint')
	latest_metrics = dict(validation.latest_metrics)
	run_metadata = _read_json(validation.full_output_root / 'run_metadata.json')
	created_at_utc = _required_string(run_metadata, 'created_at_utc', 'run metadata')
	git_commit = _required_optional_string(run_metadata, 'git_commit', 'run metadata')
	best_metric_key = validation.best_metric_key
	best_metric_value = validation.best_metric_value
	if best_metric_key is None or best_metric_value is None:
		raise ValueError('validated best checkpoint has no best metric')
	artifact_root = prepare.paths.artifact_root
	data_root = prepare.paths.parihaka_root
	checkpoint_summary = {
		'dataset': {
			'name': 'parihaka',
			'version': 'facies_benchmark_v1',
			'survey_id': 'parihaka',
			'model_tag': MODEL_TAG,
		},
		'prepared_amplitude': {
			'sha256': amplitude['sha256'],
			'shape_xyz': amplitude['shape_xyz'],
			'dtype': amplitude['dtype'],
			'order': amplitude['order'],
		},
		'precision': {
			'amp_requested': True,
			'amp_dtype_requested': 'auto',
			'resolved': validation.resolved_precision,
			'scaler_present': validation.scaler_present,
		},
		'primary_role': 'completed latest',
		'latest': {
			'path': _portable_path(latest_path, artifact_root, data_root),
			'sha256': validation.latest_sha256,
			'size_bytes': latest_path.stat().st_size,
			'schema_version': validation.checkpoint_schema_version,
			'epoch': validation.checkpoint_epoch,
			'global_step': validation.checkpoint_global_step,
			'metrics': latest_metrics,
		},
		'best': {
			'path': _portable_path(best_path, artifact_root, data_root),
			'sha256': validation.best_sha256,
			'size_bytes': best_path.stat().st_size,
			'schema_version': validation.checkpoint_schema_version,
			'epoch': validation.best_checkpoint_epoch,
			'global_step': validation.best_checkpoint_global_step,
			'metric_key': best_metric_key,
			'metric_value': best_metric_value,
			'role': 'strictly-lower training-loss diagnostic',
		},
		'finite_metric_range': [
			validation.finite_metric_min,
			validation.finite_metric_max,
		],
	}
	return {
		'artifact_type': 'parihaka_mae_pretraining_review',
		'schema_version': 2,
		'dataset': {
			'name': 'parihaka',
			'version': 'facies_benchmark_v1',
			'survey_id': 'parihaka',
			'model_tag': MODEL_TAG,
		},
		'provenance': dict(provenance),
		'source': {
			'path': _portable_path(
				prepare.inputs.amplitude_npz, artifact_root, data_root
			),
			'sha256': source['npz_sha256'],
			'size_bytes': source['npz_size_bytes'],
			'identity_qualification': {
				'aicrowd_byte_identity': provenance['aicrowd_byte_identity'],
				'redistribution_transformation': provenance[
					'redistribution_transformation'
				],
			},
		},
		'prepared_amplitude': {
			'path': _portable_path(
				prepare.outputs.amplitude_npy, artifact_root, data_root
			),
			'sha256': amplitude['sha256'],
			'size_bytes': amplitude['size_bytes'],
			'shape_xyz': amplitude['shape_xyz'],
			'dtype': amplitude['dtype'],
			'order': amplitude['order'],
			'statistics': amplitude['statistics'],
		},
		'direct_inputs': {
			'manifest': _portable_identity(
				_mapping(outputs, 'manifest'), artifact_root, data_root
			),
			'path_list': _portable_identity(
				_mapping(outputs, 'path_list'), artifact_root, data_root
			),
			'normalization_stats': _portable_identity(
				_mapping(outputs, 'normalization_stats'), artifact_root, data_root
			),
		},
		'input_boundary': {
			'modality': 'amplitude only',
			'labels_used': False,
			'label_files_opened': 0,
		},
		'scientific_scope': {
			'kind': 'survey-specific transductive self-supervised pretraining',
			'claim': (
				'Unlabeled representation learning from the Parihaka amplitude volume.'
			),
			'not_established': [
				'label leakage',
				'transfer to unseen surveys',
				'inductive holdout performance',
				'downstream accuracy improvement',
				'geological or channel interpretation performance',
			],
		},
		'full_config': {
			'data': full['data'],
			'zero_mask': full['zero_mask'],
			'model': full['model'],
			'masking': full['masking'],
			'loss': full['loss'],
			'train': full['train'],
			'visualization': full['visualization'],
			'optimizer': 'AdamW',
			'initialization': 'random from seed 42',
		},
		'precision': {
			'amp_requested': True,
			'amp_dtype_requested': 'auto',
			'resolved': validation.resolved_precision,
			'scaler_present': validation.scaler_present,
		},
		'training_invocation': {
			'created_at_utc': created_at_utc,
			'git_commit': git_commit,
			'git_dirty': None,
		},
		'summary_generation': {
			'git_commit': _git_sha(),
		},
		'training_completion': {
			'epoch': validation.checkpoint_epoch,
			'global_step': validation.checkpoint_global_step,
		},
		'checkpoints': checkpoint_summary,
		'downstream': {
			'status': 'checkpoint ready, evaluation not run',
			'embedding_jobs_executed': 0,
			'clustering_jobs_executed': 0,
			'downstream_jobs_executed': 0,
		},
	}


def _render_markdown(evidence: Mapping[str, object]) -> str:
	dataset = _mapping(evidence, 'dataset')
	precision = _mapping(evidence, 'precision')
	completion = _mapping(evidence, 'training_completion')
	checkpoints = _mapping(evidence, 'checkpoints')
	training_invocation = _mapping(evidence, 'training_invocation')
	summary_generation = _mapping(evidence, 'summary_generation')
	latest = _mapping(checkpoints, 'latest')
	best = _mapping(checkpoints, 'best')
	return (
		'# Parihaka amplitude MAE pretraining summary\n\n'
		f"- Dataset: `{dataset['name']}/{dataset['version']}` "
		f"(`{dataset['survey_id']}`)\n"
		f"- Model tag: `{dataset['model_tag']}`\n"
		f"- Training invocation: `{training_invocation['created_at_utc']}`, "
		f"git `{training_invocation['git_commit']}`\n"
		f"- Summary generation git: `{summary_generation['git_commit']}`\n"
		'- Input boundary: amplitude only; labels were not used or opened.\n'
		'- Scope: survey-specific transductive self-supervised pretraining.\n'
		f"- Completion: epoch {completion['epoch']}, "
		f"global step {completion['global_step']}\n"
		f"- Precision: AMP requested (`auto`), resolved `{precision['resolved']}`, "
		f"scaler present `{str(precision['scaler_present']).lower()}`\n"
		f"- Primary checkpoint: `{latest['path']}` (`latest.pt`, schema 2)\n"
		f"- Diagnostic checkpoint: `{best['path']}` (epoch {best['epoch']}, "
		f"{best['metric_key']}={best['metric_value']})\n"
		'- Downstream status: checkpoint ready; evaluation was not run.\n\n'
		'Training completion and loss do not establish geological interpretation, '
		'channel-estimation performance, transfer to unseen surveys, or downstream '
		'accuracy improvement.\n'
	)


def _portable_identity(
	record: Mapping[str, object], artifact_root: Path, data_root: Path
) -> dict[str, object]:
	return {
		'path': _portable_path(
			Path(cast('str', record['path'])), artifact_root, data_root
		),
		'sha256': record['sha256'],
	}


def _portable_path(path: Path, artifact_root: Path, data_root: Path) -> str:
	resolved = path.resolve(strict=False)
	for root, token in (
		(artifact_root.resolve(strict=False), '${SEIS_SSL_CLUSTER_ARTIFACT_ROOT}'),
		(data_root.resolve(strict=False), '${PARIHAKA_DATA_ROOT}'),
	):
		try:
			relative = resolved.relative_to(root)
		except ValueError:
			continue
		return f'{token}/{relative.as_posix()}'
	raise ValueError(f'cannot make machine-local path portable: {path}')


def _json_bytes(payload: object) -> bytes:
	text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n'
	return text.encode()


def _read_json(path: Path) -> Mapping[str, object]:
	payload = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(payload, Mapping):
		raise TypeError(f'expected JSON object: {path}')
	return cast('Mapping[str, object]', payload)


def _mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		raise TypeError(f'{key} must be a mapping')
	return cast('Mapping[str, object]', value)


def _required_path(value: Path | None, label: str) -> Path:
	if value is None:
		raise ValueError(f'full validation did not return {label}')
	return value


def _required_optional_string(
	payload: Mapping[str, object], key: str, label: str
) -> str | None:
	if key not in payload:
		raise ValueError(f'{label}.{key} is required')
	value = payload[key]
	if value is not None and not isinstance(value, str):
		raise TypeError(f'{label}.{key} must be a string or null')
	return cast('str | None', value)


def _required_string(
	payload: Mapping[str, object], key: str, label: str
) -> str:
	value = payload.get(key)
	if not isinstance(value, str) or not value:
		raise TypeError(f'{label}.{key} must be a non-empty string')
	return value


def _git_sha() -> str | None:
	git = shutil.which('git')
	if git is None:
		return None
	try:
		return subprocess.check_output(  # noqa: S603
			[git, 'rev-parse', 'HEAD'],
			cwd=Path(__file__).resolve().parents[3],
			text=True,
			stderr=subprocess.DEVNULL,
		).strip()
	except (OSError, subprocess.CalledProcessError):
		return None


def file_sha256(path: Path) -> str:
	"""Return SHA-256 for final reporting and focused result tests."""
	digest = sha256()
	with path.open('rb') as file_obj:
		for chunk in iter(lambda: file_obj.read(_HASH_CHUNK_BYTES), b''):
			digest.update(chunk)
	return digest.hexdigest()


__all__ = [
	'CHECKPOINT_JSON_NAME',
	'PARIHAKA_MAE_RESULT_FILES',
	'SUMMARY_JSON_NAME',
	'SUMMARY_MARKDOWN_NAME',
	'ParihakaMaeResults',
	'file_sha256',
	'summarize_parihaka_mae',
]

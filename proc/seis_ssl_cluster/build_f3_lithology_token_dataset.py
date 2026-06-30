"""Build F3 token-level lithology datasets from supervised slice locations."""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.f3 import (
	F3LithologyTokenDatasetConfig,
	F3LithologyTokenDatasetInputs,
	F3LithologyTokenDatasetOutputs,
	F3LithologyTokenPolicy,
	build_f3_lithology_token_dataset,
)

STAGE = 'build_f3_lithology_token_dataset'
DEFAULT_CONFIG = (
	Path(__file__).resolve().parents[2]
	/ 'experiments'
	/ 'f3'
	/ 'facies_benchmark_v1'
	/ '50_lithology'
	/ 'amp_mae_m075_mse_g0_patchnorm_clip8_agc65_vis01_v1'
	/ 'overlap_x16'
	/ 'png_slices_segy_labels_v1'
	/ '01_build_token_dataset.yaml'
)


def main() -> None:
	"""Build F3 lithology token datasets or print a dry-run summary."""
	parser = ArgumentParser(description='Build F3 lithology token datasets.')
	parser.add_argument(
		'--config',
		type=Path,
		default=DEFAULT_CONFIG,
		help='Path to a YAML configuration file.',
	)
	parser.add_argument(
		'--dry-run',
		action='store_true',
		help='Validate the config and print a run summary without writing outputs.',
	)
	args = parser.parse_args()

	raw_config = load_config(args.config)
	config = f3_lithology_token_dataset_config_from_mapping(raw_config)
	if args.dry_run:
		_print_summary(config)
		print('execution: dry-run; F3 lithology token dataset build skipped')
		return

	result = build_f3_lithology_token_dataset(config)
	print(f'f3_lithology_token_dataset.train_tokens: {result.train_npz}')
	print(f'f3_lithology_token_dataset.validation_tokens: {result.validation_npz}')
	print(f'f3_lithology_token_dataset.all_labeled_tokens: {result.all_labeled_npz}')
	print(f'f3_lithology_token_dataset.metadata_json: {result.metadata_json}')
	print(f'f3_lithology_token_dataset.class_counts_csv: {result.class_counts_csv}')
	print(f'f3_lithology_token_dataset.summary_markdown: {result.summary_markdown}')
	print(f'f3_lithology_token_dataset.split_manifest: {result.split_manifest_json}')
	print(f'f3_lithology_token_dataset.quicklook_count: {len(result.quicklook_paths)}')
	print(f'f3_lithology_token_dataset.train_token_count: {result.train_token_count}')
	print(
		'f3_lithology_token_dataset.validation_token_count: '
		f'{result.validation_token_count}',
	)


def f3_lithology_token_dataset_config_from_mapping(
	config: Mapping[str, object],
) -> F3LithologyTokenDatasetConfig:
	"""Validate and normalize the F3 lithology token dataset config."""
	_validate_allowed_keys(
		config,
		frozenset(
			{
				'paths',
				'dataset',
				'model',
				'embeddings',
				'labels',
				'registry',
				'lithology',
				'token_dataset',
			},
		),
		prefix='config',
	)
	paths = _required_mapping(config, 'paths')
	artifact_root = _required_absolute_path(paths, 'artifact_root', prefix='paths')
	f3_root = _required_absolute_path(paths, 'f3_root', prefix='paths')
	dataset = _required_mapping(config, 'dataset')
	model = _required_mapping(config, 'model')
	embeddings = _required_mapping(config, 'embeddings')
	labels = _required_mapping(config, 'labels')
	registry = _required_mapping(config, 'registry')
	token_dataset = _required_mapping(config, 'token_dataset')
	outputs = _outputs_from_mapping(token_dataset)
	for label, path in _output_paths(outputs):
		_validate_artifact_output_path(
			path,
			label,
			artifact_root=artifact_root,
			f3_root=f3_root,
		)
	inputs = F3LithologyTokenDatasetInputs(
		embeddings_dir=_required_absolute_path(
			embeddings,
			'input_dir',
			prefix='embeddings',
		),
		label_volume=_required_absolute_path(
			labels,
			'source_label_volume',
			prefix='labels',
		),
		seismic_volume=_required_absolute_path(
			registry,
			'seismic_volume',
			prefix='registry',
		),
		png_label_inventory=_required_absolute_path(
			labels,
			'png_label_inventory',
			prefix='labels',
		),
		class_info=_required_absolute_path(labels, 'class_info', prefix='labels'),
		segy_geometry_json=_required_absolute_path(
			labels,
			'segy_geometry_json',
			prefix='labels',
		),
		source_label_segy=_optional_absolute_path(
			labels,
			'source_label_segy',
			prefix='labels',
		),
		volume_metadata_json=_optional_absolute_path(
			registry,
			'metadata_json',
			prefix='registry',
		),
	)
	policy = _policy_from_mapping(_required_mapping(token_dataset, 'tokenization'))
	return F3LithologyTokenDatasetConfig(
		inputs=inputs,
		outputs=outputs,
		policy=policy,
		dataset=dataset,
		model=model,
		figure_dpi=_figure_dpi(token_dataset),
	)


def _outputs_from_mapping(
	token_dataset: Mapping[str, object],
) -> F3LithologyTokenDatasetOutputs:
	return F3LithologyTokenDatasetOutputs(
		output_dir=_required_absolute_path(
			token_dataset,
			'output_dir',
			prefix='token_dataset',
		),
		metadata_json=_required_absolute_path(
			token_dataset,
			'metadata_json',
			prefix='token_dataset',
		),
		class_counts_csv=_required_absolute_path(
			token_dataset,
			'class_counts_csv',
			prefix='token_dataset',
		),
		summary_markdown=_required_absolute_path(
			token_dataset,
			'summary_markdown',
			prefix='token_dataset',
		),
		split_manifest_json=_required_absolute_path(
			token_dataset,
			'split_manifest',
			prefix='token_dataset',
		),
		quicklook_dir=_required_absolute_path(
			token_dataset,
			'quicklook_dir',
			prefix='token_dataset',
		),
	)


def _policy_from_mapping(policy: Mapping[str, object]) -> F3LithologyTokenPolicy:
	for key in ('patch_size', 'patch_size_xyz'):
		if key in policy:
			msg = (
				'token_dataset.tokenization must not override patch size; '
				'patch size is read from embedding metadata'
			)
			raise ValueError(msg)
	return F3LithologyTokenPolicy(
		min_labeled_fraction=_required_fraction(
			policy,
			'min_labeled_fraction',
			prefix='token_dataset.tokenization',
		),
		min_majority_fraction=_required_fraction(
			policy,
			'min_majority_fraction',
			prefix='token_dataset.tokenization',
		),
		ignore_z_border_samples=_required_nonnegative_int(
			policy,
			'ignore_z_border_samples',
			prefix='token_dataset.tokenization',
		),
	)


def _figure_dpi(token_dataset: Mapping[str, object]) -> int:
	figure = token_dataset.get('figure')
	if figure is None:
		return 300
	if not isinstance(figure, Mapping):
		msg = f'token_dataset.figure must be a mapping; got {figure!r}'
		raise TypeError(msg)
	return _optional_positive_int(
		figure.get('dpi', 300),
		'token_dataset.figure.dpi',
	)


def _output_paths(
	outputs: F3LithologyTokenDatasetOutputs,
) -> tuple[tuple[str, Path], ...]:
	return (
		('token_dataset.output_dir', outputs.output_dir),
		('token_dataset.metadata_json', outputs.metadata_json),
		('token_dataset.class_counts_csv', outputs.class_counts_csv),
		('token_dataset.summary_markdown', outputs.summary_markdown),
		('token_dataset.split_manifest', outputs.split_manifest_json),
		('token_dataset.quicklook_dir', outputs.quicklook_dir),
	)


def _print_summary(config: F3LithologyTokenDatasetConfig) -> None:
	print(f'stage: {STAGE}')
	print(f'embeddings.input_dir: {config.inputs.embeddings_dir}')
	print(f'labels.source_label_volume: {config.inputs.label_volume}')
	print(f'labels.source_label_segy: {config.inputs.source_label_segy}')
	print(f'labels.png_label_inventory: {config.inputs.png_label_inventory}')
	print(f'labels.class_info: {config.inputs.class_info}')
	print(f'labels.segy_geometry_json: {config.inputs.segy_geometry_json}')
	print(f'registry.seismic_volume: {config.inputs.seismic_volume}')
	print(f'registry.metadata_json: {config.inputs.volume_metadata_json}')
	print('token_dataset.patch_size_source: embedding metadata')
	print(
		'token_dataset.tokenization.min_labeled_fraction: '
		f'{config.policy.min_labeled_fraction}',
	)
	print(
		'token_dataset.tokenization.min_majority_fraction: '
		f'{config.policy.min_majority_fraction}',
	)
	print(
		'token_dataset.tokenization.ignore_z_border_samples: '
		f'{config.policy.ignore_z_border_samples}',
	)
	print(f'token_dataset.output_dir: {config.outputs.output_dir}')
	print(f'token_dataset.metadata_json: {config.outputs.metadata_json}')
	print(f'token_dataset.class_counts_csv: {config.outputs.class_counts_csv}')
	print(f'token_dataset.summary_markdown: {config.outputs.summary_markdown}')
	print(f'token_dataset.split_manifest: {config.outputs.split_manifest_json}')
	print(f'token_dataset.quicklook_dir: {config.outputs.quicklook_dir}')
	print(f'token_dataset.figure.dpi: {config.figure_dpi}')


def _required_mapping(
	parent: Mapping[str, object],
	key: str,
) -> Mapping[str, Any]:
	value = parent.get(key)
	if not isinstance(value, Mapping):
		msg = f'{key} must be a mapping; got {value!r}'
		raise TypeError(msg)
	return value


def _validate_allowed_keys(
	parent: Mapping[str, object],
	allowed: frozenset[str],
	*,
	prefix: str,
) -> None:
	unexpected = sorted(set(parent) - allowed)
	if unexpected:
		msg = (
			f'{prefix} key(s) not allowed: {unexpected!r}; '
			f'allowed keys are {sorted(allowed)!r}'
		)
		raise ValueError(msg)


def _required_absolute_path(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> Path:
	path = Path(_required_str(parent, key, prefix=prefix))
	if not path.is_absolute():
		msg = f'{prefix}.{key} must be an absolute path; got {path}'
		raise ValueError(msg)
	return path


def _optional_absolute_path(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> Path | None:
	value = parent.get(key)
	if value is None:
		return None
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	path = Path(value)
	if not path.is_absolute():
		msg = f'{prefix}.{key} must be an absolute path; got {path}'
		raise ValueError(msg)
	return path


def _required_str(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> str:
	value = parent.get(key)
	if not isinstance(value, str) or not value:
		msg = f'{prefix}.{key} must be a non-empty string; got {value!r}'
		raise TypeError(msg)
	return value


def _required_fraction(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> float:
	value = parent.get(key)
	if not isinstance(value, int | float) or isinstance(value, bool):
		msg = f'{prefix}.{key} must be a number in [0, 1]; got {value!r}'
		raise TypeError(msg)
	fraction = float(value)
	if not 0.0 <= fraction <= 1.0:
		msg = f'{prefix}.{key} must be in [0, 1]; got {value!r}'
		raise ValueError(msg)
	return fraction


def _required_nonnegative_int(
	parent: Mapping[str, object],
	key: str,
	*,
	prefix: str,
) -> int:
	value = parent.get(key)
	if not isinstance(value, int) or isinstance(value, bool) or value < 0:
		msg = f'{prefix}.{key} must be a nonnegative integer; got {value!r}'
		raise ValueError(msg)
	return value


def _optional_positive_int(value: object, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
		msg = f'{label} must be a positive integer; got {value!r}'
		raise ValueError(msg)
	return value


def _validate_artifact_output_path(
	path: Path,
	label: str,
	*,
	artifact_root: Path,
	f3_root: Path,
) -> None:
	if 'runs' in path.parts:
		msg = f'{label} must not use runs/ paths; got {path}'
		raise ValueError(msg)
	if _is_relative_to(path, f3_root):
		msg = f'{label} must not be under paths.f3_root; got {path}'
		raise ValueError(msg)
	if not _is_relative_to(path, artifact_root):
		msg = (
			f'{label} must be under paths.artifact_root '
			f'({artifact_root}); got {path}'
		)
		raise ValueError(msg)


def _is_relative_to(path: Path, root: Path) -> bool:
	try:
		path.resolve(strict=False).relative_to(root.resolve(strict=False))
	except ValueError:
		return False
	return True


if __name__ == '__main__':
	main()

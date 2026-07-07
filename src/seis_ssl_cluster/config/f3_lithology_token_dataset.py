"""F3 lithology token dataset config validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from seis_ssl_cluster.config.f3_lithology_common import (
	_optional_absolute_path,
	_optional_positive_int,
	_required_absolute_path,
	_required_fraction,
	_required_mapping,
	_required_nonnegative_int,
	_validate_allowed_keys,
	_validate_artifact_path_not_f3,
)
from seis_ssl_cluster.f3.lithology.tokens import (
	F3LithologyTokenDatasetConfig,
	F3LithologyTokenDatasetInputs,
	F3LithologyTokenDatasetOutputs,
	F3LithologyTokenPolicy,
	F3ReferenceTokenDataset,
)

if TYPE_CHECKING:
	from pathlib import Path


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
				'feature_source',
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
	outputs = _token_dataset_outputs_from_mapping(token_dataset)
	for label, path in _token_dataset_output_paths(outputs):
		_validate_artifact_path_not_f3(
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
	policy = _token_dataset_policy_from_mapping(
		_required_mapping(token_dataset, 'tokenization'),
	)
	reference_token_dataset = _reference_token_dataset_from_mapping(token_dataset)
	feature_source = _feature_source(
		config,
		token_dataset,
		reference_token_dataset=reference_token_dataset,
	)
	return F3LithologyTokenDatasetConfig(
		inputs=inputs,
		outputs=outputs,
		policy=policy,
		dataset=dataset,
		model=model,
		figure_dpi=_token_dataset_figure_dpi(token_dataset),
		feature_source=feature_source,
		reference_token_dataset=reference_token_dataset,
	)


def _token_dataset_outputs_from_mapping(
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


def _token_dataset_policy_from_mapping(
	policy: Mapping[str, object],
) -> F3LithologyTokenPolicy:
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


def _token_dataset_figure_dpi(token_dataset: Mapping[str, object]) -> int:
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


def _feature_source(
	config: Mapping[str, object],
	token_dataset: Mapping[str, object],
	*,
	reference_token_dataset: F3ReferenceTokenDataset | None,
) -> Mapping[str, object] | None:
	top_level = config.get('feature_source')
	nested = token_dataset.get('feature_source')
	if top_level is not None and nested is not None and top_level != nested:
		msg = 'config.feature_source and token_dataset.feature_source must match'
		raise ValueError(msg)
	value = top_level if top_level is not None else nested
	if value is None:
		return None
	if not isinstance(value, Mapping):
		msg = f'feature_source must be a mapping; got {value!r}'
		raise TypeError(msg)
	feature_source = {
		'kind': _feature_source_str(value, 'kind'),
		'reference_model_tag': _feature_source_str(value, 'reference_model_tag'),
		'embedding_spec': _feature_source_str(value, 'embedding_spec'),
		'description': _feature_source_str(value, 'description'),
	}
	if (
		reference_token_dataset is None
		and feature_source['kind'] != 'pretrained_encoder'
	):
		msg = (
			'feature_source.kind must be "pretrained_encoder" for pretrained '
			f'token datasets; got {feature_source["kind"]!r}'
		)
		raise ValueError(msg)
	return feature_source


def _feature_source_str(value: Mapping[str, object], key: str) -> str:
	item = value.get(key)
	if not isinstance(item, str) or not item:
		msg = f'feature_source.{key} must be a non-empty string; got {item!r}'
		raise TypeError(msg)
	return item


def _reference_token_dataset_from_mapping(
	token_dataset: Mapping[str, object],
) -> F3ReferenceTokenDataset | None:
	value = token_dataset.get('reference_token_dataset')
	if value is None:
		return None
	if not isinstance(value, Mapping):
		msg = f'token_dataset.reference_token_dataset must be a mapping; got {value!r}'
		raise TypeError(msg)
	root = _optional_absolute_path(
		value,
		'root',
		prefix='token_dataset.reference_token_dataset',
	)
	train_tokens = _optional_absolute_path(
		value,
		'train_tokens',
		prefix='token_dataset.reference_token_dataset',
	)
	validation_tokens = _optional_absolute_path(
		value,
		'validation_tokens',
		prefix='token_dataset.reference_token_dataset',
	)
	metadata_json = _optional_absolute_path(
		value,
		'metadata_json',
		prefix='token_dataset.reference_token_dataset',
	)
	split_manifest_json = _optional_absolute_path(
		value,
		'split_manifest_json',
		prefix='token_dataset.reference_token_dataset',
	)
	split_manifest = _optional_absolute_path(
		value,
		'split_manifest',
		prefix='token_dataset.reference_token_dataset',
	)
	if root is not None:
		train_tokens = train_tokens or root / 'train_tokens.npz'
		validation_tokens = validation_tokens or root / 'validation_tokens.npz'
		metadata_json = metadata_json or root / 'token_dataset_metadata.json'
		split_manifest_json = (
			split_manifest_json or split_manifest or root / 'splits.json'
		)
	else:
		split_manifest_json = split_manifest_json or split_manifest
	if train_tokens is None or validation_tokens is None or metadata_json is None:
		msg = (
			'token_dataset.reference_token_dataset requires root or explicit '
			'train_tokens, validation_tokens, and metadata_json paths'
		)
		raise KeyError(msg)
	return F3ReferenceTokenDataset(
		train_tokens=train_tokens,
		validation_tokens=validation_tokens,
		metadata_json=metadata_json,
		split_manifest_json=split_manifest_json,
		root=root,
	)


def _token_dataset_output_paths(
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
__all__ = ['f3_lithology_token_dataset_config_from_mapping']

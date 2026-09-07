"""Strict provenance evidence for completed F3 five-way evaluations."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

from seis_ssl_cluster.config.f3_lithology_voxel_section_layout import (
	DATA_SIZES,
	LAYOUT_IDS,
)
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.f3.lithology.five_way_runner import (
	BEST_CHECKPOINT_NAME,
	DECODER_DIR_NAME,
	EVALUATION_DIR_NAME,
	METRICS_NAME,
	PREDICTION_DIR_NAME,
	PREDICTION_METADATA_NAME,
)
from seis_ssl_cluster.f3.lithology.voxel_prediction_artifact import (
	ARTIFACT_TYPE as PREDICTION_ARTIFACT_TYPE,
)
from seis_ssl_cluster.f3.lithology.voxel_prediction_artifact import (
	CONFIDENCE_NAME,
	PREDICTIONS_NAME,
	VALID_MASK_NAME,
)
from seis_ssl_cluster.f3.lithology.voxel_prediction_artifact import (
	SCHEMA_VERSION as PREDICTION_SCHEMA_VERSION,
)
from seis_ssl_cluster.training.voxel_decoder.checkpoint import (
	load_voxel_decoder_checkpoint,
)

_EVALUATION_ARTIFACT_TYPE = 'f3_lithology_voxel_evaluation'
_EVALUATION_METADATA_NAME = 'evaluation_metadata.json'
_EVALUATION_SCHEMA_VERSION = 2
_EXPECTED_AGGREGATION_UNIT = 'unique_validation_voxel'
_EXPECTED_PREDICTION_KIND = 'frozen_embedding_decoder'
_RESOLVED_CONFIG_NAME = 'resolved_config.json'
_VOXEL_DATASET_METADATA_NAME = 'voxel_dataset_metadata.json'
_CANONICAL_DATASET = {
	'name': 'f3_facies_benchmark',
	'version': 'facies_benchmark_v2',
}


class F3CompletedEvaluationProvenance(TypedDict):
	"""Identity-checked evidence for one completed benchmark evaluation."""

	model_id: str
	layout_id: str
	data_size: str
	aggregation_unit: str
	macro_f1: float
	evaluation_voxel_count: int
	metrics_path: str
	metrics_sha256: str
	evaluation_metadata_path: str
	evaluation_metadata_sha256: str
	prediction_metadata_path: str
	prediction_metadata_sha256: str
	voxel_predictions_path: str
	voxel_predictions_sha256: str
	voxel_confidence_path: str
	voxel_confidence_sha256: str
	voxel_valid_mask_path: str
	voxel_valid_mask_sha256: str
	decoder_checkpoint_path: str
	decoder_checkpoint_sha256: str
	decoder_resolved_config_path: str
	decoder_resolved_config_sha256: str
	voxel_dataset_metadata_path: str
	voxel_dataset_metadata_sha256: str
	label_volume_path: str
	label_volume_sha256: str
	class_info_path: str
	class_info_sha256: str
	source_label_segy_path: str
	source_label_segy_sha256: str
	png_label_inventory_path: str
	png_label_inventory_sha256: str
	segy_geometry_json_path: str
	segy_geometry_json_sha256: str


def read_f3_completed_evaluation_provenance(
	job_dir: Path,
	*,
	model_id: str,
	layout_id: str,
	data_size: str,
) -> F3CompletedEvaluationProvenance:
	"""Validate the exact evaluation-to-decoder lineage for one F3 job."""
	job_dir = Path(job_dir)
	_validate_expected_identity(
		job_dir,
		model_id=model_id,
		layout_id=layout_id,
		data_size=data_size,
	)
	label = f'{model_id}/{layout_id}/{data_size}'
	evaluation_dir = job_dir / EVALUATION_DIR_NAME
	prediction_dir = job_dir / PREDICTION_DIR_NAME
	decoder_dir = job_dir / DECODER_DIR_NAME
	metrics_path = evaluation_dir / METRICS_NAME
	evaluation_metadata_path = evaluation_dir / _EVALUATION_METADATA_NAME
	prediction_metadata_path = prediction_dir / PREDICTION_METADATA_NAME
	decoder_checkpoint_path = decoder_dir / BEST_CHECKPOINT_NAME
	decoder_config_path = decoder_dir / _RESOLVED_CONFIG_NAME

	metrics, metrics_sha256 = _read_json_with_sha(
		metrics_path, label=f'{label} metrics'
	)
	macro_f1 = _unit_interval_number(
		metrics.get('macro_f1'), label=f'{label} metrics macro_f1'
	)
	evaluation_voxel_count = _positive_int(
		metrics.get('evaluation_voxel_count'),
		label=f'{label} metrics evaluation_voxel_count',
	)
	if metrics.get('aggregation_unit') != _EXPECTED_AGGREGATION_UNIT:
		raise ValueError(
			f'{label} metrics aggregation_unit must equal '
			f'{_EXPECTED_AGGREGATION_UNIT!r}'
		)

	evaluation_metadata, evaluation_metadata_sha256 = _read_json_with_sha(
		evaluation_metadata_path, label=f'{label} evaluation metadata'
	)
	_validate_evaluation_metadata(
		evaluation_metadata,
		label=label,
		model_id=model_id,
		evaluation_metadata_path=evaluation_metadata_path,
		evaluation_voxel_count=evaluation_voxel_count,
	)
	metrics_recorded_sha256 = _verified_identity(
		evaluation_metadata.get('outputs'),
		METRICS_NAME,
		expected_path=metrics_path,
		actual_sha256=metrics_sha256,
		label=f'{label} evaluation outputs',
	)

	prediction_metadata, prediction_metadata_sha256 = _read_json_with_sha(
		prediction_metadata_path, label=f'{label} prediction metadata'
	)
	_verified_identity(
		evaluation_metadata.get('inputs'),
		'prediction_metadata',
		expected_path=prediction_metadata_path,
		actual_sha256=prediction_metadata_sha256,
		label=f'{label} evaluation inputs',
	)
	_validate_prediction_metadata(
		prediction_metadata,
		label=label,
		model_id=model_id,
		prediction_dir=prediction_dir,
		decoder_checkpoint_path=decoder_checkpoint_path,
	)

	array_evidence = _validate_prediction_arrays(
		evaluation_metadata,
		prediction_metadata,
		label=label,
		prediction_dir=prediction_dir,
	)
	decoder_checkpoint_sha256 = _current_file_sha256(
		decoder_checkpoint_path, label=f'{label} decoder checkpoint'
	)
	_verified_identity(
		prediction_metadata.get('source_identity'),
		'decoder_checkpoint',
		expected_path=decoder_checkpoint_path,
		actual_sha256=decoder_checkpoint_sha256,
		label=f'{label} prediction source_identity',
	)
	decoder_config, decoder_config_sha256 = _read_json_with_sha(
		decoder_config_path, label=f'{label} decoder resolved config'
	)
	_verified_identity(
		prediction_metadata.get('source_identity'),
		'resolved_decoder_config',
		expected_path=decoder_config_path,
		actual_sha256=decoder_config_sha256,
		label=f'{label} prediction source_identity',
	)
	decoder_checkpoint = load_voxel_decoder_checkpoint(
		decoder_checkpoint_path, map_location='cpu'
	)
	_validate_decoder_config_binding(
		decoder_config,
		decoder_checkpoint,
		evaluation_metadata,
		prediction_metadata,
		label=label,
		model_id=model_id,
		decoder_dir=decoder_dir,
	)
	ground_truth_evidence = _validate_ground_truth_provenance(
		evaluation_metadata,
		prediction_metadata,
		decoder_checkpoint,
		decoder_config,
		label=label,
		layout_id=layout_id,
		data_size=data_size,
	)

	return {
		'model_id': model_id,
		'layout_id': layout_id,
		'data_size': data_size,
		'aggregation_unit': _EXPECTED_AGGREGATION_UNIT,
		'macro_f1': macro_f1,
		'evaluation_voxel_count': evaluation_voxel_count,
		'metrics_path': str(metrics_path),
		'metrics_sha256': metrics_recorded_sha256,
		'evaluation_metadata_path': str(evaluation_metadata_path),
		'evaluation_metadata_sha256': evaluation_metadata_sha256,
		'prediction_metadata_path': str(prediction_metadata_path),
		'prediction_metadata_sha256': prediction_metadata_sha256,
		**array_evidence,
		'decoder_checkpoint_path': str(decoder_checkpoint_path),
		'decoder_checkpoint_sha256': decoder_checkpoint_sha256,
		'decoder_resolved_config_path': str(decoder_config_path),
		'decoder_resolved_config_sha256': decoder_config_sha256,
		**ground_truth_evidence,
	}


def _validate_expected_identity(
	job_dir: Path,
	*,
	model_id: str,
	layout_id: str,
	data_size: str,
) -> None:
	if not job_dir.is_absolute():
		raise ValueError('job_dir must be absolute')
	if not model_id or Path(model_id).name != model_id or model_id in {'.', '..'}:
		raise ValueError('model_id must be one non-empty path segment')
	if layout_id not in LAYOUT_IDS:
		raise ValueError(
			f'layout_id must be one of {list(LAYOUT_IDS)!r}; got {layout_id!r}'
		)
	if data_size not in DATA_SIZES:
		raise ValueError(
			f'data_size must be one of {list(DATA_SIZES)!r}; got {data_size!r}'
		)
	expected_suffix = (
		f'model={model_id}',
		f'layout={layout_id}',
		f'size={data_size}',
	)
	if job_dir.parts[-3:] != expected_suffix:
		raise ValueError(
			'evaluation job directory does not match the expected '
			f'model/layout/size: {job_dir}'
		)


def _validate_evaluation_metadata(
	metadata: Mapping[str, object],
	*,
	label: str,
	model_id: str,
	evaluation_metadata_path: Path,
	evaluation_voxel_count: int,
) -> None:
	if metadata.get('artifact_type') != _EVALUATION_ARTIFACT_TYPE:
		raise ValueError(f'{label} evaluation artifact_type is invalid')
	if metadata.get('schema_version') != _EVALUATION_SCHEMA_VERSION:
		raise ValueError(f'{label} evaluation schema_version is invalid')
	if metadata.get('dataset') != _CANONICAL_DATASET:
		raise ValueError(
			f'{label} evaluation dataset must equal {_CANONICAL_DATASET!r}'
		)
	if metadata.get('prediction_kind') != _EXPECTED_PREDICTION_KIND:
		raise ValueError(f'{label} evaluation prediction_kind is invalid')
	if metadata.get('model_tag') != model_id:
		raise ValueError(f'{label} evaluation model_tag must equal {model_id!r}')
	_exact_path(
		metadata.get('metadata_path'),
		expected=evaluation_metadata_path,
		label=f'{label} evaluation metadata_path',
	)
	aggregation = _mapping(
		metadata.get('aggregation'), label=f'{label} evaluation aggregation'
	)
	if aggregation.get('primary_unit') != _EXPECTED_AGGREGATION_UNIT:
		raise ValueError(
			f'{label} evaluation aggregation.primary_unit must equal '
			f'{_EXPECTED_AGGREGATION_UNIT!r}'
		)
	summary = _mapping(metadata.get('summary'), label=f'{label} evaluation summary')
	if summary.get('unique_validation_voxel_count') != evaluation_voxel_count:
		raise ValueError(f'{label} evaluation voxel count does not match metrics')


def _validate_prediction_metadata(
	metadata: Mapping[str, object],
	*,
	label: str,
	model_id: str,
	prediction_dir: Path,
	decoder_checkpoint_path: Path,
) -> None:
	if metadata.get('artifact_type') != PREDICTION_ARTIFACT_TYPE:
		raise ValueError(f'{label} prediction artifact_type is invalid')
	if metadata.get('schema_version') != PREDICTION_SCHEMA_VERSION:
		raise ValueError(f'{label} prediction schema_version is invalid')
	if metadata.get('prediction_kind') != _EXPECTED_PREDICTION_KIND:
		raise ValueError(f'{label} prediction prediction_kind is invalid')
	if metadata.get('model_tag') != model_id:
		raise ValueError(f'{label} prediction model_tag must equal {model_id!r}')
	inputs = _mapping(metadata.get('inputs'), label=f'{label} prediction inputs')
	_exact_path(
		inputs.get('decoder_checkpoint'),
		expected=decoder_checkpoint_path,
		label=f'{label} prediction inputs.decoder_checkpoint',
	)
	outputs = _mapping(metadata.get('outputs'), label=f'{label} prediction outputs')
	for key, name in (
		('predictions', PREDICTIONS_NAME),
		('confidence', CONFIDENCE_NAME),
		('valid_mask', VALID_MASK_NAME),
	):
		_exact_path(
			outputs.get(key),
			expected=prediction_dir / name,
			label=f'{label} prediction outputs.{key}',
		)


def _validate_prediction_arrays(
	evaluation_metadata: Mapping[str, object],
	prediction_metadata: Mapping[str, object],
	*,
	label: str,
	prediction_dir: Path,
) -> dict[str, str]:
	inputs = evaluation_metadata.get('inputs')
	outputs = _mapping(
		prediction_metadata.get('outputs'), label=f'{label} prediction outputs'
	)
	evidence: dict[str, str] = {}
	for output_key, input_key, name, result_prefix in (
		('predictions', 'voxel_predictions', PREDICTIONS_NAME, 'voxel_predictions'),
		('confidence', 'voxel_confidence', CONFIDENCE_NAME, 'voxel_confidence'),
		('valid_mask', 'voxel_valid_mask', VALID_MASK_NAME, 'voxel_valid_mask'),
	):
		expected_path = prediction_dir / name
		_exact_path(
			outputs.get(output_key),
			expected=expected_path,
			label=f'{label} prediction outputs.{output_key}',
		)
		actual_sha256 = _current_file_sha256(
			expected_path, label=f'{label} {input_key}'
		)
		sha256 = _verified_identity(
			inputs,
			input_key,
			expected_path=expected_path,
			actual_sha256=actual_sha256,
			label=f'{label} evaluation inputs',
		)
		evidence[f'{result_prefix}_path'] = str(expected_path)
		evidence[f'{result_prefix}_sha256'] = sha256
	return evidence


def _validate_decoder_config_binding(  # noqa: PLR0913
	resolved: Mapping[str, object],
	checkpoint: Mapping[str, object],
	evaluation_metadata: Mapping[str, object],
	prediction_metadata: Mapping[str, object],
	*,
	label: str,
	model_id: str,
	decoder_dir: Path,
) -> None:
	if resolved.get('dataset') != _CANONICAL_DATASET:
		raise ValueError(f'{label} decoder dataset must equal {_CANONICAL_DATASET!r}')
	model = _mapping(resolved.get('model'), label=f'{label} decoder model')
	if model.get('tag') != model_id:
		raise ValueError(f'{label} decoder model.tag must equal {model_id!r}')
	outputs = _mapping(resolved.get('outputs'), label=f'{label} decoder outputs')
	_exact_path(
		outputs.get('output_dir'),
		expected=decoder_dir,
		label=f'{label} decoder outputs.output_dir',
	)
	decoder = _mapping(resolved.get('decoder'), label=f'{label} decoder config')
	checkpoint_config = _mapping(
		checkpoint.get('resolved_config'),
		label=f'{label} checkpoint resolved_config',
	)
	if checkpoint_config != resolved:
		raise ValueError(f'{label} decoder resolved config does not match checkpoint')
	checkpoint_decoder = _mapping(
		checkpoint.get('decoder_architecture'),
		label=f'{label} checkpoint decoder_architecture',
	)
	prediction_decoder = _mapping(
		prediction_metadata.get('decoder_architecture'),
		label=f'{label} prediction decoder_architecture',
	)
	evaluation_decoder = _mapping(
		evaluation_metadata.get('decoder_architecture'),
		label=f'{label} evaluation decoder_architecture',
	)
	if not (decoder == checkpoint_decoder == prediction_decoder == evaluation_decoder):
		raise ValueError(
			f'{label} decoder architecture differs between checkpoint, '
			'resolved config, prediction, or evaluation metadata'
		)


def _validate_ground_truth_provenance(  # noqa: PLR0913
	evaluation_metadata: Mapping[str, object],
	prediction_metadata: Mapping[str, object],
	checkpoint: Mapping[str, object],
	resolved: Mapping[str, object],
	*,
	label: str,
	layout_id: str,
	data_size: str,
) -> dict[str, str]:
	voxel_dataset = _mapping(
		resolved.get('voxel_dataset'), label=f'{label} decoder voxel_dataset'
	)
	voxel_dataset_dir = _absolute_path(
		voxel_dataset.get('input_dir'),
		label=f'{label} decoder voxel_dataset.input_dir',
	)
	expected_suffix = (
		'datasets',
		f'layout={layout_id}',
		f'size={data_size}',
		'voxel_supervision',
	)
	if voxel_dataset_dir.parts[-4:] != expected_suffix:
		raise ValueError(
			f'{label} decoder voxel dataset is not the canonical layout/size'
		)
	voxel_metadata_path = voxel_dataset_dir / _VOXEL_DATASET_METADATA_NAME
	voxel_metadata, voxel_metadata_sha256 = _read_json_with_sha(
		voxel_metadata_path, label=f'{label} voxel dataset metadata'
	)

	evaluation_inputs = evaluation_metadata.get('inputs')
	_verified_identity(
		evaluation_inputs,
		'voxel_dataset_metadata',
		expected_path=voxel_metadata_path,
		actual_sha256=voxel_metadata_sha256,
		label=f'{label} evaluation inputs',
	)
	prediction_source = _mapping(
		prediction_metadata.get('source_identity'),
		label=f'{label} prediction source_identity',
	)
	prediction_artifacts = _mapping(
		prediction_source.get('artifact_identities'),
		label=f'{label} prediction artifact_identities',
	)
	checkpoint_artifacts = _mapping(
		checkpoint.get('artifact_identities'),
		label=f'{label} checkpoint artifact_identities',
	)
	for source_name, artifacts in (
		('prediction', prediction_artifacts),
		('checkpoint', checkpoint_artifacts),
	):
		if artifacts.get('name') != 'f3_voxel_decoder_sources':
			raise ValueError(f'{label} {source_name} artifact identity name is invalid')
		_verified_identity(
			artifacts,
			'voxel_dataset_metadata',
			expected_path=voxel_metadata_path,
			actual_sha256=voxel_metadata_sha256,
			label=f'{label} {source_name} artifact_identities',
		)

	_validate_voxel_metadata_identity(
		voxel_metadata,
		label=label,
		layout_id=layout_id,
		data_size=data_size,
	)
	voxel_sources = _mapping(
		voxel_metadata.get('source_identities'),
		label=f'{label} voxel dataset source_identities',
	)
	references = (
		('label_volume', 'label_volume', voxel_metadata, 'label_volume'),
		('class_info', 'class_info', voxel_sources, 'class_info'),
		('source_label_segy', 'source_label_segy', voxel_sources, 'source_label_segy'),
		('png_label_inventory', 'inventory', voxel_metadata, 'png_label_inventory'),
		(
			'segy_geometry_json',
			'segy_geometry_json',
			voxel_sources,
			'segy_geometry_json',
		),
	)
	evidence = {
		'voxel_dataset_metadata_path': str(voxel_metadata_path),
		'voxel_dataset_metadata_sha256': voxel_metadata_sha256,
	}
	for evaluation_key, reference_key, container, result_prefix in references:
		expected_path, expected_sha256 = _current_declared_identity(
			container,
			reference_key,
			label=f'{label} voxel dataset {reference_key}',
		)
		_verified_identity(
			evaluation_inputs,
			evaluation_key,
			expected_path=expected_path,
			actual_sha256=expected_sha256,
			label=f'{label} evaluation inputs',
		)
		evidence[f'{result_prefix}_path'] = str(expected_path)
		evidence[f'{result_prefix}_sha256'] = expected_sha256

	_validate_prediction_ground_truth_binding(
		prediction_metadata,
		prediction_source=prediction_source,
		prediction_artifacts=prediction_artifacts,
		checkpoint_artifacts=checkpoint_artifacts,
		label=label,
		label_volume_path=Path(evidence['label_volume_path']),
		label_volume_sha256=evidence['label_volume_sha256'],
		class_info_path=Path(evidence['class_info_path']),
		class_info_sha256=evidence['class_info_sha256'],
	)
	labels = _mapping(
		voxel_metadata.get('labels'), label=f'{label} voxel dataset labels'
	)
	_exact_path(
		labels.get('class_info'),
		expected=Path(evidence['class_info_path']),
		label=f'{label} voxel dataset labels.class_info',
	)
	_exact_path(
		labels.get('source_label_segy'),
		expected=Path(evidence['source_label_segy_path']),
		label=f'{label} voxel dataset labels.source_label_segy',
	)
	return evidence


def _validate_voxel_metadata_identity(
	metadata: Mapping[str, object],
	*,
	label: str,
	layout_id: str,
	data_size: str,
) -> None:
	if metadata.get('artifact_type') != 'f3_lithology_voxel_supervision':
		raise ValueError(f'{label} voxel dataset artifact_type is invalid')
	if metadata.get('schema_version') != 1:
		raise ValueError(f'{label} voxel dataset schema_version is invalid')
	if metadata.get('dataset') != _CANONICAL_DATASET:
		raise ValueError(f'{label} voxel dataset must equal {_CANONICAL_DATASET!r}')
	section_layout = _mapping(
		metadata.get('section_layout'),
		label=f'{label} voxel dataset section_layout',
	)
	if section_layout.get('layout_id') != layout_id:
		raise ValueError(f'{label} voxel dataset layout_id is invalid')
	if section_layout.get('data_size') != data_size:
		raise ValueError(f'{label} voxel dataset data_size is invalid')


def _validate_prediction_ground_truth_binding(  # noqa: PLR0913
	prediction_metadata: Mapping[str, object],
	*,
	prediction_source: Mapping[str, object],
	prediction_artifacts: Mapping[str, object],
	checkpoint_artifacts: Mapping[str, object],
	label: str,
	label_volume_path: Path,
	label_volume_sha256: str,
	class_info_path: Path,
	class_info_sha256: str,
) -> None:
	for source_name, artifacts in (
		('prediction', prediction_artifacts),
		('checkpoint', checkpoint_artifacts),
	):
		_verified_identity(
			artifacts,
			'label_volume',
			expected_path=label_volume_path,
			actual_sha256=label_volume_sha256,
			label=f'{label} {source_name} artifact_identities',
		)
	_verified_identity(
		prediction_source,
		'class_info',
		expected_path=class_info_path,
		actual_sha256=class_info_sha256,
		label=f'{label} prediction source_identity',
	)
	prediction_inputs = _mapping(
		prediction_metadata.get('inputs'), label=f'{label} prediction inputs'
	)
	_exact_path(
		prediction_inputs.get('class_info'),
		expected=class_info_path,
		label=f'{label} prediction inputs.class_info',
	)


def _verified_identity(
	container: object,
	key: str,
	*,
	expected_path: Path,
	actual_sha256: str,
	label: str,
) -> str:
	mapping = _mapping(container, label=label)
	entry = _mapping(mapping.get(key), label=f'{label}.{key}')
	_exact_path(entry.get('path'), expected=expected_path, label=f'{label}.{key}.path')
	recorded_sha256 = _lowercase_sha256(
		entry.get('sha256'), label=f'{label}.{key}.sha256'
	)
	if recorded_sha256 != actual_sha256:
		raise ValueError(
			f'{label}.{key} changed after its provenance was recorded: {expected_path}'
		)
	return recorded_sha256


def _read_json_with_sha(path: Path, *, label: str) -> tuple[Mapping[str, object], str]:
	if not path.is_file():
		raise FileNotFoundError(f'missing {label}: {path}')
	content = path.read_bytes()
	payload = json.loads(content)
	if not isinstance(payload, Mapping):
		raise TypeError(f'{label} must contain a JSON object: {path}')
	return payload, hashlib.sha256(content).hexdigest()


def _current_file_sha256(path: Path, *, label: str) -> str:
	if not path.is_file():
		raise FileNotFoundError(f'missing {label}: {path}')
	return file_sha256(path)


def _current_declared_identity(
	container: object,
	key: str,
	*,
	label: str,
) -> tuple[Path, str]:
	mapping = _mapping(container, label=label)
	entry = _mapping(mapping.get(key), label=f'{label}.{key}')
	path_value = entry.get('path')
	if not isinstance(path_value, str) or not path_value:
		raise ValueError(f'{label}.{key}.path must be a non-empty absolute path')
	path = Path(path_value)
	if not path.is_absolute() or path_value != str(path):
		raise ValueError(f'{label}.{key}.path must be an exact absolute path')
	sha256 = _lowercase_sha256(entry.get('sha256'), label=f'{label}.{key}.sha256')
	if _current_file_sha256(path, label=label) != sha256:
		raise ValueError(
			f'{label}.{key} changed after its provenance was recorded: {path}'
		)
	return path, sha256


def _absolute_path(value: object, *, label: str) -> Path:
	if not isinstance(value, str) or not value:
		raise ValueError(f'{label} must be a non-empty absolute path')
	path = Path(value)
	if not path.is_absolute() or value != str(path):
		raise ValueError(f'{label} must be an exact absolute path')
	return path


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
	if not isinstance(value, Mapping):
		raise TypeError(f'{label} must be a mapping')
	return value


def _exact_path(value: object, *, expected: Path, label: str) -> None:
	if not isinstance(value, str) or value != str(expected):
		raise ValueError(f'{label} must equal the exact path {expected}')


def _lowercase_sha256(value: object, *, label: str) -> str:
	if (
		not isinstance(value, str)
		or len(value) != 64
		or any(character not in '0123456789abcdef' for character in value)
	):
		raise ValueError(f'{label} must be a lowercase SHA-256')
	return value


def _unit_interval_number(value: object, *, label: str) -> float:
	if not isinstance(value, int | float) or isinstance(value, bool):
		raise TypeError(f'{label} must be numeric')
	result = float(value)
	if not math.isfinite(result):
		raise ValueError(f'{label} must be finite')
	if not 0.0 <= result <= 1.0:
		raise ValueError(f'{label} must be within [0, 1]')
	return result


def _positive_int(value: object, *, label: str) -> int:
	if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
		raise ValueError(f'{label} must be a positive integer')
	return value


__all__ = [
	'F3CompletedEvaluationProvenance',
	'read_f3_completed_evaluation_provenance',
]

"""Train F3 token-level lithology probes from frozen encoder embeddings."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from seis_ssl_cluster.f3.metrics import (
	compute_lithology_metrics,
	render_classification_report_markdown,
	write_confusion_matrix_csv,
	write_metrics_csv,
)

if TYPE_CHECKING:
	from numpy.typing import NDArray

	from seis_ssl_cluster.f3.labels import F3ClassInfo


VALID_PROBE_TYPES = frozenset({'logistic_regression', 'mlp'})
VALID_FEATURE_SCALING = frozenset({'standard', 'none'})
VALID_CLASS_WEIGHT = frozenset({'balanced'})
DEFAULT_EVALUATION_METRICS = (
	'accuracy',
	'balanced_accuracy',
	'macro_f1',
	'weighted_f1',
	'mean_iou',
)


@dataclass(frozen=True)
class F3LithologyProbeInputs:
	"""Input token datasets and class metadata for probe training."""

	train_tokens: Path
	validation_tokens: Path
	class_info: Path | None = None
	token_dataset_metadata_json: Path | None = None


@dataclass(frozen=True)
class F3LithologyProbeOutputs:
	"""Output directory contract for a trained lithology probe."""

	output_dir: Path

	@property
	def probe_joblib(self) -> Path:
		"""Return trained probe artifact path."""
		return self.output_dir / 'probe.joblib'

	@property
	def scaler_joblib(self) -> Path:
		"""Return fitted feature scaler artifact path."""
		return self.output_dir / 'scaler.joblib'

	@property
	def config_json(self) -> Path:
		"""Return resolved probe config artifact path."""
		return self.output_dir / 'probe_config_resolved.json'

	@property
	def metrics_json(self) -> Path:
		"""Return validation metrics JSON artifact path."""
		return self.output_dir / 'metrics.json'

	@property
	def metrics_csv(self) -> Path:
		"""Return validation metrics CSV artifact path."""
		return self.output_dir / 'metrics.csv'

	@property
	def confusion_matrix_csv(self) -> Path:
		"""Return raw confusion matrix CSV artifact path."""
		return self.output_dir / 'confusion_matrix.csv'

	@property
	def classification_report_md(self) -> Path:
		"""Return Markdown classification report artifact path."""
		return self.output_dir / 'classification_report.md'

	@property
	def figures_dir(self) -> Path:
		"""Return probe figure artifact directory."""
		return self.output_dir / 'figures'

	@property
	def confusion_matrix_png(self) -> Path:
		"""Return confusion matrix figure artifact path."""
		return self.figures_dir / 'confusion_matrix.png'

	@property
	def per_class_f1_png(self) -> Path:
		"""Return per-class F1 figure artifact path."""
		return self.figures_dir / 'per_class_f1.png'


@dataclass(frozen=True)
class F3LithologyProbeSettings:
	"""Validated training settings for one probe."""

	spec: str
	probe_type: str
	feature_scaling: str = 'standard'
	class_weight: str | None = 'balanced'
	max_iter: int = 2000
	hidden_dims: tuple[int, ...] = (256, 128)
	dropout: float = 0.2
	max_epochs: int = 200
	early_stopping_patience: int = 20
	batch_size: int = 1024
	learning_rate: float = 1.0e-3
	weight_decay: float = 0.0
	random_state: int = 42

	def __post_init__(self) -> None:
		"""Validate probe hyperparameters."""
		if not isinstance(self.spec, str) or not self.spec:
			msg = f'probe spec must be a non-empty string; got {self.spec!r}'
			raise TypeError(msg)
		if self.probe_type not in VALID_PROBE_TYPES:
			msg = (
				f'probe type must be one of {sorted(VALID_PROBE_TYPES)!r}; '
				f'got {self.probe_type!r}'
			)
			raise ValueError(msg)
		if self.feature_scaling not in VALID_FEATURE_SCALING:
			msg = (
				f'feature_scaling must be one of {sorted(VALID_FEATURE_SCALING)!r}; '
				f'got {self.feature_scaling!r}'
			)
			raise ValueError(msg)
		if (
			self.class_weight is not None
			and self.class_weight not in VALID_CLASS_WEIGHT
		):
			msg = 'class_weight must be "balanced" or null'
			raise ValueError(msg)
		_validate_positive_int(self.max_iter, 'max_iter')
		_validate_hidden_dims(self.hidden_dims)
		_validate_fraction(self.dropout, 'dropout')
		_validate_positive_int(self.max_epochs, 'max_epochs')
		_validate_positive_int(
			self.early_stopping_patience,
			'early_stopping_patience',
		)
		_validate_positive_int(self.batch_size, 'batch_size')
		_validate_positive_float(self.learning_rate, 'learning_rate')
		if (
			not isinstance(self.weight_decay, int | float)
			or isinstance(self.weight_decay, bool)
			or self.weight_decay < 0.0
		):
			msg = (
				f'weight_decay must be a nonnegative number; got {self.weight_decay!r}'
			)
			raise ValueError(msg)
		if not isinstance(self.random_state, int) or isinstance(
			self.random_state, bool
		):
			msg = f'random_state must be an integer; got {self.random_state!r}'
			raise TypeError(msg)

	def to_dict(self) -> dict[str, object]:
		"""Return JSON-serializable probe settings."""
		return {
			'spec': self.spec,
			'type': self.probe_type,
			'feature_scaling': self.feature_scaling,
			'class_weight': self.class_weight,
			'max_iter': self.max_iter,
			'hidden_dims': list(self.hidden_dims),
			'dropout': self.dropout,
			'max_epochs': self.max_epochs,
			'early_stopping_patience': self.early_stopping_patience,
			'batch_size': self.batch_size,
			'learning_rate': self.learning_rate,
			'weight_decay': self.weight_decay,
			'random_state': self.random_state,
		}


@dataclass(frozen=True)
class F3LithologyProbeConfig:
	"""Complete F3 lithology probe training configuration."""

	inputs: F3LithologyProbeInputs
	outputs: F3LithologyProbeOutputs
	classes: tuple[F3ClassInfo, ...]
	probe: F3LithologyProbeSettings
	dataset: Mapping[str, object]
	model: Mapping[str, object]
	embeddings: Mapping[str, object]
	labels: Mapping[str, object]
	token_dataset: Mapping[str, object]
	lithology: Mapping[str, object]
	evaluation_metrics: tuple[str, ...] = DEFAULT_EVALUATION_METRICS
	figure_dpi: int = 300


@dataclass(frozen=True)
class F3LithologyProbeResult:
	"""Paths and metrics written by one probe training run."""

	probe_joblib: Path
	scaler_joblib: Path
	config_json: Path
	metrics_json: Path
	metrics_csv: Path
	confusion_matrix_csv: Path
	classification_report_md: Path
	confusion_matrix_png: Path
	per_class_f1_png: Path
	train_token_count: int
	validation_token_count: int
	metrics: Mapping[str, object]


@dataclass(frozen=True)
class F3IdentityScaler:
	"""No-op scaler with the same transform API as sklearn scalers."""

	def fit(self, features: NDArray[np.generic]) -> F3IdentityScaler:
		"""Return self after validating feature shape."""
		_validate_feature_matrix(features, 'features')
		return self

	def transform(self, features: NDArray[np.generic]) -> NDArray[np.float32]:
		"""Return features as float32 without changing their values."""
		return np.asarray(
			_validate_feature_matrix(features, 'features'), dtype=np.float32
		)

	def fit_transform(self, features: NDArray[np.generic]) -> NDArray[np.float32]:
		"""Fit the no-op scaler and return transformed features."""
		return self.fit(features).transform(features)


@dataclass(frozen=True)
class F3TorchMLPClassifier:
	"""Joblib-serializable NumPy inference wrapper for the torch MLP probe."""

	input_dim: int
	hidden_dims: tuple[int, ...]
	dropout: float
	class_ids: tuple[int, ...]
	weights: tuple[NDArray[np.float32], ...]
	biases: tuple[NDArray[np.float32], ...]
	class_weight: dict[int, float]
	random_state: int
	training_epochs: int
	best_validation_loss: float

	@property
	def classes_(self) -> NDArray[np.int64]:
		"""Return class ids in prediction-column order."""
		return np.asarray(self.class_ids, dtype=np.int64)

	def predict_proba(self, features: NDArray[np.generic]) -> NDArray[np.float32]:
		"""Return class probabilities for a 2D feature matrix."""
		logits = self.decision_function(features)
		shifted = logits - logits.max(axis=1, keepdims=True)
		exp = np.exp(shifted)
		return np.asarray(exp / exp.sum(axis=1, keepdims=True), dtype=np.float32)

	def predict(self, features: NDArray[np.generic]) -> NDArray[np.int64]:
		"""Predict original F3 class ids."""
		probabilities = self.predict_proba(features)
		indices = np.argmax(probabilities, axis=1)
		return np.asarray([self.class_ids[index] for index in indices], dtype=np.int64)

	def decision_function(self, features: NDArray[np.generic]) -> NDArray[np.float32]:
		"""Return raw logits for a 2D feature matrix."""
		activations = np.asarray(
			_validate_feature_matrix(features, 'features'),
			dtype=np.float32,
		)
		if activations.shape[1] != self.input_dim:
			msg = (
				'features dimension does not match MLP input_dim; '
				f'got {activations.shape[1]}, expected={self.input_dim}'
			)
			raise ValueError(msg)
		last_index = len(self.weights) - 1
		for index, (weight, bias) in enumerate(
			zip(self.weights, self.biases, strict=True),
		):
			activations = activations @ weight.T + bias
			if index != last_index:
				activations = np.maximum(activations, 0.0)
		return np.asarray(activations, dtype=np.float32)


@dataclass(frozen=True)
class _TokenDataset:
	"""Loaded token feature matrix, labels, and provenance arrays."""

	path: Path
	features: NDArray[np.float32]
	labels: NDArray[np.int64]
	metadata: dict[str, NDArray[np.generic]]

	@property
	def count(self) -> int:
		return int(self.labels.shape[0])


def train_and_evaluate_f3_lithology_probe(
	config: F3LithologyProbeConfig,
) -> F3LithologyProbeResult:
	"""Train a linear or MLP lithology probe and write validation artifacts."""
	_validate_no_encoder_finetuning(config.model)
	train = load_token_dataset(config.inputs.train_tokens, label='train_tokens')
	validation = load_token_dataset(
		config.inputs.validation_tokens,
		label='validation_tokens',
	)
	_validate_label_classes(train.labels, config.classes, label='train_tokens.labels')
	_validate_label_classes(
		validation.labels,
		config.classes,
		label='validation_tokens.labels',
	)
	_validate_disjoint_token_xyz(train, validation)
	scaler, train_features, validation_features = _fit_scaler(
		train.features,
		validation.features,
		feature_scaling=config.probe.feature_scaling,
	)
	if config.probe.probe_type == 'logistic_regression':
		probe, training_summary = _fit_logistic_regression(
			train_features,
			train.labels,
			settings=config.probe,
		)
	else:
		probe, training_summary = _fit_mlp_probe(
			train_features,
			train.labels,
			validation_features,
			validation.labels,
			settings=config.probe,
		)
	predicted = np.asarray(probe.predict(validation_features), dtype=np.int64)
	metrics = compute_lithology_metrics(validation.labels, predicted, config.classes)
	_write_probe_outputs(
		config,
		scaler=scaler,
		probe=probe,
		training_summary=training_summary,
		train=train,
		validation=validation,
		metrics=metrics,
	)
	return F3LithologyProbeResult(
		probe_joblib=config.outputs.probe_joblib,
		scaler_joblib=config.outputs.scaler_joblib,
		config_json=config.outputs.config_json,
		metrics_json=config.outputs.metrics_json,
		metrics_csv=config.outputs.metrics_csv,
		confusion_matrix_csv=config.outputs.confusion_matrix_csv,
		classification_report_md=config.outputs.classification_report_md,
		confusion_matrix_png=config.outputs.confusion_matrix_png,
		per_class_f1_png=config.outputs.per_class_f1_png,
		train_token_count=train.count,
		validation_token_count=validation.count,
		metrics=metrics,
	)


def load_token_dataset(path: str | Path, *, label: str) -> _TokenDataset:
	"""Load and validate one F3 token dataset NPZ."""
	token_path = Path(path)
	if not token_path.is_file():
		msg = f'{label} does not exist: {token_path}'
		raise FileNotFoundError(msg)
	with np.load(token_path) as payload:
		if 'features' not in payload or 'labels' not in payload:
			msg = f'{label} must contain features and labels arrays: {token_path}'
			raise KeyError(msg)
		features = np.asarray(payload['features'], dtype=np.float32)
		labels = np.asarray(payload['labels'], dtype=np.int64)
		metadata = {
			key: np.asarray(payload[key])
			for key in payload.files
			if key not in {'features', 'labels'}
		}
	features = _validate_feature_matrix(features, f'{label}.features')
	labels = _validate_label_vector(labels, f'{label}.labels')
	if features.shape[0] != labels.shape[0]:
		msg = (
			f'{label} features and labels row counts differ; '
			f'features={features.shape[0]}, labels={labels.shape[0]}'
		)
		raise ValueError(msg)
	if labels.size == 0:
		msg = f'{label} must contain at least one labeled token'
		raise ValueError(msg)
	return _TokenDataset(
		path=token_path,
		features=np.asarray(features, dtype=np.float32),
		labels=labels,
		metadata=metadata,
	)


def _validate_disjoint_token_xyz(
	train: _TokenDataset,
	validation: _TokenDataset,
) -> None:
	train_token_xyz = _required_token_xyz(train, label='train_tokens')
	validation_token_xyz = _required_token_xyz(
		validation,
		label='validation_tokens',
	)
	overlap = _token_xyz_set(train_token_xyz) & _token_xyz_set(validation_token_xyz)
	if overlap:
		examples = sorted(overlap)[:5]
		msg = (
			'train_tokens and validation_tokens share token_xyz rows; '
			'validation metrics would reuse training embeddings. '
			f'overlap_count={len(overlap)}, examples={examples!r}'
		)
		raise ValueError(msg)


def _required_token_xyz(dataset: _TokenDataset, *, label: str) -> NDArray[np.int64]:
	if 'token_xyz' not in dataset.metadata:
		msg = f'{label} must contain token_xyz for leakage validation'
		raise KeyError(msg)
	token_xyz = np.asarray(dataset.metadata['token_xyz'], dtype=np.int64)
	expected_shape = (dataset.count, 3)
	if token_xyz.shape != expected_shape:
		msg = (
			f'{label}.token_xyz must have shape {expected_shape!r}; '
			f'got {token_xyz.shape!r}'
		)
		raise ValueError(msg)
	return token_xyz


def _token_xyz_set(token_xyz: NDArray[np.int64]) -> set[tuple[int, int, int]]:
	if token_xyz.size == 0:
		return set()
	return {tuple(int(axis) for axis in row) for row in token_xyz}


def _fit_scaler(
	train_features: NDArray[np.float32],
	validation_features: NDArray[np.float32],
	*,
	feature_scaling: str,
) -> tuple[object, NDArray[np.float32], NDArray[np.float32]]:
	if feature_scaling == 'standard':
		from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

		scaler = StandardScaler()
	elif feature_scaling == 'none':
		scaler = F3IdentityScaler()
	else:
		msg = f'unsupported feature_scaling: {feature_scaling!r}'
		raise ValueError(msg)
	train_scaled = np.asarray(scaler.fit_transform(train_features), dtype=np.float32)
	validation_scaled = np.asarray(
		scaler.transform(validation_features), dtype=np.float32
	)
	return scaler, train_scaled, validation_scaled


def _fit_logistic_regression(
	features: NDArray[np.float32],
	labels: NDArray[np.int64],
	*,
	settings: F3LithologyProbeSettings,
) -> tuple[object, dict[str, object]]:
	from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

	if np.unique(labels).size < 2:
		msg = 'logistic_regression requires at least two training classes'
		raise ValueError(msg)
	probe = LogisticRegression(
		class_weight=settings.class_weight,
		max_iter=settings.max_iter,
		random_state=settings.random_state,
		solver='lbfgs',
	)
	probe.fit(features, labels)
	return probe, {
		'trainer': 'sklearn.linear_model.LogisticRegression',
		'class_counts': _class_counts(labels),
		'class_weight': settings.class_weight,
		'classes': [int(class_id) for class_id in probe.classes_],
		'n_iter': [int(value) for value in np.ravel(probe.n_iter_)],
	}


def _fit_mlp_probe(
	train_features: NDArray[np.float32],
	train_labels: NDArray[np.int64],
	validation_features: NDArray[np.float32],
	validation_labels: NDArray[np.int64],
	*,
	settings: F3LithologyProbeSettings,
) -> tuple[F3TorchMLPClassifier, dict[str, object]]:
	import torch  # noqa: PLC0415
	from torch import nn  # noqa: PLC0415

	train_class_ids = tuple(int(value) for value in np.unique(train_labels))
	if len(train_class_ids) < 2:
		msg = 'mlp probe requires at least two training classes'
		raise ValueError(msg)
	torch.manual_seed(settings.random_state)
	rng = np.random.default_rng(settings.random_state)
	class_to_index = {class_id: index for index, class_id in enumerate(train_class_ids)}
	y_train = _encode_labels(train_labels, class_to_index, label='train_labels')
	validation_known = np.isin(validation_labels, np.asarray(train_class_ids))
	if np.any(validation_known):
		y_validation = _encode_labels(
			validation_labels[validation_known],
			class_to_index,
			label='validation_labels',
		)
	else:
		y_validation = y_train
	model = _build_torch_mlp(
		input_dim=int(train_features.shape[1]),
		hidden_dims=settings.hidden_dims,
		output_dim=len(train_class_ids),
		dropout=settings.dropout,
	)
	criterion = nn.CrossEntropyLoss(
		weight=_torch_class_weights(
			train_labels,
			train_class_ids=train_class_ids,
			class_weight=settings.class_weight,
		),
	)
	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=settings.learning_rate,
		weight_decay=settings.weight_decay,
	)
	train_tensor = torch.as_tensor(train_features, dtype=torch.float32)
	y_train_tensor = torch.as_tensor(y_train, dtype=torch.long)
	validation_tensor = torch.as_tensor(
		(
			validation_features[validation_known]
			if np.any(validation_known)
			else train_features
		),
		dtype=torch.float32,
	)
	y_validation_tensor = torch.as_tensor(y_validation, dtype=torch.long)
	best_state: dict[str, Any] | None = None
	best_validation_loss = float('inf')
	epochs_without_improvement = 0
	completed_epochs = 0
	for epoch in range(settings.max_epochs):
		model.train()
		indices = rng.permutation(train_features.shape[0])
		for start in range(0, indices.size, settings.batch_size):
			batch_indices = indices[start : start + settings.batch_size]
			optimizer.zero_grad()
			logits = model(train_tensor[batch_indices])
			loss = criterion(logits, y_train_tensor[batch_indices])
			loss.backward()
			optimizer.step()
		completed_epochs = epoch + 1
		validation_loss = _torch_validation_loss(
			model,
			criterion,
			validation_tensor,
			y_validation_tensor,
		)
		if validation_loss < best_validation_loss - 1.0e-7:
			best_validation_loss = validation_loss
			best_state = {
				key: value.detach().cpu().clone()
				for key, value in model.state_dict().items()
			}
			epochs_without_improvement = 0
		else:
			epochs_without_improvement += 1
			if epochs_without_improvement >= settings.early_stopping_patience:
				break
	if best_state is not None:
		model.load_state_dict(best_state)
	class_weight = _class_weight_by_id(
		train_labels,
		train_class_ids=train_class_ids,
		class_weight=settings.class_weight,
	)
	probe = _mlp_inference_wrapper(
		model,
		settings=settings,
		class_ids=train_class_ids,
		class_weight=class_weight,
		training_epochs=completed_epochs,
		best_validation_loss=best_validation_loss,
	)
	return probe, {
		'trainer': 'torch.nn.Module',
		'class_counts': _class_counts(train_labels),
		'class_weight': class_weight,
		'classes': [int(class_id) for class_id in train_class_ids],
		'epochs': completed_epochs,
		'best_validation_loss': best_validation_loss,
	}


def _write_probe_outputs(  # noqa: PLR0913
	config: F3LithologyProbeConfig,
	*,
	scaler: object,
	probe: object,
	training_summary: Mapping[str, object],
	train: _TokenDataset,
	validation: _TokenDataset,
	metrics: Mapping[str, object],
) -> None:
	import joblib  # noqa: PLC0415

	outputs = config.outputs
	outputs.output_dir.mkdir(parents=True, exist_ok=True)
	outputs.figures_dir.mkdir(parents=True, exist_ok=True)
	joblib.dump(probe, outputs.probe_joblib)
	joblib.dump(scaler, outputs.scaler_joblib)
	metrics_payload = dict(metrics)
	feature_source = _feature_source_from_config(config)
	if feature_source is not None:
		metrics_payload['feature_source'] = feature_source
	_write_json(outputs.metrics_json, metrics_payload)
	write_metrics_csv(outputs.metrics_csv, metrics, config.classes)
	write_confusion_matrix_csv(outputs.confusion_matrix_csv, metrics, config.classes)
	outputs.classification_report_md.write_text(
		render_classification_report_markdown(metrics, config.classes),
		encoding='utf-8',
	)
	_write_probe_figures(
		metrics,
		config.classes,
		confusion_matrix_png=outputs.confusion_matrix_png,
		per_class_f1_png=outputs.per_class_f1_png,
		dpi=config.figure_dpi,
	)
	_write_json(
		outputs.config_json,
		_resolved_config_payload(
			config,
			training_summary=training_summary,
			train=train,
			validation=validation,
		),
	)


def _resolved_config_payload(
	config: F3LithologyProbeConfig,
	*,
	training_summary: Mapping[str, object],
	train: _TokenDataset,
	validation: _TokenDataset,
) -> dict[str, object]:
	payload: dict[str, object] = {
		'artifact_type': 'f3_lithology_probe',
		'dataset': dict(config.dataset),
		'model': dict(config.model),
		'embeddings': dict(config.embeddings),
		'labels': dict(config.labels),
		'lithology': dict(config.lithology),
		'token_dataset': dict(config.token_dataset),
		'probe': config.probe.to_dict(),
		'evaluation': {
			'metrics': list(config.evaluation_metrics),
			'figure_dpi': config.figure_dpi,
		},
		'inputs': {
			'train_tokens': str(config.inputs.train_tokens),
			'validation_tokens': str(config.inputs.validation_tokens),
			'class_info': (
				None
				if config.inputs.class_info is None
				else str(config.inputs.class_info)
			),
			'token_dataset_metadata_json': (
				None
				if config.inputs.token_dataset_metadata_json is None
				else str(config.inputs.token_dataset_metadata_json)
			),
		},
		'outputs': {
			'probe_joblib': str(config.outputs.probe_joblib),
			'scaler_joblib': str(config.outputs.scaler_joblib),
			'probe_config_resolved_json': str(config.outputs.config_json),
			'metrics_json': str(config.outputs.metrics_json),
			'metrics_csv': str(config.outputs.metrics_csv),
			'confusion_matrix_csv': str(config.outputs.confusion_matrix_csv),
			'classification_report_md': str(config.outputs.classification_report_md),
			'confusion_matrix_png': str(config.outputs.confusion_matrix_png),
			'per_class_f1_png': str(config.outputs.per_class_f1_png),
		},
		'classes': [class_info.to_dict() for class_info in config.classes],
		'label_source_of_truth': 'segy_label_volume',
		'png_label_role': 'train_validation_slice_selection_and_visual_qc',
		'encoder_finetuning': False,
		'training_summary': dict(training_summary),
		'summary': {
			'train_tokens': train.count,
			'validation_tokens': validation.count,
			'train_class_counts': _class_counts(train.labels),
			'validation_class_counts': _class_counts(validation.labels),
		},
	}
	feature_source = _feature_source_from_config(config)
	if feature_source is not None:
		payload['feature_source'] = feature_source
	return payload


def _feature_source_from_config(
	config: F3LithologyProbeConfig,
) -> dict[str, object] | None:
	for candidate in (
		_mapping_or_none(config.token_dataset.get('feature_source')),
		_mapping_or_none(config.embeddings.get('feature_source')),
		_mapping_or_none(config.model.get('feature_source')),
	):
		if candidate:
			return dict(candidate)
	return None


def _mapping_or_none(value: object) -> Mapping[str, object] | None:
	return value if isinstance(value, Mapping) else None


def _write_probe_figures(
	metrics: Mapping[str, object],
	classes: Sequence[F3ClassInfo],
	*,
	confusion_matrix_png: Path,
	per_class_f1_png: Path,
	dpi: int,
) -> None:
	import matplotlib.pyplot as plt  # noqa: PLC0415

	class_labels = [
		f'{class_info.class_id}\n{class_info.class_name}' for class_info in classes
	]
	matrix = np.asarray(metrics['confusion_matrix'], dtype=np.int64)
	normalized = np.asarray(
		metrics['confusion_matrix_row_normalized'], dtype=np.float64
	)
	fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
	for axis, values, title, fmt in (
		(axes[0], matrix, 'Confusion matrix', 'd'),
		(axes[1], normalized, 'Row-normalized confusion matrix', '.2f'),
	):
		image = axis.imshow(values, cmap='Blues', vmin=0)
		axis.set_title(title)
		axis.set_xlabel('Predicted class')
		axis.set_ylabel('True class')
		axis.set_xticks(
			np.arange(len(classes)), labels=class_labels, rotation=45, ha='right'
		)
		axis.set_yticks(np.arange(len(classes)), labels=class_labels)
		_threshold = float(values.max()) / 2.0 if values.size else 0.0
		for row_index in range(values.shape[0]):
			for column_index in range(values.shape[1]):
				axis.text(
					column_index,
					row_index,
					format(values[row_index, column_index], fmt),
					ha='center',
					va='center',
					color='white'
					if values[row_index, column_index] > _threshold
					else 'black',
					fontsize=8,
				)
		fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
	fig.tight_layout()
	fig.savefig(confusion_matrix_png, dpi=dpi, facecolor='white')
	plt.close(fig)

	per_class_f1 = _metric_float_mapping(metrics['per_class_f1'])
	values = [per_class_f1[str(class_info.class_id)] for class_info in classes]
	colors = [class_info.hex_color for class_info in classes]
	fig, axis = plt.subplots(figsize=(8, 4.5), facecolor='white')
	axis.bar(
		np.arange(len(classes)), values, color=colors, edgecolor='black', linewidth=0.6
	)
	axis.set_ylabel('F1')
	axis.set_ylim(0.0, 1.0)
	axis.set_xticks(
		np.arange(len(classes)), labels=class_labels, rotation=45, ha='right'
	)
	axis.grid(axis='y', color='#D9D9D9', linewidth=0.8)
	axis.set_axisbelow(True)
	fig.tight_layout()
	fig.savefig(per_class_f1_png, dpi=dpi, facecolor='white')
	plt.close(fig)


def _build_torch_mlp(
	*,
	input_dim: int,
	hidden_dims: Sequence[int],
	output_dim: int,
	dropout: float,
) -> object:
	import torch  # noqa: PLC0415
	from torch import nn  # noqa: PLC0415

	layers: list[Any] = []
	previous_dim = input_dim
	for hidden_dim in hidden_dims:
		layers.append(nn.Linear(previous_dim, hidden_dim))
		layers.append(nn.ReLU())
		if dropout > 0.0:
			layers.append(nn.Dropout(dropout))
		previous_dim = hidden_dim
	layers.append(nn.Linear(previous_dim, output_dim))
	model = nn.Sequential(*layers)
	if not isinstance(model, torch.nn.Module):
		msg = 'failed to construct torch MLP model'
		raise TypeError(msg)
	return model


def _torch_validation_loss(
	model: object,
	criterion: object,
	features: object,
	labels: object,
) -> float:
	import torch  # noqa: PLC0415

	with torch.no_grad():
		model.eval()
		return float(criterion(model(features), labels).item())


def _torch_class_weights(
	labels: NDArray[np.int64],
	*,
	train_class_ids: Sequence[int],
	class_weight: str | None,
) -> object:
	if class_weight is None:
		return None
	import torch  # noqa: PLC0415

	return torch.as_tensor(
		[
			_class_weight_by_id(
				labels,
				train_class_ids=train_class_ids,
				class_weight=class_weight,
			)[class_id]
			for class_id in train_class_ids
		],
		dtype=torch.float32,
	)


def _class_weight_by_id(
	labels: NDArray[np.int64],
	*,
	train_class_ids: Sequence[int],
	class_weight: str | None,
) -> dict[int, float]:
	if class_weight is None:
		return {int(class_id): 1.0 for class_id in train_class_ids}
	if class_weight != 'balanced':
		msg = f'unsupported class_weight: {class_weight!r}'
		raise ValueError(msg)
	counts = Counter(int(label) for label in labels)
	total = int(labels.shape[0])
	class_count = len(train_class_ids)
	return {
		int(class_id): float(total / (class_count * counts[int(class_id)]))
		for class_id in train_class_ids
	}


def _mlp_inference_wrapper(  # noqa: PLR0913
	model: object,
	*,
	settings: F3LithologyProbeSettings,
	class_ids: tuple[int, ...],
	class_weight: dict[int, float],
	training_epochs: int,
	best_validation_loss: float,
) -> F3TorchMLPClassifier:
	from torch import nn  # noqa: PLC0415

	linear_layers = [
		module for module in model.modules() if isinstance(module, nn.Linear)
	]
	return F3TorchMLPClassifier(
		input_dim=int(linear_layers[0].in_features),
		hidden_dims=settings.hidden_dims,
		dropout=settings.dropout,
		class_ids=class_ids,
		weights=tuple(
			np.asarray(layer.weight.detach().cpu().numpy(), dtype=np.float32)
			for layer in linear_layers
		),
		biases=tuple(
			np.asarray(layer.bias.detach().cpu().numpy(), dtype=np.float32)
			for layer in linear_layers
		),
		class_weight=class_weight,
		random_state=settings.random_state,
		training_epochs=training_epochs,
		best_validation_loss=best_validation_loss,
	)


def _encode_labels(
	labels: NDArray[np.int64],
	class_to_index: Mapping[int, int],
	*,
	label: str,
) -> NDArray[np.int64]:
	encoded = np.empty(labels.shape, dtype=np.int64)
	for index, class_id in enumerate(labels):
		class_id_int = int(class_id)
		if class_id_int not in class_to_index:
			msg = (
				f'{label} contains class id not present in training set: {class_id_int}'
			)
			raise ValueError(msg)
		encoded[index] = class_to_index[class_id_int]
	return encoded


def _validate_feature_matrix(
	features: NDArray[np.generic],
	label: str,
) -> NDArray[np.float32]:
	array = np.asarray(features)
	if array.ndim != 2:
		msg = f'{label} must be a 2D feature matrix; got {array.shape}'
		raise ValueError(msg)
	if array.shape[0] == 0 or array.shape[1] == 0:
		msg = f'{label} must have nonzero rows and columns; got {array.shape}'
		raise ValueError(msg)
	if not np.all(np.isfinite(array)):
		msg = f'{label} contains non-finite values'
		raise ValueError(msg)
	return np.asarray(array, dtype=np.float32)


def _validate_label_vector(
	labels: NDArray[np.generic],
	label: str,
) -> NDArray[np.int64]:
	array = np.asarray(labels)
	if array.ndim != 1:
		msg = f'{label} must be a 1D label vector; got {array.shape}'
		raise ValueError(msg)
	if not np.issubdtype(array.dtype, np.integer):
		rounded = np.rint(array)
		if not np.array_equal(array, rounded):
			msg = f'{label} must contain integer class ids'
			raise ValueError(msg)
		array = rounded
	return np.asarray(array, dtype=np.int64)


def _validate_label_classes(
	labels: NDArray[np.int64],
	classes: Sequence[F3ClassInfo],
	*,
	label: str,
) -> None:
	known = {class_info.class_id for class_info in classes}
	unknown = sorted({int(class_id) for class_id in labels} - known)
	if unknown:
		msg = f'{label} contains class ids missing from class_info: {unknown!r}'
		raise ValueError(msg)


def _validate_no_encoder_finetuning(model: Mapping[str, object]) -> None:
	freeze_encoder = model.get('freeze_encoder')
	if freeze_encoder is False:
		msg = 'F3 lithology probes must use frozen encoder embeddings'
		raise ValueError(msg)


def _class_counts(labels: NDArray[np.int64]) -> dict[int, int]:
	return {
		int(class_id): int(count)
		for class_id, count in sorted(Counter(int(label) for label in labels).items())
	}


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
	)


def _metric_float_mapping(value: object) -> dict[str, float]:
	if not isinstance(value, Mapping):
		msg = f'expected mapping metric payload; got {value!r}'
		raise TypeError(msg)
	return {str(key): float(metric) for key, metric in value.items()}


def _validate_positive_int(value: int, label: str) -> None:
	if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
		msg = f'{label} must be a positive integer; got {value!r}'
		raise ValueError(msg)


def _validate_hidden_dims(values: Sequence[int]) -> None:
	if not values:
		msg = 'hidden_dims must contain at least one layer width'
		raise ValueError(msg)
	for value in values:
		_validate_positive_int(value, 'hidden_dims')


def _validate_fraction(value: float, label: str) -> None:
	if not isinstance(value, int | float) or isinstance(value, bool):
		msg = f'{label} must be a number in [0, 1); got {value!r}'
		raise TypeError(msg)
	if not 0.0 <= float(value) < 1.0:
		msg = f'{label} must be in [0, 1); got {value!r}'
		raise ValueError(msg)


def _validate_positive_float(value: float, label: str) -> None:
	if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0.0:
		msg = f'{label} must be a positive number; got {value!r}'
		raise ValueError(msg)


__all__ = [
	'DEFAULT_EVALUATION_METRICS',
	'VALID_CLASS_WEIGHT',
	'VALID_FEATURE_SCALING',
	'VALID_PROBE_TYPES',
	'F3IdentityScaler',
	'F3LithologyProbeConfig',
	'F3LithologyProbeInputs',
	'F3LithologyProbeOutputs',
	'F3LithologyProbeResult',
	'F3LithologyProbeSettings',
	'F3TorchMLPClassifier',
	'load_token_dataset',
	'train_and_evaluate_f3_lithology_probe',
]

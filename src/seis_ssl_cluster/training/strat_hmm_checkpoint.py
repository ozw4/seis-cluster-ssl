"""Checkpoint helpers for stratigraphic HMM pretext training."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

import seis_ssl_cluster
from seis_ssl_cluster.training.checkpoint import capture_rng_state


@dataclass(frozen=True)
class StratRollingCheckpointResult:
	"""Result of a rolling strat HMM checkpoint write."""

	latest_path: Path
	best_path: Path
	best_score: float | None
	best_updated: bool


def save_strat_hmm_rolling_checkpoint(  # noqa: PLR0913
	checkpoint_dir: str | Path,
	*,
	student: torch.nn.Module,
	head: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	epoch: int,
	mae_config: Mapping[str, object],
	stratigraphy_config: Mapping[str, object],
	metrics: Mapping[str, float],
	global_step: int,
	checkpoint_kind: Literal['step', 'epoch'],
	batch_index: int | None,
	amp_enabled: bool = False,
	scaler: torch.amp.GradScaler | None = None,
	rng_state: Mapping[str, object] | None = None,
	best_score: float | None = None,
) -> StratRollingCheckpointResult:
	"""Write rolling ``latest.pt`` and update ``best.pt`` on lower loss."""
	checkpoint_root = Path(checkpoint_dir)
	latest_path = save_strat_hmm_checkpoint(
		checkpoint_root / 'latest.pt',
		student=student,
		head=head,
		optimizer=optimizer,
		epoch=epoch,
		mae_config=mae_config,
		stratigraphy_config=stratigraphy_config,
		metrics=metrics,
		global_step=global_step,
		amp_enabled=amp_enabled,
		scaler=scaler,
		checkpoint_kind=checkpoint_kind,
		batch_index=batch_index,
		rng_state=rng_state,
	)
	score = _loss_score(metrics)
	best_updated = _is_improved(score, best_score)
	resolved_best_score = best_score
	best_path = checkpoint_root / 'best.pt'
	if best_updated:
		_copy_checkpoint_atomic(latest_path, best_path)
		resolved_best_score = score
	return StratRollingCheckpointResult(
		latest_path=latest_path,
		best_path=best_path,
		best_score=resolved_best_score,
		best_updated=best_updated,
	)


def save_strat_hmm_checkpoint(  # noqa: PLR0913
	path: str | Path,
	*,
	student: torch.nn.Module,
	head: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	epoch: int,
	mae_config: Mapping[str, object],
	stratigraphy_config: Mapping[str, object],
	metrics: Mapping[str, float],
	global_step: int,
	checkpoint_kind: Literal['step', 'epoch'],
	batch_index: int | None,
	amp_enabled: bool = False,
	scaler: torch.amp.GradScaler | None = None,
	rng_state: Mapping[str, object] | None = None,
) -> Path:
	"""Atomically save an extraction-compatible strat HMM checkpoint."""
	checkpoint_path = Path(path)
	checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
	payload = {
		'model_state_dict': _state_dict_cpu(student),
		'stratigraphy_state_dict': _state_dict_cpu(head),
		'stratigraphy_config': _to_plain_value(stratigraphy_config),
		'optimizer_state_dict': optimizer.state_dict(),
		'epoch': int(epoch),
		'global_step': int(global_step),
		'amp_enabled': bool(amp_enabled),
		'scaler_state_dict': None if scaler is None else scaler.state_dict(),
		'config': _to_plain_value(mae_config),
		'package_version': getattr(seis_ssl_cluster, '__version__', None),
		'metrics': dict(metrics),
		'rng_state': dict(capture_rng_state() if rng_state is None else rng_state),
		'training_state': {
			'schema_version': 1,
			'stage': 'train_strat_hmm_pretext',
			'checkpoint_kind': checkpoint_kind,
			'batch_index': batch_index,
		},
	}
	return _atomic_torch_save(checkpoint_path, payload)


def _state_dict_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
	return {
		key: value.detach().cpu()
		for key, value in module.state_dict().items()
	}


def _loss_score(metrics: Mapping[str, float]) -> float | None:
	value = metrics.get('loss')
	if isinstance(value, int | float) and not isinstance(value, bool):
		score = float(value)
		if math.isfinite(score):
			return score
	return None


def _is_improved(score: float | None, best_score: float | None) -> bool:
	if score is None:
		return False
	if best_score is None:
		return True
	return score < best_score


def _copy_checkpoint_atomic(source: Path, target: Path) -> None:
	target.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = target.with_suffix('.pt.tmp')
	shutil.copy2(source, tmp_path)
	tmp_path.replace(target)


def _atomic_torch_save(path: Path, payload: Mapping[str, object]) -> Path:
	fd, tmp_name = tempfile.mkstemp(
		prefix=f'.{path.name}.',
		suffix='.tmp',
		dir=path.parent,
	)
	tmp_path = Path(tmp_name)
	try:
		with os.fdopen(fd, 'wb') as file_obj:
			torch.save(dict(payload), file_obj)
			file_obj.flush()
			os.fsync(file_obj.fileno())
		tmp_path.replace(path)
	finally:
		if tmp_path.exists():
			tmp_path.unlink()
	return path


def _to_plain_value(value: object) -> object:
	if isinstance(value, Mapping):
		return {str(key): _to_plain_value(child) for key, child in value.items()}
	if isinstance(value, list | tuple):
		return [_to_plain_value(child) for child in value]
	if isinstance(value, Path):
		return str(value)
	return value


__all__ = [
	'StratRollingCheckpointResult',
	'save_strat_hmm_checkpoint',
	'save_strat_hmm_rolling_checkpoint',
]

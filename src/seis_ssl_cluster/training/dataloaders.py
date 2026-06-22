"""DataLoader builders for amplitude MAE pretraining."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import torch

from seis_ssl_cluster.training.collate import mae_collate_fn

if TYPE_CHECKING:
	from collections.abc import Callable


def build_mae_dataloader(  # noqa: PLR0913
	dataset: object,
	*,
	batch_size: int,
	num_workers: int = 0,
	shuffle: bool = True,
	seed: int = 42,
	device: str | torch.device = 'cpu',
) -> torch.utils.data.DataLoader:
	"""Build a deterministic amplitude MAE DataLoader."""
	if num_workers < 0:
		msg = f'num_workers must be nonnegative; got {num_workers!r}'
		raise ValueError(msg)
	generator = torch.Generator()
	generator.manual_seed(seed)
	torch_device = torch.device(device)
	return torch.utils.data.DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		collate_fn=mae_collate_fn,
		generator=generator,
		pin_memory=torch_device.type == 'cuda',
		persistent_workers=num_workers > 0,
		worker_init_fn=_make_worker_init_fn(seed),
	)


def _make_worker_init_fn(seed: int) -> Callable[[int], None]:
	def seed_worker(worker_id: int) -> None:
		worker_seed = (int(seed) + int(worker_id)) % (2**32)
		random.seed(worker_seed)
		np.random.seed(worker_seed)  # noqa: NPY002
		torch.manual_seed(worker_seed)

	return seed_worker


__all__ = ['build_mae_dataloader']

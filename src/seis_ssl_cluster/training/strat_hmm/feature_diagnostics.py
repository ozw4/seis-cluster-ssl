"""Fixed-crop teacher distances without changing the training RNG stream."""

from __future__ import annotations

import csv
import hashlib
import json
from typing import TYPE_CHECKING

import torch

from seis_ssl_cluster.training.checkpoint import capture_rng_state, restore_rng_state
from seis_ssl_cluster.training.collate import strat_multi_head_target_collate_fn
from seis_ssl_cluster.training.strat_hmm.encoding import UnmaskedEncoderPair

if TYPE_CHECKING:
	from pathlib import Path

	from seis_ssl_cluster.data import NopimsStratMultiHeadTargetDataset
	from seis_ssl_cluster.models.mae import AmplitudeMAE3D


class FixedCropFeatureDiagnostics:
	"""Cache epoch-zero crops and record a token-weighted mean cosine distance."""

	def __init__(
		self,
		dataset: NopimsStratMultiHeadTargetDataset,
		*,
		samples: int,
		output_root: Path,
	) -> None:
		"""Use fixed sample indices without iterating the training dataloader."""
		self.output_root = output_root
		state = capture_rng_state()
		previous_epoch = dataset.epoch
		try:
			dataset.set_epoch(0)
			self.batches = [
				strat_multi_head_target_collate_fn([dataset[i]])
				for i in range(min(samples, len(dataset)))
			]
		finally:
			dataset.set_epoch(previous_epoch)
			restore_rng_state({'rng_state': state})
		digest = hashlib.sha256()
		for batch in self.batches:
			for key in ('x', 'local_valid_mask'):
				tensor = batch[key].contiguous()
				digest.update(tensor.numpy().tobytes())
		self.crop_sha256 = digest.hexdigest()
		identity = {
			'epoch': 0,
			'indices': list(range(len(self.batches))),
			'crop_sha256': self.crop_sha256,
		}
		path = output_root / 'feature_diagnostics_crops.json'
		if path.exists() and json.loads(path.read_text()) != identity:
			raise ValueError('fixed diagnostic crop identity changed on resume')
		path.write_text(json.dumps(identity, indent=2) + '\n')

	@torch.no_grad()
	def record(
		self,
		student: AmplitudeMAE3D,
		teacher: AmplitudeMAE3D,
		*,
		epoch: int,
		global_step: int,
	) -> None:
		"""Evaluate fixed crops in eval mode and restore modes and all RNG states."""
		path = self.output_root / 'feature_diagnostics.csv'
		if path.exists():
			with path.open(newline='') as stream:
				if any(int(row['epoch']) == epoch for row in csv.DictReader(stream)):
					return
		state = capture_rng_state()
		flags = [(module, module.training) for module in student.modules()]
		device = next(student.parameters()).device
		total, count = 0.0, 0
		try:
			student.eval()
			teacher.eval()
			pair = UnmaskedEncoderPair(student, teacher, use_teacher=True)
			for batch in self.batches:
				encoded, reference = pair.encode(
					batch['x'].to(device),
					valid_mask=batch['local_valid_mask'].to(device),
				)
				valid = encoded['token_valid_mask'] & reference['token_valid_mask']
				distance = 1.0 - torch.nn.functional.cosine_similarity(
					encoded['tokens'], reference['tokens'], dim=-1
				)
				total += float(distance[valid].double().sum())
				count += int(valid.sum())
		finally:
			for module, flag in flags:
				module.training = flag
			restore_rng_state({'rng_state': state})
		if not count:
			raise ValueError('fixed diagnostic crops contain no valid tokens')
		row = {
			'epoch': epoch,
			'global_step': global_step,
			'teacher_cosine_distance': total / count,
			'valid_tokens': count,
			'crop_sha256': self.crop_sha256,
		}
		with path.open('a', newline='') as stream:
			writer = csv.DictWriter(stream, fieldnames=list(row))
			if stream.tell() == 0:
				writer.writeheader()
			writer.writerow(row)

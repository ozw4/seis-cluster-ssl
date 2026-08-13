"""Focused recovery-boundary contracts for periodic strat-HMM training."""
# ruff: noqa: SLF001

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from seis_ssl_cluster.training.strat_hmm import runner
from seis_ssl_cluster.training.strat_hmm.state import TrainabilitySummary

if TYPE_CHECKING:
	from pathlib import Path


@pytest.mark.parametrize(
	('epoch', 'scheduled'),
	[
		(2, True),
		(5, True),
		(8, True),
		(11, True),
		(14, True),
		(17, True),
		(20, True),
		(25, False),
		(26, False),
	],
)
def test_periodic_refresh_schedule_has_no_post_epoch_25_refresh(
	epoch: int, scheduled: bool  # noqa: FBT001
) -> None:
	assert runner._periodic_scheduled_epoch(epoch) is scheduled


@pytest.mark.parametrize(
	('checkpoint_kind', 'refresh_phase'),
	[('epoch', 'refresh_required'), ('refresh', 'refresh_complete')],
)
def test_periodic_checkpoint_recovery_replays_pre_and_post_refresh_boundaries(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	checkpoint_kind: str,
	refresh_phase: str,
) -> None:
	student = torch.nn.Linear(1, 1)
	optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
	components = SimpleNamespace(student=student, optimizer=optimizer)
	state = {
		'active_generation_id': 'refresh_0001_epoch002',
		'active_generation_manifest_sha256': 'a' * 64,
		'active_generation_content_sha256': 'b' * 64,
		'active_target_manifest_sha256': 'c' * 64,
		'source_student_state_sha256': 'd' * 64,
		'refresh_phase': refresh_phase,
	}
	events: list[dict[str, object]] = []
	monkeypatch.setattr(
		runner,
		'_append_target_refresh_event',
		lambda _root, event: events.append(dict(event)),
	)

	runner._recover_periodic_checkpoint_event(
		output_root=tmp_path,
		payload={
			'epoch': 2,
			'global_step': 8,
			'training_state': {'checkpoint_kind': checkpoint_kind},
		},
		state=state,
		components=components,
	)

	assert len(events) == 1
	assert events[0]['checkpoint_kind'] == checkpoint_kind
	assert events[0]['refresh_phase'] == refresh_phase
	assert events[0]['global_step_before'] == events[0]['global_step_after'] == 8


def test_periodic_step_checkpoint_preserves_mid_epoch_batch_rng(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	generator = torch.Generator().manual_seed(7)
	loader = torch.utils.data.DataLoader(
		torch.utils.data.TensorDataset(torch.arange(2)),
		batch_size=1,
		shuffle=True,
		generator=generator,
	)
	epoch_start_rng = loader.generator.get_state().clone()
	student = torch.nn.Linear(1, 1)
	components = SimpleNamespace(
		student=student,
		heads=torch.nn.Linear(1, 1),
		replacement_token=torch.nn.Parameter(torch.zeros(1)),
		optimizer=torch.optim.SGD(student.parameters(), lr=0.1),
		mae_checkpoint_config={},
		trainability_summary=TrainabilitySummary(1, 1, ('weight',)),
	)
	captured: dict[str, object] = {}
	monkeypatch.setattr(
		runner,
		'save_strat_hmm_rolling_checkpoint',
		lambda _output_root, **kwargs: (
			captured.update(kwargs)
			or SimpleNamespace(latest_path=tmp_path / 'latest.pt')
		),
	)

	runner._save_periodic_checkpoint(
		output_root=tmp_path,
		components=components,
		config={},
		metrics={'loss': 1.0},
		global_step=1,
		epoch=1,
		checkpoint_kind='step',
		batch_index=0,
		dataloader=loader,
		amp_enabled=False,
		scaler=None,
		control_identity=None,
		checkpoint_selection=None,
		target_refresh_state={'refresh_phase': 'training'},
		epoch_start_dataloader_rng_state=epoch_start_rng,
	)

	assert captured['checkpoint_kind'] == 'step'
	assert captured['batch_index'] == 0
	assert torch.equal(
		captured['rng_state']['dataloader_generator'], epoch_start_rng  # type: ignore[index]
	)


def test_periodic_refresh_shutdown_releases_old_dataloader_workers() -> None:
	loader = torch.utils.data.DataLoader(
		torch.utils.data.TensorDataset(torch.arange(4)),
		batch_size=1,
		num_workers=1,
		persistent_workers=True,
	)
	iterator = iter(loader)
	next(iterator)
	workers = tuple(iterator._workers)
	assert any(worker.is_alive() for worker in workers)

	runner._shutdown_strat_hmm_dataloader(loader)

	assert loader._iterator is None
	assert all(not worker.is_alive() for worker in workers)

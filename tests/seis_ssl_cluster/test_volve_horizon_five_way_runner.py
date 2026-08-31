from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import yaml

from seis_ssl_cluster.volve.horizon_five_way_config import (
	FIVE_WAY_MODEL_IDS,
	volve_horizon_five_way_config_from_mapping,
)
from seis_ssl_cluster.volve.horizon_five_way_runner import (
	FIVE_WAY_CONDITION_COUNT,
	plan_volve_horizon_five_way_jobs,
	resolve_volve_horizon_five_way_job,
	run_volve_horizon_five_way_job,
)
from seis_ssl_cluster.volve.horizon_layouts import DATA_SIZE_PREFIX, LAYOUT_IDS

if TYPE_CHECKING:
	from pathlib import Path


def _config_mapping(tmp_path: Path) -> dict[str, object]:
	artifact_root = (tmp_path / 'artifacts').resolve()
	return {
		'paths': {
			'artifact_root': str(artifact_root),
			'volve_root': str((tmp_path / 'public').resolve()),
		},
		'dataset': {'survey_id': 'volve_st10010'},
		'inputs': {
			'canonical_input_metadata': str(
				artifact_root / 'data/volve/canonical.json'
			)
		},
		'models': {
			model_id: {
				'checkpoint': str(
					artifact_root / 'checkpoints' / model_id / 'latest.pt'
				),
				'embeddings_dir': str(artifact_root / 'embeddings' / model_id),
			}
			for model_id in FIVE_WAY_MODEL_IDS
		},
		'outputs': {
			'runs_root': str(artifact_root / 'horizon/five_way/runs'),
			'summary_root': str(artifact_root / 'horizon/five_way/summary'),
		},
		'decoder': {
			'embedding_dim': 384,
			'class_count': 5,
			'hidden_channels': [128, 64, 32],
			'upsample_factors': [[2, 2, 2]] * 3,
			'upsample_mode': 'nearest',
			'normalization': 'voxelwise_layer_norm',
		},
		'tiles': {
			'patch_size': [8, 8, 8],
			'core_size_tokens': [8, 8, 27],
			'context_halo_tokens': [1, 1, 0],
			'window_start': 552,
			'window_stop': 768,
			'min_token_valid_fraction': 1.0,
		},
		'train': {
			'epochs': 50,
			'batch_size': 1,
			'learning_rate': 1.0e-3,
			'weight_decay': 1.0e-4,
			'sampling_mode': 'all_tiles_once',
			'seed': 42000,
			'amp': True,
			'gradient_clip_norm': 1.0,
		},
	}


def test_plan_contains_exactly_75_fixed_order_cells(tmp_path: Path) -> None:
	config = volve_horizon_five_way_config_from_mapping(_config_mapping(tmp_path))
	conditions = plan_volve_horizon_five_way_jobs(config)

	assert len(conditions) == FIVE_WAY_CONDITION_COUNT == 75
	assert len(set(conditions)) == 75
	assert conditions[0] == ('mae', 'layout_000', 'small')
	assert conditions[-1] == ('random', 'layout_004', 'large')
	assert {row[0] for row in conditions} == set(FIVE_WAY_MODEL_IDS)
	assert {row[1] for row in conditions} == set(LAYOUT_IDS)
	assert {row[2] for row in conditions} == set(DATA_SIZE_PREFIX)


def test_resolver_rejects_unknown_cells_and_uses_canonical_output_path(
	tmp_path: Path,
) -> None:
	config = volve_horizon_five_way_config_from_mapping(_config_mapping(tmp_path))
	job = resolve_volve_horizon_five_way_job(
		config,
		model='local_barlow_twins_hmm_k6',
		layout='layout_003',
		size='medium',
	)
	assert job.output_dir == (
		config.runs_root
		/ 'model=local_barlow_twins_hmm_k6/layout=layout_003/size=medium'
	)
	assert job.metrics_path == job.output_dir / 'metrics.json'
	with pytest.raises(ValueError, match='unknown Volve horizon five-way model'):
		resolve_volve_horizon_five_way_job(
			config, model='unknown', layout='layout_000', size='small'
		)
	with pytest.raises(ValueError, match='unknown layout'):
		resolve_volve_horizon_five_way_job(
			config, model='mae', layout='layout_999', size='small'
		)
	with pytest.raises(ValueError, match='unknown data size'):
		resolve_volve_horizon_five_way_job(
			config, model='mae', layout='layout_000', size='tiny'
		)


def test_run_rejects_completed_and_foreign_resume_before_execution(
	tmp_path: Path,
) -> None:
	output_dir = tmp_path / 'runs/model=mae/layout=layout_000/size=small'
	plan = SimpleNamespace(output_dir=output_dir)
	output_dir.mkdir(parents=True)
	(output_dir / 'metrics.json').write_text('{}', encoding='utf-8')
	with pytest.raises(FileExistsError, match='already complete'):
		run_volve_horizon_five_way_job(plan)  # type: ignore[arg-type]
	(output_dir / 'metrics.json').unlink()
	with pytest.raises(ValueError, match=r'exact cell latest\.pt'):
		run_volve_horizon_five_way_job(  # type: ignore[arg-type]
			plan,
			resume=tmp_path / 'runs/model=random/latest.pt',
		)


def test_run_accepts_exact_cell_resume_and_delegates(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	output_dir = tmp_path / 'runs/model=mae/layout=layout_000/size=small'
	plan = SimpleNamespace(output_dir=output_dir)
	resume = output_dir / 'latest.pt'
	expected = output_dir / 'metrics.json'
	captured: dict[str, object] = {}

	def fake_run(_plan, **kwargs):
		captured.update(kwargs)
		return expected

	monkeypatch.setattr(
		'seis_ssl_cluster.volve.horizon_five_way_runner.run_frozen_horizon_job',
		fake_run,
	)
	result = run_volve_horizon_five_way_job(  # type: ignore[arg-type]
		plan,
		device='cpu',
		max_steps=1,
		resume=resume,
	)

	assert result == expected
	assert captured == {'device': 'cpu', 'max_steps': 1, 'resume': resume}


def test_cli_dry_run_branch_writes_nothing(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
	capsys: pytest.CaptureFixture[str],
) -> None:
	module = importlib.import_module(
		'proc.seis_ssl_cluster.run_volve_horizon_five_way'
	)
	config_path = tmp_path / 'five_way.yaml'
	config_path.write_text(
		yaml.safe_dump(_config_mapping(tmp_path), sort_keys=False),
		encoding='utf-8',
	)
	plan = SimpleNamespace(
		model='mae',
		layout_id='layout_000',
		data_size='small',
		split_plan=SimpleNamespace(
			identity=lambda: {'selected_physical_lines': {}},
			scientific_identity_sha256='a' * 64,
		),
		run_identity={'decoder': {'initial_state_sha256': 'b' * 64}},
		effective_per_horizon_counts={'train': (1, 1, 1, 1, 1)},
		tile_records={'train': (object(),), 'validation': (), 'test': ()},
		output_dir=tmp_path / 'artifacts/runs/model=mae/layout=layout_000/size=small',
	)
	monkeypatch.setattr(
		module,
		'inspect_volve_horizon_five_way_job',
		lambda *_args, **_kwargs: plan,
	)
	monkeypatch.setattr(
		module,
		'run_volve_horizon_five_way_job',
		lambda *_args, **_kwargs: pytest.fail('dry-run executed the job'),
	)
	monkeypatch.setattr(
		'sys.argv',
		[
			'run-five-way',
			'--config',
			str(config_path),
			'--model',
			'mae',
			'--layout',
			'layout_000',
			'--size',
			'small',
			'--dry-run',
		],
	)
	before = {path.relative_to(tmp_path) for path in tmp_path.rglob('*')}
	module.main()
	after = {path.relative_to(tmp_path) for path in tmp_path.rglob('*')}

	assert before == after
	assert 'execution: dry-run; no files written' in capsys.readouterr().out


def test_cli_validates_model_after_loading_config() -> None:
	module = importlib.import_module(
		'proc.seis_ssl_cluster.run_volve_horizon_five_way'
	)
	parser = module.build_parser()
	args = parser.parse_args(
		[
			'--model',
			'not-yet-validated',
			'--layout',
			'layout_000',
			'--size',
			'small',
			'--dry-run',
		]
	)
	assert args.model == 'not-yet-validated'
	assert args.dry_run is True
	assert args.device == 'auto'

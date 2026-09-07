'''Config contracts for the Volve local Barlow Twins recipe-arm experiment.'''

from __future__ import annotations

import re
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.volve.horizon_recipe_arm import (
	RECIPE_ARM_BENCHMARK_ID,
	volve_horizon_recipe_arm_config_from_mapping,
)

ROOT = Path('experiments/volve/horizon_benchmark_v1/32_local_bt_recipe_arm_v1')
FIVE_WAY_ROOT = Path(
	'experiments/volve/horizon_benchmark_v1/31_mae_local_bt_hmm_five_way_v1'
)
CANONICAL_PRETRAINING = (
	FIVE_WAY_ROOT / '10_stage1/local_barlow_twins/02_full_100ep.yaml'
)
CANONICAL_EMBEDDING = FIVE_WAY_ROOT / '40_embeddings/03_local_barlow_twins.yaml'
CANONICAL_DOWNSTREAM = FIVE_WAY_ROOT / '50_five_way.yaml'
ARM_NAMESPACE = 'local_bt_recipe_arm_v1'
RECIPE_EPOCHS = 3
RECIPE_GLOBAL_STEPS = 1_875
ARM_CONTRACTS: dict[str, tuple[dict[str, object], list[int] | None]] = {
	'flip_3ep': ({'horizontal_flip_probability': 0.5}, None),
	'rot90_asym_g040_3ep': (
		{'policy': 'xy_rot90_asymmetric_noise_v1', 'gaussian_noise_std': 0.4},
		[2, 2, 1],
	),
	'rot90_asym_g060_3ep': (
		{'policy': 'xy_rot90_asymmetric_noise_v1', 'gaussian_noise_std': 0.6},
		[2, 2, 1],
	),
	'rot90_asym_g080_3ep': (
		{'policy': 'xy_rot90_asymmetric_noise_v1', 'gaussian_noise_std': 0.8},
		[2, 2, 1],
	),
}
ARM_IDS = tuple(ARM_CONTRACTS)


@pytest.fixture
def artifact_environment(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> Path:
	'''Point both required roots at a writable pytest directory.'''
	artifact_root = tmp_path / 'artifacts'
	monkeypatch.setenv('SEIS_SSL_CLUSTER_ARTIFACT_ROOT', str(artifact_root))
	monkeypatch.setenv('SEIS_SSL_CLUSTER_VOLVE_ROOT', str(tmp_path / 'public'))
	return artifact_root


def _mapping(config: object, key: str) -> dict[str, object]:
	assert isinstance(config, dict)
	value = config[key]
	assert isinstance(value, dict)
	return cast('dict[str, object]', value)


def test_every_arm_has_a_pretraining_embedding_and_downstream_config() -> None:
	for stage in ('10_pretraining', '20_embeddings', '30_downstream'):
		found = {path.stem for path in (ROOT / stage).glob('*.yaml')}
		assert found == set(ARM_IDS), stage


@pytest.mark.usefixtures('artifact_environment')
@pytest.mark.parametrize('arm_id', ARM_IDS)
def test_pretraining_differs_from_canonical_only_in_the_declared_recipe(
	arm_id: str,
) -> None:
	augmentations, window = ARM_CONTRACTS[arm_id]
	canonical = deepcopy(load_config(CANONICAL_PRETRAINING))
	arm = load_config(ROOT / '10_pretraining' / f'{arm_id}.yaml')
	_mapping(canonical, 'paths')['output_root'] = _mapping(arm, 'paths')[
		'output_root'
	]
	canonical['augmentations'] = dict(augmentations)
	_mapping(canonical, 'train')['epochs'] = RECIPE_EPOCHS
	if window is not None:
		_mapping(canonical, 'barlow_twins')['positive_window_tokens'] = list(window)
	assert arm == canonical


@pytest.mark.usefixtures('artifact_environment')
@pytest.mark.parametrize('arm_id', ARM_IDS)
def test_pretraining_writes_into_the_recipe_arm_namespace(arm_id: str) -> None:
	arm = load_config(ROOT / '10_pretraining' / f'{arm_id}.yaml')
	output_root = str(_mapping(arm, 'paths')['output_root'])
	stem = arm_id.removesuffix('_3ep')
	assert output_root.endswith(f'/{ARM_NAMESPACE}/{stem}/full_3ep')


@pytest.mark.usefixtures('artifact_environment')
@pytest.mark.parametrize('arm_id', ARM_IDS)
def test_embedding_extraction_differs_only_in_its_source_and_output(
	arm_id: str,
) -> None:
	canonical = deepcopy(load_config(CANONICAL_EMBEDDING))
	arm = load_config(ROOT / '20_embeddings' / f'{arm_id}.yaml')
	canonical_embeddings = _mapping(canonical, 'embeddings')
	arm_embeddings = _mapping(arm, 'embeddings')
	canonical_embeddings['checkpoint'] = arm_embeddings['checkpoint']
	canonical_embeddings['output_dir'] = arm_embeddings['output_dir']
	assert arm == canonical


@pytest.mark.usefixtures('artifact_environment')
@pytest.mark.parametrize('arm_id', ARM_IDS)
def test_embedding_extraction_reads_the_arm_pretraining_checkpoint(
	arm_id: str,
) -> None:
	pretraining = load_config(ROOT / '10_pretraining' / f'{arm_id}.yaml')
	embedding = load_config(ROOT / '20_embeddings' / f'{arm_id}.yaml')
	output_root = str(_mapping(pretraining, 'paths')['output_root'])
	assert _mapping(embedding, 'embeddings')['checkpoint'] == (
		f'{output_root}/latest.pt'
	)


@pytest.mark.usefixtures('artifact_environment')
@pytest.mark.parametrize('arm_id', ARM_IDS)
def test_downstream_keeps_the_five_way_decoder_contract(arm_id: str) -> None:
	canonical = load_config(CANONICAL_DOWNSTREAM)
	arm = load_config(ROOT / '30_downstream' / f'{arm_id}.yaml')
	for block in ('decoder', 'tiles', 'train'):
		assert arm[block] == canonical[block], block
	assert _mapping(arm, 'dataset') == _mapping(canonical, 'dataset')
	assert _mapping(arm, 'inputs') == _mapping(canonical, 'inputs')
	assert _mapping(arm, 'paths') == _mapping(canonical, 'paths')


@pytest.mark.usefixtures('artifact_environment')
@pytest.mark.parametrize('arm_id', ARM_IDS)
def test_downstream_baseline_is_the_canonical_random_encoder(arm_id: str) -> None:
	canonical = load_config(CANONICAL_DOWNSTREAM)
	arm = load_config(ROOT / '30_downstream' / f'{arm_id}.yaml')
	expected = _mapping(_mapping(canonical, 'models'), 'random')
	assert _mapping(arm, 'baseline') == expected


@pytest.mark.usefixtures('artifact_environment')
@pytest.mark.parametrize('arm_id', ARM_IDS)
def test_downstream_recipe_matches_its_pretraining_config(arm_id: str) -> None:
	pretraining = load_config(ROOT / '10_pretraining' / f'{arm_id}.yaml')
	downstream = load_config(ROOT / '30_downstream' / f'{arm_id}.yaml')
	arm_block = _mapping(downstream, 'arm')
	recipe = _mapping(arm_block, 'recipe')
	barlow_twins = _mapping(pretraining, 'barlow_twins')
	train = _mapping(pretraining, 'train')
	assert arm_block['arm_id'] == arm_id
	assert recipe['augmentations'] == pretraining['augmentations']
	assert recipe['method'] == barlow_twins['method']
	assert recipe['local_pairs_per_crop'] == barlow_twins['local_pairs_per_crop']
	assert recipe['positive_window_tokens'] == barlow_twins.get(
		'positive_window_tokens'
	)
	assert recipe['epochs'] == train['epochs']
	assert recipe['samples_per_epoch'] == train['samples_per_epoch']
	assert recipe['batch_size'] == train['batch_size']
	assert recipe['learning_rate'] == train['lr']
	assert recipe['weight_decay'] == train['weight_decay']
	assert recipe['seed'] == train['seed']


@pytest.mark.usefixtures('artifact_environment')
@pytest.mark.parametrize('arm_id', ARM_IDS)
def test_downstream_binds_the_arm_checkpoint_and_embeddings(arm_id: str) -> None:
	pretraining = load_config(ROOT / '10_pretraining' / f'{arm_id}.yaml')
	embedding = load_config(ROOT / '20_embeddings' / f'{arm_id}.yaml')
	downstream = load_config(ROOT / '30_downstream' / f'{arm_id}.yaml')
	arm_block = _mapping(downstream, 'arm')
	output_root = str(_mapping(pretraining, 'paths')['output_root'])
	assert arm_block['checkpoint'] == f'{output_root}/latest.pt'
	assert arm_block['embeddings_dir'] == (
		_mapping(embedding, 'embeddings')['output_dir']
	)


@pytest.mark.usefixtures('artifact_environment')
def test_every_downstream_config_resolves_and_shares_one_runs_root() -> None:
	runs_roots: set[str] = set()
	summary_roots: set[str] = set()
	for arm_id in ARM_IDS:
		raw = load_config(ROOT / '30_downstream' / f'{arm_id}.yaml')
		config = volve_horizon_recipe_arm_config_from_mapping(raw)
		assert config.arm_id == arm_id
		assert config.benchmark_id == RECIPE_ARM_BENCHMARK_ID
		assert config.recipe.epochs == RECIPE_EPOCHS
		assert config.recipe.global_steps == RECIPE_GLOBAL_STEPS
		runs_roots.add(str(config.runs_root))
		summary_roots.add(str(config.summary_root))
	assert len(runs_roots) == 1
	assert len(summary_roots) == len(ARM_IDS)


@pytest.mark.usefixtures('artifact_environment')
def test_recipe_arm_namespace_is_not_shared_with_another_experiment() -> None:
	owners = {
		path.parts[3]
		for path in Path('experiments').rglob('*.yaml')
		if ARM_NAMESPACE in path.read_text(encoding='utf-8')
	}
	assert owners == {ROOT.name}


def test_readme_documents_the_workflow_and_result_provenance() -> None:
	readme = (ROOT / 'README.md').read_text(encoding='utf-8')
	for stage in ('10_pretraining', '20_embeddings', '30_downstream'):
		assert stage in readme
	for cli in (
		'train_amp_barlow_twins.py',
		'extract_embeddings.py',
		'run_volve_horizon_recipe_arm.py',
		'summarize_volve_horizon_recipe_arm.py',
	):
		assert cli in readme
	assert 'local_bt_rot90_asymmetric_noise_recipe.md' in readme
	assert 'Gate tokens' not in readme
	assert '```python' not in readme


def test_readme_shell_blocks_are_valid_bash(tmp_path: Path) -> None:
	readme = (ROOT / 'README.md').read_text(encoding='utf-8')
	blocks = re.findall(r'```bash\n(.*?)```', readme, flags=re.DOTALL)
	assert blocks
	for index, block in enumerate(blocks):
		path = tmp_path / f'runbook_block_{index}.sh'
		path.write_text(block, encoding='utf-8')
		subprocess.run(  # noqa: S603
			['bash', '-n', str(path)],  # noqa: S607
			check=True,
			capture_output=True,
			text=True,
		)

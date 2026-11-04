"""Fixed experiment matrix, CLI planning, and exact publication checks."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pytest

from seis_ssl_cluster.clustering.kmeans_definition import (
	LAYOUTS,
	SIZES,
	SOURCES,
	SURVEYS,
	read_yaml,
	validate_study,
)
from seis_ssl_cluster.clustering.kmeans_study import (
	PUBLISHED_FILES,
	command_plan,
	comparison_rows,
	publish_summary,
	validate_cells,
)

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / 'experiments/kmeans/kmeans_k6810_multi_source_v1/matrix.yaml'


def _args(**kwargs):
	return argparse.Namespace(
		first='targets',
		last='summary',
		survey=None,
		candidate=None,
		layout=None,
		size=None,
		resume=False,
		**kwargs,
	)


def _cells():
	return [
		{
			'survey': s,
			'source_family': f,
			'layout_id': l,
			'data_size': z,
			'primary': 0.6 + i / 100,
			'control_primary': 0.5,
			'secondary': 0.7,
			'control_secondary': 0.6,
		}
		for s in SURVEYS
		for f in SOURCES
		for i, l in enumerate(LAYOUTS)
		for z in SIZES
	]


def test_study_has_nine_matched_training_arms_and_135_cells():
	matrix = validate_study(MATRIX, ROOT)
	assert sum(len(arms) for arms in matrix['surveys'].values()) == 9
	commands = command_plan(MATRIX, ROOT, _args())
	assert sum(stage == 'full' for stage, _ in commands) == 9
	assert sum(stage == 'downstream' for stage, _ in commands) == 135
	assert all('__main__' not in command for _, command in commands)


@pytest.mark.parametrize('source', SOURCES)
def test_volve_downstream_binds_its_existing_training_recipe(source):
	arm = read_yaml(MATRIX)['surveys']['volve'][source]
	downstream = read_yaml(ROOT / arm['configs']['downstream'])
	recipe = downstream['arm']['multi_head_training_config']
	resolved = Path(recipe.replace('${SEIS_SSL_CLUSTER_WORKSPACE}', str(ROOT)))
	assert resolved == ROOT / arm['configs']['training']
	assert resolved.is_file()


def test_command_plan_rejects_invalid_resume_and_partial_summary():
	args = _args()
	args.resume = True
	with pytest.raises(ValueError, match='resume requires'):
		command_plan(MATRIX, ROOT, args)
	args = _args()
	args.survey = 'f3'
	with pytest.raises(ValueError, match='complete three-survey'):
		command_plan(MATRIX, ROOT, args)


@pytest.mark.parametrize('kind', ['missing', 'duplicate'])
def test_summary_requires_all_unique_cells(kind):
	cells = _cells()
	if kind == 'missing':
		cells.pop()
	else:
		cells[-1] = copy.deepcopy(cells[0])
	with pytest.raises(ValueError, match='135 unique'):
		validate_cells(cells)


def test_summary_publishes_exact_review_files(tmp_path):
	cells = _cells()
	rows = comparison_rows(cells)
	assert len(rows) == 27
	assert all(row['primary_difference_mean'] == pytest.approx(0.12) for row in rows)
	assert all(row['layout_count'] == 5 for row in rows)
	output = tmp_path / 'summary'
	publish_summary(output, {'cells': cells, 'rows': rows, 'interpretation': 'fixture'})
	assert {path.name for path in output.iterdir()} == PUBLISHED_FILES
	with pytest.raises(FileExistsError):
		publish_summary(output, {'cells': cells, 'rows': rows})

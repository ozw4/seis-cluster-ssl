# ruff: noqa: INP001
"""Apply the experiment-local screen or final overlap-subcrop PoC gate."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path

LAYOUT_IDS = tuple(f'layout_{index:03d}' for index in range(5))
SCREEN_LAYOUT_ID = 'layout_001'
DATA_SIZE = 'medium'
EPOCHS = 10
RANDOM_MODEL_ID = 'random'
CANDIDATE_MODEL_ID = 'local_barlow_twins'
AGGREGATION_UNIT = 'unique_validation_voxel'
RESULT_METRICS = (
	'macro_f1',
	'mean_iou',
	'balanced_accuracy',
	'weighted_f1',
)


def build_parser() -> argparse.ArgumentParser:
	"""Build the two-mode decision parser."""
	parser = argparse.ArgumentParser(
		description='Apply the F3 overlap-subcrop PoC adoption gate.'
	)
	parser.add_argument('--candidate-id', required=True)
	parser.add_argument('--random-runs-root', required=True, type=Path)
	parser.add_argument('--candidate-runs-root', required=True, type=Path)
	parser.add_argument('--mode', required=True, choices=('screen', 'final'))
	return parser


def decide(
	*,
	candidate_id: str,
	random_runs_root: Path,
	candidate_runs_root: Path,
	mode: str,
) -> dict[str, object]:
	"""Read completed metrics and return the strict screen/final decision."""
	if Path(candidate_id).name != candidate_id or candidate_id in {'', '.', '..'}:
		raise ValueError('candidate-id must be one non-empty path segment')
	if mode not in {'screen', 'final'}:
		raise ValueError("mode must be 'screen' or 'final'")
	layout_ids = (SCREEN_LAYOUT_ID,) if mode == 'screen' else LAYOUT_IDS
	layouts = [
		_layout_result(
			layout_id,
			random_runs_root=random_runs_root,
			candidate_runs_root=candidate_runs_root,
		)
		for layout_id in layout_ids
	]
	wins = sum(bool(layout['strict_win']) for layout in layouts)
	screen_layout = next(
		layout for layout in layouts if layout['layout_id'] == SCREEN_LAYOUT_ID
	)
	screen_passed = bool(screen_layout['strict_win'])
	adopted = mode == 'final' and wins >= 4
	return {
		'candidate_id': candidate_id,
		'epoch': EPOCHS,
		'mode': mode,
		'primary_metric': 'macro_f1',
		'aggregation_unit': AGGREGATION_UNIT,
		'screen_passed': screen_passed,
		'wins': wins,
		'losses_or_ties': len(layouts) - wins,
		'adopted': adopted,
		'layouts': layouts,
	}


def _layout_result(
	layout_id: str,
	*,
	random_runs_root: Path,
	candidate_runs_root: Path,
) -> dict[str, object]:
	random_path = _metrics_path(
		random_runs_root, model_id=RANDOM_MODEL_ID, layout_id=layout_id
	)
	candidate_path = _metrics_path(
		candidate_runs_root,
		model_id=CANDIDATE_MODEL_ID,
		layout_id=layout_id,
	)
	random = _read_metrics(random_path, label=f'random/{layout_id}')
	candidate = _read_metrics(candidate_path, label=f'candidate/{layout_id}')
	delta = candidate['macro_f1'] - random['macro_f1']
	return {
		'layout_id': layout_id,
		'random': random,
		'candidate': candidate,
		'paired_delta': delta,
		'strict_win': delta > 0.0,
		'random_metrics_path': str(random_path),
		'candidate_metrics_path': str(candidate_path),
	}


def _metrics_path(runs_root: Path, *, model_id: str, layout_id: str) -> Path:
	return (
		runs_root
		/ f'model={model_id}'
		/ f'layout={layout_id}'
		/ f'size={DATA_SIZE}'
		/ 'evaluation'
		/ 'metrics.json'
	)


def _read_metrics(path: Path, *, label: str) -> dict[str, float | str | int]:
	if not path.is_file():
		raise FileNotFoundError(f'missing {label} metrics: {path}')
	payload = json.loads(path.read_text(encoding='utf-8'))
	if not isinstance(payload, Mapping):
		raise TypeError(f'{label} metrics must contain a JSON object')
	if payload.get('aggregation_unit') != AGGREGATION_UNIT:
		raise ValueError(
			f'{label} aggregation_unit must equal {AGGREGATION_UNIT!r}'
		)
	metrics: dict[str, float | str | int] = {
		'aggregation_unit': AGGREGATION_UNIT,
	}
	for metric in RESULT_METRICS:
		value = payload.get(metric)
		if (
			isinstance(value, bool)
			or not isinstance(value, int | float)
			or not math.isfinite(float(value))
		):
			raise ValueError(f'{label} {metric} must be finite numeric')
		metrics[metric] = float(value)
	voxel_count = payload.get('evaluation_voxel_count')
	if (
		isinstance(voxel_count, bool)
		or not isinstance(voxel_count, int)
		or voxel_count <= 0
	):
		raise ValueError(f'{label} evaluation_voxel_count must be positive')
	metrics['evaluation_voxel_count'] = voxel_count
	return metrics


def main() -> None:
	"""Write the decision JSON before returning the gate-specific exit code."""
	args = build_parser().parse_args()
	result = decide(
		candidate_id=args.candidate_id,
		random_runs_root=args.random_runs_root,
		candidate_runs_root=args.candidate_runs_root,
		mode=args.mode,
	)
	output = args.candidate_runs_root.parent / f'{args.mode}_decision.json'
	output.parent.mkdir(parents=True, exist_ok=True)
	text = json.dumps(result, indent=2, sort_keys=True) + '\n'
	output.write_text(text, encoding='utf-8')
	print(text, end='')
	print(f'decision_json: {output}')
	passed = result['screen_passed'] if args.mode == 'screen' else result['adopted']
	raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
	main()

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export SEIS_SSL_CLUSTER_WORKSPACE="$PWD"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
: "${SEIS_SSL_CLUSTER_ARTIFACT_ROOT:?Set the existing artifact root}"
: "${F3_ROOT:?Use the frozen F3 label path}"
: "${SEIS_SSL_CLUSTER_VOLVE_ROOT:?Set the existing Volve data root}"
matrix=experiments/hmm_v2/zero_distillation_paper_sections_v1/barlow_sources.yaml
operations="$SEIS_SSL_CLUSTER_ARTIFACT_ROOT/operations/hmm_v2/barlow_zero_distillation_paper_sources_v1"
mkdir -p "$operations"
exec 9>"$operations/study.lock"
flock -n 9
run() {
  python -m seis_ssl_cluster.hmm.distillation_sensitivity --matrix "$matrix" "$@"
}
run --phase validate
for phase in full embeddings; do
  for survey in f3 parihaka volve; do
    run --phase "$phase" --survey "$survey" --weight 0.0 --resume
  done
done
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from seis_ssl_cluster.config import load_config
from seis_ssl_cluster.embedding.writer import file_sha256
from seis_ssl_cluster.hmm.distillation_sensitivity import _cell_path, _run

matrix_path = Path('experiments/hmm_v2/zero_distillation_paper_sections_v1/barlow_sources.yaml')
matrix = load_config(matrix_path)
root = Path(os.environ['SEIS_SSL_CLUSTER_ARTIFACT_ROOT'])
logs = root / 'operations/hmm_v2/barlow_zero_distillation_paper_sources_v1/execution'
evidence = []
for survey, arms in matrix['surveys'].items():
    arm = arms['0.0']
    path = _cell_path(survey, arm, 'layout_000', 'large')
    command = [os.sys.executable, '-m', 'seis_ssl_cluster.hmm.multi_source_cell',
               '--survey', survey, '--config', arm['configs']['downstream']['path'],
               '--candidate', arm['candidate_id'], '--layout', 'layout_000', '--size', 'large']
    if not path.is_file():
        _run([*command, '--dry-run'], logs, f'decoder_{survey}_dry_run')
        checkpoint = path.parent.parent / 'decoder/latest.pt' if survey == 'f3' else path.parent / 'latest.pt'
        if checkpoint.exists():
            command += ['--resume', str(checkpoint)]
        _run(command, logs, f'decoder_{survey}_layout_000_large')
    paths = [_cell_path(survey, arms[w], 'layout_000', 'large') for w in ('0.0', '0.2')]
    evidence.append({'survey': survey, 'weight': '0.0', 'layout': 'layout_000', 'size': 'large',
                     'candidate_id': arm['candidate_id'], 'configs': arm['configs'],
                     'metrics_paths': [str(p) for p in paths],
                     'metrics_sha256': [file_sha256(p) for p in paths]})
output = root / 'studies/hmm_v2/zero_distillation_paper_sections_v1/barlow_sources.json'
if not output.exists():
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump({'study': matrix['study'], 'status': 'complete',
                   'scope': matrix['scope'], 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                   'new_encoder_trainings': 3, 'new_embedding_extractions': 3, 'new_decoder_cells': 3,
                   'matrix_sha256': file_sha256(matrix_path), 'evidence': evidence}, stream, indent=2)
        stream.write('\n')
else:
    previous = json.loads(output.read_text())
    if previous['evidence'] != evidence or previous['matrix_sha256'] != file_sha256(matrix_path):
        raise ValueError('completed Barlow source receipt differs from current artifacts')
print(output)
PY

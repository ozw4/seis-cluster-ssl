# F3 Strat HMM Pretraining M2-A Boundary Weighting

The complete ordered/resumable runbook and failure policy are in
[`docs/f3_strat_hmm_m2a_boundary_weighting.md`](../../../../docs/f3_strat_hmm_m2a_boundary_weighting.md).
M1 is complete/strong positive and is not rerun as part of this experiment.

This experiment changes only the pseudo-target boundary weight from the M1
pretraining condition. The preregistered candidate uses `alpha=0.5` and
`tau=2.0` tokens, so tokens adjacent to a transition have weight `0.5` and the
weight approaches `1.0` with distance. No parameter sweep is part of M2-A.

The model tag is
`strat_hmm_pretext_m2a_boundary_a050_t2_k6_topblock1_distill`. Teacher and
student initialization both use the M1 `full_100ep/mae_latest.pt` checkpoint.

## Export and parity check

Validate both exports without writing artifacts:

```bash
bash experiments/f3/facies_benchmark_v1/84_strat_hmm_pretraining_m2a_boundary/01_export_alpha0_parity_bootstrap.sh --dry-run

bash experiments/f3/facies_benchmark_v1/84_strat_hmm_pretraining_m2a_boundary/02_export_boundary_weighted_bootstrap.sh --dry-run
```

Run the exports after the plans resolve:

```bash
bash experiments/f3/facies_benchmark_v1/84_strat_hmm_pretraining_m2a_boundary/01_export_alpha0_parity_bootstrap.sh

bash experiments/f3/facies_benchmark_v1/84_strat_hmm_pretraining_m2a_boundary/02_export_boundary_weighted_bootstrap.sh
```

The alpha-zero export is deliberately separate from the candidate artifact.
Use this check to confirm that its labels, confidence, and validity arrays are
exactly equal to M1, its boundary weight is one for valid tokens and zero for
invalid tokens, and its schema-v2 metadata records `alpha=0` and `tau=2`:

```bash
PYTHONPATH=src python - <<'PY'
import json
from pathlib import Path

import numpy as np

base = Path('/workspace/artifacts/seis_ssl_cluster/pseudo_targets/f3/facies_benchmark_v1')
m1 = base / 'strat_hmm_k6_pca64_resid_token_phase_edge8_expected3_iter10_bootstrap' / 'k6'
parity = base / 'strat_hmm_k6_pca64_resid_token_phase_edge8_expected3_iter10_bootstrap_boundary_a000_t2_parity' / 'k6'
for labels_path in sorted(parity.glob('*.hmm_labels_token.npy')):
    survey = labels_path.name.removesuffix('.hmm_labels_token.npy')
    for suffix in ('hmm_labels_token.npy', 'hmm_confidence_token.npy', 'valid_tokens.npy'):
        assert np.array_equal(np.load(m1 / f'{survey}.{suffix}'), np.load(parity / f'{survey}.{suffix}'))
    valid = np.load(parity / f'{survey}.valid_tokens.npy')
    weight = np.load(parity / f'{survey}.hmm_boundary_weight_token.npy')
    assert np.array_equal(weight, valid.astype(np.float32))
    metadata = json.loads((parity / f'{survey}.pseudo_target_metadata.json').read_text())
    assert metadata['schema_version'] == 2
    weighting = metadata['source']['boundary_weighting']
    assert weighting['alpha'] == 0.0 and weighting['tau'] == 2.0
print('alpha0 parity: OK')
PY
```

## Smoke and full training

Resolve the CPU-friendly two-step smoke config, then run it:

```bash
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config experiments/f3/facies_benchmark_v1/84_strat_hmm_pretraining_m2a_boundary/03_train_boundary_smoke.yaml \
  --dry-run --device cpu --max-steps 2

python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config experiments/f3/facies_benchmark_v1/84_strat_hmm_pretraining_m2a_boundary/03_train_boundary_smoke.yaml \
  --device cpu --max-steps 2
```

The full run retains the M1 crop, AGC, zero-mask, model, optimizer, batch-size,
epoch, and seed conditions:

```bash
python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config experiments/f3/facies_benchmark_v1/84_strat_hmm_pretraining_m2a_boundary/04_train_boundary_full.yaml \
  --dry-run

python proc/seis_ssl_cluster/train_strat_hmm_pretext.py \
  --config experiments/f3/facies_benchmark_v1/84_strat_hmm_pretraining_m2a_boundary/04_train_boundary_full.yaml
```

## Downstream evaluation

```bash
python proc/seis_ssl_cluster/extract_embeddings.py \
  --config experiments/f3/facies_benchmark_v1/84_strat_hmm_pretraining_m2a_boundary/05_extract_student_embeddings.yaml

python proc/seis_ssl_cluster/build_f3_lithology_token_dataset.py \
  --config experiments/f3/facies_benchmark_v1/84_strat_hmm_pretraining_m2a_boundary/06_build_lithology_token_dataset.yaml

python proc/seis_ssl_cluster/train_f3_lithology_probe.py \
  --config experiments/f3/facies_benchmark_v1/84_strat_hmm_pretraining_m2a_boundary/07_train_lithology_probe.yaml

python proc/seis_ssl_cluster/build_f3_lithology_report.py \
  --config experiments/f3/facies_benchmark_v1/84_strat_hmm_pretraining_m2a_boundary/08_build_lithology_report.yaml
```

Embedding extraction uses the M2-A full-run `best.pt` and the same
`overlap_x16` geometry as M1. The token dataset, balanced linear probe, scaler,
class weighting, and random seed are unchanged from M1. Checkpoints and NumPy
artifacts remain under `/workspace/artifacts/seis_ssl_cluster`; `reports/`
receives only the bounded lithology report publication.

# seis-cluster-ssl

Experimental codebase for evaluating **HMM Prompt Clustering as a self-supervised pretraining method for 3D seismic interpretation**.

The main research question is whether representations learned with HMM Prompt Clustering improve downstream seismic interpretation across different surveys and tasks. **HMM Prompt Clustering is the primary method under study; MAE and contrastive learning are comparison baselines.**

The Python package name is `seis_ssl_cluster`.

## Experimental scope

The main benchmarks use three seismic datasets:

- F3
- Parihaka
- Volve

Experiments compare downstream performance under matched training conditions using combinations of:

- no self-supervised pretraining
- MAE pretraining
- contrastive pretraining
- HMM Prompt Clustering pretraining

The repository is intended to answer a method-level question: **does HMM Prompt Clustering provide useful pretraining for seismic interpretation, and is that benefit consistent across different downstream tasks?**

## Repository layout

```text
seis-cluster-ssl/
├── src/seis_ssl_cluster/   # reusable implementation
├── proc/seis_ssl_cluster/  # thin CLI entrypoints
├── experiments/            # experiment definitions and conditions
├── reports/                # lightweight tracked results and summaries
├── artifacts/              # local execution products; not tracked
├── docs/                   # runbooks and technical documentation
├── tests/
└── README.md
```

## Output policy

- `experiments/` defines experimental conditions.
- `artifacts/` stores checkpoints, embeddings, predictions, and other execution products and is not tracked by Git.
- `reports/` stores lightweight, human-readable summaries used to compare experiments.

Detailed execution procedures and experiment-specific settings belong in `docs/` and `experiments/`, rather than in this README.

## Development

```bash
python -m compileall -q src proc tests
python -m ruff check .
pytest -q
```

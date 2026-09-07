# seis-cluster-ssl

Research codebase for 3D seismic self-supervised pretraining, clustering, and
downstream interpretation.

The main research question is whether stratigraphic HMM pretext training
improves downstream seismic interpretation across surveys and tasks. MAE and
joint-embedding methods provide comparison baselines.

The Python package name is `seis_ssl_cluster`.

## Scope

NOPIMS provides multi-survey pretraining definitions. Downstream benchmarks cover
three seismic datasets:

- F3
- Parihaka
- Volve

Experiments compare downstream performance under matched conditions using:

- no self-supervised pretraining
- MAE pretraining
- joint-embedding pretraining
- stratigraphic HMM pretext training

## Documentation

- [Documentation index](docs/README.md)
- [Experiment definitions](experiments/)
- [Repository output policy](docs/report_sharing_policy.md)
- [Contributor guide](AGENTS.md)

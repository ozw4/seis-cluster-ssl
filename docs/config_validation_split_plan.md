# Config Validation Split Plan

This historical plan records the separation of config validation from the
former monolithic `config/validate.py` module. The active implementation uses
explicit YAML and CLI paths as its source of truth.

## Public API

`config/validate.py` remains the public resolver facade and dispatcher. Stage
implementations live in focused modules such as `manifest.py`,
`normalization.py`, `pretraining.py`, `embedding.py`, `clustering.py`, and
`cluster_visualization.py`.

## Dependency Rules

- `common.py` owns primitive type, mapping, and explicit-path parsing helpers.
- `base.py` owns stage routing and required top-level sections.
- Stage modules may import `common.py`, `base.py`, and `config.schema`.
- Stage modules must not import `config/validate.py`.
- `config/validate.py` re-exports public symbols for caller compatibility.

## Path Policy

Resolvers preserve configured input and output path values. They validate
required fields, absolute-path requirements where applicable, input/output
collisions, source-data protection, and runtime overwrite behavior. They do not
derive output locations from dataset, version, model, or spec identifiers.

Large local outputs belong outside the repository. Only lightweight,
reproducible review files belong in `results/`.

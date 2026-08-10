# Configuration Validation

Configuration validation is split by responsibility while preserving one public
stage-dispatch surface.

## Public surface

`seis_ssl_cluster.config.validate` exposes `validate_config` and the core stage
resolvers. `validate_config` receives an explicit stage selected by caller code
and dispatches to the matching resolver.

F3 and other domain-specific configuration builders live in focused config
modules and may be imported directly by their owning entrypoints. Compatibility
exports from `config.validate` resolve to those focused modules without changing
their behavior.

## Module responsibilities

- `common.py` owns primitive mapping, type, scalar, sequence, and path
  validators.
- `base.py` owns allowed and required top-level sections, stage injection, and
  shared path-root checks.
- `schema.py` owns stage identifiers, fixed constants, and shared schema
  definitions.
- `manifest.py`, `normalization.py`, `pretraining.py`, `embedding.py`,
  `clustering.py`, `cluster_visualization.py`, and
  `strat_hmm_pseudo_targets.py` own the corresponding core stage resolvers.
- Focused F3 config modules own their domain-specific mapping-to-config
  functions.
- `validate.py` depends on the focused resolvers and dispatches to them. Focused
  resolver modules do not depend on `validate.py`.

## Stage ownership

The proc entrypoint selects the stage. Raw YAML with a top-level `stage` field
is rejected.

Each resolver accepts only the top-level sections declared for its stage.
Unknown sections and missing required sections fail validation. Fixed
code-owned settings are injected into the resolved mapping rather than accepted
as user overrides.

## Path policy

Input and output paths remain exactly as configured. Resolvers do not derive
paths from dataset names, versions, model tags, or run identifiers.

Generated output paths must be non-empty absolute paths. Registry stages protect
the configured raw NOPIMS root from generated outputs. Stage-specific checks
also reject invalid input/output collisions and enforce overwrite or reuse
rules before writing.

`paths.artifact_root` identifies the local artifact area but does not impose a
repository-wide directory hierarchy. Inputs may live outside that root when the
stage contract permits it.

## Failure policy

Invalid types, unsupported values, stale fixed fields, unknown sections,
missing inputs, and identity mismatches fail explicitly. A resolver must not
silently ignore an unknown field or repair a scientific identity.

Configuration resolution completes before stage execution. Dry-run entrypoints
use the same resolver and validation path as write execution.

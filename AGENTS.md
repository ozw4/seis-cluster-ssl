# Repository Guidelines

## Project Structure & Module Organization

Reusable Python code lives in `src/seis_ssl_cluster/`, organized by domain (`data/`, `models/`, `training/`, `embedding/`, `clustering/`, `visualization/`, and related packages). Keep command-line wrappers thin in `proc/seis_ssl_cluster/`; place their generic YAML in `proc/configs/seis_ssl_cluster/`. Versioned experiment definitions and configurations belong under `experiments/nopims/` or `experiments/f3/`. Tests mirror package and CLI behavior in `tests/seis_ssl_cluster/`. Use `docs/` for contracts and runbooks and `tools/` for repository checks. Keep complete execution outputs, intermediate products, and downstream inputs in ignored `artifacts/`; keep only lightweight human-readable summaries in tracked `reports/`, and never use `reports/` as pipeline input.

## Setup, Test, and Development Commands

- `python -m pip install -e ".[dev,cluster,visualization]"` installs the package and common development extras (Python 3.10+).
- `python -m compileall -q src proc tests` catches syntax/import compilation errors.
- `python -m ruff check .` runs the configured lint suite and safe fixes.
- `pytest -q` runs the full test suite.
- `pytest -q -m "not slow and not requires_segy and not requires_cuda"` runs the portable local subset.
- `python tools/check_seis_ssl_cluster_isolation.py` verifies independence from legacy namespaces.

Pipeline stages are config-driven. Prefer a supported `--dry-run` before execution, for example `python proc/seis_ssl_cluster/build_nopims_manifests.py --config proc/configs/seis_ssl_cluster/build_nopims_manifests.yaml --dry-run`.

## Coding Style & Naming Conventions

Ruff targets Python 3.10 and selects all lint families with repository-specific exclusions. Follow its formatter settings: tabs for indentation, single-quoted strings, and formatted code in docstrings. Use type annotations, `snake_case` for modules/functions/variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Import through `seis_ssl_cluster`; reusable logic does not belong in `proc/` scripts.

## Testing Guidelines

Use pytest files and functions named `test_*.py` and `test_*`. Add focused regression tests beside the corresponding package or CLI contract. Mark expensive or environment-dependent coverage with the registered `integration`, `smoke`, `slow`, `requires_segy`, or `requires_cuda` markers. No numeric coverage threshold is configured; new behavior should cover success, validation, and failure paths.

## Commit & Pull Request Guidelines

Recent history favors concise imperative subjects (`Add ...`, `Update ...`) and issue-focused forms such as `chore: address issue #245` or `Batch: address issues #241-#243 (#247)`. Keep commits focused and reference issues when applicable. PRs should explain scope, configuration/artifact effects, and exact validation run; include representative figures or report links for visual changes. Never commit raw data, checkpoints, embeddings, or machine-specific paths. Keep `reports/` limited to reviewable Markdown, JSON, CSV, and representative figures; verify the concrete producer file set in focused tests and inspect `git diff` during review.

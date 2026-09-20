# Repository Guidelines

## Repository boundaries

Reusable Python code lives in `src/seis_ssl_cluster/`; import through `seis_ssl_cluster` and preserve independence from legacy namespaces. Keep command-line wrappers thin in `proc/seis_ssl_cluster/` and their generic YAML in `proc/configs/seis_ssl_cluster/`. Versioned definitions belong under the matching survey in `experiments/`, tests in `tests/seis_ssl_cluster/`, shared guidance in `docs/`, and repository checks in `tools/`.

Keep execution outputs and downstream inputs in `artifacts/`. Tracked `reports/` contain curated review outputs and must never be pipeline inputs. Never commit raw data, checkpoints, embeddings, or machine-specific paths. Consult [the report policy](docs/report_sharing_policy.md) when changing artifact production or publication; report producers must define and test their exact published file set.

## Development and completion

Follow `ruff.toml` for formatting and lint rules. Annotate reusable Python APIs, respecting existing test exceptions. Use [the development guide](docs/development.md) for setup and validation commands and [configuration guidance](docs/configuration.md) when changing YAML resolution or artifact handoffs. Consult the relevant scientific note or experiment runbook for changes to data use, comparisons, or execution. Prefer a supported `--dry-run` before executing a pipeline stage.

For behavior changes, add or update focused regression coverage beside the affected package or CLI contract. Cover meaningful success and failure behavior, preserving explicit contract tests. Use marker meanings from `pytest.ini`; mark slow, SEG-Y-dependent, and CUDA-dependent tests explicitly.

Complete the requested change, including affected documentation and appropriate validation. Reuse successful checks unless new changes, failures, or unresolved concerns justify another run. For an Issue Forge phase, complete the assigned phase and return control to its orchestrator, which owns the configured Checks and subsequent flow phases.

Do not write tests for reversible, low-impact changes that mirror the implementation

## Commits and pull requests

Use concise, focused commit subjects and reference relevant issues. PRs should explain scope, configuration/artifact effects, and exact validation run; include representative figures or report links for visual changes.

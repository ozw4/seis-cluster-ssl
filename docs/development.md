# Development

Use these commands according to the change being made. They are a reference,
not a checklist for every task.

## Setup

Python 3.10 or newer is required. Install the package and common development
extras when preparing an environment:

```bash
python -m pip install -e ".[dev,cluster,visualization]"
```

SEG-Y-dependent work additionally needs the `segy` extra and its relevant data.

## Validation

| Change or purpose | Check |
|---|---|
| Isolated behavior change | Run the affected test files or pytest node IDs. |
| Shared Python behavior or fixtures | `pytest -q -m "not slow and not requires_segy and not requires_cuda"` |
| Dependency/test configuration changes or full-suite verification | `pytest -q` in an environment with the required dependencies, data, and hardware. |
| Python lint validation | `python -m ruff check --no-fix <changed-paths>` |
| Formatting validation | `python -m ruff format --check <changed-paths>` |
| Broad Python syntax compilation | `python -m compileall -q src proc tests` |
| Imports, package boundaries, or CLI dependencies | `python tools/check_seis_ssl_cluster_isolation.py` |

Compilation checks syntax; it does not verify runtime import resolution.
The portable pytest selection excludes expensive and environment-dependent
markers, but still includes applicable integration and smoke tests. Run relevant
excluded tests when changing their behavior and the needed environment is
available; report any validation that could not be performed.

`ruff.toml` enables automatic fixes: `python -m ruff check <changed-paths>` can
modify files. Use `--no-fix` for read-only validation. To format an authorized
change, use `python -m ruff format <changed-paths>`.

For documentation-only edits, check the affected links, paths, and command
descriptions. Behavior and publication changes need their corresponding
regression coverage, including the exact file-set tests required by the
[report policy](report_sharing_policy.md).

### Issue Forge checks

The consumer hook is `.issue_forge/checks/run_changed.sh <fixed-base-commit>`.
It considers committed changes since that base plus staged, unstaged, and
untracked files, excluding `.work/` and `vendor/issue_forge/`.

- Changed shell files receive ShellCheck.
- Changes limited to existing `test_*.py` files run those files, including any
  marked tests they contain. Their required environment must be available.
- Python source changes, shared test helpers/fixtures, and deleted tests run the
  portable suite. The hook does not guess a source-to-test dependency mapping.
- Dependency and pytest configuration changes run the full suite.
- Changes to the hook itself include its focused regression test.
- Ordinary documentation changes do not trigger pytest.

The hook is non-interactive and validates without fixing tracked files. The
orchestrator owns its Checks phase and recorded evidence; a Fixer's focused
validation does not replace that phase. Check selection does not change the
engine's review, recovery, or publication contracts.

## Pipeline execution

Use the owning YAML and resolver described in
[configuration guidance](configuration.md). Prefer a supported dry-run before
executing a stage, for example:

```bash
python proc/seis_ssl_cluster/build_nopims_manifests.py \
  --config proc/configs/seis_ssl_cluster/build_nopims_manifests.yaml --dry-run
```

Use the relevant experiment runbook for scientific controls, phase ordering,
checkpoint identity, and completion evidence.
